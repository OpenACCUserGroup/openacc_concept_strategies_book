#ifndef _cg_solve_hpp_
#define _cg_solve_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// ************************************************************************
//@HEADER

#include <cmath>
#include <limits>

#include <Vector_functions.hpp>
#include <mytimer.hpp>

#include <outstream.hpp>

#include <cuda_runtime.h>
namespace miniFE {

template<typename Scalar>
void print_vec(const std::vector<Scalar>& vec, const std::string& name)
{
  for(size_t i=0; i<vec.size(); ++i) {
    std::cout << name << "["<<i<<"]: " << vec[i] << std::endl;
  }
}

template<typename VectorType>
bool breakdown(typename VectorType::ScalarType inner,
               const VectorType& v,
               const VectorType& w)
{
  typedef typename VectorType::ScalarType Scalar;
  typedef typename TypeTraits<Scalar>::magnitude_type magnitude;

//This is code that was copied from Aztec, and originally written
//by my hero, Ray Tuminaro.
//
//Assuming that inner = <v,w> (inner product of v and w),
//v and w are considered orthogonal if
//  |inner| < 100 * ||v||_2 * ||w||_2 * epsilon

  magnitude vnorm = std::sqrt(dot(v,v));
  magnitude wnorm = std::sqrt(dot(w,w));
  return std::abs(inner) <= 100*vnorm*wnorm*std::numeric_limits<magnitude>::epsilon();
}

__global__ void matvec_kernel(int nrows, const int* A_row_offsets, const int* A_cols, const double* A_vals, double* y, const double* x) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if(row>=nrows) return;

  int row_start=A_row_offsets[row];
  int row_end=A_row_offsets[row+1];
  double sum = 0.0;
  for(int i=row_start + threadIdx.x; i<row_end; i+=blockDim.x) 
    sum += A_vals[i]*x[A_cols[i]];
  // Reduce over blockDim.x
  int delta = 1;
  while(delta < blockDim.x) {
    sum += __shfl_down(sum,delta,blockDim.x);
    delta*=2;
  }
  if(threadIdx.x == 0)
    y[row] = sum;
}

void matvec(int nrows, const int* A_row_offsets, const int* A_cols, const double* A_vals, double* y, const double* x) {
  dim3 blocks((nrows+63)/64,1,1);
  dim3 threads(8,64,1);
  matvec_kernel<<<blocks,threads>>>(nrows, A_row_offsets, A_cols, A_vals, y, x);
  cudaDeviceSynchronize();
}

void __global__ dot_kernel(int n, const double* x, const double* y, double* buffer) {
  int i_start = blockIdx.x*blockDim.x + threadIdx.x;
  double sum = 0;
  for(int i=i_start; i<n; i+=gridDim.x*blockDim.x)
    sum += x[i]*y[i];

  // Do per Block Reduction
  __shared__ double local_buffer[256];
  local_buffer[threadIdx.x] = sum;
  int delta = 1;
  while(delta<256) {
    __syncthreads();
    if(threadIdx.x%(delta*2)==0)
    local_buffer[threadIdx.x] += local_buffer[threadIdx.x+delta];
    delta*=2;
  }
  if(threadIdx.x==0)
    buffer[blockIdx.x] = local_buffer[0];
}

double dot(int n, const double* x, const double* y) {
  static double * buffer = NULL;
  if(buffer == NULL) {
    cudaMallocHost(&buffer,4096*sizeof(double));
  }
  
  int nblocks = 4096 * 256 < n?4096:(n+255)/256;
  dot_kernel<<<nblocks,256>>> (n,x,y,buffer);
  cudaDeviceSynchronize();
  
  double sum = 0.0;
  for(int i=0; i<nblocks; i++)
    sum += buffer[i];

  return sum;
}

void __global__ axpby_kernel(int n, double* z, double alpha, const double* x, double beta, const double* y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n)
    z[i] = alpha*x[i] + beta*y[i];
}

void axpby(int n, double* z, double alpha, const double* x, double beta, const double* y) {
  axpby_kernel<<<(n+255)/256,256>>>(n,z,alpha,x,beta,y);
  cudaDeviceSynchronize();
}

template<typename OperatorType,
         typename VectorType,
         typename Matvec>
void
cg_solve(OperatorType& A,
         const VectorType& b_in,
         VectorType& x_in,
         Matvec MatVec,
         typename OperatorType::LocalOrdinalType max_iter,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance,
         typename OperatorType::LocalOrdinalType& num_iters,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr,
         timer_type* my_cg_times)
{
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
  typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
  typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  timer_type t0 = 0, tWAXPY = 0, tDOT = 0, tMATVEC = 0, tMATVECDOT = 0;
  timer_type total_time = mytimer();

  int myproc = 0;

  if (!A.has_local_indices) {
    std::cerr << "miniFE::cg_solve ERROR, A.has_local_indices is false, needs to be true. This probably means "
       << "miniFE::make_local_matrix(A) was not called prior to calling miniFE::cg_solve."
       << std::endl;
    return;
  }

  size_t nrows = A.rows.size();
  size_t A_size = A.packed_coefs.size();
  LocalOrdinalType ncols = A.num_cols;

  double* h_r = new double[nrows];
  double* h_p = new double[ncols];
  double* h_Ap = new double[nrows];
 
  double* h_x = &x_in.coefs[0];
  const double* h_b = &b_in.coefs[0];
  
  double* h_A_vals = &A.packed_coefs[0];
  int* h_A_cols = &A.packed_cols[0];
  int* h_A_rows = &A.row_offsets[0]; 


  // Allocate Device Data
  double* r; cudaMalloc(&r,nrows*sizeof(double));
  double* p; cudaMalloc(&p,nrows*sizeof(double));
  double* Ap; cudaMalloc(&Ap,nrows*sizeof(double));
  double* x; cudaMalloc(&x,nrows*sizeof(double));
  double* b; cudaMalloc(&b,nrows*sizeof(double));
  double* A_vals; cudaMalloc(&A_vals,A_size*sizeof(double));
  int* A_cols; cudaMalloc(&A_cols,A_size*sizeof(int));
  int* A_rows; cudaMalloc(&A_rows,(nrows+1)*sizeof(int));

  
  // Copy to Device

  cudaMemcpy(r,h_r,nrows*sizeof(double),cudaMemcpyDefault);
  cudaMemcpy(p,h_p,nrows*sizeof(double),cudaMemcpyDefault);
  cudaMemcpy(Ap,h_Ap,nrows*sizeof(double),cudaMemcpyDefault);
  cudaMemcpy(x,h_x,nrows*sizeof(double),cudaMemcpyDefault);
  cudaMemcpy(b,h_b,nrows*sizeof(double),cudaMemcpyDefault);
  cudaMemcpy(A_vals,h_A_vals,A_size*sizeof(double),cudaMemcpyDefault);
  cudaMemcpy(A_cols,h_A_cols,A_size*sizeof(int),cudaMemcpyDefault);
  cudaMemcpy(A_rows,h_A_rows,(nrows+1)*sizeof(int),cudaMemcpyDefault);

  normr = 0;
  magnitude_type rtrans = 0;
  magnitude_type oldrtrans = 0;

  LocalOrdinalType print_freq = max_iter/10;
  if (print_freq>50) print_freq = 50;
  if (print_freq<1)  print_freq = 1;

  ScalarType one = 1.0;
  ScalarType zero = 0.0;

  TICK(); axpby(nrows, p, one, x, zero, x); TOCK(tWAXPY);

  TICK();
  matvec(nrows,A_rows, A_cols, A_vals, Ap, p);
  TOCK(tMATVEC);

  TICK(); axpby(nrows,r, one, b, -one, Ap); TOCK(tWAXPY);

  TICK(); rtrans = dot(nrows,r, r); TOCK(tDOT);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

  for(LocalOrdinalType k=1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      TICK(); axpby(nrows, p, one, r, zero, r); TOCK(tWAXPY);
    }
    else {
      oldrtrans = rtrans;
      TICK(); rtrans = dot(nrows, r, r); TOCK(tDOT);
      magnitude_type beta = rtrans/oldrtrans;
      TICK(); axpby(nrows, p, one, r, beta, p); TOCK(tWAXPY);
    }

    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    TICK(); matvec(nrows,A_rows, A_cols, A_vals, Ap, p); TOCK(tMATVEC);

    TICK(); p_ap_dot = dot(nrows, Ap, p); TOCK(tDOT);

    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 ) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
        //update the timers before jumping out.
        my_cg_times[WAXPY] = tWAXPY;
        my_cg_times[DOT] = tDOT;
        my_cg_times[MATVEC] = tMATVEC;
        my_cg_times[TOTAL] = mytimer() - total_time;
        return;
      }
      else brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans/p_ap_dot;
    
    TICK(); axpby(nrows, x, one, x, alpha, p);
            axpby(nrows, r, one, r, -alpha, Ap); TOCK(tWAXPY);

    num_iters = k;
  }

  my_cg_times[WAXPY] = tWAXPY;
  my_cg_times[DOT] = tDOT;
  my_cg_times[MATVEC] = tMATVEC;
  my_cg_times[MATVECDOT] = tMATVECDOT;
  my_cg_times[TOTAL] = mytimer() - total_time;

}

}//namespace miniFE

#endif

