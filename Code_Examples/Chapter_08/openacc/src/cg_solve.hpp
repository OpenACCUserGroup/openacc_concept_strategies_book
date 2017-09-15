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
#include <omp.h>
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

#ifdef MINIFE_USE_FLAT_SPMV
void matvec(int nrows, int nnz, const int* A_row_offsets, const int* A_cols, const double* A_vals, double* y, const double* x) {
  #pragma acc parallel loop gang, vector\
      present(y[0:nrows], x[0:nrows], A_row_offsets[0:nrows+1], A_cols[0:nnz], A_vals[0:nnz])
  for(int row=0; row<nrows; ++row) {
    const int row_start=A_row_offsets[row];
    const int row_end=A_row_offsets[row+1];

    double sum = 0.0;
    for(int i=row_start; i<row_end; ++i) {
      sum += A_vals[i]*x[A_cols[i]];
    }
    y[row] = sum;
  }
}
#else
void matvec(int nrows, int nnz, const int* A_row_offsets, const int* A_cols, const double* A_vals, double* y, const double* x) {
  #pragma acc parallel num_workers(64) vector_length(8) \
      present(y[0:nrows], x[0:nrows], A_row_offsets[0:nrows+1], A_cols[0:nnz], A_vals[0:nnz])
  {
    #pragma acc loop gang worker
    for(int row=0; row<nrows; ++row) {
      const int row_start=A_row_offsets[row];
      const int row_end=A_row_offsets[row+1];

      double sum = 0.0;
      #pragma acc loop vector reduction(+: sum)
      for(int i=row_start; i<row_end; ++i) {
        sum += A_vals[i]*x[A_cols[i]];
      }
      y[row] = sum;
    }
  }
}
#endif

double dot(int n, const double* x, const double* y) {
  double sum = 0.0;
  #pragma acc parallel loop reduction(+: sum) \
     present(x[0:n],y[0:n])
  for(int i=0; i<n; i++)
    sum += x[i]*y[i];
  return sum;
}

void axpby(int n, double* z, double alpha, const double* x, double beta, const double* y) {
  #pragma acc parallel loop \ 
     present(x[0:n],y[0:n],z[0:n])
  for(int i=0; i<n; i++)
    z[i] = alpha*x[i] + beta*y[i];
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
  LocalOrdinalType ncols = A.num_cols;

  ScalarType* r = new double[nrows];
  ScalarType* p = new double[ncols];
  ScalarType* Ap = new double[nrows];
 
  ScalarType* x = &x_in.coefs[0];
  const ScalarType* b = &b_in.coefs[0];
 
  int nnz = A.packed_coefs.size(); 
  ScalarType* A_vals = &A.packed_coefs[0];
  GlobalOrdinalType* A_cols = &A.packed_cols[0];
  LocalOrdinalType* A_rows = &A.row_offsets[0]; 

  #pragma acc data \
     copyin(r[0:nrows], p[0:nrows],Ap[0:nrows], b[0:nrows], \
            A_vals[0:nnz], A_cols[0:nnz], A_rows[0:nrows+1]) \
     copy(x[0:nrows])
  {
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
  matvec(nrows,nnz,A_rows, A_cols, A_vals, Ap, p);
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

    TICK(); matvec(nrows,nnz,A_rows, A_cols, A_vals, Ap, p); TOCK(tMATVEC);

    TICK(); p_ap_dot = dot(nrows, Ap, p); TOCK(tDOT);

    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 ) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
        //update the timers before jumping out.
        my_cg_times[WAXPY] = tWAXPY;
        my_cg_times[DOT] = tDOT;
        my_cg_times[MATVEC] = tMATVEC;
        my_cg_times[TOTAL] = mytimer() - total_time;
      }
      else brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans/p_ap_dot;
    
    TICK(); axpby(nrows, x, one, x, alpha, p);
            axpby(nrows, r, one, r, -alpha, Ap); TOCK(tWAXPY);

    num_iters = k;
  }
  }
  my_cg_times[WAXPY] = tWAXPY;
  my_cg_times[DOT] = tDOT;
  my_cg_times[MATVEC] = tMATVEC;
  my_cg_times[MATVECDOT] = tMATVECDOT;
  my_cg_times[TOTAL] = mytimer() - total_time;
}

}//namespace miniFE

#endif

