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
#include <tbb/tbb.h>


class pinning_observer: public tbb::task_scheduler_observer {
    cpu_set_t *mask;
    int ncpus;

    const int pinning_step;
    tbb::atomic<int> thread_index;
public:
    pinning_observer( int pinning_step=1 ) : pinning_step(pinning_step), thread_index() {
        for ( ncpus = sizeof(cpu_set_t)/CHAR_BIT; ncpus < 16*1024 /* some reasonable limit */; ncpus <<= 1 ) {
            mask = CPU_ALLOC( ncpus );
            if ( !mask ) break;
            const size_t size = CPU_ALLOC_SIZE( ncpus );
            CPU_ZERO_S( size, mask );
            const int err = sched_getaffinity( 0, size, mask );
            if ( !err ) break;

            CPU_FREE( mask );
            mask = NULL;
            if ( errno != EINVAL )  break;
        }
        if ( !mask )
            std::cout << "Warning: Failed to obtain process affinity mask. Thread affinitization is disabled." << std::endl;
    }

/*override*/ void on_scheduler_entry( bool ) {
    if ( !mask ) return;

    const size_t size = CPU_ALLOC_SIZE( ncpus );
    const int num_cpus = CPU_COUNT_S( size, mask );
    int thr_idx =
#if USE_TASK_ARENA_CURRENT_SLOT
        tbb::task_arena::current_slot();
#else
        thread_index++;
#endif
#if __MIC__
    thr_idx += 1; // To avoid logical thread zero for the master thread on Intel(R) Xeon Phi(tm)
#endif
    thr_idx %= num_cpus; // To limit unique number in [0; num_cpus-1] range

        // Place threads with specified step
        int cpu_idx = 0;
        for ( int i = 0, offset = 0; i<thr_idx; ++i ) {
            cpu_idx += pinning_step;
            if ( cpu_idx >= num_cpus )
                cpu_idx = ++offset;
        }

        // Find index of 'cpu_idx'-th bit equal to 1
        int mapped_idx = -1;
        while ( cpu_idx >= 0 ) {
            if ( CPU_ISSET_S( ++mapped_idx, size, mask ) )
                --cpu_idx;
        }

        cpu_set_t *target_mask = CPU_ALLOC( ncpus );
        CPU_ZERO_S( size, target_mask );
        CPU_SET_S( mapped_idx, size, target_mask );
        const int err = sched_setaffinity( 0, size, target_mask );

        if ( err ) {
            std::cout << "Failed to set thread affinity!n";
            exit( EXIT_FAILURE );
        }
#if LOG_PINNING
        else {
            std::stringstream ss;
            ss << "Set thread affinity: Thread " << thr_idx << ": CPU " << mapped_idx << std::endl;
            std::cerr << ss.str();
        }
#endif
        CPU_FREE( target_mask );
    }

    ~pinning_observer() {
        if ( mask )
            CPU_FREE( mask );
    }
};

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
void matvec(int nrows, const int* A_row_offsets, const int* A_cols, const double* A_vals, double* y, const double* x) {
  tbb::parallel_for(0,nrows,[&] (const int& row) {
    double sum = 0.0;

    int row_start=A_row_offsets[row];
    int row_end=A_row_offsets[row+1];
    for(int i=row_start; i<row_end; ++i) {
      sum += A_vals[i]*x[A_cols[i]];
    }
    y[row] = sum;
  });
}
#else
void matvec(int nrows, const int* A_row_offsets, const int* A_cols, const double* A_vals, double* y, const double* x) {
  tbb::parallel_for(0,nrows,[&] (const int& row) {
    int row_start=A_row_offsets[row];
    int row_end=A_row_offsets[row+1];
    y[row] = tbb::parallel_reduce(tbb::blocked_range<int>(row_start,row_end), 0.0,
    [&] (const tbb::blocked_range<int>& r, double lsum)->double {
      for(int i = r.begin(); i<r.end(); i++)
        lsum += A_vals[i]*x[A_cols[i]];
      return lsum;
    }, std::plus<double>()
    ); 
  });
}
#endif

double dot(int n, const double* x, const double* y) {
  return tbb::parallel_reduce(tbb::blocked_range<int>(0,n), 0.0, 
    [&] (const tbb::blocked_range<int>& r, double lsum)->double {
    for(int i = r.begin(); i<r.end(); i++)
      lsum += x[i]*y[i];
    return lsum;
  }, std::plus<double>() 
  );
}

void axpby(int n, double* z, double alpha, const double* x, double beta, const double* y) {
  tbb::parallel_for(0,n,[&] (const int& i) {
    z[i] = alpha*x[i] + beta*y[i];
  });
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

  pinning_observer pinner( 4 );
  pinner.observe( true );
  ScalarType* r = new double[nrows];
  ScalarType* p = new double[ncols];
  ScalarType* Ap = new double[nrows];
 
  ScalarType* x = &x_in.coefs[0];
  const ScalarType* b = &b_in.coefs[0];
  
  ScalarType* A_vals = &A.packed_coefs[0];
  GlobalOrdinalType* A_cols = &A.packed_cols[0];
  LocalOrdinalType* A_rows = &A.row_offsets[0]; 

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

