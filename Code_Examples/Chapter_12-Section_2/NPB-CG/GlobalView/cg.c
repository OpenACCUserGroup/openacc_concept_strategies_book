#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpinpb.h"
#include "timing.h"
#include "npbparams.h"
#include "../common/c_timers.h"
#include "../common/c_print_results.h"
#include "../common/c_rand.h"

#ifdef _TLOG
#include "tlog.h"
#define TLOG_LOG(log) do{tlog_log((log));}while(0)
#else
#define TLOG_LOG(log) do{}while(0)
#endif
/*** EVENT description
   2: Calculation w/o SpMV
   3: Scalar reduction
   4: SpMV
   5: Array Reduction
   6: Transpose
   7: Two element array reduction
  10: total
***/

//#define _XMPACC
#include <xmp.h> 
#pragma xmp nodes proc(num_proc_cols, num_proc_rows)
#pragma xmp nodes subproc(num_proc_cols) = proc(:,*)
#pragma xmp template t(0:na-1,0:na-1)
#pragma xmp distribute t(block, block) onto proc

static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
static int icnvrt(double x, int ipwr2);

static void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[],
		   double aelt[], int firstrow, int lastrow, double x[], int mark[],
		   int nzloc[], int nnza);
static void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], int mark[]);
static void setup_proc_info( int num_procs_, int num_proc_rows_, int num_proc_cols_ );
static void setup_submatrix_info(double [na], double [na]);
static void makea(int n, int nz, double a[], int colidx[], int rowstr[], int nonzer,
		  int firstrow, int lastrow, int firstcol, int lastcol,
		  double rcond, int arow[], int acol[], double aelt[], double v[], int iv[],
		  double shift);
static void conj_grad(int colidx[restrict],
		      int rowstr[restrict],
		      double x[restrict],
		      double z[restrict],
		      double a[restrict],
		      double p[restrict],
		      double q[restrict],
		      double r[restrict],
		      double w[restrict],
		      double * restrict rnorm);

static void initialize_mpi();
static void *alloc_mem(size_t elmtsize, int length);

double amult, tran;
int naa, nzz, 
    npcols, nprows,
    proc_col, proc_row,
    firstrow, 
    lastrow, 
    firstcol, 
    lastcol,
    exch_proc,
    exch_recv_length,
    send_start,
    send_len;
int nz;

static inline
int max(int a, int b){
  return (a > b)? a : b;
}

int main(int argc, char** argv){

/*--------------------------------------------------------------------
c  num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows.
c  num_proc_cols and num_proc_cols are to be found in npbparams.h.
c  When num_procs is not square, then num_proc_cols must be = 2*num_proc_rows.
c-------------------------------------------------------------------*/
  const int num_procs = num_proc_cols * num_proc_rows;

/*--------------------------------------------------------------------
c  Class specific parameters: 
c  It appears here for reference only.
c  These are their values, however, this info is imported in the npbparams.h
c  include file, which is written by the sys/setparams.c program.
c-------------------------------------------------------------------*/

/*---------
C  Class S:
C----------
CC       parameter( na=1400, 
CC      >           nonzer=7, 
CC      >           shift=10., 
CC      >           niter=15,
CC      >           rcond=1.0d-1 )
C----------
C  Class W:
C----------
CC       parameter( na=7000,
CC      >           nonzer=8, 
CC      >           shift=12., 
CC      >           niter=15,
CC      >           rcond=1.0d-1 )
C----------
C  Class A:
C----------
CC       parameter( na=14000,
CC      >           nonzer=11, 
CC      >           shift=20., 
CC      >           niter=15,
CC      >           rcond=1.0d-1 )
C----------
C  Class B:
C----------
CC       parameter( na=75000, 
CC      >           nonzer=13, 
CC      >           shift=60., 
CC      >           niter=75,
CC      >           rcond=1.0d-1 )
C----------
C  Class C:
C----------
CC       parameter( na=150000, 
CC      >           nonzer=15, 
CC      >           shift=110., 
CC      >           niter=75,
CC      >           rcond=1.0d-1 )
C----------
C  Class D:
C----------
CC       parameter( na=1500000, 
CC      >           nonzer=21, 
CC      >           shift=500., 
CC      >           niter=100,
CC      >           rcond=1.0d-1 )
C----------
C  Class E:
C----------
CC       parameter( na=9000000, 
CC      >           nonzer=26, 
CC      >           shift=1500., 
CC      >           niter=100,
CC      >           rcond=1.0d-1 ) */


  nz = na*(nonzer+1)/num_procs*(nonzer+1)+nonzer
                 + na*(nonzer+2+num_procs/256)/num_proc_cols;

  int *colidx, *rowstr, *iv, *arow, *acol;

  colidx = alloc_mem(sizeof(int), nz);
  rowstr = alloc_mem(sizeof(int), na+1);
  iv = alloc_mem(sizeof(int), 2*na+1);
  arow = alloc_mem(sizeof(int), nz);
  acol = alloc_mem(sizeof(int), nz);

  double * restrict v, * restrict aelt, * restrict a;
  v = alloc_mem(sizeof(double), na+1);
  aelt = alloc_mem(sizeof(double), nz);
  a = alloc_mem(sizeof(double), nz);

double p[na], x[na], z[na],w[na], q[na], r[na];
#pragma xmp align [i] with t(*,i) :: w
#pragma xmp align [i] with t(i,*) :: p, x, z, q, r

  int i, j, k, it;

  double zeta;
  double rnorm;
  double norm_temp1[2], norm_temp2[2];

  double t, tmax, mflops;
  char class;
  int verified; //bool
  double zeta_verify_value, epsilon, err;

  double tsum[t_last+2], t1[t_last+2], tming[t_last+2], tmaxg[t_last+2];
  char t_recs[6][9] = {"total", "conjg", "rcomm", "ncomm", " totcomp", " totcomm"}; //6 = t_last+2
  

/*--------------------------------------------------------------------
c  Set up mpi initialization and number of proc testing
c-------------------------------------------------------------------*/
  initialize_mpi();


  if( na == 1400 && 
      nonzer == 7 && 
      niter == 15 &&
      shift == 10.e0 ){
    class = 'S';
    zeta_verify_value = 8.5971775078648e0;
  }else if( na == 7000 && 
	    nonzer == 8 && 
	    niter == 15 &&
	    shift == 12.e0 ){
    class = 'W';
    zeta_verify_value = 10.362595087124e0;
  }else if( na == 14000 && 
	    nonzer == 11 && 
	    niter == 15 &&
	    shift == 20.e0 ){
    class = 'A';
    zeta_verify_value = 17.130235054029e0;
  }else if( na == 75000 && 
	    nonzer == 13 && 
	    niter == 75 &&
	    shift == 60.e0 ){
    class = 'B';
    zeta_verify_value = 22.712745482631e0;
  }else if( na == 150000 && 
	    nonzer == 15 && 
	    niter == 75 &&
	    shift == 110.e0 ){
    class = 'C';
    zeta_verify_value = 28.973605592845e0;
  }else if( na == 1500000 && 
	    nonzer == 21 && 
	    niter == 100 &&
	    shift == 500.e0 ){
    class = 'D';
    zeta_verify_value = 52.514532105794e0;
  }else if( na == 9000000 && 
	    nonzer == 26 && 
	    niter == 100 &&
	    shift == 1.5e3 ){
    class = 'E';
    zeta_verify_value = 77.522164599383e0;
  }else{
    class = 'U';
  }

  if( me == root ){
    printf("\n\n NAS Parallel Benchmarks 3.3 -- CG Benchmark\n\n");
    printf(" Size: %10d\n", na );
    printf(" Iterations: %5d\n", niter );
    printf(" Number of active processes: %5d\n", nprocs);
    printf(" Number of nonzeroes per row: %8d\n", nonzer);
    printf(" Eigenvalue shift: %8.3e\n", shift);
  }


  naa = na;
  nzz = nz;


/*--------------------------------------------------------------------
c  Set up processor info, such as whether sq num of procs, etc
c-------------------------------------------------------------------*/
  setup_proc_info( num_procs, 
		   num_proc_rows, 
		   num_proc_cols );


/*--------------------------------------------------------------------
c  Set up partition's submatrix info: firstcol, lastcol, firstrow, lastrow
c-------------------------------------------------------------------*/
  setup_submatrix_info(x, w);

   for( i = 1; i<= t_last; i++){
     timer_clear(i);
   }

/*--------------------------------------------------------------------
c  Inialize random number generator
c-------------------------------------------------------------------*/
   tran    = 314159265.0e0;
   amult   = 1220703125.0e0;
   zeta    = randlc( &(tran), &(amult) );

/*--------------------------------------------------------------------
c  Set up partition's sparse random matrix for given class size
c-------------------------------------------------------------------*/
   makea(naa, nzz, a, colidx, rowstr, nonzer,
	  firstrow, lastrow, firstcol, lastcol, 
	  rcond, arow, acol, aelt, v, iv, shift);



/*--------------------------------------------------------------------
c  Note: as a result of the above call to makea:
c        values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1
c        values of colidx which are col indexes go from firstcol --> lastcol
c        So:
c        Shift the col index vals from actual (firstcol --> lastcol ) 
c        to local, i.e., (1 --> lastcol-firstcol+1)
c-------------------------------------------------------------------*/
   //shift for xmp loop index. rowstr[firstrow-1:lastrow-firstrow+2] <= rowstr[0:lastrow-firstrow+2]
   for(j = lastrow - firstrow + 1; j >= 0; j--){
     rowstr[j + firstrow - 1] = rowstr[j];
   }

#pragma xmp loop on t(*,j)
   for(j=0; j < na; j++){
     for( k=rowstr[j]; k < rowstr[j+1]; k++){
       colidx[k] = colidx[k] - firstcol + 1;
     }
   }

/*--------------------------------------------------------------------
c  set starting vector to (1, 1, .... 1)
c-------------------------------------------------------------------*/
#pragma xmp loop on t(i,*)
   for( i = 0; i < na; i++){
     x[i] = 1.0e0;
   }

   zeta  = 0.0e0;

   int size_x = na/num_proc_rows+2; //for acc
   int size_rowstr = na + 1; //for acc
#pragma acc data pcopyin(colidx[0:nz], rowstr[0:size_rowstr], a[0:nz])  pcopyin(w, q, r, p, x, z)
   {
/*--------------------------------------------------------------------
c---->
c  Do one iteration untimed to init all code and data page tables
c---->                    (then reinit, start timing, to niter its)
c-------------------------------------------------------------------*/
   for( it = 1; it <= 1; it++){

/*--------------------------------------------------------------------
c  The call to the conjugate gradient routine:
c-------------------------------------------------------------------*/
     conj_grad( colidx,
		rowstr,
		x,
		z,
		a,
		p,
		q,
		r,
		w,
		&rnorm );

/*--------------------------------------------------------------------
c  zeta = shift + 1/(x.z)
c  So, first: (x.z)
c  Also, find norm of z
c  So, first: (z.z)
c-------------------------------------------------------------------*/
     double norm1 = 0.0e0, norm2 = 0.0e0;
     int size_x = na/num_proc_rows+2; //for acc
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:norm1,norm2) pcopy(x[0:size_x], z[0:size_x])
     for( j=0; j < na; j++){
       norm1 = norm1 + x[j]*z[j];
       norm2 = norm2 + z[j]*z[j];
     }
     norm_temp1[0] = norm1;
     norm_temp1[1] = norm2;

     if (timeron) timer_start(t_ncomm);
#pragma xmp reduction(+:norm_temp1) on subproc
     if (timeron) timer_stop(t_ncomm);

     norm_temp1[1] = 1.0e0 / sqrt( norm_temp1[1] );


/*--------------------------------------------------------------------
c  Normalize z to obtain x
c-------------------------------------------------------------------*/
     norm2 = norm_temp1[1];
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(x[0:size_x], z[0:size_x])
     for( j=0; j < na; j++){
       x[j] = norm2*z[j];    
     }


   } // end of do one iteration untimed


/*--------------------------------------------------------------------
c  set starting vector to (1, 1, .... 1)
c---------------------------------------------------------------------
c
c  NOTE: a questionable limit on size:  should this be na/num_proc_cols+1 ?
c */
#pragma xmp loop on t(i,*)
#pragma acc parallel loop pcopy(x[0:size_x])
   for( i = 0; i < na; i++){
     x[i] = 1.0e0;
   }

   zeta = 0.0e0;

/*--------------------------------------------------------------------
c  Synchronize and start timing
c-------------------------------------------------------------------*/
   for( i = 1; i <= t_last; i++){
     timer_clear(i);
   }
#ifdef _TLOG
   tlog_initialize();
#endif   
#pragma xmp barrier

   timer_clear( 1 );
   timer_start( 1 );
   TLOG_LOG(TLOG_EVENT_10_IN);
/*--------------------------------------------------------------------
c---->
c  Main Iteration for inverse power method
c---->
c-------------------------------------------------------------------*/
#if 0
   for( it = 1; it <= 15; it++){
#else
   for( it = 1; it <= niter; it++){
#endif

/*--------------------------------------------------------------------
c  The call to the conjugate gradient routine:
c-------------------------------------------------------------------*/
     conj_grad ( colidx,
		 rowstr,
		 x,
		 z,
		 a,
		 p,
		 q,
		 r,
		 w,
		 &rnorm );


/*--------------------------------------------------------------------
c  zeta = shift + 1/(x.z)
c  So, first: (x.z)
c  Also, find norm of z
c  So, first: (z.z)
c-------------------------------------------------------------------*/
     TLOG_LOG(TLOG_EVENT_2_IN);
     double norm1 = 0.0e0, norm2 = 0.0e0;
     int size_x = na/num_proc_rows+2; //for acc
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:norm1, norm2) pcopy(x[0:size_x], z[0:size_x])
     for( j=0; j < na; j++){
       norm1 = norm1 + x[j]*z[j];
       norm2 = norm2 + z[j]*z[j];
     }
     norm_temp1[0] = norm1;
     norm_temp1[1] = norm2;
     TLOG_LOG(TLOG_EVENT_2_OUT);

     TLOG_LOG(TLOG_EVENT_7_IN);
     if (timeron) timer_start(t_ncomm);
#pragma xmp reduction(+:norm_temp1) on subproc
     if (timeron) timer_stop(t_ncomm);
     TLOG_LOG(TLOG_EVENT_7_OUT);

     norm_temp1[1] = 1.0e0 / sqrt( norm_temp1[1] );


     if( me == root ){
       zeta = shift + 1.0e0 / norm_temp1[0];
       if( it == 1 ) printf("\n   iteration           ||r||                 zeta\n");
       printf("    %5d       %20.14e%20.13f\n", it, rnorm, zeta);
     }

/*--------------------------------------------------------------------
c  Normalize z to obtain x
c-------------------------------------------------------------------*/
     norm2 = norm_temp1[1];
     TLOG_LOG(TLOG_EVENT_2_IN);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(x[0:size_x], z[0:size_x])
     for( j=0; j < na; j++){
       x[j] = norm2*z[j];
     }
     TLOG_LOG(TLOG_EVENT_2_OUT);


   } // end of main iter inv pow meth

   TLOG_LOG(TLOG_EVENT_10_OUT);
   timer_stop( 1 );

   }//end of acc data

#ifdef _TLOG
   tlog_finalize();
#endif
/*--------------------------------------------------------------------
c  End of timed section
c-------------------------------------------------------------------*/

   tmax = t = timer_read( 1 );

#pragma xmp reduction(max:tmax)


   if( me == root ){
     printf(" Benchmark completed \n");

     epsilon = 1.e-10;
     if (class != 'U'){

       err = fabs( zeta - zeta_verify_value )/zeta_verify_value;
       if( err <= epsilon ){
	 verified = 1; //.TRUE.
	 printf(" VERIFICATION SUCCESSFUL \n");
	 printf(" Zeta is    %20.13e\n", zeta);
	 printf(" Error is   %20.13e\n", err);
       }else{
	 verified = 0; //.FALSE.
	 printf(" VERIFICATION FAILED\n");
	 printf(" Zeta                %20.13e\n", zeta);
	 printf(" The correct zeta is %20.13e\n", zeta_verify_value);
       }
     }else{
       verified = 0; //.FALSE.
       printf(" Problem size unknown\n");
       printf(" NO VERIFICATION PERFORMED\n");
     }


     if( tmax != 0. ){
       mflops = (double)( 2*niter*na )
	 * ( 3.+(double)( nonzer*(nonzer+1) )
	     + 25.*(5.+(double)( nonzer*(nonzer+1) ))
	     + 3. ) / tmax / 1000000.0;
     }else{
       mflops = 0.0;
     }
     c_print_results("CG", class, na, 0, 0,
		     niter, nnodes_compiled, (nprocs), tmax,
		     mflops, "          floating point", 
		     verified, NPBVERSION, COMPILETIME,
		     MPICC, CLINK, CMPI_LIB, CMPI_INC, CFLAGS, CLINKFLAGS);

   }


   if (! timeron) goto continue999;

   for( i = 1; i <= t_last; i++){
     t1[i-1] = timer_read(i);
   }
   t1[t_conjg-1] = t1[t_conjg-1] - t1[t_rcomm-1];
   t1[t_last+2-1] = t1[t_rcomm-1] + t1[t_ncomm-1];
   t1[t_last+1-1] = t1[t_total-1] - t1[t_last+2-1];

   for( i = 0; i < t_last+2; i++){
     tsum[i] = tming[i] = tmaxg[i] = t1[i];
   }
#pragma xmp reduction(+:tsum)
#pragma xmp reduction(min:tming)
#pragma xmp reduction(max:tmaxg)


   if (me == 0){
     printf(" nprocs =%6d           minimum     maximum     average\n", nprocs);
     for( i = 1; i <= t_last+2; i++){
       tsum[i-1] = tsum[i-1] / nprocs;
       printf(" timer %2d(%8s) :  %10.4f  %10.4f  %10.4f\n", i, t_recs[i-1], tming[i-1], tmaxg[i-1], tsum[i-1]);
     }
   }
 continue999:
   return 0;

} //! end main



void initialize_mpi()
{
  me = xmp_node_num() - 1;
  nprocs = xmp_num_nodes();
  root = 0;

  if (me == root){
    FILE *fp = fopen("timer.flag", "r");
    timeron = 0; // .false.
    if (fp != NULL){
      timeron = 1; // .true.
      fclose(fp);
    }
  }

#pragma xmp bcast(timeron)

  return;
}


void setup_proc_info( int num_procs,
		      int num_proc_rows_,
		      int num_proc_cols_ )
{
  int i;
  int log2nprocs;
/*--------------------------------------------------------------------
c  num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows
c  When num_procs is not square, then num_proc_cols = 2*num_proc_rows
c---------------------------------------------------------------------
c  First, number of procs must be power of two. 
c-------------------------------------------------------------------*/
  if( nprocs != num_procs ){
    if( me == root ){
      printf("Error: num of procs allocated (%d) is not equal to \
compiled number of procs (%d)", nprocs, num_procs);
    } 
    exit(1); //   stop
  }


  i = num_proc_cols;
 continue100:
  if( i != 1 && i/2*2 != i ){
    if ( me == root ){
      printf("Error: num_proc_cols is %d which is not a power of two\n", num_proc_cols);
    }
    exit(1);
  }
  i = i / 2;
  if( i != 0 ){
    goto continue100;
  }
      
  i = num_proc_rows;
 continue200:
  if( i != 1 && i/2*2 != i ){
    if ( me == root ){
      printf("Error: num_proc_rows is %d which is not a power of two\n", num_proc_rows);
    }
    exit(1);
  }
  i = i / 2;
  if( i != 0 ){
    goto continue200;
  }
      
  log2nprocs = 0;
  i = nprocs;
 continue300:
  if( i != 1 && i/2*2 != i ){
    printf("Error: nprocs is %d which is not a power of two\n", nprocs);
    exit(1);
  }
  i = i / 2;
  if( i != 0 ){
    log2nprocs = log2nprocs + 1;
    goto continue300;
  }
      
  npcols = num_proc_cols;
  nprows = num_proc_rows;

  return;
}


void setup_submatrix_info(double x[na], double w[na])
{
#pragma xmp align x[i] with t(i,*)
#pragma xmp align w[i] with t(*,i)

  firstcol = xmp_array_gcllbound(xmp_desc_of(x), 1) + 1;
  lastcol = xmp_array_gclubound(xmp_desc_of(x), 1) + 1;
  firstrow = xmp_array_gcllbound(xmp_desc_of(w), 1) + 1;
  lastrow = xmp_array_gclubound(xmp_desc_of(w), 1) + 1;

/*--------------------------------------------------------------------
c  If naa evenly divisible by npcols, then it is evenly divisible 
c  by nprows 
c-------------------------------------------------------------------*/

/*--------------------------------------------------------------------
c  If naa not evenly divisible by npcols, then first subdivide for nprows
c  and then, if npcols not equal to nprows (i.e., not a sq number of procs), 
c  get col subdivisions by dividing by 2 each row subdivision.
c-------------------------------------------------------------------*/

/*--------------------------------------------------------------------
c  Transpose exchange processor
c-------------------------------------------------------------------*/

/*--------------------------------------------------------------------
c  Set up the reduce phase schedules...
c-------------------------------------------------------------------*/

  return;
}



void conj_grad(int colidx[restrict],
	       int rowstr[restrict],
	       double x[restrict na],
	       double z[restrict na],
	       double a[restrict],
	       double p[restrict na],
	       double q[restrict na],
	       double r[restrict na],
	       double w[restrict na],
	       double * restrict rnorm)
{
#pragma xmp align [i] with t(*,i) :: w
#pragma xmp align [i] with t(i,*) :: p, x, z, q, r
#pragma xmp static_desc :: w, p, x, z, q, r
/*--------------------------------------------------------------------
c  Floaging point arrays here are named as in NPB1 spec discussion of 
c  CG algorithm
c-------------------------------------------------------------------*/
  int i,j;
  int cgit,cgitmax=25;
  double d, sum, rho, rho0, alpha, beta;

  if (timeron) timer_start(t_conjg);
  TLOG_LOG(TLOG_EVENT_2_IN);
  int size_x = na/num_proc_rows+2; //for acc
  int size_rowstr = na+1; //for acc
#pragma acc data pcopy(colidx[0:nz], rowstr[0:size_rowstr], a[0:nz]) pcopy(q, r, w, p, x, z)
  {
/*--------------------------------------------------------------------
c  Initialize the CG algorithm:
c-------------------------------------------------------------------*/
  TLOG_LOG(TLOG_EVENT_1);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(p[0:size_x], q[0:size_x], r[0:size_x], z[0:size_x], w[0:size_x], x[0:size_x])
  for(j=0; j < na; j++){
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
  }
#pragma xmp loop on t(*,j)
#pragma acc parallel loop pcopy(w[0:size_x])
  for (j = 0; j < na; j++){
    w[j] = 0.0;
  }


/*--------------------------------------------------------------------
c  rho = r.r
c  Now, obtain the norm of r: First, sum squares of r elements locally...
c-------------------------------------------------------------------*/
  sum = 0.0;
  TLOG_LOG(TLOG_EVENT_1);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:sum) pcopy(r[0:size_x])
  for(j=0; j< na; j++){
    sum = sum + r[j]*r[j];
  }

/*--------------------------------------------------------------------
c  Exchange and sum with procs identified in reduce_exch_proc
c  (This is equivalent to mpi_allreduce.)
c  Sum the partial sums of rho, leaving rho on all processors
c-------------------------------------------------------------------*/
  TLOG_LOG(TLOG_EVENT_2_OUT);
  TLOG_LOG(TLOG_EVENT_3_IN);
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:sum) on subproc
  if (timeron) timer_stop(t_rcomm);
  rho = sum;
  TLOG_LOG(TLOG_EVENT_3_OUT);


/*--------------------------------------------------------------------
c---->
c  The conj grad iteration loop
c---->
c-------------------------------------------------------------------*/
  for(cgit = 1; cgit <= cgitmax; cgit++){


/*--------------------------------------------------------------------
c  q = A.p
c  The partition submatrix-vector multiply: use workspace w
c-------------------------------------------------------------------*/
    TLOG_LOG(TLOG_EVENT_4_IN);
#pragma xmp loop on t(*,j)
#pragma acc parallel loop gang pcopy(a[0:nz], p[0:size_x], colidx[0:nz], rowstr[0:size_rowstr], w[0:size_x])
    for(int j=0; j < na; j++){
      double sum = 0.0;
      int rowstr_j = rowstr[j];
      int rowstr_j1 = rowstr[j+1];
#pragma acc loop vector reduction(+:sum)
      for(int k=rowstr_j; k < rowstr_j1; k++){
	sum = sum + a[k]*p[colidx[k]];
      }
      w[j] = sum;
    }
    TLOG_LOG(TLOG_EVENT_4_OUT);

/*--------------------------------------------------------------------
c  Sum the partition submatrix-vec A.p's across rows
c  Exchange and sum piece of w with procs identified in reduce_exch_proc
c-------------------------------------------------------------------*/
    TLOG_LOG(TLOG_EVENT_5_IN);

#ifdef _XMPACC
#pragma acc update host(w)
#pragma xmp reduction(+:w) on subproc(:)
#else
#pragma xmp reduction(+:w) on subproc(:) acc
#endif
    TLOG_LOG(TLOG_EVENT_5_OUT);
      

/*--------------------------------------------------------------------
c  Exchange piece of q with transpose processor:
c-------------------------------------------------------------------*/
    TLOG_LOG(TLOG_EVENT_6_IN);
    if (timeron) timer_start(t_rcomm);
#ifdef _XMPACC
#pragma xmp gmove
#else
#pragma xmp gmove acc
#endif
    q[:] = w[:];
#ifdef _XMPACC
#pragma acc update device(q)
#endif
    if (timeron) timer_stop(t_rcomm);
    TLOG_LOG(TLOG_EVENT_6_OUT);


    TLOG_LOG(TLOG_EVENT_2_IN);
/*--------------------------------------------------------------------
c  Clear w for reuse...
c-------------------------------------------------------------------*/
#pragma xmp loop on t(*,j)
#pragma acc parallel loop pcopy(w[0:size_x])
    for(j=0; j< na; j++){
      w[j] = 0.0;
    }
         

/*--------------------------------------------------------------------
c  Obtain p.q
c-------------------------------------------------------------------*/
    sum = 0.0;
    TLOG_LOG(TLOG_EVENT_1);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:sum) pcopy(p[0:size_x], q[0:size_x])
    for(j=0; j< na; j++){
      sum = sum + p[j]*q[j];
    }
    TLOG_LOG(TLOG_EVENT_2_OUT);

/*--------------------------------------------------------------------
c  Obtain d with a sum-reduce
c-------------------------------------------------------------------*/
    TLOG_LOG(TLOG_EVENT_3_IN);
    if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:sum) on subproc
    if (timeron) timer_stop(t_rcomm);
    d = sum;
    TLOG_LOG(TLOG_EVENT_3_OUT);


/*--------------------------------------------------------------------
c  Obtain alpha = rho / (p.q)
c-------------------------------------------------------------------*/
    alpha = rho / d;

/*--------------------------------------------------------------------
c  Save a temporary of rho
c-------------------------------------------------------------------*/
    rho0 = rho;

/*--------------------------------------------------------------------
c  Obtain z = z + alpha*p
c  and    r = r - alpha*q
c-------------------------------------------------------------------*/
    TLOG_LOG(TLOG_EVENT_2_IN);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(z[0:size_x], r[0:size_x], p[0:size_x], q[0:size_x]) 
    for(j=0; j< na; j++){
      z[j] = z[j] + alpha*p[j];
      r[j] = r[j] - alpha*q[j];
    }
            
/*--------------------------------------------------------------------
c  rho = r.r
c  Now, obtain the norm of r: First, sum squares of r elements locally...
c-------------------------------------------------------------------*/
    sum = 0.0;
    TLOG_LOG(TLOG_EVENT_1);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:sum) pcopy(r[0:size_x])
    for(j=0; j < na; j++){
      sum = sum + r[j]*r[j];
    }
    TLOG_LOG(TLOG_EVENT_2_OUT);

/*--------------------------------------------------------------------
c  Obtain rho with a sum-reduce
c-------------------------------------------------------------------*/
    TLOG_LOG(TLOG_EVENT_3_IN);
    if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:sum) on subproc
    if (timeron) timer_stop(t_rcomm);
    rho = sum;
    TLOG_LOG(TLOG_EVENT_3_OUT);

/*--------------------------------------------------------------------
c  Obtain beta:
c-------------------------------------------------------------------*/
    beta = rho / rho0;

/*--------------------------------------------------------------------
c  p = r + beta*p
c-------------------------------------------------------------------*/
    TLOG_LOG(TLOG_EVENT_2_IN);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(p[0:size_x],r[0:size_x])
    for(j=0; j< na; j++){
      p[j] = r[j] + beta*p[j];
    }
    TLOG_LOG(TLOG_EVENT_2_OUT);



  }       // end of do cgit=1,cgitmax



/*--------------------------------------------------------------------
c  Compute residual norm explicitly:  ||r|| = ||x - A.z||
c  First, form A.z
c  The partition submatrix-vector multiply
c-------------------------------------------------------------------*/
  int size_rowstr = na+1; //for acc
  TLOG_LOG(TLOG_EVENT_4_IN);
#pragma xmp loop on t(*,j)
#pragma acc parallel loop gang pcopy(a[0:nz], z[0:size_x], colidx[0:nz], rowstr[0:size_rowstr], w[0:size_x])
  for(int j=0; j < na; j++){
    double sum = 0.0;
    int rowstr_j = rowstr[j];
    int rowstr_j1 = rowstr[j+1];
#pragma acc loop vector reduction(+:sum)
    for(int k=rowstr_j; k < rowstr_j1; k++){
      sum = sum + a[k]*z[colidx[k]];
    }
    w[j] = sum;
  }
  TLOG_LOG(TLOG_EVENT_4_OUT);



/*--------------------------------------------------------------------
c  Sum the partition submatrix-vec A.z's across rows
c-------------------------------------------------------------------*/
  TLOG_LOG(TLOG_EVENT_5_IN);
#ifdef _XMPACC
  if (timeron) timer_start(t_rcomm);
#pragma acc update host(w)
#pragma xmp reduction(+:w) on subproc(:)
  if (timeron) timer_stop(t_rcomm);
#else
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:w) on subproc(:) acc
  if (timeron) timer_stop(t_rcomm);
#endif
  TLOG_LOG(TLOG_EVENT_5_OUT);
      

/*--------------------------------------------------------------------
c  Exchange piece of q with transpose processor:
c-------------------------------------------------------------------*/
  TLOG_LOG(TLOG_EVENT_6_IN);
  if (timeron) timer_start(t_rcomm);
#ifdef _XMPACC
#pragma xmp gmove
#else
#pragma xmp gmove acc
#endif
  r[:] = w[:];
#ifdef _XMPACC
#pragma acc update device(r)
#endif
  if (timeron) timer_stop(t_rcomm);
  TLOG_LOG(TLOG_EVENT_6_OUT);


/*--------------------------------------------------------------------
c  At this point, r contains A.z
c-------------------------------------------------------------------*/
  sum = 0.0;
  TLOG_LOG(TLOG_EVENT_2_IN);
#pragma xmp loop on t(j,*)
#pragma acc parallel loop private(d) reduction(+:sum) pcopy(x[0:size_x], r[0:size_x])
  for(j=0; j < na; j++){
    d   = x[j] - r[j];
    sum = sum + d*d;
  }
  TLOG_LOG(TLOG_EVENT_2_OUT);
         
/*--------------------------------------------------------------------
c  Obtain d with a sum-reduce
c-------------------------------------------------------------------*/
  TLOG_LOG(TLOG_EVENT_3_IN);
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:sum) on subproc
  if (timeron) timer_stop(t_rcomm);
  d = sum;
  TLOG_LOG(TLOG_EVENT_3_OUT);


  if( me == root ) *rnorm = sqrt( d );
} //end of acc data

  if (timeron) timer_stop(t_conjg);


} //      ! end of routine conj_grad


void makea(int n, int nz, double a[], int colidx[], int rowstr[], int nonzer,
	   int firstrow, int lastrow, int firstcol, int lastcol,
	   double rcond, int arow[], int acol[], double aelt[], double v[], int iv[],
	   double shift)
{
/*--------------------------------------------------------------------
c       generate the test problem for benchmark 6
c       makea generates a sparse matrix with a
c       prescribed sparsity distribution
c
c       parameter    type        usage
c
c       input
c
c       n            i           number of cols/rows of matrix
c       nz           i           nonzeros as declared array size
c       rcond        r*8         condition number
c       shift        r*8         main diagonal shift
c
c       output
c
c       a            r*8         array for nonzeros
c       colidx       i           col indices
c       rowstr       i           row pointers
c
c       workspace
c
c       iv, arow, acol i
c       v, aelt        r*8
c-------------------------------------------------------------------*/

  int i, nnza, iouter, ivelt, ivelt1, irow, nzv, jcol;

/*--------------------------------------------------------------------
c      nonzer is approximately  (int(sqrt(nnza /n)));
c-------------------------------------------------------------------*/

  double size, ratio, scale;
  size = 1.0;
  ratio = pow(rcond, 1.0 / (double)(n)); //*rcond ** (1.0 / dfloat(n));
  nnza = 0;

/*--------------------------------------------------------------------
c  Initialize iv(n+1 .. 2n) to zero.
c  Used by sprnvc to mark nonzero positions
c-------------------------------------------------------------------*/

  for( i = 1; i <= n; i++){
    iv[n+i-1] = 0;
  }
  for(iouter = 1; iouter <= n; iouter++){
    nzv = nonzer;
    sprnvc( n, nzv, v, colidx, &iv[1-1], &iv[n+1-1] );
    vecset( n, v, colidx, &nzv, iouter, 0.5);
    for(ivelt = 1; ivelt <= nzv; ivelt++){
      jcol = colidx[ivelt-1];
      if (jcol >= firstcol && jcol <= lastcol){
	scale = size * v[ivelt-1];
	for(ivelt1 = 1; ivelt1 <= nzv; ivelt1++){
	  irow = colidx[ivelt1-1];
	  if (irow >= firstrow && irow <= lastrow){
	    nnza = nnza + 1;
	    if (nnza > nz) goto continue9999;
	    acol[nnza-1] = jcol;
	    arow[nnza-1] = irow;
	    aelt[nnza-1] = v[ivelt1-1] * scale;
	  }
	}
      }
    }
    size = size * ratio;
  }


/*--------------------------------------------------------------------
c       ... add the identity * rcond to the generated matrix to bound
c           the smallest eigenvalue from below by rcond
c-------------------------------------------------------------------*/
  for(i = firstrow; i <= lastrow; i++){
    if (i >= firstcol && i <= lastcol){
      iouter = n + i;
      nnza = nnza + 1;
      if (nnza > nz) goto continue9999;
      acol[nnza-1] = i;
      arow[nnza-1] = i;
      aelt[nnza-1] = rcond - shift;
    }
  }


/*--------------------------------------------------------------------
c       ... make the sparse matrix from list of elements with duplicates
c           (v and iv are used as  workspace)
c-------------------------------------------------------------------*/
  sparse( a, colidx, rowstr, n, arow, acol, aelt,
	  firstrow, lastrow,
	  v, &iv[1-1], &iv[n+1-1], nnza );
  return;

 continue9999:
  printf("Space for matrix elements exceeded in makea\n");
  int nzmax = -1;
  printf("nnza, nzmax = %d %d\n", nnza, nzmax);
  printf(" iouter = %d\n", iouter);
}


void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[],
	    double aelt[], int firstrow, int lastrow, double x[], int mark[],
	    int nzloc[], int nnza)
{
/*--------------------------------------------------------------------
c       rows range from firstrow to lastrow
c       the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
c-------------------------------------------------------------------*/
  int nrows;
/*--------------------------------------------------
c       generate a sparse matrix from a list of
c       [col, row, element] tri
c-------------------------------------------------*/
  int i,j, jajp1, nza, k, nzrow;
  double xi;
/*--------------------------------------------------------------------
c    how many rows of result
c-------------------------------------------------------------------*/
  nrows = lastrow - firstrow + 1;

/*--------------------------------------------------------------------
c     ...count the number of triples in each row
c-------------------------------------------------------------------*/
  for(j = 0; j < n; j++){
    rowstr[j] = 0;
    mark[j] = 0; //false
  }
  rowstr[n] = 0;

  for(nza = 0; nza < nnza; nza++){
    j = (arow[nza] - firstrow + 1);
    rowstr[j] = rowstr[j] + 1;
  }

  rowstr[0] = 0; //set 0 for 0-basing
  for( j = 1; j < nrows+1; j++){
    rowstr[j] = rowstr[j] + rowstr[j-1];
  }


/*--------------------------------------------------------------------
c     ... rowstr(j) now is the location of the first nonzero
c           of row j of a
c-------------------------------------------------------------------*/


/*--------------------------------------------------------------------
c     ... do a bucket sort of the triples on the row index
c-------------------------------------------------------------------*/
  for( nza = 0; nza < nnza; nza++){
    j = arow[nza] - firstrow;
    k = rowstr[j]; // - 1;
    a[k] = aelt[nza];
    colidx[k] = acol[nza] - 1; //0-basing
    rowstr[j] = rowstr[j] + 1;
  }


/*--------------------------------------------------------------------
c       ... rowstr(j) now points to the first element of row j+1
c-------------------------------------------------------------------*/
  for(j = nrows - 1; j >= 0; j--){
    rowstr[j+1] = rowstr[j];
  }
  rowstr[0] = 0; //0-basing  1;


/*--------------------------------------------------------------------
c       ... generate the actual output rows by adding elements
c-------------------------------------------------------------------*/
  nza = 0;
  for(i = 0; i < n; i++){
    x[i] = 0.0;
    mark[i] = 0; //.false.
  }

  jajp1 = rowstr[0] + 1;
  for(j = 0; j < nrows; j++){
    nzrow = 0;

/*--------------------------------------------------------------------
c          ...loop over the jth row of a
c-------------------------------------------------------------------*/
    for( k = jajp1 - 1 ; k < rowstr[j+1]; k++){
      i = colidx[k];
      x[i] = x[i] + a[k];
      if ( (! mark[i]) && (x[i] != 0.0)){
	mark[i] = 1; //.true.
	nzrow = nzrow + 1;
	nzloc[nzrow-1] = i + 1;
      }
    }

/*--------------------------------------------------------------------
c          ... extract the nonzeros of this row
c-------------------------------------------------------------------*/
    for( k = 0; k < nzrow;k++){
      i = nzloc[k] - 1;
      mark[i] = 0; //.false.
      xi = x[i];
      x[i] = 0.0;
      if (xi != 0.0) {
	nza = nza + 1;
	a[nza-1] = xi;
	colidx[nza-1] = i;
      }
    }
    jajp1 = rowstr[j+1]+1;
    rowstr[j+1] = nza + rowstr[0];
  }
}


void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], int mark[])
{
/*--------------------------------------------------------------------
c       generate a sparse n-vector (v, iv)
c       having nzv nonzeros
c
c       mark(i) is set to 1 if position i is nonzero.
c       mark is all zero on entry and is reset to all zero before exit
c       this corrects a performance bug found by John G. Lewis, caused by
c       reinitialization of mark on every one of the n calls to sprnvc
c-------------------------------------------------------------------*/
  int nzrow, nzv, ii, i, nn1;
  double vecelt, vecloc;

  nzv = 0;
  nzrow = 0;
  nn1 = 1;
 continue50:
  nn1 = 2 * nn1;
  if (nn1 < n) goto continue50;

/*--------------------------------------------------------------------
c    nn1 is the smallest power of two not less than n
c-------------------------------------------------------------------*/

 continue100:
  if (nzv >= nz) goto continue110;
  vecelt = randlc( &(tran), &(amult) );

/*--------------------------------------------------------------------
c   generate an integer between 1 and n in a portable manner
c-------------------------------------------------------------------*/
  vecloc = randlc( &(tran), &(amult));
  i = icnvrt(vecloc, nn1) + 1;
  if (i > n) goto continue100;

/*--------------------------------------------------------------------
c  was this integer generated already?
c-------------------------------------------------------------------*/
  if (mark[i-1] == 0){
    mark[i-1] = 1;
    nzrow = nzrow + 1;
    nzloc[nzrow-1] = i;
    nzv = nzv + 1;
    v[nzv-1] = vecelt;
    iv[nzv-1] = i;
  }
  goto continue100;
 continue110:
  for(ii = 1; ii <= nzrow; ii++){
    i = nzloc[ii-1];
    mark[i-1] = 0;
  }
}


int icnvrt(double x, int ipwr2)
{
/*--------------------------------------------------------------------
c    scale a double precision number x in (0,1) by a power of 2 and chop it
c-------------------------------------------------------------------*/
  int icnvrt = (int)(ipwr2 * x);

  return icnvrt;
}


void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
/*--------------------------------------------------------------------
c       set ith element of sparse vector (v, iv) with
c       nzv nonzeros to val
c-------------------------------------------------------------------*/
  int k, set;

  set = 0; //.false.
  for(k = 0; k < *nzv; k++){
    if(iv[k] == i){
      v[k] = val;
      set = 1; //.true.
    }
  }
  if(! set){
    v[*nzv] = val;
    iv[*nzv] = i;
    (*nzv) = (*nzv) + 1;
  }
}


void *alloc_mem(size_t elmtsize, int length)
{
  void *p = malloc(elmtsize * length);
  if(p == NULL){
    printf("Error: cannot allocate memory\n");
    exit(1);
  }
  return p;
}
