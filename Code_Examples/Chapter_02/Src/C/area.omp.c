#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

# define NPOINTS 4000
# define MAXITER 4000


struct complex{
  double real;
  double imag;
};

int main(){
  int i, j, iter, numoutside = 0;
  int num_threads;
  double area, error, ztemp;
  struct complex z, c;

  double start, end, cpu_time_used;
     
  start = omp_get_wtime();
/*
 *   
 *
 *     Outer loops run over npoints, initialise z=c
 *
 *     Inner loop has the iteration z=z*z+c, and threshold test
 */
#pragma omp parallel for default(shared) private(i,j,c,z,ztemp,iter) reduction(+:numoutside) collapse(2)
  for (i=0; i<NPOINTS; i++) {
    for (j=0; j<NPOINTS; j++) {
      c.real = -2.0+2.5*(double)(i)/(double)(NPOINTS)+1.0e-7;
      c.imag = 1.125*(double)(j)/(double)(NPOINTS)+1.0e-7;
      z=c;
      for (iter=0; iter<MAXITER; iter++){
	ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
	z.imag=z.real*z.imag*2+c.imag; 
	z.real=ztemp; 
	if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
	  numoutside++;
          break; 
	}
      }
    }
  }



  end = omp_get_wtime();
  cpu_time_used = ((double) (end - start));

#pragma omp parallel
{
  num_threads = omp_get_num_threads();
}


/*
 *  Calculate area and error and output the results
 */

   area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
   error=area/(double)NPOINTS;

   printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
   printf("Time taken for calculation: %f on %d threads\n",cpu_time_used,num_threads);    
}

