#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

# define NPOINTS 4000
# define MAXITER 4000


struct complex{
  double real;
  double imag;
};

int main(){
  int i, j, iter, numoutside = 0, found;
  double area, error, ztemp;
  struct complex z, c;

  clock_t start, end;
  double cpu_time_used;
     
  start = clock();
/*
 *   
 *
 *     Outer loops run over npoints, initialise z=c
 *
 *     Inner loop has the iteration z=z*z+c, and threshold test
 */
#pragma acc parallel loop reduction(+:numoutside) private(i,j) 
  for (i=0; i<NPOINTS; i++) {
#pragma acc loop vector private(c,z,found,c,z,iter,ztemp)
    for (j=0; j<NPOINTS; j++) {
      c.real = -2.0+2.5*(double)(i)/(double)(NPOINTS)+1.0e-7;
      c.imag = 1.125*(double)(j)/(double)(NPOINTS)+1.0e-7;
      z.real=c.real;
      z.imag=c.imag;
      found  = 0;
      iter = 0;
      while(iter<MAXITER && found == 0){
	ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
	z.imag=z.real*z.imag*2+c.imag; 
	z.real=ztemp; 
	if ((z.real*z.real+z.imag*z.imag)>4.0e0 && found == 0) {
	  numoutside++; 
          found = 1;
	}
	iter =  iter + 1;
      }
    }
  }



  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;



/*
 *  Calculate area and error and output the results
 */

   area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
   error=area/(double)NPOINTS;

   printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
   printf("Time taken for calculation: %f\n",cpu_time_used);    
}

