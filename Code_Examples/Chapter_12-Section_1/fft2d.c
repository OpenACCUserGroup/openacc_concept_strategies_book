#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "fft_config.h"

#ifndef DELAY_ELEMENTS_ON_LOCAL
#define DELAY_ELEMENTS_ON_LOCAL 1
#endif

#ifndef COALESCED_BUFFERING
#define COALESCED_BUFFERING 0
#endif

// Function prototypes
static void test_fft(int mangle, int inverse);
static void fetch(flt2 * src, int mangle);
static void fft2d(int inverse);
static void transpose(flt2 * dest, int mangle);
static int coord(int iteration, int i);
static void fourier_transform_gold(int inverse, int lognr_points, dbl2 * data);
static void fourier_stage(int lognr_points, dbl2 * data);
static int mangle_bits(int x);
double my_timer();

// Host memory buffers
flt2 *h_inData, *h_outData, *h_tmp;
dbl2 *h_verify, *h_verify_tmp;
flt2 *chan0, *chan1, *chan2, *chan3, *chan4, *chan5, *chan6, *chan7;
flt2 *chanin0, *chanin1, *chanin2, *chanin3, *chanin4, *chanin5, *chanin6, *chanin7;
#pragma acc declare pipe(chan0[0:N*N/8], chan1[0:N*N/8], chan2[0:N*N/8], chan3[0:N*N/8], chan4[0:N*N/8], chan5[0:N*N/8], chan6[0:N*N/8], chan7[0:N*N/8], chanin0[0:N*N/8], chanin1[0:N*N/8], chanin2[0:N*N/8], chanin3[0:N*N/8], chanin4[0:N*N/8], chanin5[0:N*N/8], chanin6[0:N*N/8], chanin7[0:N*N/8])

int main( int argc, char** argv ) {
	printf("==> Input LOGN : %d\n", LOGN);
	// Allocate host memory
	posix_memalign((void **)(&h_inData), AOCL_ALIGNMENT, sizeof(flt2)*N*N);
	posix_memalign((void **)(&h_outData), AOCL_ALIGNMENT, sizeof(flt2)*N*N);
	posix_memalign((void **)(&h_tmp), AOCL_ALIGNMENT, sizeof(flt2)*N*N);
	posix_memalign((void **)(&h_verify), AOCL_ALIGNMENT, sizeof(dbl2)*N*N);
	posix_memalign((void **)(&h_verify_tmp), AOCL_ALIGNMENT, sizeof(dbl2)*N*N);
	posix_memalign((void **)(&chan0), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chan1), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chan2), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chan3), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chan4), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chan5), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chan6), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chan7), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin0), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin1), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin2), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin3), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin4), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin5), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin6), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	posix_memalign((void **)(&chanin7), AOCL_ALIGNMENT, sizeof(flt2)*N*N/8);
	if (!(h_inData && h_outData && h_verify && h_verify_tmp)) {
		printf("ERROR: Couldn't create host buffers\n");
		return _false;
	}

	test_fft(_false, _false); // test FFT transform with ordered memory layout
	test_fft(_false, _true); // test inverse FFT transform with ordered memory layout

	test_fft(_true, _false); // test FFT transform with alternative memory layout
	test_fft(_true, _true); // test inverse FFT transform with alternative memory layout

	// Free the resources allocated
	free(h_inData);
	free(h_outData);
	free(h_tmp);
	free(h_verify);
	free(h_verify_tmp);
	free(chan0);
	free(chan1);
	free(chan2);
	free(chan3);
	free(chan4);
	free(chan5);
	free(chan6);
	free(chan7);
	free(chanin0);
	free(chanin1);
	free(chanin2);
	free(chanin3);
	free(chanin4);
	free(chanin5);
	free(chanin6);
	free(chanin7);

	return 0;
}

void test_fft(int mangle, int inverse) {
	double time;
	double totaltime;
	int i, j;
	double gpoints_per_sec;
	double gflops;
	double fpga_snr = 200;
	double mag_sum = 0;
	double noise_sum = 0;
	int where;
	double magnitude;
	double noise;
	double db;

	printf("Launching %sFFT transform (%s data layout)\n", inverse ? "inverse " : "", mangle ? "alternative" : "ordered");

	// Initialize input and produce verification data
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			where = mangle ? mangle_bits(coord(i, j)) :  coord(i, j);
			h_verify[coord(i, j)].x = h_inData[where].x = (float)((double)rand() / (double)RAND_MAX);
			h_verify[coord(i, j)].y = h_inData[where].y = (float)((double)rand() / (double)RAND_MAX);
		}
	}
	totaltime = my_timer();	

	#pragma acc data copyin(h_inData[0:N*N]) copyout(h_outData[0:N*N]) create(h_tmp[0:N*N]) 
	{
		printf("Kernel initialization is complete.\n");

		// Get the iterationstamp to evaluate performance
		time = my_timer();	
		
		for(i = 0; i<2; i++) {
			fetch((flt2 *)(i == 0 ? h_inData : h_tmp), mangle);
			fft2d(inverse);
			transpose((flt2 *)(i == 0 ? h_tmp : h_outData), mangle);
		}

		// Record execution time
		time = my_timer() - time;
	}

	totaltime = my_timer() - totaltime;
	printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
	printf("\tMain execution time = %.4fms\n", (float)(totaltime * 1E3));
	gpoints_per_sec = ((double) N * N / time) * 1E-9;
	gflops = 2 * 5 * N * N * (log((float)N)/log((float)2))/(time * 1E9);
	printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);

	// Check signal to noise ratio

	// Run reference code
	for (i = 0; i < N; i++) {
		fourier_transform_gold(inverse, LOGN, h_verify + coord(i, 0));
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			h_verify_tmp[coord(j, i)] = h_verify[coord(i, j)];
		}
	}

	for (i = 0; i < N; i++) {
		fourier_transform_gold(inverse, LOGN, h_verify_tmp + coord(i, 0));
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			h_verify[coord(j, i)] = h_verify_tmp[coord(i, j)];
		}
	}

	mag_sum = 0;
	noise_sum = 0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			where = mangle ? mangle_bits(coord(i, j)) : coord(i, j);
			magnitude = (double)h_verify[coord(i, j)].x * (double)h_verify[coord(i, j)].x + (double)h_verify[coord(i, j)].y * (double)h_verify[coord(i, j)].y;
			noise = (h_verify[coord(i, j)].x - (double)h_outData[where].x) * (h_verify[coord(i, j)].x - (double)h_outData[where].x) + (h_verify[coord(i, j)].y - (double)h_outData[where].y) * (h_verify[coord(i, j)].y - (double)h_outData[where].y);

			mag_sum += magnitude;
			noise_sum += noise;
		}
	}
	db = 10 * log(mag_sum / noise_sum) / log(10.0);
	printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");

}

static void fetch(flt2 * src, int mangle) {
	int gid, wid;
#pragma acc kernels loop gang(N/8) present(src[0:N*N]) pipeout(chanin0, chanin1, chanin2, chanin3, chanin4, chanin5, chanin6, chanin7)
	for(gid=0; gid<N/8; gid++) {
		// Local memory for storing 8 rows
		flt2 buf[8 * N]; 
#pragma acc loop worker(N)
		for(wid=0; wid<N; wid++) {
#if COALESCED_BUFFERING == 1
			flt2x8 data;
#endif

			// Each read fetches 8 matrix points
			int x = (gid*N + wid) << LOGPOINTS;

			/* When using the alternative memory layout, each row consists of a set of
			* segments placed far apart in memory. Instead of reading all segments from
			* one row in order, read one segment from each row before switching to the 
			*  next segment. This requires swapping bits log(N) + 2 ... log(N) with 
			*  bits log(N) / 2 + 2 ... log(N) / 2 in the offset. 
			*/  

			int inrow, incol, where, where_global;
			if (mangle) {
				int a1210 = x & ((POINTS - 1) << LOGN);
				int a75 = x & ((POINTS - 1) << (LOGN / 2));
				int mask = ((POINTS - 1) << (LOGN / 2)) | ((POINTS - 1) << LOGN);
				a1210 >>= (LOGN / 2);
				a75 <<= (LOGN / 2);
				where = (x & ~mask) | a1210 | a75;
				where_global = mangle_bits(where);
			} else {
				where = x;
				where_global = where;
			}
		
			/* Read 8 points in a single coalesced access - the cast to ulong is
			* performed to avoid a rare coalescing issue preventing the compiler from
			* fully coalescing the accesses - this will be fixed in a later version of
			* the compiler
			* This is the intended code:
			* *(local flt2x8 *)&buf[where & ((1 << (LOGN + LOGPOINTS)) - 1)] =
			*                                   *(global flt2x8 *)&src[where_global];
			*/

#if COALESCED_BUFFERING == 1
			//[DEBUG] for now, there is no way to express this  type of casting in OpenACC.
			*(local ulong8 *)&buf[where & ((1 << (LOGN + LOGPOINTS)) - 1)] =
                                         *(global ulong8 *)&src[where_global];
#else
			buf[where & ((1 << (LOGN + LOGPOINTS)) - 1)] = src[where_global];
			buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))  + 1] = src[where_global + 1];
			buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))  + 2] = src[where_global + 2];
			buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))  + 3] = src[where_global + 3];
			buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))  + 4] = src[where_global + 4];
			buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))  + 5] = src[where_global + 5];
			buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))  + 6] = src[where_global + 6];
			buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))  + 7] = src[where_global + 7];
#endif
			#pragma acc barrier(acc_mem_fence_local)

			int row = wid >> (LOGN - LOGPOINTS);
			int col = wid & (N / POINTS - 1);

			// Stream fetched data over 8 channels to the FFT engine

			chanin0[gid*N + wid] = buf[row * N + col];
			chanin1[gid*N + wid] = buf[row * N + 4 * N / 8 + col];
			chanin2[gid*N + wid] = buf[row * N + 2 * N / 8 + col];
			chanin3[gid*N + wid] = buf[row * N + 6 * N / 8 + col];
			chanin4[gid*N + wid] = buf[row * N + N / 8 + col];
			chanin5[gid*N + wid] = buf[row * N + 5 * N / 8 + col];
			chanin6[gid*N + wid] = buf[row * N + 3 * N / 8 + col];
			chanin7[gid*N + wid] = buf[row * N + 7 * N / 8 + col];
		}
	}
}

static void fft2d(int inverse) {

	/* The FFT engine requires a sliding window for data reordering; data stored
	* in this array is carried across loop iterations and shifted by 1 element
	* every iteration; all loop dependencies derived from the uses of this 
	* array are simple transfers between adjacent array elements
	*/  

#if DELAY_ELEMENTS_ON_LOCAL == 0
	flt2 fft_delay_elements[N + POINTS * (LOGN - 2)];
#endif

#if DELAY_ELEMENTS_ON_LOCAL == 1
	#pragma acc parallel num_gangs(1) num_workers(1) pipein(chanin0, chanin1, chanin2, chanin3, chanin4, chanin5, chanin6, chanin7) pipeout(chan0, chan1, chan2, chan3, chan4, chan5, chan6, chan7)
#else
	#pragma acc parallel num_gangs(1) num_workers(1) pipein(chanin0, chanin1, chanin2, chanin3, chanin4, chanin5, chanin6, chanin7) pipeout(chan0, chan1, chan2, chan3, chan4, chan5, chan6, chan7) create(fft_delay_elements[0: N + POINTS * (LOGN -2)])
#endif
	{
	int i;
#if DELAY_ELEMENTS_ON_LOCAL == 1
	flt2 fft_delay_elements[N + POINTS * (LOGN - 2)];
#endif
	int k = 0;
	// needs to run "N / 8 - 1" additional iterations to drain the last outputs
	#pragma acc loop seq
	for (i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
		flt2x8 data;
	
		// Read data from channels
		if (i < N * (N / POINTS)) {
			data.i0 = chanin0[i];
			data.i1 = chanin1[i];
			data.i2 = chanin2[i];
			data.i3 = chanin3[i];
			data.i4 = chanin4[i];
			data.i5 = chanin5[i];
			data.i6 = chanin6[i];
			data.i7 = chanin7[i];
		} else {
			data.i0.x = data.i1.x = data.i2.x = data.i3.x = data.i4.x = data.i5.x = data.i6.x = data.i7.x = 0.0F;
			data.i0.y = data.i1.y = data.i2.y = data.i3.y = data.i4.y = data.i5.y = data.i6.y = data.i7.y = 0.0F;
		}   

		// Perform one FFT step
		data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

		// Write result to channels
		if (i >= N / POINTS - 1) {
			chan0[k] = data.i0;
			chan1[k] = data.i1;
			chan2[k] = data.i2;
			chan3[k] = data.i3;
			chan4[k] = data.i4;
			chan5[k] = data.i5;
			chan6[k] = data.i6;
			chan7[k] = data.i7;
			k++;
		}   
	}
	}
}

static void transpose(flt2 *dest, int mangle) {
	int gid, wid;
#pragma acc kernels loop gang(N/8) present(dest) pipein(chan0, chan1, chan2, chan3, chan4, chan5, chan6, chan7)
	for(gid=0; gid<N/8; gid++) {
		flt2 buf[POINTS * N]; 
#pragma acc loop worker(N)
		for(wid=0; wid<N; wid++) {
#if COALESCED_BUFFERING == 1
			flt2x8 data;
			data.i0 = chan0[gid*N + wid];
			data.i1 = chan1[gid*N + wid];
			data.i2 = chan2[gid*N + wid];
			data.i3 = chan3[gid*N + wid];
			data.i4 = chan4[gid*N + wid];
			data.i5 = chan5[gid*N + wid];
			data.i6 = chan6[gid*N + wid];
			data.i7 = chan7[gid*N + wid];
			*(local ulong8 *)&buf[8 * wid] = *(ulong8 *)&data;
			#pragma acc barrier(acc_mem_fence_local)
			int colt = wid;
			int revcolt = bit_reversed(colt, LOGN);
			data.i0 = buf[revcolt];
			data.i1 = buf[N + revcolt];
			data.i2 = buf[2 * N + revcolt];
			data.i3 = buf[3 * N + revcolt];
			data.i4 = buf[4 * N + revcolt];
			data.i5 = buf[5 * N + revcolt];
			data.i6 = buf[6 * N + revcolt];
			data.i7 = buf[7 * N + revcolt];
			int i = (gid*N + wid) >> LOGN;
			int where = colt * N + i * POINTS;
			if (mangle) where = mangle_bits(where);
			*(global ulong8 *)&dest[where] = *(ulong8 *)&data;
#else
			buf[8*wid] = chan0[gid*N + wid];
			buf[8*wid + 1] = chan1[gid*N + wid];
			buf[8*wid + 2] = chan2[gid*N + wid];
			buf[8*wid + 3] = chan3[gid*N + wid];
			buf[8*wid + 4] = chan4[gid*N + wid];
			buf[8*wid + 5] = chan5[gid*N + wid];
			buf[8*wid + 6] = chan6[gid*N + wid];
			buf[8*wid + 7] = chan7[gid*N + wid];
			#pragma acc barrier(acc_mem_fence_local)
			int colt = wid;
			int revcolt = bit_reversed(colt, LOGN);
			int i = (gid*N + wid) >> LOGN;
			int where = colt * N + i * POINTS;
			if (mangle) where = mangle_bits(where);
			dest[where] = buf[revcolt];
			dest[where + 1] = buf[N + revcolt];
			dest[where + 2] = buf[2 * N + revcolt];
			dest[where + 3] = buf[3 * N + revcolt];
			dest[where + 4] = buf[4 * N + revcolt];
			dest[where + 5] = buf[5 * N + revcolt];
			dest[where + 6] = buf[6 * N + revcolt];
			dest[where + 7] = buf[7 * N + revcolt];
#endif
		}
	}
}

/////// HELPER FUNCTIONS ///////

// provides a linear offset in the input array
int coord(int iteration, int i) {
  return iteration * N + i;
}

// This modifies the linear matrix access offsets to provide an alternative
// memory layout to improve the efficiency of the memory accesses
int mangle_bits(int x) {
   const int NB = LOGN / 2;
   int a95 = x & (((1 << NB) - 1) << NB);
   int a1410 = x & (((1 << NB) - 1) << (2 * NB));
   int mask = ((1 << (2 * NB)) - 1) << NB;
   a95 = a95 << NB;
   a1410 = a1410 >> NB;
   return (x & ~mask) | a95 | a1410;
}


// Reference Fourier transform
void fourier_transform_gold(int inverse, const int lognr_points, dbl2 *data) {
	int i,j;
	double tmp;
	int fwd;
	int bit_rev;
	const int nr_points = 1 << lognr_points;

	// The inverse requires swapping the real and imaginary component

	if (inverse) {
		for (i = 0; i < nr_points; i++) {
			tmp = data[i].x;
			data[i].x = data[i].y;
			data[i].y = tmp;;
		}
	}
	// Do a FT recursively
	fourier_stage(lognr_points, data);

	// The inverse requires swapping the real and imaginary component
	if (inverse) {
		for (i = 0; i < nr_points; i++) {
			tmp = data[i].x;
			data[i].x = data[i].y;
			data[i].y = tmp;;
		}
	}
}

void fourier_stage(int lognr_points, dbl2 *data) {
	int i;
	int nr_points = 1 << lognr_points;
	if (nr_points == 1) return;
	dbl2 *half1 = (dbl2 *)alloca(sizeof(dbl2) * nr_points / 2);
	dbl2 *half2 = (dbl2 *)alloca(sizeof(dbl2) * nr_points / 2);
	for (i = 0; i < nr_points / 2; i++) {
		half1[i] = data[2 * i];
		half2[i] = data[2 * i + 1];
	}
	fourier_stage(lognr_points - 1, half1);
	fourier_stage(lognr_points - 1, half2);
	for (i = 0; i < nr_points / 2; i++) {
		data[i].x = half1[i].x + cos (2 * M_PI * i / nr_points) * half2[i].x + sin (2 * M_PI * i / nr_points) * half2[i].y;
		data[i].y = half1[i].y - sin (2 * M_PI * i / nr_points) * half2[i].x + cos (2 * M_PI * i / nr_points) * half2[i].y;
		data[i + nr_points / 2].x = half1[i].x - cos (2 * M_PI * i / nr_points) * half2[i].x - sin (2 * M_PI * i / nr_points) * half2[i].y;
		data[i + nr_points / 2].y = half1[i].y + sin (2 * M_PI * i / nr_points) * half2[i].x - cos (2 * M_PI * i / nr_points) * half2[i].y;
	}
}

double my_timer ()
{
    struct timeval time;
    gettimeofday (&time, 0); 
    return time.tv_sec + time.tv_usec / 1000000.0;
}

/////////////////////
//Device functions //
/////////////////////

// This utility function bit-reverses an integer 'x' of width 'bits'.

#pragma acc routine nohost
int bit_reversed(int x, int bits) {
  int i;
  int y = 0;
  #pragma unroll
  for (i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  return y;
}

