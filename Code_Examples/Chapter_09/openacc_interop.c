/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <stdint.h>
#include <assert.h>

#ifdef USE_CUFFT

#include <cufft.h>
typedef float complex fft_complex_t;

#define CUFFT_CALL( call )                                                                                                           \
{                                                                                                                                    \
    cufftResult cufftStatus = call;                                                                                                  \
    if ( CUFFT_SUCCESS != cufftStatus )                                                                                              \
        fprintf(stderr, "ERROR: CUFFT call \"%s\" in line %d of file %s failed with %d.\n", #call, __LINE__, __FILE__, cufftStatus); \
}

#else /* USE_CUFFT */

#include <fftw3.h>
typedef fftwf_complex fft_complex_t;

#endif /* USE_CUFFT */

#ifdef _OPENACC
#include <openacc.h>
#endif

#include "pgm_io.h"

const float sharpening_filter[3][3] = {  
   { 0.0f,  -0.25f, 0.0f   },
   {-0.25f,  1.0f,  -0.25f },
   { 0.0f,  -0.25f, 0.0f   }
};
const int sharpening_filter_columns = sizeof(sharpening_filter[0])/sizeof(float);
const int sharpening_filter_rows = sizeof(sharpening_filter)/sizeof(sharpening_filter[0]);

//from http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm 
const float gaussian_blur_filter[5][5] = {
   { 1.0f/273.0f,   4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f },
   { 4.0f/273.0f,  16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f },
   { 7.0f/273.0f,  26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f },
   { 4.0f/273.0f,  16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f },
   { 1.0f/273.0f,   4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f }
};
const int gaussian_blur_filter_columns = sizeof(gaussian_blur_filter[0])/sizeof(float);
const int gaussian_blur_filter_rows = sizeof(gaussian_blur_filter)/sizeof(gaussian_blur_filter[0]);

int main(int argc, char *argv[])
{
    if ( 3 != argc )
    {
        fprintf(stderr, "Usage: %s input.pgm output.pgm\n", argv[0]);
        return 1;
    }

    int width = 0;
    int height = 0;
    if ( 0 != read_pgm_file_header( &width,  &height, argv[1] ) )
    {
        fprintf(stderr, "ERROR: reading header of input from %s failed.\n", argv[1]);
        return 1;
    }
    
    float * restrict const picture_real = (float*) malloc( height*width*sizeof(float) );
    float * restrict const sharpening_filter_real = (float*) malloc(width*height*sizeof(float));
    float * restrict const gaussian_blur_filter_real = (float*) malloc(width*height*sizeof(float));
    const int width_in_freq_domain = (width/2+1);
    fft_complex_t * restrict const sharpening_filter_freq = (fft_complex_t*) malloc( height*width_in_freq_domain*sizeof(fft_complex_t) );
    fft_complex_t * restrict const gaussian_blur_filter_freq = (fft_complex_t*) malloc( height*width_in_freq_domain*sizeof(fft_complex_t) );
    fft_complex_t * restrict const combined_filter_freq = (fft_complex_t*) malloc( height*width_in_freq_domain*sizeof(fft_complex_t) );
    fft_complex_t * restrict const picture_freq = (fft_complex_t*) malloc( height*width_in_freq_domain*sizeof(fft_complex_t) );
    
    //create FFT plans
#ifdef USE_CUFFT
    cufftHandle r2c_plan;
    cufftHandle c2r_plan;
    CUFFT_CALL( cufftPlan2d(&r2c_plan, height,width, CUFFT_R2C) );
    CUFFT_CALL( cufftPlan2d(&c2r_plan, height,width, CUFFT_C2R) );
    CUFFT_CALL( cufftSetStream(r2c_plan, acc_get_cuda_stream ( acc_async_sync ) ) );
    CUFFT_CALL( cufftSetStream(c2r_plan, acc_get_cuda_stream ( acc_async_sync ) ) );
#else /* USE_CUFFT */
    fftwf_plan sharpening_filter_plan = fftwf_plan_dft_r2c_2d( height,width, sharpening_filter_real, sharpening_filter_freq, FFTW_ESTIMATE );
    fftwf_plan gaussian_blur_filter_plan = fftwf_plan_dft_r2c_2d( height,width, gaussian_blur_filter_real, gaussian_blur_filter_freq, FFTW_ESTIMATE );
    fftwf_plan r2c_picture_plan = fftwf_plan_dft_r2c_2d( height,width, picture_real, picture_freq, FFTW_ESTIMATE );
    fftwf_plan c2r_picture_plan = fftwf_plan_dft_c2r_2d( height,width, picture_freq, picture_real, FFTW_ESTIMATE );
#endif /* USE_CUFFT */
    
    int error_code = 0;
    #pragma acc data create( combined_filter_freq[0:height*width_in_freq_domain] )
    {
        //prepare filters
        #pragma acc data create( sharpening_filter_real[0:height*width], sharpening_filter_freq[0:height*width_in_freq_domain], gaussian_blur_filter_real[0:height*width], gaussian_blur_filter_freq[0:height*width_in_freq_domain] ) copyin( sharpening_filter,gaussian_blur_filter )
        {
            #pragma acc parallel loop collapse(2)
            for (int row=0; row<height; ++row) {
                for (int column=0; column<width; column++) {
                    if ( row < sharpening_filter_rows && column < sharpening_filter_columns )
                        sharpening_filter_real[row*width+column] = sharpening_filter[row][column];
                    else
                        sharpening_filter_real[row*width + column] = 0.0f;
                    if ( row < gaussian_blur_filter_rows && column < gaussian_blur_filter_columns )
                        gaussian_blur_filter_real[row*width+column] = gaussian_blur_filter[row][column];
                    else
                        gaussian_blur_filter_real[row*width + column] = 0.0f;
                }
            }
        
#ifdef USE_CUFFT
            #pragma acc host_data use_device( sharpening_filter_real, sharpening_filter_freq )
            {
                //Alignment of C11 float complex does not match cufftComplex alignment. This is no issue here as
                //sufficient alignment is guaranteed by the allocator but it could be when custom allocators or
                //user defined types are used.
                assert( 0 == ((uintptr_t)sharpening_filter_freq%alignof(cufftComplex)));
                CUFFT_CALL( cufftExecR2C(r2c_plan, sharpening_filter_real, (cufftComplex*) sharpening_filter_freq) );
            }
            #pragma acc host_data use_device( gaussian_blur_filter_real, gaussian_blur_filter_freq )
            {
                //Alignment of C11 float complex does not match cufftComplex alignment. This is no issue here as
                //sufficient alignment is guaranteed by the allocator but it could be when custom allocators or
                //user defined types are used.
                assert( 0 == ((uintptr_t)gaussian_blur_filter_freq%alignof(cufftComplex)));
                CUFFT_CALL( cufftExecR2C(r2c_plan, gaussian_blur_filter_real, (cufftComplex*) gaussian_blur_filter_freq) );
            }
#else /* USE_CUFFT */
            #pragma acc update self( sharpening_filter_real[0:height*width], gaussian_blur_filter_real[0:height*width] )
            fftwf_execute_dft_r2c(sharpening_filter_plan,sharpening_filter_real,sharpening_filter_freq);
            fftwf_execute_dft_r2c(gaussian_blur_filter_plan,gaussian_blur_filter_real,gaussian_blur_filter_freq);
            #pragma acc update device( sharpening_filter_freq[0:height*width_in_freq_domain], gaussian_blur_filter_freq[0:height*width_in_freq_domain] )
#endif /* USE_CUFFT */

            //combine filters
            #pragma acc parallel loop collapse(2)
            for (int row=0; row<height; ++row) {
                for (int column=0; column<width_in_freq_domain; ++column) {
                    const float complex sharpening_filter_freq_val = sharpening_filter_freq[row*width_in_freq_domain + column];
                    const float complex gaussian_blur_filter_freq_val = gaussian_blur_filter_freq[row*width_in_freq_domain + column];
                    combined_filter_freq[row*width_in_freq_domain + column] = sharpening_filter_freq_val*gaussian_blur_filter_freq_val;
                }
            }
        } //#pragma acc data create( sharpening_filter_real[0:height*width], sharpening_filter_freq[0:height*width_in_freq_domain], gaussian_blur_filter_real[0:height*width], gaussian_blur_filter_freq[0:height*width_in_freq_domain], ) copyin( sharpening_filter,gaussian_blur_filter )

        //read, process and write image
        #pragma acc data create( picture_real[0:height*width], picture_freq[0:height*width_in_freq_domain] )
        {
            if ( 0 == read_pgm_file( picture_real, width,  height, argv[1] ) )
            {    
                #pragma acc update device( picture_real[0:height*width] )

#ifdef USE_CUFFT
                #pragma acc host_data use_device( picture_real, picture_freq )
                {
                    //Alignment of C11 float complex does not match cufftComplex alignment. This is no issue here as
                    //sufficient alignment is guaranteed by the allocator but it could be when custom allocators or
                    //user defined types are used.
                    assert( 0 == ((uintptr_t)picture_freq%alignof(cufftComplex)));
                    CUFFT_CALL( cufftExecR2C(r2c_plan, picture_real, (cufftComplex*) picture_freq) );
                }
#else /* USE_CUFFT */
                #pragma acc update self( picture_real[0:height*width] )
                fftwf_execute_dft_r2c(r2c_picture_plan,picture_real,picture_freq);
                #pragma acc update device( picture_freq[0:height*width_in_freq_domain] )
#endif /* USE_CUFFT */
                
                //apply filter
                #pragma acc parallel loop collapse(2)
                for (int row=0; row<height; ++row) {
                    for (int column=0; column<width_in_freq_domain; ++column) {
                        picture_freq[row*width_in_freq_domain + column] *= combined_filter_freq[row*width_in_freq_domain + column];
                    }
                }

#ifdef USE_CUFFT
                #pragma acc host_data use_device( picture_freq, picture_real )
                {
                    //Alignment of C11 float complex does not match cufftComplex alignment. This is no issue here as
                    //sufficient alignment is guaranteed by the allocator but it could be when custom allocators or
                    //user defined types are used.
                    assert( 0 == ((uintptr_t)picture_freq%alignof(cufftComplex)));
                    CUFFT_CALL( cufftExecC2R(c2r_plan, (cufftComplex*) picture_freq, picture_real) );
                }
#else /* USE_CUFFT */
                #pragma acc update self( picture_freq[0:height*width_in_freq_domain] )
                fftwf_execute_dft_c2r(c2r_picture_plan,picture_freq,picture_real);
                #pragma acc update device( picture_real[0:height*width] )
#endif /* USE_CUFFT */
                
                //normalize output
                const float scale_factor = 1.0f/(height*width);
                #pragma acc parallel loop collapse(2)
                for (int row=0; row<height; ++row) {
                    for (int column=0; column<width; column++) {
                        picture_real[row*width + column] *= scale_factor;
                    }
                }
                
                #pragma acc update self( picture_real[0:height*width] )
                
                if ( 0 != write_pgm_file( argv[2], width, height, picture_real ) )
                {
                    fprintf(stderr, "ERROR: writing output to %s failed.\n", argv[2]);
                    error_code = 1;
                }
                else
                {
                    printf("Wrote output image %s of size %d x %d\n", argv[2], width, height );
                }
            }
        } //#pragma acc data create( picture_real[0:height*width], picture_freq[0:height*width_in_freq_domain] )
    } //#pragma acc data create( combined_filter_freq[0:height*width_in_freq_domain] )

    //destroy plans
#ifdef USE_CUFFT
    cufftDestroy(c2r_plan);
    cufftDestroy(r2c_plan);
#else /* USE_CUFFT */
    fftwf_destroy_plan(c2r_picture_plan);
    fftwf_destroy_plan(r2c_picture_plan);
    fftwf_destroy_plan(gaussian_blur_filter_plan);
    fftwf_destroy_plan(sharpening_filter_plan);
#endif /* USE_CUFFT */
    
    free( picture_freq );
    free( combined_filter_freq );
    free( gaussian_blur_filter_freq );
    free( sharpening_filter_freq );
    free( gaussian_blur_filter_real );
    free( sharpening_filter_real );
    free( picture_real);
    return error_code;
}
