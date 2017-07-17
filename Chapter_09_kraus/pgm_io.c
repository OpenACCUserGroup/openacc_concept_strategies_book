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
#include "pgm_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

const unsigned long PGM_MAXVAL = 255;

int read_pgm_file_header_impl( int* width_p, int* height_p, FILE * image_file )
{
    char pgm_type[256];
    int n_read = fscanf(image_file, "%s\n", pgm_type);
    if ( EOF == n_read )
    {
        fprintf( stderr, "ERROR: could not read type from image file header: %d.\n", ferror(image_file) );
        fclose( image_file );
        return 1;
    }
    
    const char* p2_type = "P2";
    if (0 != strcmp(p2_type, pgm_type))
    {
        fprintf( stderr, "ERROR: only images of type %s are supported input is %s.\n", p2_type, pgm_type );
        fclose( image_file );
        return 1;
    }
    
    n_read = fscanf(image_file, "%d %d\n", width_p, height_p);
    if ( EOF == n_read || 2 != n_read )
    {
        fprintf( stderr, "ERROR: could not read image dimensions from image file header: %d.\n", ferror(image_file) );
        fclose( image_file );
        return 1;
    }
    
    if ( *width_p <= 0 || *height_p <= 0 )
    {
        fprintf( stderr, "ERROR: invalid image dimensions: %d %d.\n", *width_p,*height_p );
        fclose( image_file );
        return 1;
    }
    
    int maxval = 0;
    n_read = fscanf(image_file, "%d\n", &maxval);
    if ( EOF == n_read || 1 != n_read )
    {
        fprintf( stderr, "ERROR: could not maxval from image file header: %d.\n", ferror(image_file) );
        fclose( image_file );
        return 1;
    }
    
    if ( PGM_MAXVAL != maxval )
    {
        fprintf( stderr, "ERROR: invalid maxval: %d only %d is supported.\n", maxval,PGM_MAXVAL );
        fclose( image_file );
        return 1;
    }
    return 0;
}

int read_pgm_file_header( int* width_p, int* height_p, const char* const filename )
{
    FILE * image_file = fopen(filename, "r" );
    if ( NULL == image_file )
    {
        fprintf( stderr, "ERROR: could not open %s for reading.\n", filename );
        return 1;
    }
    
    int retval = read_pgm_file_header_impl( width_p, height_p, image_file );
    
    fclose( image_file );
    return retval;
}

int read_pgm_file( float* picture, int width, int height, const char* const filename )
{
    FILE * image_file = fopen(filename, "r" );
    if ( NULL == image_file )
    {
        fprintf( stderr, "ERROR: could not open %s for reading.\n", filename );
        return 1;
    }
    
    int width_header = 0;
    int height_header = 0;
    int retval = read_pgm_file_header_impl( &width_header, &height_header, image_file );
    if ( 0 != retval )
    {
        fclose( image_file );
        return 1;
    }
    if ( width_header != width || height_header != height )
    {
        fprintf( stderr, "ERROR: image header does not match passed width and height: width = %d, height = %d - header.width = %d, header.height = %d.\n", width, height, width_header, height_header );
        fclose( image_file );
        return 1;
    }
    
    for (int row=0; row<height; ++row) {
        for (int column=0; column<width; column++) {
            unsigned long pixel_val = 0;
            int n_read = fscanf(image_file, "%d", &pixel_val);
            if ( EOF == n_read || 1 != n_read )
            {
                fprintf( stderr, "ERROR: reading image content from %s failed: %d.\n", filename, ferror(image_file) );
                fclose( image_file );
                return 1;
            }
            if ( 255 < pixel_val )
            {
                fprintf( stderr, "ERROR: invalid pixel value %d at %d %d in %s.\n", pixel_val, column, row, filename );
                fclose( image_file );
                return 1;
            }
            picture[row*width + column] = fmaxf(0.0f,fminf(1.0f,(1.0f*pixel_val)/(1.0f*PGM_MAXVAL)));
        }
    }

    fclose( image_file );
    return 0;
}

int write_pgm_file( const char* const filename, int width, int height, const float* const picture )
{
    FILE * image_file = fopen(filename, "w" );
    if ( NULL == image_file )
    {
        fprintf( stderr, "ERROR: could not open %s for writing.\n", filename );
        return 1;
    }
    
    fprintf(image_file, "P2\n");
    fprintf(image_file, "%d %d\n",width,height);
    fprintf(image_file, "%d\n",PGM_MAXVAL);
   
    for (int row=0; row<height; ++row) {
        for (int column=0; column<width; column++) {
            unsigned long pixel_val = PGM_MAXVAL * picture[row*width + column];
            pixel_val = pixel_val > PGM_MAXVAL ? PGM_MAXVAL : pixel_val;
            pixel_val = pixel_val < 0 ? 0 : pixel_val;
            fprintf(image_file, "%d ",pixel_val);
        }
        fprintf(image_file, "\n");
    }

    fclose( image_file );
    return 0;
}
