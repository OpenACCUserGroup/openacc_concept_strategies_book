`openacc_interop` is an example program to demonstrate OpenACC interoperability
features. The application reads a grayscale image in pgm format, applies an
edge detector (combination of gaussian blur and sharpening filter) and writes
the output as grayscale image in pgm format. The filters are applied in
frequencies space and the necessary DFTs are carried out with using either a
FFT library with a FFTW compatible interface, e.g. Intel MKL, in case a CPU is
used for the compute or cuFFT in case a NVIDIA GPU is used.

### Usage
```
./openacc_interop input.pgm output.pgm 
```
E.g. using the example image from the subfolder images:
```
./openacc_interop images/617019_NVIDIA_HQ_bldg.pgm 617019_NVIDIA_HQ_bldg.out.pgm
Wrote output image 617019_NVIDIA_HQ_bldg.out.pgm of size 2400 x 1600
```

### Requirements
* GNU make
* C compiler for the serial CPU version or OpenACC capable C compiler for a
  parallel version. The provided Makefile is written to support GCC and PGI.
* For the CPU versions: FFT library with a FFTW compatible Interface. The
  provided Makefile is written to use Intel MKL
* For the NVIDIA GPU version: cuFFT 

### Compiling
The makefile has the following targets:
* openacc_interop(default): builds the executable with the configuration
                            specified via environment (COMPILER)
* clean:                    deletes the executable, object files and the output
                            image produced by the run target
* run:                      runs the executable on the provided example image
* profile:                  profile the executable with nvprof using the
                            provided example image as input

With the environment variable COMPILER three different build configurations can
be selected:

* COMPILER = GCC:           A serial CPU version using GCC and the FFTW
                            compatible interface of Intel MKL
* COMPILER = PGI-tesla:     A parallel GPU version using the PGI compiler and
                            cuFFT
* COMPILER = PGI-multicore: A parallel CPU version using the PGI compiler and
                            the FFTW compatible interface of Intel MKL
