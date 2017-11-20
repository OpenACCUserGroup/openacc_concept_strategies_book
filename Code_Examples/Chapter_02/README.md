# Mandelbrot

This directory contains an example program and implements the calculation of the area of a [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set). This calculation was used because it involves performing a large number of independent calculations to produce the overall area. These can be trivially parallelised using OpenMP or OpenACC, and therefore it is an easy example to provide in multiple formats and langauges.

The example has been implemented in both C and Fortran, with serial, OpenMP, and OpenACC implementations for both languages.  Each different has a corresponding makefile that can be used to compile an executable for that version. To use the makefile simple use the `make` command with the `-f` flag to specify the makefile you want to use, i.e.:
```
make -f makefile.ser
```  

Makefiles ending with `.ser` will compile the serial version of the code.  Makefiles ending with `.omp` will compile the OpenMP version of the code. Makefiles ending with `.acc` will compile the OpenACC version of the code.

The makefiles are configured to use the PGI compilers, but also include the relevant flags for the GNU compilers.  To use the GNU compilers instead simply uncomment the 'CC' or 'FC' line that has been commented out in the makefile you are using, i.e. the following enables the GNU C compiler:

```
CC= gcc -O3 -fopenmp
#CC=     pgcc -O3 -acc -ta=nvidia -Minfo=accel
```
