--------------------------------------------------------------------------
COMPILE
--------------------------------------------------------------------------
To compile and run this example, please refer to README.md in the OpenARC repository. (To access the OpenARC repository, visit http://ft.ornl.gov/research/openarc.)

	- Set environment variable, openarc to the root directory of the OpenARC repository.

	- Adjust configurations in the Makefile as necessary. (At least, AOCL_BOARD should be set to a correct Altera board name if targeting an FPGA.)

	- run O2GBuild.script to translate the input OpenACC program into output CUDA or OpenCL program.
	$ O2GBuild.script

	- Compile the generated output program. (If targeting Altera FPGA, it may take several hours to compile.)
	$ make
	
	- Run the compiled output program
	$ cd bin; fft2d_ACC

To compile this example using other OpenACC compilers targeting GPUs, the following OpenACC clauses should be manually changed (and a separate Makefile should be provided for the OpenACC compiler):

In fft2d.c

pipe => create

pipein => present

pipeout => present

--------------------------------------------------------------------------
SAMPLE OUTPUT
--------------------------------------------------------------------------
[Sample output on NVIDIA P100 GPU]

==> Input LOGN : 8
Launching FFT transform (ordered data layout)
Kernel initialization is complete.
	Processing time = 433.6591ms
	Main execution time = 434.9752ms
	Throughput = 0.0002 Gpoints / sec (0.0121 Gflops)
	Signal to noise ratio on output sample: 137.670912 --> PASSED

Launching inverse FFT transform (ordered data layout)
Kernel initialization is complete.
	Processing time = 416.4381ms
	Main execution time = 416.7881ms
	Throughput = 0.0002 Gpoints / sec (0.0126 Gflops)
	Signal to noise ratio on output sample: 138.489590 --> PASSED

Launching FFT transform (alternative data layout)
Kernel initialization is complete.
	Processing time = 416.5590ms
	Main execution time = 416.8489ms
	Throughput = 0.0002 Gpoints / sec (0.0126 Gflops)
	Signal to noise ratio on output sample: 138.194197 --> PASSED

Launching inverse FFT transform (alternative data layout)
Kernel initialization is complete.
	Processing time = 416.4832ms
	Main execution time = 416.7791ms
	Throughput = 0.0002 Gpoints / sec (0.0126 Gflops)
	Signal to noise ratio on output sample: 137.648670 --> PASSED

[Sample output on Altera Arria 10 FPGA]

Reprogramming device with handle 1
==> Input LOGN : 8
Launching FFT transform (ordered data layout)
Kernel initialization is complete.
	Processing time = 0.1750ms
	Main execution time = 4.8048ms
	Throughput = 0.3745 Gpoints / sec (29.9594 Gflops)
	Signal to noise ratio on output sample: 138.137525 --> PASSED

Launching inverse FFT transform (ordered data layout)
Kernel initialization is complete.
	Processing time = 0.1891ms
	Main execution time = 4.3550ms
	Throughput = 0.3466 Gpoints / sec (27.7304 Gflops)
	Signal to noise ratio on output sample: 138.266486 --> PASSED

Launching FFT transform (alternative data layout)
Kernel initialization is complete.
	Processing time = 0.1822ms
	Main execution time = 4.2400ms
	Throughput = 0.3598 Gpoints / sec (28.7830 Gflops)
	Signal to noise ratio on output sample: 138.865104 --> PASSED

Launching inverse FFT transform (alternative data layout)
Kernel initialization is complete.
	Processing time = 0.2091ms
	Main execution time = 4.4119ms
	Throughput = 0.3134 Gpoints / sec (25.0744 Gflops)
	Signal to noise ratio on output sample: 137.389024 --> PASSED

