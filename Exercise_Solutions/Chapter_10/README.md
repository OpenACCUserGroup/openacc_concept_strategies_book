Chapter 10 Exercise Solutions
=============================

1. The source code for the image-filtering example used in this chapter has
   been provided at the GitHub site shown in the Preface. You should attempt to
   build and understand this sample application.

The provided code should build and run with the PGI compiler without any
issues. It has not been tested against other OpenACC compilers, but is expected
to work regardless of the compiler used.

2. You should experiment to determine the best block size for your machine.
   Does changing the image size affect the result? Does changing the accelerator
   type change the result?

The performance will vary wildly depending on the specific GPU and the size of
the image used. Below is an example varying the blocksize from 2 - 64 blocks on
3 different NVIDIA GPUs using a 3267 x 1736 JPG image.

+--------+------------------+-------------------+-------------------+
| Blocks | NVIDIA Telsa K40 | NVIDIA Telsa P100 | NVIDIA Tesla V100 |
+--------+------------------+-------------------+-------------------+
|      2 |         0.013841 |          0.005604 |          0.004532 |
+--------+------------------+-------------------+-------------------+
|      4 |         0.011639 |          0.004418 |          0.003358 |
+--------+------------------+-------------------+-------------------+
|      8 |         0.010635 |          0.003782 |          0.002906 |
+--------+------------------+-------------------+-------------------+
|     16 |         0.010092 |          0.003528 |          0.003237 |
+--------+------------------+-------------------+-------------------+
|     32 |         0.009823 |          0.003884 |          0.003312 |
+--------+------------------+-------------------+-------------------+
|     64 |         0.009763 |          0.004643 |          0.004760 |
+--------+------------------+-------------------+-------------------+
