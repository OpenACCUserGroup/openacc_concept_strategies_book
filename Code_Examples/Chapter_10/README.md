Ch 10. - OpenACC Image Pipeline
===============================

This example demonstrates a pipelined approach to optimizing an OpenACC program
to overlap computation and data transfers. For more details, please see Chapter
10 of OpenACC for Programmers: Concepts and Strategies, First Edition.

Required Software
-----------------
1. OpenACC-aware compiler. This has been tested with PGI 16.10 and newer.
2. OpenCV

Build Instructions
------------------
A Makefile has been provided to build the example. It may be necessary to
modify the Makefile to point to your installation of OpenCV or your OpenACC
Compiler.

Running
-------
The resulting executable, filter.x, expects 2 commandline parameters. The first
parameter is the input image and the second is the name of the output image.
You can use any image type supported by OpenCV.

    $ ./filter.x input.jpg output.jpg

License
-------
Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

