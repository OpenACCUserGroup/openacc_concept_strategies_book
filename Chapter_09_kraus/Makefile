# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
COMPILER ?= PGI-tesla
ifeq ($(COMPILER),GCC)
	CC = gcc
	CFLAGS += -m64 -I$(MKLROOT)/include/fftw -march=native -O3 -std=c99
	LDFLAGS += -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
else ifeq ($(COMPILER),PGI-tesla)
	CC = pgcc
	CFLAGS += -c11 -DUSE_CUFFT -Minfo=accel -fast -acc -ta=tesla:cuda8.0
	LDFLAGS += -L$(PGI)/cuda/8.0/lib64 -lcufft
else ifeq ($(COMPILER),PGI-multicore)
	CC = pgcc
	CFLAGS += -c11 -I$(MKLROOT)/include/fftw -Minfo=accel -fast -acc -ta=multicore
	LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
endif
all: openacc_interop

pgm_io.o: pgm_io.c Makefile
	$(CC) $(CFLAGS) pgm_io.c -c -o pgm_io.o

openacc_interop: openacc_interop.c Makefile pgm_io.o
	$(CC) $(CFLAGS) openacc_interop.c pgm_io.o $(LDFLAGS) -o openacc_interop

.PHONY: clean
clean:
	rm -f *.o openacc_interop *.out.pgm

run: openacc_interop
	./openacc_interop images/617019_NVIDIA_HQ_bldg.pgm 617019_NVIDIA_HQ_bldg.out.pgm

profile: openacc_interop
	nvprof ./openacc_interop images/617019_NVIDIA_HQ_bldg.pgm 617019_NVIDIA_HQ_bldg.out.pgm
