#-----------------------------------------------------------------------

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
# MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -O3
LDFLAGS = 

CXXFLAGS = $(CFLAGS)
CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) \
	-DMINIFE_REPORT_RUSAGE

#Use flat SPMV kernel
#CPPFLAGS += -DMINIFE_USE_FLAT_SPMV

#PGI compiler compiling for NVIDIA Pascal P100
CPPFLAGS += -acc -ta=tesla:cc60 
LDFLAGS += -acc -ta=tesla:cc60
CXX=pgc++
CC=pgc++

LIBS=


include make_targets

