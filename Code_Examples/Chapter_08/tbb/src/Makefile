#-----------------------------------------------------------------------

MINIFE_TYPES =  \
	-DMINIFE_SCALAR=double   \
	-DMINIFE_LOCAL_ORDINAL=int      \
	-DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX

TBB_PATH = /projects/sems/install/rhel6-x86_64/sems/compiler/intel/16.0.3/base/tbb
# MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -g -O3
CXXFLAGS = -g -O3

CPPFLAGS = -std=c++11 -I. -I../utils -I../fem $(MINIFE_TYPES) -I${TBB_PATH}/include\
	$(MINIFE_MATRIX_TYPE)
CPPFLAGS += -DMINIFE_USE_FLAT_SPMV

LDFLAGS = -L${TBB_PATH}/lib/intel64/gcc4.4 -ltbb
LIBS=-lm

CXX=icpc
CC=icc

include make_targets

