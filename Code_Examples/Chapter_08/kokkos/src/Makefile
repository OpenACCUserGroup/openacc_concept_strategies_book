KOKKOS_PATH = ${HOME}/kokkos
vpath %.cpp ../utils

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX

SRC = $(wildcard *.cpp)
SRC += $(wildcard ../utils/*.cpp)
HEADERS = $(wildcard *.hpp)
HEADERS += $(wildcard ../utils/*.hpp)

KOKKOS_DEVICES=OpenMP
KOKKOS_CUDA_OPTIONS=enable_lambda,rdc

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/config/nvcc_wrapper
KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = g++
LINK = ${CXX}
KOKKOS_ARCH = "SNB"
endif
CXXFLAGS = -O3 -g -I./ -I../utils -I../fem ${MINIFE_TYPES} ${MINIFE_MATRIX_TYPE}
#CXXFLAGS += -DMINIFE_USE_FLAT_MATVEC

LINK = ${CXX}
LINKFLAGS = 
EXE = miniFE.x

DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
