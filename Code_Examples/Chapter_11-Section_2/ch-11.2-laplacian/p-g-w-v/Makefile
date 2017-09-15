TARGET = laplacian
CC=uhcc
LDLIBS=
CUDAPATH=/home/sw/cuda/hammer-8.0/linux86-64

CFLAGS = -fopenacc -nvcc,-arch=sm_35,-fmad=false -nvpath,$(CUDAPATH)/bin/ -Wb,-accarch:nvidia$(CONSTFLAG)$(SRFLAGS)
LDFLAGS = $(CFLAGS) -lopenacc -L$(CUDAPATH)/lib64/

OPT= 

OBJS=$(TARGET).o 

all:		$(TARGET)

$(TARGET):		$(OBJS)
		$(CC) $(CFLAGS) $(OPT) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
		rm -f $(TARGET) $(OBJS)


.SUFFIXES:	.c.o

.c.o:
		$(CC) $(CFLAGS) $(OPT) -c $< -o $*.o

