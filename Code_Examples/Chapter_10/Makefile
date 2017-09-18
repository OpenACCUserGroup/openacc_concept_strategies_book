CC=pgcc
CFLAGS=-ta=tesla:pinned -Minfo=all
CXX=pgc++
CXXFLAGS=-ta=tesla:pinned -Minfo=all
# It may be necessary to edit the line below to point to your installation of
# OpenCV.
OPENCV_FLAGS=-lopencv_highgui -lopencv_core

filter.x: filter.o main.o
	$(CXX) $(CXXFLAGS) -mp $^ $(OPENVC_FLAGS) -lm -o $@ -Mcuda -Minfo=all

.SUFFIXES: .o .cpp .c
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core filter.x
