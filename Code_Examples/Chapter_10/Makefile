CC=pgcc
CFLAGS=-Mcuda=cc60 -Minfo=all
CXX=pgc++
CXXFLAGS=-Mcuda=cc60 -Minfo=all
# It may be necessary to edit the line below to point to your installation of
# OpenCV.
OPENCV_FLAGS=-lopencv_highgui -lopencv_core

filter.x: filter.o main.o
	$(CXX) $(CXXFLAGS) -mp $^ $(OPENCV_FLAGS) -lm -o $@

.SUFFIXES: .o .cpp .c
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core filter.x
