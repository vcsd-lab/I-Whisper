# this Makefile is for building the ukernel as a shared library
CXX     := g++
CXXFLAGS:= -O3 -march=armv8-a+simd -std=c++17 -fPIC

SOURCES := w8a8_ukernel_gemv_neon.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS)