PROJECT  := testlib
CXX      := g++
CXXFLAGS := -g3 -O3 -march=armv8-a+simd -std=c++17 -I. -Iukernel
LDFLAGS  :=

SOURCES  := test.cpp w8a8.cpp
OBJECTS  := $(SOURCES:.cpp=.o)
UKERNEL_O:= ukernel/w8a8_ukernel_gemv_neon.o

.PHONY: all clean dump

all: build_ukernel $(PROJECT) dump

build_ukernel:
	$(MAKE) -C ukernel

$(PROJECT): $(OBJECTS) $(UKERNEL_O)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dump: $(PROJECT)
	objdump -d $(PROJECT) > $(PROJECT).log

clean:
	rm -f $(OBJECTS) $(PROJECT) $(PROJECT).log
	$(MAKE) -C ukernel clean
