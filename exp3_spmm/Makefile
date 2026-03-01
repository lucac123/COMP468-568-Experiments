NVCC ?= nvcc
ARCH ?= -arch=sm_70
CXX = g++
CXXFLAGS ?= -O3 -std=c++17

all: spmm_baseline spmm_opt

spmm_baseline: spmm_baseline.cu spmm_ref.cpp
	$(NVCC) $(ARCH) $(CXXFLAGS) spmm_baseline.cu spmm_ref.cpp -o spmm_baseline

spmm_opt: spmm_opt.cu spmm_ref.cpp
	$(NVCC) $(ARCH) $(CXXFLAGS) spmm_opt.cu spmm_ref.cpp -o spmm_opt

clean:
	rm -f spmm_baseline spmm_opt
