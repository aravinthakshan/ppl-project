# Builds dct_cpu (always) and dct_gpu (only if nvcc is on PATH).
# Usage:
#   make                -> builds everything available
#   make cpu            -> CPU only
#   make gpu            -> CUDA only
#   make OMP=1          -> CPU with OpenMP
#   make clean

CC      ?= cc
CFLAGS  ?= -O3 -ffast-math -march=native -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

ifdef OMP
  # clang on macOS needs libomp via homebrew; gcc on linux: -fopenmp works.
  ifneq (,$(findstring clang,$(shell $(CC) --version 2>/dev/null)))
    OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp 2>/dev/null)/include
    OMP_LDLIBS ?= -lomp -L$(shell brew --prefix libomp 2>/dev/null)/lib
  else
    OMP_CFLAGS ?= -fopenmp
    OMP_LDLIBS ?= -fopenmp
  endif
  CFLAGS += $(OMP_CFLAGS)
  LDLIBS += $(OMP_LDLIBS)
endif

NVCC    := $(shell command -v nvcc 2>/dev/null)
NVCCFLAGS ?= -O3 -std=c++14 -arch=sm_70

.PHONY: all cpu gpu clean
all: cpu $(if $(NVCC),gpu,)

cpu: dct_cpu
gpu: dct_gpu

dct_cpu: dct_cpu.c common.c common.h
	$(CC) $(CFLAGS) -o $@ dct_cpu.c common.c $(LDLIBS)

dct_gpu: dct_gpu.cu common.c common.h
ifdef NVCC
	$(NVCC) $(NVCCFLAGS) -o $@ dct_gpu.cu common.c
else
	@echo "nvcc not found on PATH; skipping CUDA build."
	@echo "To build: install CUDA toolkit and re-run 'make gpu'."
endif

clean:
	rm -f dct_cpu dct_gpu *.o
