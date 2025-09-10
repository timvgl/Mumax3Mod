# Build mumax3 CUDA kernels and create Go wrappers for the compute capabilities listed in $CUDA_CC.
# If $CUDA_CC is not defined, it defaults to "30".
#
# The compiler ${CUDA_HOME}/bin/nvcc is used to compile the CUDA kernels.
# If CUDA_HOME is not defined, it looks for nvcc in the PATH.
#
# Examples:
#    make
#    make CUDA_CC=70
#    make CUDA_CC="30 32 35 37 50 52 53 60 61 62 70"
#    make CUDA_HOME="/usr/local/cuda-9.0" CUDA_CC="30 32 35 37 50 52 53 60 61 62 70"
#
# Different CUDA versions support different compute capabilities:
# CUDA SDK  9.0 supports: 30 32 35 37 50 52 53 60 61 62 70
# CUDA SDK  9.1 supports: 30 32 35 37 50 52 53 60 61 62 70
# CUDA SDK  9.2 supports: 30 32 35 37 50 52 53 60 61 62 70 72
# CUDA SDK 10.0 supports: 30 32 35 37 50 52 53 60 61 62 70 72 75
# CUDA SDK 10.1 supports: 30 32 35 37 50 52 53 60 61 62 70 72 75
# CUDA SDK 10.2 supports: 30 32 35 37 50 52 53 60 61 62 70 72 75
# CUDA SDK 11.0 supports: 30 32 35 37 50 52 53 60 61 62 70 72 75 80

SHELL = /bin/bash

# Export CUDA_CC so that subprocesses can see it.
export CUDA_CC = 30 32 35 37 50 52 53 60 61 62 70 72 75 80

# Get number of cores for parallel builds.
NUM_CORES := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

# If the user does not provide a -j option and we're in the top-level make,
# add -j<NUM_CORES> to MAKEFLAGS.
ifneq ($(filter -j%,$(MAKEFLAGS)),)
	# User provided -j, do nothing.
else ifneq ($(MAKELEVEL),0)
	# Sub-make: do not override -j.
else
	MAKEFLAGS += -j$(NUM_CORES)
endif

# Use NVCC from CUDA_HOME if set, otherwise look for nvcc in PATH.
ifeq ($(CUDA_HOME),)
	NVCC = nvcc
else 
	NVCC = $(CUDA_HOME)/bin/nvcc
endif

# If CUDA_CC is not defined, default to compute capability 3.0.
ifeq ($(CUDA_CC),)
	CUDA_CC = 30
endif

# Set the gcc host compiler for nvcc.
ifeq ($(NVCC_CCBIN),)
	NVCC_CCBIN = /usr/bin/gcc
endif

# Retrieve CUDA version from nvcc.
CUDA_VERSION := $(shell $(NVCC) --version | grep "Cuda compilation" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | head -n 1)

# Set compatibility flags; for CUDA versions 7.0, 7.5, and 8.0 the flag is not needed.
NVCC_COMPATIBILITY_FLAGS := -std=c++03
ifneq (,$(filter 7.0 7.5 8.0,$(CUDA_VERSION)))
	NVCC_COMPATIBILITY_FLAGS :=
endif

NVCCFLAGS = $(NVCC_COMPATIBILITY_FLAGS) -ccbin=$(NVCC_CCBIN) --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx

.PHONY: all split_cuda_functions rebuild wrappers awrappers clean realclean

# The "all" target depends on "rebuild", which in turn depends on "split_cuda_functions".
all: rebuild

# Run the split script that generates additional CUDA files (e.g., *.f.cu).
split_cuda_functions: cuda_functions.cu
	./split_cuda_functions.sh

# After splitting, re-run the make targets so that newly generated CUDA files are detected.
rebuild: #split_cuda_functions
	$(MAKE) wrappers awrappers
	@echo "Built with CUDA version $(CUDA_VERSION)"
	go install -v

# Build Go wrappers for all *.cu files (excluding cuda_functions.cu and *.a.cu files).
wrappers:
	$(MAKE) $(patsubst %.cu,%_wrapper.go,$(filter-out cuda_functions.cu $(wildcard *.a.cu),$(wildcard *.cu)))

# Pattern rule: create a Go wrapper from a .cu file.
%_wrapper.go: %.cu cuda2go
	@rm -f $(basename $<)*.ptx
	@for cc in $(CUDA_CC); do \
		echo $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
		$(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $<)_$$cc.ptx ;\
	done
	@./cuda2go $< > /dev/null
	@gofmt -w -s -l $@ > /dev/null

# Build the cuda2go executable from its source.
cuda2go: cuda2go.go
	rm -vf cuda2go
	go build $<

# Build Go wrappers for all *.a.cu files.
awrappers:
	$(MAKE) $(patsubst %.a.cu,%_wrapper.a.go,$(wildcard *.a.cu))

# Pattern rule: create a Go wrapper from a .a.cu file.
%_wrapper.a.go: %.a.cu acuda2go
	@rm -f $(basename $(basename $<))*.ptx.a
	@for cc in $(CUDA_CC); do \
		echo $(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $(basename $<))_$$cc.ptx.a ;\
		$(NVCC) $(NVCCFLAGS) -arch=compute_$$cc -code=sm_$$cc $< -o $(basename $(basename $<))_$$cc.ptx.a ;\
	done
	@./acuda2go $< > /dev/null
	@gofmt -w -s -l $@ > /dev/null

# Build the acuda2go executable from its source.
acuda2go: acuda2go.go
	rm -vf acuda2go
	go build $<

# Clean temporary files.
clean:
	rm -vf *.ptx cuda2go
	rm -vf *.ptx.a acuda2go

# Remove all generated files.
realclean:
	rm -vf *_wrapper.go *.ptx cuda2go
	rm -vf *_wrapper.a.go *.ptx.a acuda2go
	rm -vf *.f.cu
