#!/bin/bash
# Rebuild PTX + wrapper for a single NP kernel file: ./rebuild_one.sh melasforcenp
set -e
export PATH=/usr/local/cuda/bin:/usr/local/go/bin:$PATH
cd "$(dirname "$0")"
f="$1"
NVCCFLAGS="-ccbin=/usr/bin/gcc --compiler-options -Werror,-Wall,-U_GNU_SOURCE -Xptxas -O3 -ptx -Wno-deprecated-gpu-targets"
for cc in 50 52 53 60 61 62 70 72 75 80; do
    nvcc $NVCCFLAGS -arch=compute_$cc -code=sm_$cc $f.cu -o ${f}_${cc}.ptx
done
if [ ! -x ./cuda2go ]; then go build cuda2go.go; fi
./cuda2go $f.cu > /dev/null
gofmt -w -s ${f}_wrapper.go > /dev/null
echo REBUILD_OK $f
