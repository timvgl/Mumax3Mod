#!/bin/bash
# Build PTX + Go wrappers for the magnum.np-port (NP) magnetoelastic kernels.
# Usage: ./build_melas_np.sh [quick]
#   quick: only compile cc=50 (syntax check), no wrapper generation.
set -e
export PATH=/usr/local/cuda/bin:/usr/local/go/bin:$PATH
cd "$(dirname "$0")"

FILES="melasgradjumpnp melasbsigmnp melasbepsnp melasstressnp melasforcenp melasassemblenp melashfieldnp melasenergynp melasstrainnp melasdudnp melasgilbertnp"
CCS="50 52 53 60 61 62 70 72 75 80"
if [ "$1" = "quick" ]; then
    CCS="50"
fi

# NOTE: -std=c++03 (used by the Makefile for ancient CUDA versions) breaks against the
# glibc headers with CUDA >= 12; the default C++ dialect is used instead.
# -U_GNU_SOURCE avoids the CUDA 12.x vs glibc >= 2.41 sinpi/cospi declaration clash.
NVCCFLAGS="-ccbin=/usr/bin/gcc --compiler-options -Werror,-Wall,-U_GNU_SOURCE -Xptxas -O3 -ptx -Wno-deprecated-gpu-targets"

for f in $FILES; do
    for cc in $CCS; do
        echo "nvcc $f.cu -> ${f}_${cc}.ptx"
        nvcc $NVCCFLAGS -arch=compute_$cc -code=sm_$cc $f.cu -o ${f}_${cc}.ptx
    done
done

if [ "$1" != "quick" ]; then
    if [ ! -x ./cuda2go ]; then
        go build cuda2go.go
    fi
    for f in $FILES; do
        echo "cuda2go $f.cu"
        ./cuda2go $f.cu > /dev/null
        gofmt -w -s ${f}_wrapper.go > /dev/null
    done
fi
echo BUILD_OK
