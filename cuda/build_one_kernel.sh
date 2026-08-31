#!/bin/bash
# Build the PTX variants + Go wrapper for ONE mumax CUDA kernel.
#
# Usage:  ./build_one_kernel.sh <name>.cu
#
# Why this exists (and why plain `make` fails here):
#   glibc >= 2.41 declares the C23 functions cospi/sinpi/tanpi/rsqrt in
#   <bits/mathcalls.h>. CUDA <= 12.8 declares the same names as __device__
#   builtins in crt/math_functions.h without an exception specification, so
#   nvcc's frontend reports
#       "exception specification is incompatible with that of previous function".
#   This breaks EVERY kernel in this directory, not just new ones (verify with
#   `nvcc ... exchange.cu`). Fixed in CUDA >= 12.9.
#
#   Workaround used below: shadow <bits/mathcalls.h> with a stub on the include
#   path that switches the C23 feature gate off and then chain-loads the real
#   header via #include_next. Nothing else in the build changes.
#
# Remove this script (and use `make`) once CUDA >= 12.9 is installed.

set -e

CU="$1"
if [ -z "$CU" ] || [ ! -f "$CU" ]; then
    echo "usage: $0 <kernel>.cu" >&2
    exit 1
fi
BASE="${CU%.cu}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
GO="${GO:-/usr/local/go/bin/go}"
CUDA_CC="${CUDA_CC:-50 52 53 60 61 62 70 72 75 80}"

# --- glibc/CUDA shim ---------------------------------------------------------
SHIM="$(mktemp -d)"
mkdir -p "$SHIM/bits"
cat > "$SHIM/bits/mathcalls.h" <<'EOF'
/* Build shim: hide the glibc >= 2.41 C23 declarations (cospi/sinpi/tanpi/rsqrt)
   that clash with the __device__ builtins of CUDA <= 12.8. */
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#include_next <bits/mathcalls.h>
EOF
trap 'rm -rf "$SHIM"' EXIT

NVCCFLAGS="-std=c++03 -ccbin=/usr/bin/gcc --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx -Wno-deprecated-gpu-targets -I$SHIM"

rm -f "${BASE}"_*.ptx
for cc in $CUDA_CC; do
    echo "  nvcc -arch=compute_$cc  $CU"
    "$NVCC" $NVCCFLAGS -arch=compute_"$cc" -code=sm_"$cc" "$CU" -o "${BASE}_${cc}.ptx"
done

# --- Go wrapper --------------------------------------------------------------
if [ ! -x ./cuda2go ]; then
    echo "  building cuda2go"
    "$GO" build cuda2go.go
fi
./cuda2go "$CU" > /dev/null
GOFMT="$(dirname "$GO")/gofmt"
[ -x "$GOFMT" ] || GOFMT="gofmt"
"$GOFMT" -w -s "${BASE}_wrapper.go"

echo "OK: ${BASE}_wrapper.go ($(wc -c < "${BASE}_wrapper.go") bytes)"
