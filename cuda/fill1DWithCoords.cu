#include "stencil.h"
extern "C" __global__ void
fill1DWithCoords(float* __restrict__  dst, float factor, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = float(idx)*factor;
    }
}
