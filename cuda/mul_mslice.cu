#include "amul.h"
// dst[i] = a[i] * b[i]
extern "C" __global__ void
mul_mslice(float* __restrict__  dst, float* __restrict__  a_, float a_mul, float* __restrict__ b_, float b_mul, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] = amul(a_, a_mul, i) * amul(b_, b_mul, i);
    }
}

