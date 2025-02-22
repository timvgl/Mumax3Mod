#include "amul.h"

// dst[i] = a[i] / b[i]
extern "C" __global__ void
pointwise_div_mslice(float* __restrict__  dst, float* __restrict__  a_, float a_mul, float* __restrict__ b_, float b_mul, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        float a = amul(a_, a_mul, i);
        float b = amul(b_, b_mul, i);
        if (b != 0.0f) {
            dst[i] = a / b;
        } else {
            dst[i] = 0.0f;
        }
    }
}

