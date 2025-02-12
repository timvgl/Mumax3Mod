#include <cuda_runtime.h>

extern "C" __global__ void
ldexpGovaluate3X1(float* __restrict__ output, float* __restrict__ input, float input2, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = ldexpf(input[idx], int(input2));
    }
}