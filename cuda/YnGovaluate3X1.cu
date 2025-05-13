#define __CUDA_NO_MATH_OVERLOAD
#include <cuda_runtime.h>
extern "C" __global__ void
YnGovaluate3X1(float* __restrict__ output, float* __restrict__ input, float input2, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = ynf(int(input[idx]), input2);
    }
}