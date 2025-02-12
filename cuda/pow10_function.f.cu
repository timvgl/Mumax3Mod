#include <cuda_runtime.h>
extern "C" __global__ void pow10Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        int n = int(value[idx]);
        // sinc(x) = sin(x)/x for x != 0, and 1 for x == 0.
        value[idx] = powf(10.0f, float(n));
    }
}