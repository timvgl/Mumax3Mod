extern "C" __global__ void expm1Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = expm1f(value[idx]);
    }
}
