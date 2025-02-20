extern "C" __global__ void expGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = expf(value[idx]);
    }
}
