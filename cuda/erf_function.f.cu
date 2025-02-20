extern "C" __global__ void erfGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = erff(value[idx]);
    }
}
