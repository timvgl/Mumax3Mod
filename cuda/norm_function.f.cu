extern "C" __global__ void normGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Assuming norm(x) is defined as the absolute value.
        value[idx] = fabsf(value[idx]);
    }
}
