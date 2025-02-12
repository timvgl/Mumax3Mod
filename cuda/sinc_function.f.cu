extern "C" __global__ void sincGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = value[idx];
        // sinc(x) = sin(x)/x for x != 0, and 1 for x == 0.
        value[idx] = (fabsf(x) < 1e-7f) ? 1.0f : sinf(x) / x;
    }
}
