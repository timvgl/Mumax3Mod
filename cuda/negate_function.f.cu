extern "C" __global__ void negateGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        // sinc(x) = sin(x)/x for x != 0, and 1 for x == 0.
        value[idx] = -value[idx];
    }
}