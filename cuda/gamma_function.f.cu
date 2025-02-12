extern "C" __global__ void gammaGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Use tgammaf to compute the Gamma function.
        value[idx] = tgammaf(value[idx]);
    }
}
