extern "C" __global__ void
atan2Govaluate3X1(float* __restrict__ output, float* __restrict__ input, float input2, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = atan2f(input[idx], input2);
    }
}