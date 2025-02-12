extern "C" __global__ void
divGovaluate1X3(float* __restrict__ output, float input2, float* __restrict__ input, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input2 / input[idx];
    }
}