extern "C" __global__ void
mulGovaluate3X1(float* __restrict__ output, float* __restrict__ input, float input2, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * input2;
    }
}