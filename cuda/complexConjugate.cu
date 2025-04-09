extern "C" __global__ void
complexConjugate(float* __restrict__ output, float* __restrict__ input, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Store amplitude and phase alternately
        output[2 * idx] = input[2 * idx];
        output[2 * idx + 1] = -input[2 * idx + 1];
    }
}