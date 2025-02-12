extern "C" __global__ void
real(float* __restrict__ output, float* __restrict__ input, int N) {
    int idx =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[2 * idx];  // Real parts are at even indices
    }
}