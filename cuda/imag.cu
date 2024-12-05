extern "C" __global__ void
imag(float* __restrict__ output, float* __restrict__ input, int N) {
    int idx =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[2 * idx +1];  // Real parts are at even indices
    }
}