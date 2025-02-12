extern "C" __global__ void ilogbGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        // ilogbf returns an int; convert the result to float.
        value[idx] = (float) ilogbf(value[idx]);
    }
}
