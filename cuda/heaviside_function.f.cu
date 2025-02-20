extern "C" __global__ void heavisideGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = value[idx];
        if (x < 0.0f)
            value[idx] = 0.0f;
        else if (x > 0.0f)
            value[idx] = 1.0f;
        else
            value[idx] = 0.5f;
    }
}
