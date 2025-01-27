extern "C" __global__ void
complexToPolar(float* __restrict__ output, float* __restrict__ input, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < N) {
        float real = input[2 * idx];      // Real part at even indices
        float imag = input[2 * idx + 1]; // Imaginary part at odd indices

        // Compute amplitude
        float amplitude = sqrtf(real * real + imag * imag);
        
        // Compute phase
        float phase = atan2f(imag, real);

        // Store amplitude and phase alternately
        output[2 * idx] = amplitude;
        output[2 * idx + 1] = phase;
    }
}