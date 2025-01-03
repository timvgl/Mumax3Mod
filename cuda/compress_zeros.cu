#include <stdint.h>

// rle_compressor_single_kernel.cu
extern "C" __global__ void
compressRLESingleKernel(uint8_t* __restrict__ d_output, float* __restrict__ d_input, size_t numElements, size_t* d_outputSize) {
    extern __shared__ unsigned int s_data[];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Mark run starts
    unsigned char runStart = 0;
    if (idx < numElements) {
        if (d_input[idx] == 0.0f) {
            if (idx == 0 || idx > 0 && d_input[idx - 1] != 0.0f) {
                runStart = 1; // Start of zero run
            }
        }
        else {
            runStart = 1; // Non-zero element as single run
        }
    }
    s_data[threadIdx.x] = runStart;
    __syncthreads();

    // Step 2: Perform exclusive scan within the block
    unsigned int scan = 0;
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        if (threadIdx.x >= offset)
            scan += s_data[threadIdx.x - offset];
        __syncthreads();
        if (threadIdx.x >= offset)
            s_data[threadIdx.x] += s_data[threadIdx.x - offset];
        __syncthreads();
    }
    // Now, s_data contains inclusive scan
    unsigned int runIdx = s_data[threadIdx.x] - s_data[threadIdx.x - 1] * runStart;
    if (runStart == true) {
        if (d_input[idx] == 0.0f) {
            // Zero run: count the number of consecutive zeros
            unsigned int runLength = 1;
            while ((idx + runLength) < numElements && d_input[idx + runLength] == 0.0f) {
                runLength++;
                if (runLength >= 4294967295) break; // Prevent overflow
            }

            // Write ZERO_MARKER
            d_output[runIdx * 5] = 0x00;
            // Write runLength as 4 bytes (little endian)
            d_output[runIdx * 5 + 1] = (uint8_t)(runLength & 0xFF);
            d_output[runIdx * 5 + 2] = (uint8_t)((runLength >> 8) & 0xFF);
            d_output[runIdx * 5 + 3] = (uint8_t)((runLength >> 16) & 0xFF);
            d_output[runIdx * 5 + 4] = (uint8_t)((runLength >> 24) & 0xFF);
        }
        else {
            // Non-zero run: write NON_ZERO_MARKER and float value
            d_output[runIdx * 5] = 0xFF;
            // Copy float bytes (assuming little endian)
            uint8_t* floatBytes = (uint8_t*)&d_input[idx];
            d_output[runIdx * 5 + 1] = floatBytes[0];
            d_output[runIdx * 5 + 2] = floatBytes[1];
            d_output[runIdx * 5 + 3] = floatBytes[2];
            d_output[runIdx * 5 + 4] = floatBytes[3];
        }
    }

    // Step 3: Atomic add to output size
    if (runStart && threadIdx.x == 0) {
        atomicAdd((unsigned long long*)d_outputSize, (unsigned long long)((runIdx + 1) * 5));
    }
}
