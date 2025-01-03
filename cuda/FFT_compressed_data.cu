#include <cuda_runtime.h>
#include <stdint.h>
#include <cuComplex.h>

// Example dummy function that doubles both real and imaginary parts.
// You can replace this with any complex operation you like.
__device__ void dummyComplexOp(float& realVal, float& imagVal)
{
    // e.g., multiply the complex number by 2
    realVal *= 2.0f;
    imagVal *= 2.0f;
}

__device__ __forceinline__ cuComplex my_cexpf(float phase) {
    cuComplex res;
    sincosf(phase, &res.y, &res.x); // Compute sin(phase) and cos(phase)
    return res; // Return the result as a cuComplex
}


__device__ __forceinline__ cuComplex FFT_Stepper(float phase, cuComplex src2Complex) {
    cuComplex expComplex = my_cexpf(phase);
    float dstReal = src2Complex.x * expComplex.x - src2Complex.y * expComplex.y; // Real part
    float dstImag = src2Complex.x * expComplex.y + src2Complex.y * expComplex.x; // Imaginary part
    return make_cuComplex(dstReal, dstImag);
}

/**
 * transformCompressedComplexKernel:
 *  - d_inCompressed:  Input compressed array, in 10-byte "complex runs"
 *  - d_outCompressed: Output compressed array (also 10-byte runs)
 *  - totalRuns:       Number of complex runs (each run = 10 bytes)
 *
 * Each run layout:
 *   0: RealMarker (1 byte)
 *   1..4: Real payload (4 bytes) => either runLength or float bits
 *   5: ImagMarker (1 byte)
 *   6..9: Imag payload (4 bytes) => either runLength or float bits
 */
extern "C" __global__ void
transformCompressedComplexKernel(uint8_t* __restrict__ d_outCompressed,
                                 uint8_t* __restrict__ d_inCompressed,
                                 float phase,
                                 size_t totalRuns)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each run is 10 bytes
    if (idx < totalRuns)
    {
        size_t offset = idx * 10;

        // 1) ----------------------- REAL PART -----------------------
        uint8_t realMarker = d_inCompressed[offset + 0];
        // Write the real marker to the output as is
        d_outCompressed[offset + 0] = realMarker;

        float realVal = 0.0f;

        if (realMarker == 0x00) 
        {
            // Zero run => read run length from next 4 bytes
            // We'll do a "dummyComplexOp(0.0, something)" for the real part
            // But typically we just keep it zero
            d_outCompressed[offset + 1] = d_inCompressed[offset + 1];
            d_outCompressed[offset + 2] = d_inCompressed[offset + 2];
            d_outCompressed[offset + 3] = d_inCompressed[offset + 3];
            d_outCompressed[offset + 4] = d_inCompressed[offset + 4];

            // So realVal stays 0.0f
        }
        else if (realMarker == 0xFF)
        {
            // Non-zero run => read float bits
            union {
                uint8_t b[4];
                float   f;
            } u;

            u.b[0] = d_inCompressed[offset + 1];
            u.b[1] = d_inCompressed[offset + 2];
            u.b[2] = d_inCompressed[offset + 3];
            u.b[3] = d_inCompressed[offset + 4];

            realVal = u.f;

            // We won't update run marker or structure, but we will store back updated float
        }
        else 
        {
            // Unknown marker => just copy
            for (int i = 1; i <= 4; i++) {
                d_outCompressed[offset + i] = d_inCompressed[offset + i];
            }
        }

        // 2) ----------------------- IMAG PART -----------------------
        uint8_t imagMarker = d_inCompressed[offset + 5];
        // Write the imag marker to the output as is
        d_outCompressed[offset + 5] = imagMarker;

        float imagVal = 0.0f;

        if (imagMarker == 0x00) 
        {
            // Zero run for imaginary part
            d_outCompressed[offset + 6] = d_inCompressed[offset + 6];
            d_outCompressed[offset + 7] = d_inCompressed[offset + 7];
            d_outCompressed[offset + 8] = d_inCompressed[offset + 8];
            d_outCompressed[offset + 9] = d_inCompressed[offset + 9];

            // imagVal stays 0.0f
        }
        else if (imagMarker == 0xFF)
        {
            // Non-zero run => read float bits for imaginary part
            union {
                uint8_t b[4];
                float   f;
            } u;

            u.b[0] = d_inCompressed[offset + 6];
            u.b[1] = d_inCompressed[offset + 7];
            u.b[2] = d_inCompressed[offset + 8];
            u.b[3] = d_inCompressed[offset + 9];

            imagVal = u.f;
        }
        else 
        {
            // Unknown marker => just copy
            for (int i = 6; i <= 9; i++) {
                d_outCompressed[offset + i] = d_inCompressed[offset + i];
            }
        }

        // 3) --------------- Apply the dummy complex operation ---------------
        // realVal, imagVal are now the "uncompressed" values (0 or the actual float)
        cuComplex FFT_data = FFT_Stepper(phase, make_cuComplex(realVal, imagVal));
        realVal = FFT_data.x;
        imagVal = FFT_data.y;

        // 4) --------------- Write updated results back ---------------
        // (We only overwrite if the marker was 0xFF; 
        //  if it's 0x00, we keep the same run length, meaning still zeros.)

        // Real part
        if (realMarker == 0xFF)
        {
            union {
                float   f;
                uint8_t b[4];
            } u;
            u.f = realVal;
            d_outCompressed[offset + 1] = u.b[0];
            d_outCompressed[offset + 2] = u.b[1];
            d_outCompressed[offset + 3] = u.b[2];
            d_outCompressed[offset + 4] = u.b[3];
        }

        // Imag part
        if (imagMarker == 0xFF)
        {
            union {
                float   f;
                uint8_t b[4];
            } u;
            u.f = imagVal;
            d_outCompressed[offset + 6] = u.b[0];
            d_outCompressed[offset + 7] = u.b[1];
            d_outCompressed[offset + 8] = u.b[2];
            d_outCompressed[offset + 9] = u.b[3];
        }
    }
}
