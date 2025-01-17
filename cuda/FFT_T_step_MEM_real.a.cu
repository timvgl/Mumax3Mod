#include "stencil.h"
#include <cuComplex.h>

__device__ __forceinline__ cuComplex my_cexpf(cuComplex z) {
    cuComplex res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

extern "C" __global__ void
FFT_Step_MEM_real(float* __restrict__ dst, float* __restrict__ src1, float* __restrict__ src2, int Nx, int Ny, int Nz, int Nf, float minF, float dF, float t, float n) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= Nx * Ny * Nz / 2) {
        return;
    }
    // Calculate 3D indices for src2
    int ix = i % Nx;
    int iy = ((i - ix) / Nx) % Ny;
    int iz = ((i - iy * Nx - ix) / (Nx * Ny)) % Nz;

    int IIReal = index(ix, iy, iz, Nx/2, Ny, Nz);

    for (int fi = 0;
         fi < Nf;
         fi++) {
        int IReal = idx4D(2 * ix, iy, iz, fi);
        int IImag = idx4D(2 * ix + 1, iy, iz, fi);

        // Load src2 data
        cuComplex src2Complex = make_cuComplex(src2[IIReal], 0);

        // Calculate phase and exponential
        float phase = -2 * M_PI * (minF + dF * float(fi)) * t;
        cuComplex angle = make_cuComplex(0, phase);
        cuComplex expComplex = my_cexpf(angle);

        // Calculate real and imaginary parts
        float dstReal = src2Complex.x * expComplex.x - src2Complex.y * expComplex.y;
        float dstImag = src2Complex.x * expComplex.y + src2Complex.y * expComplex.x;

        // Update dst with the new data and accumulated data from src1
        dst[IReal] = dstReal / n + src1[IReal];
        dst[IImag] = dstImag / n + src1[IImag];
    }
}