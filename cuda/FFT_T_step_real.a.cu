#include "stencil.h"
#include <cuComplex.h>

__device__ __forceinline__ cuComplex my_cexpf (cuComplex z) {
    cuComplex res;
    float t = expf (z.x);
    sincosf (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

extern "C" __global__ void
FFT_Step_Real(float* __restrict__ dst, float* __restrict__ src1, float* __restrict__ src2, int Nx, int Ny, int Nz, float phase, float n) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int IReal = idx(2*ix, iy, iz);
    int IImag = idx(2*ix+1, iy, iz);
    cuComplex src2Complex = make_cuComplex(src2[idx(ix, iy, iz)], 0.);
    cuComplex angle = make_cuComplex(0, phase);
    cuComplex expComplex = my_cexpf(angle);
    float dstReal = src2Complex.x * expComplex.x - src2Complex.y * expComplex.y; // Real part
    float dstImag = src2Complex.x * expComplex.y + src2Complex.y * expComplex.x; // Imaginary part
    dst[IReal] = dstReal / n + src1[IReal];
    dst[IImag] = dstImag / n + src1[IImag];
}
