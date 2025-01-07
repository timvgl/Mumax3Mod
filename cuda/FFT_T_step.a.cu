#include "stencil.h"
#include <cuComplex.h>
#include <stdint.h>

__device__ __forceinline__ cuComplex my_cexpf (cuComplex z) {
    cuComplex res;
    float t = expf (z.x);
    sincosf (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

__device__ __forceinline__ cuComplex FFT_Stepper (float angleReal, float angleImag, cuComplex src2Complex) {
    cuComplex angle = make_cuComplex(angleReal, angleImag);
    cuComplex expComplex = my_cexpf(angle);
    float dstReal = src2Complex.x * expComplex.x - src2Complex.y * expComplex.y; // Real part
    float dstImag = src2Complex.x * expComplex.y + src2Complex.y * expComplex.x; // Imaginary part
    return make_cuComplex(dstReal, dstImag);
}


extern "C" __global__ void
FFT_Step(uint8_t* __restrict__ dst, uint8_t* __restrict__ src1, uint8_t* __restrict__ src2, int Nx, int Ny, int Nz, float angleReal, float angleImag, int n) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx / 2 || iy >= Ny || iz >= Nz) {
        return;
    }
    int IReal = idx(2*ix, iy, iz);
    int IImag = idx(2*ix+1, iy, iz);
    cuComplex src2Complex = make_cuComplex(src2[IReal], src2[IImag]);
    cuComplex dstVal = FFT_Stepper(angleReal, angleImag, src2Complex);
    
    dst[IReal] = dstVal.x / n + src1[IReal];
    dst[IImag] = dstVal.y / n + src1[IImag];
}