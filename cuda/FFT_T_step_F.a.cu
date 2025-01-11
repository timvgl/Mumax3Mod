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
FFT_Step_F(float* __restrict__ dst, float* __restrict__ src1, float* __restrict__ src2, int Nx, int Ny, int Nz, int Nf, float minF, float dF, float t, float n) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= Nx*Ny*Nz*Nf) {
        return;
    }
    int ix = i % Nx;
    int iy = ((i - ix) / Nx) % Ny;
    int iz = ((i - iy * Nx - ix) / (Nx * Ny)) % Nz;
    int fi = ((i - iz * Nx * Ny - iy * Nx - ix) / (Nx * Ny * Nz)) % Nf;
    int IReal = idx4D(2*ix, iy, iz, fi);
    int IImag = idx4D(2*ix+1, iy, iz, fi);
    int iix = i % Nx;
    int iiy = ((i - iix) / Nx) % Ny;
    int iiz = ((i - iiy * Nx - iix) / (Nx * Ny)) % Nz;
    int IIReal = idx(2*iix, iiy, iiz);
    int IIImag = idx(2*iix+1, iiy, iiz);
    //printf("%d\n",iix)
    cuComplex src2Complex = make_cuComplex(src2[IIReal], src2[IIImag]);
    //printf("%f\n", minF + dF * fi);
    float phase = -2 * M_PI * (minF + dF * fi) * t;
    cuComplex angle = make_cuComplex(0, phase);
    cuComplex expComplex = my_cexpf(angle);
    float dstReal = src2Complex.x * expComplex.x - src2Complex.y * expComplex.y; // Real part
    float dstImag = src2Complex.x * expComplex.y + src2Complex.y * expComplex.x; // Imaginary part
    dst[IReal] = dstReal / n + src1[IReal];
    dst[IImag] = dstImag / n + src1[IImag];
}