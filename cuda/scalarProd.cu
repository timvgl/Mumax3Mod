#include <stdint.h>
#include <stdio.h>
#include "stencil.h"
#include "amul.h"

// dst[i] = grad(a[3])
extern "C" __global__ void
scalarProd(float* __restrict__ res,
            float* __restrict__  ax, float* __restrict__  ay, float* __restrict__  az,
            float* __restrict__  bx, float* __restrict__  by, float* __restrict__  bz,
            int Nx, int Ny, int Nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;


    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
        return;
    }

	int I = idx(ix, iy, iz);
    res[I] = ax[I]*bx[I] + ay[I]*by[I] + az[I]*bz[I];
}