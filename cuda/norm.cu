#include <stdint.h>
#include <stdio.h>
#include "stencil.h"
#include "amul.h"

// dst[i] = grad(a[3])
extern "C" __global__ void
pointwise_norm(	float* __restrict__  dst,
				float* __restrict__  inputx, float* __restrict__  inputy, float* __restrict__  inputz,
				int Nx, int Ny, int Nz, 
				uint8_t PBC) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;


    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
    	return;
    }

	int I = idx(ix, iy, iz);
    dst[I] = sqrt(pow(inputx[I], 2) + pow(inputy[I], 2) + pow(inputz[I], 2));
}

