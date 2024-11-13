#include <stdint.h>
#include <stdio.h>
#include "stencil.h"
#include "amul.h"

// dst[i] = grad(a[3])
extern "C" __global__ void
scalarToVector(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
								float* __restrict__ a_, float a_mul,
								float* __restrict__ b_, float b_mul,
								float* __restrict__ c_, float c_mul,
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
    dstx[I] = amul(a_, a_mul, I);
	dsty[I] = amul(b_, b_mul, I);
	dstz[I] = amul(c_, c_mul, I);
}

