#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
KineticEnergy(float* __restrict__ energy, 
                 float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz,
                 float* __restrict__ rho_, float rho_mul, int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    //Do nothing if cell position is not in mesh
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // Central cell
    int I = idx(ix, iy, iz);

    float rho = amul(rho_, rho_mul, I);

    energy[I] = 0.5* rho* (dux[I]*dux[I]+duy[I]*duy[I]+duz[I]*duz[I]);
}

