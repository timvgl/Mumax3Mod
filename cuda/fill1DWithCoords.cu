#include "stencil.h"
extern "C" __global__ void
fill1DWithCoords(float* __restrict__  dst, float factor, int Nx, int Ny, int Nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix < Nx && iy < Ny && iz < Nz) {
        dst[idx(ix, iy, iz)] = (float(ix) + float(iy) + float(iz))*factor;
    }
}
