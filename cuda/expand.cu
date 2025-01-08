#include "stencil.h"

// See crop.go
extern "C" __global__ void
expand(float* __restrict__  dst, int Dx, int Dy, int Dz,
     float* __restrict__  src, int Sx, int Sy, int Sz,
     int Offx, int Offy, int Offz, float value) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < Dx && iy < Dy && iz < Dz) {
        // Calculate corresponding source indices by removing the offset
        int src_x = ix - Offx;
        int src_y = iy - Offy;
        int src_z = iz - Offz;

        // Check if the source indices are within the valid range
        if (src_x >= 0 && src_x < Sx &&
            src_y >= 0 && src_y < Sy &&
            src_z >= 0 && src_z < Sz) {
            // Copy the value from the source to the destination
            dst[index(ix, iy, iz, Dx, Dy, Dz)] = src[index(src_x, src_y, src_z, Sx, Sy, Sz)];
        }
        else {
            // Assign a default value (e.g., 0) for the padded regions
            dst[index(ix, iy, iz, Dx, Dy, Dz)] = value;
        }
    }
}

