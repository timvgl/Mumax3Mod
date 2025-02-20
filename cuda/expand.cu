#include "stencil.h"

// See crop.go
extern "C" __global__ void
expand(float* __restrict__  dst, int Dx, int Dy, int Dz,
     float* __restrict__  src, int Sx, int Sy, int Sz,
     int Offx, int Offy, int Offz,
     int ShiftX, int ShiftY, int ShiftZ, float value) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < Dx && iy < Dy && iz < Dz) {
        int adjust_x = (Sx % 2 == 1 && Dx % 2 == 0 || Sx % 2 == 0 && Dx % 2 == 1) ? 1 : 0;
        int adjust_y = (Sy % 2 == 1 && Dy % 2 == 0 || Sy % 2 == 0 && Dy % 2 == 1) ? 1 : 0;
        int adjust_z = (Sz % 2 == 1 && Dz % 2 == 0 || Sz % 2 == 0 && Dz % 2 == 1) ? 1 : 0;
        int adjust_x = (Sx % 2 == 1 && Dx % 2 == 0 || Sx % 2 == 0 && Dx % 2 == 1) ? 1 : 0;
        int adjust_y = (Sy % 2 == 1 && Dy % 2 == 0 || Sy % 2 == 0 && Dy % 2 == 1) ? 1 : 0;
        int adjust_z = (Sz % 2 == 1 && Dz % 2 == 0 || Sz % 2 == 0 && Dz % 2 == 1) ? 1 : 0;
        // Calculate corresponding source indices by removing the offset
        int src_x = ix - Offx - adjust_x - ShiftX;
        int src_y = iy - Offy - adjust_y - ShiftY;
        int src_z = iz - Offz - adjust_z - ShiftZ;

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
