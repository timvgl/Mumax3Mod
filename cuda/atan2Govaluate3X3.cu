#include "stencil.h"
// CUDA kernel for 3D addition with support for both broadcast and full-size inputs.
// Inputs:
//   a      - pointer to a flattened array representing input A.
//   b      - pointer to a flattened array representing input B.
//   out    - pointer to the flattened output array of shape (Nx, Ny, Nz).
//   Nx, Ny, Nz - dimensions of the output array.
//   a_axis - if >= 0, indicates the axis along which A is non-singleton (0 for (Nx,1,1), 1 for (1,Ny,1), 2 for (1,1,Nz)).
//            if -1, A is assumed to be a full-size array of shape (Nx,Ny,Nz).
//   b_axis - similar to a_axis but for input B.
extern "C" __global__ void
atan2Govaluate3X3(float *out, float *a, float *b, 
    int Nx, int Ny, int Nz,
    int aNx, int aNy, int aNz,
    int bNx, int bNy, int bNz) {
    // Compute the 3D coordinates for this thread.
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate (0 <= ix < Nx)
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate (0 <= iy < Ny)
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // z-coordinate (0 <= iz < Nz)

    // Check bounds.
    if (ix < Nx && iy < Ny && iz < Nz) {
        // Compute the flattened index into the output array.
        int outIndex = idx(ix, iy, iz);

        // For A, if the dimension is 1 then broadcast (always index 0); otherwise, use the output coordinate.
        int a_ix = (aNx == 1 ? 0 : ix);
        int a_iy = (aNy == 1 ? 0 : iy);
        int a_iz = (aNz == 1 ? 0 : iz);
        int aIndex = index(a_ix, a_iy, a_iz, aNx, aNy, aNz);

        // For B, do the same.
        int b_ix = (bNx == 1 ? 0 : ix);
        int b_iy = (bNy == 1 ? 0 : iy);
        int b_iz = (bNz == 1 ? 0 : iz);
        int bIndex = index(b_ix, b_iy, b_iz, bNx, bNy, bNz);

        float a_val = a[aIndex];
        float b_val = b[bIndex];
        out[outIndex] = atan2f(a_val, b_val);
    }
}
