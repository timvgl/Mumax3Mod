#include "stencil.h"

// CUDA kernel for 3D addition with flexible broadcasting.
// Inputs:
//   out  - pointer to the flattened output array of shape (Nx, Ny, Nz).
//   a    - pointer to a flattened array representing input A.
//   b    - pointer to a flattened array representing input B.
//   Nx, Ny, Nz 
//         - dimensions of the output array.
//   aNx, aNy, aNz 
//         - dimensions of array A. Each dimension should either equal the corresponding
//           output dimension or be 1 (in which case that axis is broadcast).
//   bNx, bNy, bNz 
//         - dimensions of array B, following the same convention.
extern "C" __global__ void
addGovaluate3X3(float *out, float *a, float *b, 
                 int Nx, int Ny, int Nz,
                 int aNx, int aNy, int aNz,
                 int bNx, int bNy, int bNz)
{
    // Compute the 3D coordinates for this thread.
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate [0, Nx)
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate [0, Ny)
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // z-coordinate [0, Nz)

    // Process only threads that fall within the output bounds.
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
        // Write the elementwise sum to the output.
        out[outIndex] = a[aIndex] + b[bIndex];
    }
}
