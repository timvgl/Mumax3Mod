#include "stencil.h"
#include <cuda_runtime.h>
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
ldexpGovaluate3X3(float *out, float *a, float *b, 
    int Nx, int Ny, int Nz,
    int a_axis, int b_axis)
{
    // Compute the 3D coordinates for this thread.
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate (0 <= ix < Nx)
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate (0 <= iy < Ny)
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // z-coordinate (0 <= iz < Nz)

    // Check bounds.
    if (ix < Nx && iy < Ny && iz < Nz) {
        // Compute the flattened index into the output array.
        int outIndex = ix * (Ny * Nz) + iy * Nz + iz;

        float a_val, b_val;

        // Determine the corresponding value from A.
        if (a_axis == -1) {
        // A is full-size (Nx, Ny, Nz)
            a_val = a[idx(ix, iy, iz)];
        }
        else if (a_axis == 0) {
        // A is broadcast along x: shape (Nx, 1, 1)
            a_val = a[ix];
        }
        else if (a_axis == 1) {
        // A is broadcast along y: shape (1, Ny, 1)
            a_val = a[iy];
        }
        else if (a_axis == 2) {
        // A is broadcast along z: shape (1, 1, Nz)
            a_val = a[iz];
        }

        // Determine the corresponding value from B.
        if (b_axis == -1) {
        // B is full-size (Nx, Ny, Nz)
            b_val = b[idx(ix, iy, iz)];
        }
        else if (b_axis == 0) {
        // B is broadcast along x: shape (Nx, 1, 1)
            b_val = b[ix];
        }
        else if (b_axis == 1) {
        // B is broadcast along y: shape (1, Ny, 1)
            b_val = b[iy];
        }
        else if (b_axis == 2) {
        // B is broadcast along z: shape (1, 1, Nz)
            b_val = b[iz];
        }

        // Write the sum to the output array.
        out[outIndex] = ldexpf(a_val, int(b_val));
    }
}
