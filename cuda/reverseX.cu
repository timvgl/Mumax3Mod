// fftshift for 3D data, shifting only y and z axes
#include "stencil.h" 
extern "C" __global__ void
reverseX(float* __restrict__ data_out, float* __restrict__ data_in,
                                   int Nx, int Ny, int Nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x index (first axis)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y index (second axis)
    int z = blockIdx.z * blockDim.z + threadIdx.z; // z index (third axis)

    if (x < Nx && y < Ny && z < Nz) {
        // First axis (x) is not shifted
        int x_left = 2*x;
        int x_right = 4*Nx - 2*x - 2;

        // Calculate linear indices for input and output arrays
        int idx_leftReal = index(x_left, y, z, Nx*4, Ny, Nz);
        int idx_leftImag = index(x_left+1, y, z, Nx*4, Ny, Nz);
        int idx_rightReal = index(x_right, y, z, Nx*4, Ny, Nz);
        int idx_rightImag = index(x_right+1, y, z, Nx*4, Ny, Nz);

        float dataLeftReal = data_in[idx_leftReal];
        float dataLeftImag = data_in[idx_leftImag];
        float dataRightReal = data_in[idx_rightReal];
        float dataRightImag = data_in[idx_rightImag];
        // Copy data from input to output at the shifted position
        data_out[idx_leftReal] = dataRightReal;
        data_out[idx_leftImag] = dataRightImag;
        data_out[idx_rightReal] = dataLeftReal;
        data_out[idx_rightImag] = dataLeftImag;
    }
}
//NxTotal = 256
//Nx = 64
//x_left = 2*x
//x_right = 4*(Nx - x/2) -2
//x = 0
//x_left = 0
//x_right = 4*(64 - 0) -2 = 254
//x = 1
//x_left = 2
//x_right = 4*(64 - 1/2) -2 = 252
//x = 2
//x_left = 4
//x_right = 4*(64 - 1) -2 = 250