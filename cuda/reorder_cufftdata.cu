// fftshift for 3D data, shifting only y and z axes
extern "C" __global__ void
fftshift3D_partial(float* __restrict__ data_out, float* __restrict__ data_in,
                                   int Nx, int Ny, int Nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x index (first axis)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y index (second axis)
    int z = blockIdx.z * blockDim.z + threadIdx.z; // z index (third axis)

    if (x < Nx && y < Ny && z < Nz) {
        // First axis (x) is not shifted
        int new_x = x;

        // Shift the second (y) and third (z) axes
        int new_y = (y + (Ny / 2)) % Ny;
        int new_z = (z + (Nz / 2)) % Nz;

        // Calculate linear indices for input and output arrays
        int idx_in = (z * Ny + y) * Nx + x;
        int idx_out = (new_z * Ny + new_y) * Nx + new_x;

        // Copy data from input to output at the shifted position
        data_out[idx_out] = data_in[idx_in];
    }
}