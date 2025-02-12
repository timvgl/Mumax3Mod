extern "C" __global__ void
extractZSlice(
    float* __restrict__ output,      // Output array for the Z slice
    float* __restrict__ input,       // Flattened 3D input array
    int X,                           // Size along the X dimension
    int Y,                           // Size along the Y dimension
    int Z,                           // Size along the Z dimension
    int x,                           // Fixed X index
    int y                            // Fixed Y index
) {
    // Calculate the thread's Z index
    int z = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (z < Z) {
        // Compute the 1D index for the (x, y, z) element
        int index = x * (Y * Z) + y * Z + z;

        // Retrieve the value from the input and store it in the output
        output[z] = input[index];
    }
}