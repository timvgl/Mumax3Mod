extern "C" __global__ void
extractYSlice(
    float* __restrict__ output,       // Output array for the Y slice
    float* __restrict__ input,        // Flattened 3D input array
    int X,                            // Size along the X dimension
    int Y,                            // Size along the Y dimension
    int Z,                            // Size along the Z dimension
    int x,                            // Fixed X index
    int z                             // Fixed Z index
) {
    // Calculate the thread's Y index
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (y < Y) {
        // Compute the 1D index for the (x, y, z) element
        int index = x * (Y * Z) + y * Z + z;

        // Retrieve the value from the input and store it in the output
        output[y] = input[index];
    }
}