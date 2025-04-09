// fftshift for 3D data, shifting only y and z axes
extern "C" __global__ void
fftshift3D_partial(float* __restrict__ data_out, float* __restrict__ data_in,
    int Nx, int Ny, int Nz) {
// Jeder Thread arbeitet mit einem komplexen Element (bestehend aus 2 Floats)
int x = blockIdx.x * blockDim.x + threadIdx.x; // komplexer x-Index
int y = blockIdx.y * blockDim.y + threadIdx.y; // komplexer y-Index
int z = blockIdx.z * blockDim.z + threadIdx.z; // komplexer z-Index

if (x < Nx && y < Ny && z < Nz) {
// x bleibt unverändert:
int new_x = x;
// y und z werden um die Hälfte der jeweiligen Dimension verschoben (fftshift-artig):
int new_y = (y + (Ny / 2)) % Ny;
int new_z = (z + (Nz / 2)) % Nz;

// Berechne den linearen Index in einem Array, in dem jedes Element
// ein komplexer Wert ist (jeweils 2 Floats).
// Die Gesamtzahl komplexer Elemente pro "Ebene" ist Nx * Ny.
int idx_in  = 2 * (x + Nx * (y + Ny * z));
int idx_out = 2 * (new_x + Nx * (new_y + Ny * new_z));

// Kopiere das gesamte komplexe Element (Real- und Imaginärteil)
data_out[idx_out]     = data_in[idx_in];       // Realteil
data_out[idx_out + 1] = data_in[idx_in + 1];   // Imaginärteil
}
}