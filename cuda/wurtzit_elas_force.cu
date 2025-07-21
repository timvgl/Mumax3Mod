#include "stencil.h"

// 3×3‐Matrix‐Container (bleibt unverändert)
struct Mat3x3 {
    float e[3][3];
    __host__ __device__
    Mat3x3() {
        #pragma unroll
        for(int i=0;i<3;i++)
            #pragma unroll
            for(int j=0;j<3;j++)
                e[i][j] = 0.0f;
    }
    __host__ __device__
    float& operator()(int i,int j)       { return e[i][j]; }
    __host__ __device__
    const float& operator()(int i,int j) const { return e[i][j]; }
};

//  ―――――――――――――――――――――――――――――――――――――――
// Hilfs-Device-Funktion statt Lambda:
__device__
Mat3x3 fetchSigma(
    int xi, int yi, int zi,
    const float *sigmaxx, const float *sigmayy, const float *sigmazz,
    const float *sigmaxy, const float *sigmaxz, const float *sigmayz,
    int Nx, int Ny, int Nz
) {
    // Neumann-Randbedingung: ∂σ/∂n = 0 ⇒ außerhalb = 0
    if (xi < 0 || xi >= Nx ||
        yi < 0 || yi >= Ny ||
        zi < 0 || zi >= Nz) {
        return Mat3x3();
    }
    int I = idx(xi, yi, zi);
    Mat3x3 sigma;
    sigma(0,0) = sigmaxx[I];
    sigma(1,1) = sigmayy[I];
    sigma(2,2) = sigmazz[I];
    sigma(0,1) = sigma(1,0) = sigmaxy[I];
    sigma(0,2) = sigma(2,0) = sigmaxz[I];
    sigma(1,2) = sigma(2,1) = sigmayz[I];
    return sigma;
}
//  ―――――――――――――――――――――――――――――――――――――――

extern "C" __global__ void
divergenceStress(
    float *fx, float *fy, float *fz,
    float *sigmaxx, float *sigmayy, float *sigmazz,
    float *sigmaxy, float *sigmaxz, float *sigmayz,
    int Nx, int Ny, int Nz,
    float dx, float dy, float dz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < 0 || x >= Nx ||
        y < 0 || y >= Ny ||
        z < 0 || z >= Nz) return;

    int I = idx(x, y, z);

    // Nachbar-Matrizen holen
    Mat3x3 s_px = fetchSigma(x+1, y,   z,   sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz, Nx, Ny, Nz);
    Mat3x3 s_mx = fetchSigma(x-1, y,   z,   sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz, Nx, Ny, Nz);
    Mat3x3 s_py = fetchSigma(x,   y+1, z,   sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz, Nx, Ny, Nz);
    Mat3x3 s_my = fetchSigma(x,   y-1, z,   sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz, Nx, Ny, Nz);
    Mat3x3 s_pz = fetchSigma(x,   y,   z+1, sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz, Nx, Ny, Nz);
    Mat3x3 s_mz = fetchSigma(x,   y,   z-1, sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz, Nx, Ny, Nz);

    // zentrale Differenzen (Beispiel: in x-Richtung)
    float dsxx_dx = (s_px(0,0) - s_mx(0,0)) / (2.0f * dx);
    float dsxy_dy = (s_py(0,1) - s_my(0,1)) / (2.0f * dy);
    float dsxz_dz = (s_pz(0,2) - s_mz(0,2)) / (2.0f * dz);

    float dsyx_dx = (s_px(1,0) - s_mx(1,0)) / (2.0f * dx);
    float dsyy_dy = (s_py(1,1) - s_my(1,1)) / (2.0f * dy);
    float dsyz_dz = (s_pz(1,2) - s_mz(1,2)) / (2.0f * dz);

    float dszx_dx = (s_px(2,0) - s_mx(2,0)) / (2.0f * dx);
    float dszy_dy = (s_py(2,1) - s_my(2,1)) / (2.0f * dy);
    float dszz_dz = (s_pz(2,2) - s_mz(2,2)) / (2.0f * dz);

    // f_i = ∑_j ∂_j σ_ij
    fx[I] = dsxx_dx + dsxy_dy + dsxz_dz;
    fy[I] = dsyx_dx + dsyy_dy + dsyz_dz;
    fz[I] = dszx_dx + dszy_dy + dszz_dz;
}
