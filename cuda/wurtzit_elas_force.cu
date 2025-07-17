#include "stencil.h"
// float3 ist CUDA-vorhanden; Mat3x3 wie oben definiert.

// Berechnet f = ∇·σ an (x,y,z)
// sigma: flaches Array von Mat3x3 mit den Spannungs-Tensoren
// dx: Gitterabstand (gleich in x,y,z)
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

__device__
float3 divergenceStress(
    float *fx, float *fy, float *fz,
    float *sigmaxx, float *sigmayy, float *sigmazz,
    float *sigmaxy, float *sigmaxz, float *sigmayz,
    int Nx, int Ny, int Nz,
    float dx, float dy, float dz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int I = idx(x, y, z);
    // Helper: liefert sigma_ij an Nachbar oder 0, falls außerhalb
    auto s = [&](int xi,int yi,int zi)->Mat3x3 {
        if(xi<0||xi>=Nx||yi<0||yi>=Ny||zi<0||zi>=Nz)
            return Mat3x3(); // randwärts null → Neumann ∂σ/∂n=0
        Mat3x3 sigma;
        int I_ = idx(xi,yi,zi);
        sigma(0,0) = sigmaxx[I_];
        sigma(1,1) = sigmayy[I_];
        sigma(2,2) = sigmazz[I_];
        sigma(0,1) = sigmaxy[I_];
        sigma(0,2) = sigmaxz[I_];
        sigma(1,0) = sigmaxy[I_];
        sigma(1,2) = sigmayz[I_];
        sigma(2,0) = sigmaxz[I_];
        sigma(2,1) = sigmayz[I_];
        return sigma;
    };

    // zentrale Differenzen für Ableitungen
    float dsxx_dx = ( s(x+1,y,z)(0,0) - s(x-1,y,z)(0,0) ) / (2.0f*dx);
    float dsxy_dy = ( s(x,y+1,z)(0,1) - s(x,y-1,z)(0,1) ) / (2.0f*dy);
    float dsxz_dz = ( s(x,y,z+1)(0,2) - s(x,y,z-1)(0,2) ) / (2.0f*dz);

    float dsyx_dx = ( s(x+1,y,z)(1,0) - s(x-1,y,z)(1,0) ) / (2.0f*dx);
    float dsyy_dy = ( s(x,y+1,z)(1,1) - s(x,y-1,z)(1,1) ) / (2.0f*dy);
    float dsyz_dz = ( s(x,y,z+1)(1,2) - s(x,y,z-1)(1,2) ) / (2.0f*dz);

    float dszx_dx = ( s(x+1,y,z)(2,0) - s(x-1,y,z)(2,0) ) / (2.0f*dx);
    float dszy_dy = ( s(x,y+1,z)(2,1) - s(x,y-1,z)(2,1) ) / (2.0f*dy);
    float dszz_dz = ( s(x,y,z+1)(2,2) - s(x,y,z-1)(2,2) ) / (2.0f*dz);

    // f_i = ∑_j ∂_j σ_ij
    fx[I] = dsxx_dx + dsxy_dy + dsxz_dz;
    fy[I] = dsyx_dx + dsyy_dy + dsyz_dz;
    fz[I] = dszx_dx + dszy_dy + dszz_dz;
}
