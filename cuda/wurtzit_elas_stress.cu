#include "stencil.h"
#include "amul.h"
// Mat6x6: einfacher Container für eine 6×6-Matrix
struct Mat6x6 {
    float e[6][6];
    __host__ __device__
    Mat6x6() {
        // Matrix initial auf 0 setzen
        #pragma unroll
        for(int i=0;i<6;i++)
            #pragma unroll
            for(int j=0;j<6;j++)
                e[i][j] = 0.0f;
    }
    __host__ __device__
    float& operator()(int i,int j)       { return e[i][j]; }
    __host__ __device__
    const float& operator()(int i,int j) const { return e[i][j]; }
};

__host__ __device__
Mat6x6 makeWurtzitStiffness(
    float C11, float C12, float C13,
    float C33, float C44, float C66 = -1.0f
) {
    if(C66 < 0.0f) C66 = 0.5f*(C11 - C12);

    Mat6x6 C;
    // Normalspannungen
    C(0,0) = C11; C(0,1) = C12; C(0,2) = C13;
    C(1,0) = C12; C(1,1) = C11; C(1,2) = C13;
    C(2,0) = C13; C(2,1) = C13; C(2,2) = C33;
    // Schubspannungen (Voigt-Indizes 4,5,6 → 3,4,5 in 0-basiertem C[])
    C(3,3) = C44;
    C(4,4) = C44;
    C(5,5) = C66;

    return C;
}

// 3×3‐Matrix‐Container für Host und Device (falls noch nicht definiert)
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


// Holen eines Feldwertes mit Neumann-BC
__device__
inline float fetchU(
    const float *u,
    int xi, int yi, int zi,
    int Nx, int Ny, int Nz,
    int x, int y, int z
) {
    if(xi<0||xi>=Nx || yi<0||yi>=Ny || zi<0||zi>=Nz)
        return u[idx(x,y,z)];
    return u[idx(xi,yi,zi)];
}

// fünf-Punkte-Ableitung mit Richtung (di,dj,dk)
__device__
inline float deriv(
    float *u,
    int x, int y, int z,
    int di, int dj, int dk,
    int Nx, int Ny, int Nz,
    float dx, float dy, float dz
) {
    float delta = (di!=0 ? dx : dj!=0 ? dy : dz);
    float up2 = fetchU(u, x+2*di, y+2*dj, z+2*dk, Nx,Ny,Nz, x,y,z);
    float up1 = fetchU(u, x+1*di, y+1*dj, z+1*dk, Nx,Ny,Nz, x,y,z);
    float um1 = fetchU(u, x-1*di, y-1*dj, z-1*dk, Nx,Ny,Nz, x,y,z);
    float um2 = fetchU(u, x-2*di, y-2*dj, z-2*dk, Nx,Ny,Nz, x,y,z);
    return (-up2 + 8.0f*up1 - 8.0f*um1 + um2) / (12.0f * delta);
}


__device__
Mat3x3 computeStrainNeumann5pt(
    float* __restrict__ ux,
    float* __restrict__ uy,
    float* __restrict__ uz,
    int x, int y, int z,
    int Nx, int Ny, int Nz,
    float dx, float dy, float dz
) {
    Mat3x3 eps;
    // Diagonal
    eps(0,0) = deriv(ux, x,y,z, 1,0,0, Nx,Ny,Nz, dx,dy,dz);
    eps(1,1) = deriv(uy, x,y,z, 0,1,0, Nx,Ny,Nz, dx,dy,dz);
    eps(2,2) = deriv(uz, x,y,z, 0,0,1, Nx,Ny,Nz, dx,dy,dz);
    // Off-Diagonal
    float du_yx = deriv(ux, x,y,z, 0,1,0, Nx,Ny,Nz, dx,dy,dz);
    float du_xy = deriv(uy, x,y,z, 1,0,0, Nx,Ny,Nz, dx,dy,dz);
    float du_zx = deriv(ux, x,y,z, 0,0,1, Nx,Ny,Nz, dx,dy,dz);
    float du_xz = deriv(uz, x,y,z, 1,0,0, Nx,Ny,Nz, dx,dy,dz);
    float du_zy = deriv(uy, x,y,z, 0,0,1, Nx,Ny,Nz, dx,dy,dz);
    float du_yz = deriv(uz, x,y,z, 0,1,0, Nx,Ny,Nz, dx,dy,dz);

    eps(0,1) = eps(1,0) = 0.5f*(du_yx + du_xy);
    eps(0,2) = eps(2,0) = 0.5f*(du_zx + du_xz);
    eps(1,2) = eps(2,1) = 0.5f*(du_zy + du_yz);

    return eps;
}


// Annahme: Mat6x6 und Mat3x3 sind wie oben definiert und idx(x,y,z) existiert.

__host__ __device__
Mat3x3 computeStress(
    Mat6x6& C,    // 6×6 Steifigkeitsmatrix
    Mat3x3& eps   // 3×3 Strain‐Tensor
) {
    // 1) Strain in Voigt-Vektor überführen (engineering strains):
    //    [ε₀,ε₁,ε₂,ε₃,ε₄,ε₅] = [εxx, εyy, εzz, 2εyz, 2εzx, 2εxy]
    float eps_v[6];
    eps_v[0] = eps(0,0);
    eps_v[1] = eps(1,1);
    eps_v[2] = eps(2,2);
    eps_v[3] = 2.0f*eps(1,2);
    eps_v[4] = 2.0f*eps(0,2);
    eps_v[5] = 2.0f*eps(0,1);

    // 2) C (6×6) · eps_v (6×1) → sigma_v (6×1)
    float sigma_v[6] = {0,0,0,0,0,0};
    #pragma unroll
    for(int i=0;i<6;i++){
        float s = 0.0f;
        #pragma unroll
        for(int j=0;j<6;j++){
            s += C.e[i][j] * eps_v[j];
        }
        sigma_v[i] = s;
    }

    // 3) Voigt-Vektor zurück in 3×3 Stress-Tensor:
    Mat3x3 sigma;
    sigma(0,0) = sigma_v[0];
    sigma(1,1) = sigma_v[1];
    sigma(2,2) = sigma_v[2];
    // Mischkomponenten (sie stehen in Voigt-Index 3,4,5)
    sigma(1,2) = sigma(2,1) = sigma_v[3];  // σyz
    sigma(0,2) = sigma(2,0) = sigma_v[4];  // σzx
    sigma(0,1) = sigma(1,0) = sigma_v[5];  // σxy

    return sigma;
}
extern "C" __global__ void
stressWurtzit(  float *sigmaxx, float *sigmayy, float *sigmazz,
                float *sigmaxy, float *sigmaxz, float *sigmayz,
                float *ux, float *uy, float *uz,
                float* __restrict__  C11_, float  C11_mul,
                float* __restrict__  C12_, float  C12_mul,
                float* __restrict__  C13_, float  C13_mul,
                float* __restrict__  C33_, float  C33_mul,
                float* __restrict__  C44_, float  C44_mul,
                float dx, float dy, float dz,
                int Nx, int Ny, int Nz) {
    // Compute the 3D coordinates for this thread.
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate (0 <= ix < Nx)
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate (0 <= iy < Ny)
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // z-coordinate (0 <= iz < Nz)

    // Check bounds.
    if (ix < Nx && iy < Ny && iz < Nz) {
        int I = idx(ix, iy, iz);
        Mat6x6 C = makeWurtzitStiffness(amul(C11_, C11_mul, I), 
                             amul(C12_, C12_mul, I),
                             amul(C13_, C13_mul, I),
                             amul(C33_, C33_mul, I),
                             amul(C44_, C44_mul, I));
        Mat3x3 eps = computeStrainNeumann5pt(ux, uy, uz, ix, iy, iz, Nx, Ny, Nz, dx, dy, dz);
        Mat3x3 sigma = computeStress(C, eps);
        sigmaxx[I] = sigma(0,0);
        sigmayy[I] = sigma(1,1);
        sigmazz[I] = sigma(2,2);
        sigmaxy[I] = sigma(0,1);
        sigmaxz[I] = sigma(0,2);
        sigmayz[I] = sigma(1,2);
    }
}