#include "stencil.h"
#include "amul.h"
#include <stdint.h>
#include "deriv_axis.cuh"

// Mat6x6: einfacher Container für eine 6×6-Matrix
struct Mat6x6 {
    float e[6][6];
    __host__ __device__
    Mat6x6() {
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
    float C33, float C44, float C66
) {
    Mat6x6 C;
    // Normalspannungen
    C(0,0) = C11; C(0,1) = C12; C(0,2) = C13;
    C(1,0) = C12; C(1,1) = C11; C(1,2) = C13;
    C(2,0) = C13; C(2,1) = C13; C(2,2) = C33;
    // Schubspannungen (Voigt-Indizes 4,5,6 → 3,4,5)
    C(3,3) = C44;
    C(4,4) = C44;
    C(5,5) = C66;
    return C;
}

// 3×3‐Matrix‐Container
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

// ========== Strain aus u mit BC-tauglichen Ableitungen ======================
__device__
Mat3x3 computeStrainWithBC(
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    int i, int j, int k,
    int Nx, int Ny, int Nz,
    uint8_t PBC,
    float dx, float dy, float dz)
{
    Mat3x3 eps;
    // Normal
    eps(0,0) = deriv_axis(ux,i,j,k,Nx,Ny,Nz,PBC,dx,0,0,0); // ∂x ux
    eps(1,1) = deriv_axis(uy,i,j,k,Nx,Ny,Nz,PBC,dy,1,0,0); // ∂y uy
    eps(2,2) = (Nz > 5 ? deriv_axis(uz,i,j,k,Nx,Ny,Nz,PBC,dz,2,0,0): 0.0f); // ∂z uz
    // Scher (symmetrisiert)
    float du_yx = deriv_axis(ux,i,j,k,Nx,Ny,Nz,PBC,dy,1,0,0);
    float du_xy = deriv_axis(uy,i,j,k,Nx,Ny,Nz,PBC,dx,0,0,0);
    float du_zx = (Nz > 5 ? deriv_axis(ux,i,j,k,Nx,Ny,Nz,PBC,dz,2,0,0): 0.0f); // only if enough cells in z
    float du_xz = (Nz > 5 ? deriv_axis(uz,i,j,k,Nx,Ny,Nz,PBC,dx,0,0,0): 0.0f); // only if enough cells in z
    float du_zy = (Nz > 5 ? deriv_axis(uy,i,j,k,Nx,Ny,Nz,PBC,dz,2,0,0): 0.0f); // only if enough cells in z
    float du_yz = (Nz > 5 ? deriv_axis(uz,i,j,k,Nx,Ny,Nz,PBC,dy,1,0,0): 0.0f); // only if enough cells in z
    eps(0,1) = du_yx; eps(1,0) = du_xy;
    eps(0,2) = du_zx; eps(2,0) = du_xz;
    eps(1,2) = du_zy; eps(2,1) = du_yz;
    return eps;
}

// ========== Stress aus C und ε ==============================================
__host__ __device__
Mat3x3 computeStress(
    Mat6x6& C,
    Mat3x3& eps
) {
    float eps_v[6];
    eps_v[0] = eps(0,0);
    eps_v[1] = eps(1,1);
    eps_v[2] = eps(2,2);
    eps_v[3] = eps(1,2)+eps(2,1);
    eps_v[4] = eps(0,2)+eps(2,0);
    eps_v[5] = eps(0,1)+eps(1,0);

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

    Mat3x3 sigma;
    sigma(0,0) = sigma_v[0];
    sigma(1,1) = sigma_v[1];
    sigma(2,2) = sigma_v[2];
    sigma(1,2) = sigma(2,1) = sigma_v[3];  // stress_yz
    sigma(0,2) = sigma(2,0) = sigma_v[4];  // stress_zx
    sigma(0,1) = sigma(1,0) = sigma_v[5];  // stress_xy
    return sigma;
}

// --- Magnetostriktive "Spannung" (engineering-Voigt) ------------------------
__device__ inline void sigma_m_from_B(
    float B1, float B2, float mx, float my, float mz,
    float sv[6])
{
    sv[0] = B1 * mx*mx;           // stress_xx^m
    sv[1] = B1 * my*my;           // stress_yy^m
    sv[2] = B1 * mz*mz;           // stress_zz^m
    sv[3] = 2.f * B2 * (my*mz);   // stress_yz^m
    sv[4] = 2.f * B2 * (mz*mx);   // stress_zx^m
    sv[5] = 2.f * B2 * (mx*my);   // stress_xy^m
}

// --- ε^m = S * stress_^m (hex/cubic, ohne 6×6-Inversion) --------------------------
__device__ inline void eps_m_from_sigma_m_hex_cubic(
    float C11,float C12,float C13,float C33,float C44,float C66,
    const float sv[6], float ev[6], int Nz)
{
    const float a=C11, b=C12, c=C13, d=C33;
    const float eps = 1e-30f;
    const float Delta2 = d*(a+b) - 2.f*c*c;
    const float Delta3 = (a-b) * Delta2;
    const float iD2 = 1.f/(Delta2 + eps);
    const float iD3 = 1.f/(Delta3 + eps);

    // Compliance des Normalblocks
    const float S00 = (a*d - c*c) * iD3;
    const float S01 = (-b*d + c*c) * iD3;
    const float S02 = (-c) * iD2;
    const float S22 = (a+b) * iD2;

    const float sx = sv[0], sy = sv[1], sz = sv[2];

    ev[0] = S00*sx + S01*sy + S02*sz;  // εxx^m
    ev[1] = S01*sx + S00*sy + S02*sz;  // εyy^m
    ev[2] = S02*sx + S02*sy + S22*sz;  // εzz^m

    // Shears (engineering: ev[3]=2εyz, ...)
    ev[3] = sv[3] / (C44 + eps);  // 2εyz
    ev[4] = sv[4] / (C44 + eps);  // 2εzx
    ev[5] = sv[5] / (C66 + eps);  // 2εxy
}

// --- Voigt (engineering) -> 3×3-Strain-Tensor -------------------------------
__device__ inline void voigt6_to_mat3x3_strain(const float ev[6], Mat3x3& E){
    E(0,0)=ev[0]; E(1,1)=ev[1]; E(2,2)=ev[2];
    E(1,2)=E(2,1)= 0.5f*ev[3]; // γ/2
    E(0,2)=E(2,0)= 0.5f*ev[4];
    E(0,1)=E(1,0)= 0.5f*ev[5];
}

// ============================================================================
// Stress mit magnetostriktiver Eigen-Dehnung, konsistente Indizes
// ============================================================================
extern "C" __global__ void
stressWurtzit(  float *stressxx, float *stressyy, float *stresszz,
                float *stressxy, float *stressxz, float *stressyz,
                float *epsilonxx, float *epsilonyy, float *epsilonzz,
                float *epsilonxy, float *epsilonxz, float *epsilonyz,
                float *ux, float *uy, float *uz,
                float *mx, float *my, float *mz,
                float* __restrict__  C11_, float  C11_mul,
                float* __restrict__  C12_, float  C12_mul,
                float* __restrict__  C13_, float  C13_mul,
                float* __restrict__  C33_, float  C33_mul,
                float* __restrict__  C44_, float  C44_mul,
                float* __restrict__ B1_,  float  B1_mul,
                float* __restrict__ B2_,  float  B2_mul,
                float dx, float dy, float dz,
                int Nx, int Ny, int Nz,
                uint8_t PBC, bool cubic)
{
    // Thread-Koordinaten
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Harte Bounds (wie Divergenz)
    if ((!PBCx && (ix < 0 || ix >= Nx)) ||
        (!PBCy && (iy < 0 || iy >= Ny)) ||
        (!PBCz && (iz < 0 || iz >= Nz))) return;

    // Konsistente zentrale Zelle (clamp/wrap)
    int i = hclampx(lclampx(ix));
    int j = hclampy(lclampy(iy));
    int k = hclampz(lclampz(iz));
    int IClamp = idx(i, j, k);

    // --- 1) Material einlesen (auf derselben Zelle!)
    float c11 = amul(C11_, C11_mul, IClamp);
    float c12 = amul(C12_, C12_mul, IClamp);
    float c13 = amul(C13_, C13_mul, IClamp);
    float c33 = amul(C33_, C33_mul, IClamp);
    float c44 = amul(C44_, C44_mul, IClamp);
    float c66;

    if (cubic) { c13 = c12; c33 = c11; }
    c66 = 0.5f * (c11 - c12);

    // --- 2) Steifigkeit
    Mat6x6 C = makeWurtzitStiffness(c11, c12, c13, c33, c44, c66);

    // --- 3) Mechanische Dehnung ε(u) mit denselben Indizes/BC
    Mat3x3 eps = computeStrainWithBC(ux, uy, uz, i, j, k, Nx, Ny, Nz, PBC, dx, dy, dz);
    Mat3x3 sigma = computeStress(C, eps);

    // --- 4) Magnetostriktive Eigen-Dehnung ε^m aus B1,B2,m (gleiche Zelle)
    float B1 = amul(B1_, B1_mul, IClamp);
    float B2 = amul(B2_, B2_mul, IClamp);
    float sv[6];
    sigma_m_from_B(B1, B2, mx[IClamp], my[IClamp], mz[IClamp], sv);

    //float ev[6];
    //eps_m_from_sigma_m_hex_cubic(c11, c12, c13, c33, c44, c66, sv, ev, Nz);

    Mat3x3 sigmam;
    voigt6_to_mat3x3_strain(sv, sigmam);
    Mat3x3 sigmaeff;
    #pragma unroll
    for(int a=0;a<3;a++)
        #pragma unroll
        for(int b=0;b<3;b++)
            sigmaeff(a,b) = sigma(a,b) - sigmam(a,b);

    // --- 6) Spannung: stress_ = C : (ε - ε^m)
    
    // --- 7) Output: exakt dieselbe (geclampte) Zelle
    stressxx[IClamp] = sigmaeff(0,0);
    stressyy[IClamp] = sigmaeff(1,1);
    stresszz[IClamp] = sigmaeff(2,2);
    stressxy[IClamp] = sigmaeff(0,1);
    stressxz[IClamp] = sigmaeff(0,2);
    stressyz[IClamp] = sigmaeff(1,2);
    epsilonxx[IClamp] = eps(0,0);
    epsilonyy[IClamp] = eps(1,1);
    epsilonzz[IClamp] = eps(2,2);
    epsilonxy[IClamp] = eps(0,1)+eps(1,0);
    epsilonxz[IClamp] = eps(0,2)+eps(2,0);
    epsilonyz[IClamp] = eps(1,2)+eps(2,1);
}
