#include "stencil.h"
#include "amul.h"
#include <stdint.h>

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

// ========== Index-Helfer =====================================================
__device__ __forceinline__ int ID3(int ix,int iy,int iz, int Nx,int Ny,int Nz, uint8_t PBC){
    int cx = hclampx(lclampx(ix));
    int cy = hclampy(lclampy(iy));
    int cz = hclampz(lclampz(iz));
    return idx(cx,cy,cz);
}

// ========== 1D-Stencils ======================================================
__device__ __forceinline__
float d1_central_5pt(const float* u, int I_m2,int I_m1,int I_p1,int I_p2, float h){
    float um2=u[I_m2], um1=u[I_m1], up1=u[I_p1], up2=u[I_p2];
    return (-up2 + 8.f*up1 - 8.f*um1 + um2) / (12.f*h);
}
__device__ __forceinline__
float d1_one_sided_left_4(const float* u, int I0,int I1,int I2,int I3,int I4, float h){
    return (-25.f*u[I0] + 48.f*u[I1] - 36.f*u[I2] + 16.f*u[I3] - 3.f*u[I4]) / (12.f*h);
}
__device__ __forceinline__
float d1_one_sided_right_4(const float* u, int I0,int I1,int I2,int I3,int I4, float h){
    return ( 25.f*u[I0] - 48.f*u[I1] + 36.f*u[I2] - 16.f*u[I3] + 3.f*u[I4]) / (12.f*h);
}

// ========== 1D-Ableitung mit PBC/Clamping + Randfallbehandlung ==============
__device__ __forceinline__
float deriv_axis(
    const float* __restrict__ u,
    int i,int j,int k,
    int Nx,int Ny,int Nz,
    uint8_t PBC,
    float h, int axis,
    const float* __restrict__ HookLeft,
    const float* __restrict__ HookRight)
{
    const bool pbcAxis = (axis==0 ? PBCx : (axis==1 ? PBCy : PBCz));
    if (pbcAxis) {
        if (axis==0) return d1_central_5pt(u, ID3(i-2,j,k,Nx,Ny,Nz,PBC), ID3(i-1,j,k,Nx,Ny,Nz,PBC),
                                            ID3(i+1,j,k,Nx,Ny,Nz,PBC), ID3(i+2,j,k,Nx,Ny,Nz,PBC), h);
        if (axis==1) return d1_central_5pt(u, ID3(i,j-2,k,Nx,Ny,Nz,PBC), ID3(i,j-1,k,Nx,Ny,Nz,PBC),
                                            ID3(i,j+1,k,Nx,Ny,Nz,PBC), ID3(i,j+2,k,Nx,Ny,Nz,PBC), h);
        return d1_central_5pt(u, ID3(i,j,k-2,Nx,Ny,Nz,PBC), ID3(i,j,k-1,Nx,Ny,Nz,PBC),
                                ID3(i,j,k+1,Nx,Ny,Nz,PBC), ID3(i,j,k+2,Nx,Ny,Nz,PBC), h);
    }
    // inside sample (≥2 from edge) -> central 5-point
    if ((axis==0 && i>=2 && i<=Nx-3) ||
        (axis==1 && j>=2 && j<=Ny-3) ||
        (axis==2 && k>=2 && k<=Nz-3)) {
        if(axis==0) return d1_central_5pt(u, ID3(i-2,j,k,Nx,Ny,Nz,PBC), ID3(i-1,j,k,Nx,Ny,Nz,PBC),
                                             ID3(i+1,j,k,Nx,Ny,Nz,PBC), ID3(i+2,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_central_5pt(u, ID3(i,j-2,k,Nx,Ny,Nz,PBC), ID3(i,j-1,k,Nx,Ny,Nz,PBC),
                                             ID3(i,j+1,k,Nx,Ny,Nz,PBC), ID3(i,j+2,k,Nx,Ny,Nz,PBC), h);
        return d1_central_5pt(u, ID3(i,j,k-2,Nx,Ny,Nz,PBC), ID3(i,j,k-1,Nx,Ny,Nz,PBC),
                                 ID3(i,j,k+1,Nx,Ny,Nz,PBC), ID3(i,j,k+2,Nx,Ny,Nz,PBC), h);
    }
    // edge case: at the boundary of index being zero in at least one of the three directions
    if((axis==0 && i==0)||(axis==1 && j==0)||(axis==2 && k==0)){
        if(HookLeft) return HookLeft[idx(i,j,k)];
        if(axis==0) return d1_one_sided_left_4 (u, ID3(0,j,k,Nx,Ny,Nz,PBC), ID3(1,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(2,j,k,Nx,Ny,Nz,PBC), ID3(3,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(4,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_one_sided_left_4 (u, ID3(i,0,k,Nx,Ny,Nz,PBC), ID3(i,1,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,2,k,Nx,Ny,Nz,PBC), ID3(i,3,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,4,k,Nx,Ny,Nz,PBC), h);
        return             d1_one_sided_left_4 (u, ID3(i,j,0,Nx,Ny,Nz,PBC), ID3(i,j,1,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,2,Nx,Ny,Nz,PBC), ID3(i,j,3,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,4,Nx,Ny,Nz,PBC), h);
    }
    if((axis==0 && i==1)||(axis==1 && j==1)||(axis==2 && k==1)){
        if(axis==0){
            // symmetrisch zu i==Nx-2
            float u0=u[ID3(0,j,k,Nx,Ny,Nz,PBC)], u1=u[ID3(1,j,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(2,j,k,Nx,Ny,Nz,PBC)], u3=u[ID3(3,j,k,Nx,Ny,Nz,PBC)], u4=u[ID3(4,j,k,Nx,Ny,Nz,PBC)];
            return (-3.f*u0 -10.f*u1 +18.f*u2 -6.f*u3 +1.f*u4)/(12.f*h);
        }
        if(axis==1){
            float u0=u[ID3(i,0,k,Nx,Ny,Nz,PBC)], u1=u[ID3(i,1,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(i,2,k,Nx,Ny,Nz,PBC)], u3=u[ID3(i,3,k,Nx,Ny,Nz,PBC)], u4=u[ID3(i,4,k,Nx,Ny,Nz,PBC)];
            return (-3.f*u0 -10.f*u1 +18.f*u2 -6.f*u3 +1.f*u4)/(12.f*h);
        }
        // axis==2: bei Nz==1 sowieso thin → hier nie hinein
    }
    // edge case: one cell before the boundary of index being max in at least one of the three directions
    if((axis==0 && i==Nx-2)||(axis==1 && j==Ny-2)||(axis==2 && k==Nz-2)){
        if(axis==0){
            float u0=u[ID3(Nx-1,j,k,Nx,Ny,Nz,PBC)], u1=u[ID3(Nx-2,j,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(Nx-3,j,k,Nx,Ny,Nz,PBC)], u3=u[ID3(Nx-4,j,k,Nx,Ny,Nz,PBC)], u4=u[ID3(Nx-5,j,k,Nx,Ny,Nz,PBC)];
            return (3.f*u0 +10.f*u1 -18.f*u2 +6.f*u3 -1.f*u4)/(12.f*h);
        }
        if(axis==1){
            float u0=u[ID3(i,Ny-1,k,Nx,Ny,Nz,PBC)], u1=u[ID3(i,Ny-2,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(i,Ny-3,k,Nx,Ny,Nz,PBC)], u3=u[ID3(i,Ny-4,k,Nx,Ny,Nz,PBC)], u4=u[ID3(i,Ny-5,k,Nx,Ny,Nz,PBC)];
            return (3.f*u0 +10.f*u1 -18.f*u2 +6.f*u3 -1.f*u4)/(12.f*h);
        }
        float u0=u[ID3(i,j,Nz-1,Nx,Ny,Nz,PBC)], u1=u[ID3(i,j,Nz-2,Nx,Ny,Nz,PBC)];
        float u2=u[ID3(i,j,Nz-3,Nx,Ny,Nz,PBC)], u3=u[ID3(i,j,Nz-4,Nx,Ny,Nz,PBC)], u4=u[ID3(i,j,Nz-5,Nx,Ny,Nz,PBC)];
        return (3.f*u0 +10.f*u1 -18.f*u2 +6.f*u3 -1.f*u4)/(12.f*h);
    }
    // edge case: at the boundary of index being max in at least one of the three directions
    if((axis==0 && i==Nx-1)||(axis==1 && j==Ny-1)||(axis==2 && k==Nz-1)){
        if(HookRight) return HookRight[idx(i,j,k)];
        if(axis==0) return d1_one_sided_right_4(u, ID3(Nx-1,j,k,Nx,Ny,Nz,PBC), ID3(Nx-2,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(Nx-3,j,k,Nx,Ny,Nz,PBC), ID3(Nx-4,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(Nx-5,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_one_sided_right_4(u, ID3(i,Ny-1,k,Nx,Ny,Nz,PBC), ID3(i,Ny-2,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,Ny-3,k,Nx,Ny,Nz,PBC), ID3(i,Ny-4,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,Ny-5,k,Nx,Ny,Nz,PBC), h);
        return          d1_one_sided_right_4(u, ID3(i,j,Nz-1,Nx,Ny,Nz,PBC), ID3(i,j,Nz-2,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,Nz-3,Nx,Ny,Nz,PBC), ID3(i,j,Nz-4,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,Nz-5,Nx,Ny,Nz,PBC), h);
    }
    printf("Error in deriv_axis: should never reach here!\n");
    // Fallback (2. Ordnung)
    if(axis==0) return (u[ID3(i+1,j,k,Nx,Ny,Nz,PBC)] - u[ID3(i-1,j,k,Nx,Ny,Nz,PBC)])/(2.f*h);
    if(axis==1) return (u[ID3(i,j+1,k,Nx,Ny,Nz,PBC)] - u[ID3(i,j-1,k,Nx,Ny,Nz,PBC)])/(2.f*h);
    return          (u[ID3(i,j,k+1,Nx,Ny,Nz,PBC)] - u[ID3(i,j,k-1,Nx,Ny,Nz,PBC)])/(2.f*h);
}

// ========== Strain aus u mit BC-tauglichen Ableitungen ======================
__device__
Mat3x3 computeStrainWithBC(
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    int i, int j, int k,
    int Nx, int Ny, int Nz,
    uint8_t PBC,
    float dx, float dy, float dz,
    // Hooks für ∂x ux, ∂x uy (x-Ränder)
    const float* __restrict__ hxL_ux, const float* __restrict__ hxR_ux,
    const float* __restrict__ hxL_uy, const float* __restrict__ hxR_uy,
    // Hooks für ∂y ux, ∂y uy (y-Ränder)
    const float* __restrict__ hyL_ux, const float* __restrict__ hyR_ux,
    const float* __restrict__ hyL_uy, const float* __restrict__ hyR_uy )
{
    Mat3x3 eps;
    // Normal
    eps(0,0) = deriv_axis(ux,i,j,k,Nx,Ny,Nz,PBC,dx,0,hxL,hxR); // ∂x ux
    eps(1,1) = deriv_axis(uy,i,j,k,Nx,Ny,Nz,PBC,dy,1,hyL,hyR); // ∂y uy
    eps(2,2) = (Nz > 5 ? deriv_axis(uz,i,j,k,Nx,Ny,Nz,PBC,dz,2,hzL,hzR): 0.0f); // ∂z uz
    // Scher (symmetrisiert)
    float du_yx = deriv_axis(ux,i,j,k,Nx,Ny,Nz,PBC,dy,1,0,0);
    float du_xy = deriv_axis(uy,i,j,k,Nx,Ny,Nz,PBC,dx,0,0,0);
    float du_zx = (Nz > 5 ? deriv_axis(ux,i,j,k,Nx,Ny,Nz,PBC,dz,2,0,0): 0.0f); // only if enough cells in z
    float du_xz = (Nz > 5 ? deriv_axis(uz,i,j,k,Nx,Ny,Nz,PBC,dx,0,0,0): 0.0f); // only if enough cells in z
    float du_zy = (Nz > 5 ? deriv_axis(uy,i,j,k,Nx,Ny,Nz,PBC,dz,2,0,0): 0.0f); // only if enough cells in z
    float du_yz = (Nz > 5 ? deriv_axis(uz,i,j,k,Nx,Ny,Nz,PBC,dy,1,0,0): 0.0f); // only if enough cells in z
    eps(0,1) = eps(1,0) = 0.5f*(du_yx + du_xy);
    eps(0,2) = eps(2,0) = 0.5f*(du_zx + du_xz);
    eps(1,2) = eps(2,1) = 0.5f*(du_zy + du_yz);
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
    eps_v[3] = 2.0f*eps(1,2);
    eps_v[4] = 2.0f*eps(0,2);
    eps_v[5] = 2.0f*eps(0,1);

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
    ev[3] = (Nz > 5 ? sv[3] / (C44 + eps): 0.0f);  // 2εyz
    ev[4] = (Nz > 5 ? sv[4] / (C44 + eps): 0.0f);  // 2εzx
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
stressWurtzit(  float *sigmaxx, float *sigmayy, float *sigmazz,
                float *sigmaxy, float *sigmaxz, float *sigmayz,
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

    // --- 4) Magnetostriktive Eigen-Dehnung ε^m aus B1,B2,m (gleiche Zelle)
    float B1 = amul(B1_, B1_mul, IClamp);
    float B2 = amul(B2_, B2_mul, IClamp);
    float sv[6];
    sigma_m_from_B(B1, B2, mx[IClamp], my[IClamp], mz[IClamp], sv);

    float ev[6];
    eps_m_from_sigma_m_hex_cubic(c11, c12, c13, c33, c44, c66, sv, ev, Nz);

    Mat3x3 epsm;
    voigt6_to_mat3x3_strain(ev, epsm);

    // --- 5) Effektive Dehnung ε_eff = ε - ε^m
    if (Nz <= 5) { // if not enough cells calc ezz from condition of zero normal stress in z-direction
        eps(2, 2) = epsm(2, 2) - c13/c33*((eps(0,0)-epsm(0,0))+(eps(1,1)-epsm(1,1))); // only normal strain zz
        eps(0, 2) = epsm(0, 2); // set to zero, since it cannot be calculated properly
        eps(1, 2) = epsm(1, 2); // set
        eps(2, 0) = epsm(2, 0);
        eps(2, 1) = epsm(2, 1);
        //eps(2, 2) = 0.0f; // set to zero, since it cannot be calculated properly
    }
    Mat3x3 epseff;
    #pragma unroll
    for(int a=0;a<3;a++)
        #pragma unroll
        for(int b=0;b<3;b++)
            epseff(a,b) = eps(a,b) - epsm(a,b);

    // --- 6) Spannung: stress_ = C : (ε - ε^m)
    Mat3x3 sigma = computeStress(C, epseff);

    // --- 7) Output: exakt dieselbe (geclampte) Zelle
    sigmaxx[IClamp] = sigma(0,0);
    sigmayy[IClamp] = sigma(1,1);
    sigmazz[IClamp] = sigma(2,2);
    sigmaxy[IClamp] = sigma(0,1);
    sigmaxz[IClamp] = sigma(0,2);
    sigmayz[IClamp] = sigma(1,2);
}
