#include "stencil.h"
#include <stdint.h>

// ========== Index-Helfer (nutzt hclamp*/lclamp*) ============================
__device__ __forceinline__ int ID3(int ix,int iy,int iz, int Nx,int Ny,int Nz, uint8_t PBC){
    int cx = hclampx(lclampx(ix));
    int cy = hclampy(lclampy(iy));
    int cz = hclampz(lclampz(iz));
    return idx(cx,cy,cz);
}

// ========== 1D-Stencils ======================================================
// Zentral 5-Punkt (4. Ordnung)
__device__ __forceinline__
float d1_central_5pt(const float* u, int I_m2,int I_m1,int I_p1,int I_p2, float h){
    float um2=u[I_m2], um1=u[I_m1], up1=u[I_p1], up2=u[I_p2];
    return (-up2 + 8.f*up1 - 8.f*um1 + um2) / (12.f*h);
}
// Einseitig links (i=0)
__device__ __forceinline__
float d1_one_sided_left_4(const float* u, int I0,int I1,int I2,int I3,int I4, float h){
    return (-25.f*u[I0] + 48.f*u[I1] - 36.f*u[I2] + 16.f*u[I3] - 3.f*u[I4]) / (12.f*h);
}
// Einseitig rechts (i=N-1)
__device__ __forceinline__
float d1_one_sided_right_4(const float* u, int I0,int I1,int I2,int I3,int I4, float h){
    return ( 25.f*u[I0] - 48.f*u[I1] + 36.f*u[I2] - 16.f*u[I3] + 3.f*u[I4]) / (12.f*h);
}

// ========== 1D-Ableitung mit PBC/Clamping + Randfallbehandlung ==============
// axis: 0=x, 1=y, 2=z
// HookLeft/HookRight: optional (0 -> ignorieren)
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
    // Bei PBC in der jeweiligen Achse: immer zentraler 5-Punkt (wrap via ID3)
    const bool pbcAxis = (axis==0 ? PBCx : (axis==1 ? PBCy : PBCz));
    if (pbcAxis) {
        if (axis==0) return d1_central_5pt(u, ID3(i-2,j,k,Nx,Ny,Nz,PBC), ID3(i-1,j,k,Nx,Ny,Nz,PBC),
                                            ID3(i+1,j,k,Nx,Ny,Nz,PBC), ID3(i+2,j,k,Nx,Ny,Nz,PBC), h);
        if (axis==1) return d1_central_5pt(u, ID3(i,j-2,k,Nx,Ny,Nz,PBC), ID3(i,j-1,k,Nx,Ny,Nz,PBC),
                                            ID3(i,j+1,k,Nx,Ny,Nz,PBC), ID3(i,j+2,k,Nx,Ny,Nz,PBC), h);
        // axis==2
        return d1_central_5pt(u, ID3(i,j,k-2,Nx,Ny,Nz,PBC), ID3(i,j,k-1,Nx,Ny,Nz,PBC),
                                ID3(i,j,k+1,Nx,Ny,Nz,PBC), ID3(i,j,k+2,Nx,Ny,Nz,PBC), h);
    }

    // Innen (≥2 vom Rand entfernt) -> zentral 5-Punkt
    if ((axis==0 && i>=2 && i<=Nx-3) ||
        (axis==1 && j>=2 && j<=Ny-3) ||
        (axis==2 && k>=2 && k<=Nz-3)) {
        if(axis==0) return d1_central_5pt(u, ID3(i-2,j,k,Nx,Ny,Nz,PBC), ID3(i-1,j,k,Nx,Ny,Nz,PBC),
                                             ID3(i+1,j,k,Nx,Ny,Nz,PBC), ID3(i+2,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_central_5pt(u, ID3(i,j-2,k,Nx,Ny,Nz,PBC), ID3(i,j-1,k,Nx,Ny,Nz,PBC),
                                             ID3(i,j+1,k,Nx,Ny,Nz,PBC), ID3(i,j+2,k,Nx,Ny,Nz,PBC), h);
        // axis==2
        return d1_central_5pt(u, ID3(i,j,k-2,Nx,Ny,Nz,PBC), ID3(i,j,k-1,Nx,Ny,Nz,PBC),
                                 ID3(i,j,k+1,Nx,Ny,Nz,PBC), ID3(i,j,k+2,Nx,Ny,Nz,PBC), h);
    }

    // Linker Rand
    if((axis==0 && i==0)||(axis==1 && j==0)||(axis==2 && k==0)){
        if(HookLeft) return HookLeft[idx(i,j,k)];
        if(axis==0) return d1_one_sided_left_4 (u, ID3(0,j,k,Nx,Ny,Nz,PBC), ID3(1,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(2,j,k,Nx,Ny,Nz,PBC), ID3(3,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(4,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_one_sided_left_4 (u, ID3(i,0,k,Nx,Ny,Nz,PBC), ID3(i,1,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,2,k,Nx,Ny,Nz,PBC), ID3(i,3,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,4,k,Nx,Ny,Nz,PBC), h);
        return          d1_one_sided_left_4 (u, ID3(i,j,0,Nx,Ny,Nz,PBC), ID3(i,j,1,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,2,Nx,Ny,Nz,PBC), ID3(i,j,3,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,4,Nx,Ny,Nz,PBC), h);
    }

    // Rechter Rand
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

    // Fallback (2. Ordnung zentral)
    if(axis==0) return (u[ID3(i+1,j,k,Nx,Ny,Nz,PBC)] - u[ID3(i-1,j,k,Nx,Ny,Nz,PBC)])/(2.f*h);
    if(axis==1) return (u[ID3(i,j+1,k,Nx,Ny,Nz,PBC)] - u[ID3(i,j-1,k,Nx,Ny,Nz,PBC)])/(2.f*h);
    return          (u[ID3(i,j,k+1,Nx,Ny,Nz,PBC)] - u[ID3(i,j,k-1,Nx,Ny,Nz,PBC)])/(2.f*h);
}

// ============================================================================
// Divergenz stress: f_i = ∑_j ∂_j stress__ij   (4. Ordnung konsistent mit Strain-Kernel)
// ============================================================================
extern "C" __global__ void
divergenceStress(
    float *fx, float *fy, float *fz,
    float * __restrict__ sigmaxx,
    float * __restrict__ sigmayy,
    float * __restrict__ sigmazz,
    float * __restrict__ sigmaxy, // = stress_xy = stress_yx
    float * __restrict__ sigmaxz, // = stress_xz = stress_zx
    float * __restrict__ sigmayz, // = stress_yz = stress_zy
    int Nx, int Ny, int Nz,
    float dx, float dy, float dz,
    uint8_t PBC
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Harte Bounds (wie im Stress-Kernel); innen clamping/wrap via ID3.
    if ((!PBCx && (ix < 0 || ix >= Nx)) ||
        (!PBCy && (iy < 0 || iy >= Ny)) ||
        (!PBCz && (iz < 0 || iz >= Nz))) return;

    // Zentrale (geclampte/gewrapte) Zelle
    int i = hclampx(lclampx(ix));
    int j = hclampy(lclampy(iy));
    int k = hclampz(lclampz(iz));
    int IClamp = idx(i,j,k);

    // Ableitungen (alle über deriv_axis, gleiche BC-/Stencil-Logik)
    float dstress_xx_dx = deriv_axis(sigmaxx, i,j,k, Nx,Ny,Nz, PBC, dx, 0, 0, 0);
    float dstress_xy_dy = deriv_axis(sigmaxy, i,j,k, Nx,Ny,Nz, PBC, dy, 1, 0, 0);
    float dstress_xz_dz = (Nz > 5 ? deriv_axis(sigmaxz, i,j,k, Nx,Ny,Nz, PBC, dz, 2, 0, 0): 0.0f); // only if enough cells in z

    float dstress_yx_dx = deriv_axis(sigmaxy, i,j,k, Nx,Ny,Nz, PBC, dx, 0, 0, 0);
    float dstress_yy_dy = deriv_axis(sigmayy, i,j,k, Nx,Ny,Nz, PBC, dy, 1, 0, 0);
    float dstress_yz_dz = (Nz > 5 ? deriv_axis(sigmayz, i,j,k, Nx,Ny,Nz, PBC, dz, 2, 0, 0): 0.0f); // only if enough cells in z

    float dstress_zx_dx = deriv_axis(sigmaxz, i,j,k, Nx,Ny,Nz, PBC, dx, 0, 0, 0);
    float dstress_zy_dy = deriv_axis(sigmayz, i,j,k, Nx,Ny,Nz, PBC, dy, 1, 0, 0);
    float dstress_zz_dz = (Nz > 5 ? deriv_axis(sigmazz, i,j,k, Nx,Ny,Nz, PBC, dz, 2, 0, 0): 0.0f); // only if enough cells in z

    // Write auf exakt dieselbe (geclampte) Zelle wie im Stress-Kernel
    fx[IClamp] = dstress_xx_dx + dstress_xy_dy + dstress_xz_dz;
    fy[IClamp] = dstress_yx_dx + dstress_yy_dy + dstress_yz_dz;
    fz[IClamp] = dstress_zx_dx + dstress_zy_dy + dstress_zz_dz;
}
