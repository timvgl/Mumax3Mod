#include "stencil.h"
#include <stdint.h>
#include "deriv_axis.cuh"

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

    float dstress_zx_dx = (Nz > 5 ? deriv_axis(sigmaxz, i,j,k, Nx,Ny,Nz, PBC, dx, 0, 0, 0): 0.0f); // only if enough cells in z
    float dstress_zy_dy = (Nz > 5 ? deriv_axis(sigmayz, i,j,k, Nx,Ny,Nz, PBC, dy, 1, 0, 0): 0.0f); // only if enough cells in z
    float dstress_zz_dz = (Nz > 5 ? deriv_axis(sigmazz, i,j,k, Nx,Ny,Nz, PBC, dz, 2, 0, 0): 0.0f); // only if enough cells in z

    // Write auf exakt dieselbe (geclampte) Zelle wie im Stress-Kernel
    fx[IClamp] = dstress_xx_dx + dstress_xy_dy + dstress_xz_dz;
    fy[IClamp] = dstress_yx_dx + dstress_yy_dy + dstress_yz_dz;
    fz[IClamp] = dstress_zx_dx + dstress_zy_dy + dstress_zz_dz;
}
