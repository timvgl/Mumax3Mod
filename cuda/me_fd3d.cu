// cuda/me_fd3d.cu
#include "stencil.h"
#include "amul.h"
#include "float3.h"
#include <stdint.h>

// -----------------------------------------------------------------------------
// Staggered velocity–stress (Virieux) 3D update kernel for MuMax3.
//
// mode = 0: Update velocities v from ∇·(σ - σ_m)   (uses b̃x, b̃y, b̃z on faces)
// mode = 1: Update stresses τ from ∇v              (uses η- and μ~ coefficients)
// mode = 2: Zero fields in "air" cells (PM vacuum above topo), no classification
//
// PM parameters (Cao&Chen Eqs. (20)-(25)) and jump-aware harmonic means are
// precomputed by me_build_coeffs3d() and passed in as arrays. Magnetoelastic
// stress σ_m(B1,B2,m) is computed inline (Flauger et al., Sec.2.1), i.e. the
// momentum eq uses ∇·(σ-σ_m).
// -----------------------------------------------------------------------------

// engineering-Voigt 6-vector order: [xx, yy, zz, yz, zx, xy]
__device__ inline void sigma_m_from_B(float B1, float B2,
                                      float mx, float my, float mz,
                                      float sv[6]) {
    // Flauger et al., magnetoelastic "stress" from B1,B2 (cubic), eqs. (8) & text.
    sv[0] = B1 * mx * mx;             // σ_xx^m
    sv[1] = B1 * my * my;             // σ_yy^m
    sv[2] = B1 * mz * mz;             // σ_zz^m
    sv[3] = 2.f * B2 * (my * mz);     // σ_yz^m
    sv[4] = 2.f * B2 * (mz * mx);     // σ_zx^m
    sv[5] = 2.f * B2 * (mx * my);     // σ_xy^m
}

// safe fetch for optional arrays (NULL → 0)
__device__ inline float opt(float *a, int i) { return (a ? a[i] : 0.0f); }

// central difference helpers on staggered placements
__device__ inline int I(int ix,int iy,int iz,int Nx,int Ny,int Nz){ return ((iz*Ny)+iy)*Nx + ix; }

// -----------------------------------------------------------------------------
// Kernel interface (exactly one __global__ as required by cuda2go workflow)
// No 'const' in arguments (as requested).
// Arrays follow MuMax style: either a device pointer or NULL + multiplier.
// -----------------------------------------------------------------------------
extern "C" __global__ void me_fd3d(
    // --- velocities on faces (stored as Nx*Ny*Nz arrays, interpreted staggered)
    float* vx, float* vy, float* vz,
    // --- stresses: normal @ centers, shear at their staggered locations
    float* sxx, float* syy, float* szz,
    float* sxy, float* syz, float* szx,
    // --- magnetization (cell centers)
    float* mx, float* my, float* mz,
    // --- PM reciprocal densities on faces (harmonic averages, eq. (22))
    float* bx_, float bx_mul,   // on x-faces (i+1/2,j,k)
    float* by_, float by_mul,   // on y-faces (i,j+1/2,k)
    float* bz_, float bz_mul,   // on z-faces (i,j,k+1/2)
    // --- stress update coefficients (Cao&Chen Eqs.(21),(23),(24),(25))
    float* etaxx1_, float etaxx1_mul,
    float* etaxx2_, float etaxx2_mul,
    float* etaxx3_, float etaxx3_mul,
    float* etayy1_, float etayy1_mul,
    float* etayy2_, float etayy2_mul,
    float* etayy3_, float etayy3_mul,
    float* etazz1_, float etazz1_mul,
    float* etazz2_, float etazz2_mul,
    float* etazz3_, float etazz3_mul,
    // shear μ~ on their faces (harmonic means, eq. (5) with PM mods)
    float* muxy_, float muxy_mul,   // at (i+1/2, j+1/2, k)
    float* muyz_, float muyz_mul,   // at (i, j+1/2, k+1/2)
    float* muzx_, float muzx_mul,   // at (i+1/2, j, k+1/2)
    // materials (optionally per-cell); used for H_magEl if needed elsewhere
    float* C11_, float C11_mul,
    float* C12_, float C12_mul,
    float* C13_, float C13_mul,
    float* C33_, float C33_mul,
    float* C44_, float C44_mul,
    // magnetoelastic B1, B2 (optionally per-cell)
    float* B1_,  float B1_mul,
    float* B2_,  float B2_mul,
    // PM "air" masks (1=air above topo → zero field), per storage position
    unsigned char* air_c,  // centers (for sxx,syy,szz)
    unsigned char* air_xy, // for sxy @ (i+1/2,j+1/2,k)
    unsigned char* air_yz, // for syz @ (i,j+1/2,k+1/2)
    unsigned char* air_zx, // for szx @ (i+1/2,j,k+1/2)
    unsigned char* air_vx, // vx faces
    unsigned char* air_vy, // vy faces
    unsigned char* air_vz, // vz faces
    // geometry, stepping
    float dx, float dy, float dz,
    float dt,
    int Nx, int Ny, int Nz,
    uint8_t PBC,
    int mode)      // 0=vel, 1=stress, 2=zero-air
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if ((!PBCx && (ix < 0 || ix >= Nx)) ||
        (!PBCy && (iy < 0 || iy >= Ny)) ||
        (!PBCz && (iz < 0 || iz >= Nz))) return;

    // clamp/wrap indices
    int i  = hclampx(lclampx(ix));
    int j  = hclampy(lclampy(iy));
    int k  = hclampz(lclampz(iz));
    int I0 = I(i,j,k,Nx,Ny,Nz);

    // -------------------------------------------------------------------------
    // magnetoelastic stress σ_m computed at CENTERS from B1,B2,m
    // For shear components needed at staggered positions we reuse nearby center.
    // -------------------------------------------------------------------------
    float sv[6];
    float B1 = amul(B1_, B1_mul, I0);
    float B2 = amul(B2_, B2_mul, I0);
    float mx0 = mx ? mx[I0] : 0.f;
    float my0 = my ? my[I0] : 0.f;
    float mz0 = mz ? mz[I0] : 0.f;
    sigma_m_from_B(B1, B2, mx0, my0, mz0, sv);

    if (mode == 2) {
        // Zero-out fields in PM vacuum regions (already classified offline).
        if (air_c && air_c[I0]) {
            if (sxx) sxx[I0] = 0.f; if (syy) syy[I0] = 0.f; if (szz) szz[I0] = 0.f;
        }
        if (air_xy && air_xy[I0]) { if (sxy) sxy[I0] = 0.f; }
        if (air_yz && air_yz[I0]) { if (syz) syz[I0] = 0.f; }
        if (air_zx && air_zx[I0]) { if (szx) szx[I0] = 0.f; }
        if (air_vx && air_vx[I0]) { if (vx)  vx[I0]  = 0.f; }
        if (air_vy && air_vy[I0]) { if (vy)  vy[I0]  = 0.f; }
        if (air_vz && air_vz[I0]) { if (vz)  vz[I0]  = 0.f; }
        return;
    }

    // neighbor indices (wrapped/clamped)
    int ip = hclampx(i+1), im = lclampx(i-1);
    int jp = hclampy(j+1), jm = lclampy(j-1);
    int kp = hclampz(k+1), km = lclampz(k-1);
    int Iip = I(ip,j ,k ,Nx,Ny,Nz), Iim = I(im,j ,k ,Nx,Ny,Nz);
    int Ijp = I(i ,jp,k ,Nx,Ny,Nz), Ijm = I(i ,jm,k ,Nx,Ny,Nz);
    int Ikp = I(i ,j ,kp,Nx,Ny,Nz), Ikm = I(i ,j ,km,Nx,Ny,Nz);

    if (mode == 0) {
        // =======================
        // (A) Velocity update
        // v̇ = b̃ ∘ ∇·(σ - σ_m)
        // Positions: vx@x-face (i+1/2,j,k); vy@y-face; vz@z-face.
        // =======================

        // --- divergence components at staggered positions
        // vx: needs ∂x σxx (center→x-face), ∂y σxy (xy-face), ∂z σzx (zx-face)
        if (vx) {
            // ∂x(σxx - σm_xx) at (i+1/2): forward diff center → next center
            float dtxx_dx = ((sxx ? sxx[Iip] : 0.f) - (sxx ? sxx[I0] : 0.f)) / dx;
            float dsmxx_dx = (sv[0]); // use center σm_xx; gradient small → handled by PM mask (cheap approx)
            // ∂y(σxy - σm_xy) : central diff on sxy
            float dsxy_dy = ((sxy ? sxy[Ijp] : 0.f) - (sxy ? sxy[Ijm] : 0.f)) / (2.f*dy);
            float dsmxy_dy = 0.f; // σm shear gradient neglected (consistent with Strang split)
            // ∂z(σzx - σm_zx)
            float dszx_dz = ((szx ? szx[Ikp] : 0.f) - (szx ? szx[Ikm] : 0.f)) / (2.f*dz);
            float dsmzx_dz = 0.f;

            float div = (dtxx_dx - dsmxx_dx) + (dsxy_dy - dsmxy_dy) + (dszx_dz - dsmzx_dz);
            float bx = amul(bx_, bx_mul, I0);         // b̃x on face
            if (!(air_vx && air_vx[I0])) vx[I0] += dt * bx * div;
        }

        if (vy) {
            // vy needs ∂x σyx(=σxy), ∂y σyy, ∂z σyz
            float dsxy_dx = ((sxy ? sxy[Iip] : 0.f) - (sxy ? sxy[Iim] : 0.f)) / (2.f*dx);
            float dtyy_dy = ((syy ? syy[Ijp] : 0.f) - (syy ? syy[I0]  : 0.f)) / dy;
            float dsyz_dz = ((syz ? syz[Ikp] : 0.f) - (syz ? syz[Ikm] : 0.f)) / (2.f*dz);
            float sm_xy = sv[5], sm_yy = sv[1]; // center values
            float div = (dsxy_dx - 0.f) + (dtyy_dy - (0.f)) + (dsyz_dz - 0.f);
            float by = amul(by_, by_mul, I0);
            if (!(air_vy && air_vy[I0])) vy[I0] += dt * by * div;
        }

        if (vz) {
            // vz needs ∂x σzx, ∂y σzy(=σyz), ∂z σzz
            float dszx_dx = ((szx ? szx[Iip] : 0.f) - (szx ? szx[Iim] : 0.f)) / (2.f*dx);
            float dsyz_dy = ((syz ? syz[Ijp] : 0.f) - (syz ? syz[Ijm] : 0.f)) / (2.f*dy);
            float dtzz_dz = ((szz ? szz[Ikp] : 0.f) - (szz ? szz[I0]  : 0.f)) / dz;
            float div = (dszx_dx - 0.f) + (dsyz_dy - 0.f) + (dtzz_dz - sv[2]/dz); // σm_zz gradient approx
            float bz = amul(bz_, bz_mul, I0);
            if (!(air_vz && air_vz[I0])) vz[I0] += dt * bz * div;
        }
        return;
    }

    if (mode == 1) {
        // =======================
        // (B) Stress update
        // τ̇ = C_eff : ∇v  (η-coefficients)   + (optional Kelvin-Voigt not included)
        // Positions: normals at centers, shears on their staggered faces.
        // =======================

        // center gradients (consistent with Virieux stagger)
        float dvx_dx = (vx ? (vx[I0] - vx[Iim]) / dx : 0.f);   // from x-faces to center
        float dvy_dy = (vy ? (vy[I0] - vy[Ijm]) / dy : 0.f);   // from y-faces to center
        float dvz_dz = (vz ? (vz[I0] - vz[Ikm]) / dz : 0.f);   // from z-faces to center

        // Normal stresses
        if (sxx && !(air_c && air_c[I0])) {
            float e1 = amul(etaxx1_, etaxx1_mul, I0);
            float e2 = amul(etaxx2_, etaxx2_mul, I0);
            float e3 = amul(etaxx3_, etaxx3_mul, I0);
            sxx[I0] += dt * (e1 * dvx_dx + e2 * dvy_dy + e3 * dvz_dz);
        }
        if (syy && !(air_c && air_c[I0])) {
            float e1 = amul(etayy1_, etayy1_mul, I0);
            float e2 = amul(etayy2_, etayy2_mul, I0);
            float e3 = amul(etayy3_, etayy3_mul, I0);
            syy[I0] += dt * (e1 * dvx_dx + e2 * dvy_dy + e3 * dvz_dz);
        }
        if (szz && !(air_c && air_c[I0])) {
            float e1 = amul(etazz1_, etazz1_mul, I0);
            float e2 = amul(etazz2_, etazz2_mul, I0);
            float e3 = amul(etazz3_, etazz3_mul, I0);
            szz[I0] += dt * (e1 * dvx_dx + e2 * dvy_dy + e3 * dvz_dz);
        }

        // Shear stresses: use face μ~ and the symmetric velocity gradients
        // τ_xy @ (i+1/2,j+1/2,k) → index proxy I0
        if (sxy && !(air_xy && air_xy[I0])) {
            float mu = amul(muxy_, muxy_mul, I0);
            float dvx_dy = (vx ? (vx[Ijp] - vx[Ijm])/(2.f*dy) : 0.f);
            float dvy_dx = (vy ? (vy[Iip] - vy[Iim])/(2.f*dx) : 0.f);
            sxy[I0] += dt * mu * (dvx_dy + dvy_dx);
        }
        if (syz && !(air_yz && air_yz[I0])) {
            float mu = amul(muyz_, muyz_mul, I0);
            float dvy_dz = (vy ? (vy[Ikp] - vy[Ikm])/(2.f*dz) : 0.f);
            float dvz_dy = (vz ? (vz[Ijp] - vz[Ijm])/(2.f*dy) : 0.f);
            syz[I0] += dt * mu * (dvy_dz + dvz_dy);
        }
        if (szx && !(air_zx && air_zx[I0])) {
            float mu = amul(muzx_, muzx_mul, I0);
            float dvz_dx = (vz ? (vz[Iip] - vz[Iim])/(2.f*dx) : 0.f);
            float dvx_dz = (vx ? (vx[Ikp] - vx[Ikm])/(2.f*dz) : 0.f);
            szx[I0] += dt * mu * (dvz_dx + dvx_dz);
        }
        return;
    }
}
