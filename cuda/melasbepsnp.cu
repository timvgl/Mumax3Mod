#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Displacement-gradient part of the B jump parameters, iterative step
// (magnum.np strain.py::_get_B_jump_conditions), added onto the magnetic-stress part.
//
// Interface values of transversal gradient components are estimated by the harmonic mean
// weighted by the corresponding stiffness component over the neighboring cell pair:
//   hm = (C_i g_i + C_n g_n) / (C_i + C_n)
// l-side of cell i: pair (i-1, i); r-side: pair (i, i+1). Prefactor is the local C of cell i.
// Per direction (paper Table I, B column without the sig_m part):
//   q=x: Bx = C12 hm(dyuy) + C13 hm(dzuz); By = C66 hm(dyux); Bz = C55 hm(dzux)
//   q=y: Bx = C66 hm(dxuy); By = C12 hm(dxux) + C23 hm(dzuz); Bz = C44 hm(dzuy)
//   q=z: Bx = C55 hm(dxuz); By = C44 hm(dyuz); Bz = C13 hm(dxux) + C23 hm(dyuy)
// Output: B<l/r><q><comp> = sigM part (input) + eps part.
// Neighbor indices wrap when the PBC bit is set, otherwise clamp (the clamped edge values are
// never used by the non-periodic difference formulas).
//
// gu<c> = jump-aware gradient of u_<c> with components (d_x, d_y, d_z).
extern "C" __global__ void
MelasBEpsNP(float* __restrict__ Blxx, float* __restrict__ Blxy, float* __restrict__ Blxz,
            float* __restrict__ Brxx, float* __restrict__ Brxy, float* __restrict__ Brxz,
            float* __restrict__ Blyx, float* __restrict__ Blyy, float* __restrict__ Blyz,
            float* __restrict__ Bryx, float* __restrict__ Bryy, float* __restrict__ Bryz,
            float* __restrict__ Blzx, float* __restrict__ Blzy, float* __restrict__ Blzz,
            float* __restrict__ Brzx, float* __restrict__ Brzy, float* __restrict__ Brzz,
            float* __restrict__ SlBlxx, float* __restrict__ SlBlxy, float* __restrict__ SlBlxz,
            float* __restrict__ SlBrxx, float* __restrict__ SlBrxy, float* __restrict__ SlBrxz,
            float* __restrict__ SlBlyx, float* __restrict__ SlBlyy, float* __restrict__ SlBlyz,
            float* __restrict__ SlBryx, float* __restrict__ SlBryy, float* __restrict__ SlBryz,
            float* __restrict__ SlBlzx, float* __restrict__ SlBlzy, float* __restrict__ SlBlzz,
            float* __restrict__ SlBrzx, float* __restrict__ SlBrzy, float* __restrict__ SlBrzz,
            float* __restrict__ guxx, float* __restrict__ guxy, float* __restrict__ guxz,
            float* __restrict__ guyx, float* __restrict__ guyy, float* __restrict__ guyz,
            float* __restrict__ guzx, float* __restrict__ guzy, float* __restrict__ guzz,
            float* __restrict__ C12_, float C12_mul,
            float* __restrict__ C13_, float C13_mul,
            float* __restrict__ C23_, float C23_mul,
            float* __restrict__ C44_, float C44_mul,
            float* __restrict__ C55_, float C55_mul,
            float* __restrict__ C66_, float C66_mul,
            int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    // neighbor indices (wrap with PBC, clamp otherwise)
    int Ixm = idx(lclampx(ix - 1), iy, iz);
    int Ixp = idx(hclampx(ix + 1), iy, iz);
    int Iym = idx(ix, lclampy(iy - 1), iz);
    int Iyp = idx(ix, hclampy(iy + 1), iz);
    int Izm = idx(ix, iy, lclampz(iz - 1));
    int Izp = idx(ix, iy, hclampz(iz + 1));

    float c12 = amul(C12_, C12_mul, I);
    float c13 = amul(C13_, C13_mul, I);
    float c23 = amul(C23_, C23_mul, I);
    float c44 = amul(C44_, C44_mul, I);
    float c55 = amul(C55_, C55_mul, I);
    float c66 = amul(C66_, C66_mul, I);

    // ---- x direction ----
    {
        float c12m = amul(C12_, C12_mul, Ixm), c12p = amul(C12_, C12_mul, Ixp);
        float c13m = amul(C13_, C13_mul, Ixm), c13p = amul(C13_, C13_mul, Ixp);
        float c55m = amul(C55_, C55_mul, Ixm), c55p = amul(C55_, C55_mul, Ixp);
        float c66m = amul(C66_, C66_mul, Ixm), c66p = amul(C66_, C66_mul, Ixp);

        // l: pair (i-1, i); r: pair (i, i+1)
        float dyuy_l = melas_harmonic(guyy[I], c12, guyy[Ixm], c12m);
        float dyuy_r = melas_harmonic(guyy[I], c12, guyy[Ixp], c12p);
        float dzuz_l = melas_harmonic(guzz[I], c13, guzz[Ixm], c13m);
        float dzuz_r = melas_harmonic(guzz[I], c13, guzz[Ixp], c13p);
        float dyux_l = melas_harmonic(guxy[I], c66, guxy[Ixm], c66m);
        float dyux_r = melas_harmonic(guxy[I], c66, guxy[Ixp], c66p);
        float dzux_l = melas_harmonic(guxz[I], c55, guxz[Ixm], c55m);
        float dzux_r = melas_harmonic(guxz[I], c55, guxz[Ixp], c55p);

        Blxx[I] = SlBlxx[I] + c12 * dyuy_l + c13 * dzuz_l;
        Brxx[I] = SlBrxx[I] + c12 * dyuy_r + c13 * dzuz_r;
        Blxy[I] = SlBlxy[I] + c66 * dyux_l;
        Brxy[I] = SlBrxy[I] + c66 * dyux_r;
        Blxz[I] = SlBlxz[I] + c55 * dzux_l;
        Brxz[I] = SlBrxz[I] + c55 * dzux_r;
    }

    // ---- y direction ----
    {
        float c12m = amul(C12_, C12_mul, Iym), c12p = amul(C12_, C12_mul, Iyp);
        float c23m = amul(C23_, C23_mul, Iym), c23p = amul(C23_, C23_mul, Iyp);
        float c44m = amul(C44_, C44_mul, Iym), c44p = amul(C44_, C44_mul, Iyp);
        float c66m = amul(C66_, C66_mul, Iym), c66p = amul(C66_, C66_mul, Iyp);

        float dxuy_l = melas_harmonic(guyx[I], c66, guyx[Iym], c66m);
        float dxuy_r = melas_harmonic(guyx[I], c66, guyx[Iyp], c66p);
        float dxux_l = melas_harmonic(guxx[I], c12, guxx[Iym], c12m);
        float dxux_r = melas_harmonic(guxx[I], c12, guxx[Iyp], c12p);
        float dzuz_l = melas_harmonic(guzz[I], c23, guzz[Iym], c23m);
        float dzuz_r = melas_harmonic(guzz[I], c23, guzz[Iyp], c23p);
        float dzuy_l = melas_harmonic(guyz[I], c44, guyz[Iym], c44m);
        float dzuy_r = melas_harmonic(guyz[I], c44, guyz[Iyp], c44p);

        Blyx[I] = SlBlyx[I] + c66 * dxuy_l;
        Bryx[I] = SlBryx[I] + c66 * dxuy_r;
        Blyy[I] = SlBlyy[I] + c12 * dxux_l + c23 * dzuz_l;
        Bryy[I] = SlBryy[I] + c12 * dxux_r + c23 * dzuz_r;
        Blyz[I] = SlBlyz[I] + c44 * dzuy_l;
        Bryz[I] = SlBryz[I] + c44 * dzuy_r;
    }

    // ---- z direction ----
    {
        float c13m = amul(C13_, C13_mul, Izm), c13p = amul(C13_, C13_mul, Izp);
        float c23m = amul(C23_, C23_mul, Izm), c23p = amul(C23_, C23_mul, Izp);
        float c44m = amul(C44_, C44_mul, Izm), c44p = amul(C44_, C44_mul, Izp);
        float c55m = amul(C55_, C55_mul, Izm), c55p = amul(C55_, C55_mul, Izp);

        float dxuz_l = melas_harmonic(guzx[I], c55, guzx[Izm], c55m);
        float dxuz_r = melas_harmonic(guzx[I], c55, guzx[Izp], c55p);
        float dyuz_l = melas_harmonic(guzy[I], c44, guzy[Izm], c44m);
        float dyuz_r = melas_harmonic(guzy[I], c44, guzy[Izp], c44p);
        float dxux_l = melas_harmonic(guxx[I], c13, guxx[Izm], c13m);
        float dxux_r = melas_harmonic(guxx[I], c13, guxx[Izp], c13p);
        float dyuy_l = melas_harmonic(guyy[I], c23, guyy[Izm], c23m);
        float dyuy_r = melas_harmonic(guyy[I], c23, guyy[Izp], c23p);

        Blzx[I] = SlBlzx[I] + c55 * dxuz_l;
        Brzx[I] = SlBrzx[I] + c55 * dxuz_r;
        Blzy[I] = SlBlzy[I] + c44 * dyuz_l;
        Brzy[I] = SlBrzy[I] + c44 * dyuz_r;
        Blzz[I] = SlBlzz[I] + c13 * dxux_l + c23 * dyuy_l;
        Brzz[I] = SlBrzz[I] + c13 * dxux_r + c23 * dyuy_r;
    }
}
