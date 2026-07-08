#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Bulk force columns of the fully coupled magnetoelastic solver
// (magnum.np solver.dpd: f_ij minus the magnetic force terms get_fm).
//
// Output fc<q><i> = sum of all force terms of row i whose OUTER derivative is along q,
// including minus the corresponding magnetic-stress derivative terms:
//   f_i = sum_q fc<q><i>  (before Neumann boundary overrides, see MelasAssembleNP).
//
// Terms per column (block-diagonal C, paper Eq. 24):
//  col x: row x: D2x[C11,ux] + Dx(C12 dyuy) + Dx(C13 dzuz) - 3 l100 (C11 mx dxmx + C12 my dxmy + C13 mz dxmz)
//         row y: D2x[C66,uy] + Dx(C66 dyux) - 3 l111 C66 (mx dxmy + my dxmx)
//         row z: D2x[C55,uz] + Dx(C55 dzux) - 3 l111 C55 (mx dxmz + mz dxmx)
//  col y: row x: D2y[C66,ux] + Dy(C66 dxuy) - 3 l111 C66 (mx dymy + my dymx)
//         row y: D2y[C22,uy] + Dy(C12 dxux) + Dy(C23 dzuz) - 3 l100 (C12 mx dymx + C22 my dymy + C23 mz dymz)
//         row z: D2y[C44,uz] + Dy(C44 dzuy) - 3 l111 C44 (my dymz + mz dymy)
//  col z: row x: D2z[C55,ux] + Dz(C55 dxuz) - 3 l111 C55 (mx dzmz + mz dzmx)
//         row y: D2z[C44,uy] + Dz(C44 dyuz) - 3 l111 C44 (my dzmz + mz dzmy)
//         row z: D2z[C33,uz] + Dz(C13 dxux) + Dz(C23 dyuy) - 3 l100 (C13 mx dzmx + C23 my dzmy + C33 mz dzmz)
//
// D2q = jump-aware second derivative (B jump data B<q><comp>), Dq(C g) = jump-aware first
// derivative of the product with jump coefficient C and B=0. so != 0 applies the 3-point
// boundary stencil to the product fields at non-periodic mesh boundaries (boundary_nodes=3).
extern "C" __global__ void
MelasForceNP(float* __restrict__ fcxx, float* __restrict__ fcxy, float* __restrict__ fcxz,
             float* __restrict__ fcyx, float* __restrict__ fcyy, float* __restrict__ fcyz,
             float* __restrict__ fczx, float* __restrict__ fczy, float* __restrict__ fczz,
             float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
             float* __restrict__ guxx, float* __restrict__ guxy, float* __restrict__ guxz,
             float* __restrict__ guyx, float* __restrict__ guyy, float* __restrict__ guyz,
             float* __restrict__ guzx, float* __restrict__ guzy, float* __restrict__ guzz,
             float* __restrict__ gmxx, float* __restrict__ gmxy, float* __restrict__ gmxz,
             float* __restrict__ gmyx, float* __restrict__ gmyy, float* __restrict__ gmyz,
             float* __restrict__ gmzx, float* __restrict__ gmzy, float* __restrict__ gmzz,
             float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
             float* __restrict__ Blxx, float* __restrict__ Blxy, float* __restrict__ Blxz,
             float* __restrict__ Brxx, float* __restrict__ Brxy, float* __restrict__ Brxz,
             float* __restrict__ Blyx, float* __restrict__ Blyy, float* __restrict__ Blyz,
             float* __restrict__ Bryx, float* __restrict__ Bryy, float* __restrict__ Bryz,
             float* __restrict__ Blzx, float* __restrict__ Blzy, float* __restrict__ Blzz,
             float* __restrict__ Brzx, float* __restrict__ Brzy, float* __restrict__ Brzz,
             float* __restrict__ l100_, float l100_mul,
             float* __restrict__ l111_, float l111_mul,
             float* __restrict__ C11_, float C11_mul,
             float* __restrict__ C12_, float C12_mul,
             float* __restrict__ C13_, float C13_mul,
             float* __restrict__ C22_, float C22_mul,
             float* __restrict__ C23_, float C23_mul,
             float* __restrict__ C33_, float C33_mul,
             float* __restrict__ C44_, float C44_mul,
             float* __restrict__ C55_, float C55_mul,
             float* __restrict__ C66_, float C66_mul,
             int Nx, int Ny, int Nz,
             float cx, float cy, float cz,
             int so, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    float l100 = amul(l100_, l100_mul, I);
    float l111 = amul(l111_, l111_mul, I);
    float c11 = amul(C11_, C11_mul, I);
    float c12 = amul(C12_, C12_mul, I);
    float c13 = amul(C13_, C13_mul, I);
    float c22 = amul(C22_, C22_mul, I);
    float c23 = amul(C23_, C23_mul, I);
    float c33 = amul(C33_, C33_mul, I);
    float c44 = amul(C44_, C44_mul, I);
    float c55 = amul(C55_, C55_mul, I);
    float c66 = amul(C66_, C66_mul, I);

    float m0x = mx[I], m0y = my[I], m0z = mz[I];

    // ================= column x (outer derivative along x) =================
    {
        bool hasm = PBCx || (ix > 0);
        bool hasp = PBCx || (ix < Nx - 1);
        int Im = idx(lclampx(ix - 1), iy, iz);
        int Ip = idx(hclampx(ix + 1), iy, iz);

        float c11m = amul(C11_, C11_mul, Im), c11p = amul(C11_, C11_mul, Ip);
        float c12m = amul(C12_, C12_mul, Im), c12p = amul(C12_, C12_mul, Ip);
        float c13m = amul(C13_, C13_mul, Im), c13p = amul(C13_, C13_mul, Ip);
        float c55m = amul(C55_, C55_mul, Im), c55p = amul(C55_, C55_mul, Ip);
        float c66m = amul(C66_, C66_mul, Im), c66p = amul(C66_, C66_mul, Ip);

        // ---- row x ----
        float f = 0.0f;
        {
            // D2x[C11, ux]
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(ux[I], ux[Ip], c11, c11p, Brxx[I], Blxx[Ip], cx);
            if (hasm) a += melas_d2_minus(ux[I], ux[Im], c11, c11m, Brxx[Im], Blxx[I], cx);
            f += a / cx; // note: melas_d2_* already carries one 1/h via denom (magnum.np a/dx_denom)

            // Dx(C12 dyuy) + Dx(C13 dzuz)
            f += melas_dmix(c12 * guyy[I], c12m * guyy[Im], c12p * guyy[Ip], c12, c12m, c12p, hasm, hasp, cx);
            f += melas_dmix(c13 * guzz[I], c13m * guzz[Im], c13p * guzz[Ip], c13, c13m, c13p, hasm, hasp, cx);

            // magnetic force terms
            f -= 3.0f * l100 * (c11 * m0x * gmxx[I] + c12 * m0y * gmyx[I] + c13 * m0z * gmzx[I]);
        }
        // 3-pt boundary override for the mixed terms at open x boundaries
        if (so && !PBCx && Nx > 2 && (ix == 0 || ix == Nx - 1)) {
            int s = (ix == 0) ? 1 : -1;
            int I1 = idx(ix + s, iy, iz);
            int I2 = idx(ix + 2 * s, iy, iz);
            float sgn = (ix == 0) ? 1.0f : -1.0f;
            // recompute the mixed contributions with the 3-point one-sided stencil
            float fmix3 = 0.0f;
            fmix3 += sgn * (-(amul(C12_, C12_mul, I2) * guyy[I2]) + 4.0f * (amul(C12_, C12_mul, I1) * guyy[I1]) - 3.0f * (c12 * guyy[I])) / (2.0f * cx);
            fmix3 += sgn * (-(amul(C13_, C13_mul, I2) * guzz[I2]) + 4.0f * (amul(C13_, C13_mul, I1) * guzz[I1]) - 3.0f * (c13 * guzz[I])) / (2.0f * cx);
            // remove the one-sided estimates that were added above
            f -= melas_dmix(c12 * guyy[I], c12m * guyy[Im], c12p * guyy[Ip], c12, c12m, c12p, hasm, hasp, cx);
            f -= melas_dmix(c13 * guzz[I], c13m * guzz[Im], c13p * guzz[Ip], c13, c13m, c13p, hasm, hasp, cx);
            f += fmix3;
        }
        fcxx[I] = f;

        // ---- row y ----
        f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(uy[I], uy[Ip], c66, c66p, Brxy[I], Blxy[Ip], cx);
            if (hasm) a += melas_d2_minus(uy[I], uy[Im], c66, c66m, Brxy[Im], Blxy[I], cx);
            f += a / cx; // note: melas_d2_* already carries one 1/h via denom (magnum.np a/dx_denom)

            f += melas_dmix(c66 * guxy[I], c66m * guxy[Im], c66p * guxy[Ip], c66, c66m, c66p, hasm, hasp, cx);

            f -= 3.0f * l111 * c66 * (m0x * gmyx[I] + m0y * gmxx[I]);
        }
        if (so && !PBCx && Nx > 2 && (ix == 0 || ix == Nx - 1)) {
            int s = (ix == 0) ? 1 : -1;
            int I1 = idx(ix + s, iy, iz);
            int I2 = idx(ix + 2 * s, iy, iz);
            float sgn = (ix == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c66 * guxy[I], c66m * guxy[Im], c66p * guxy[Ip], c66, c66m, c66p, hasm, hasp, cx);
            f += sgn * (-(amul(C66_, C66_mul, I2) * guxy[I2]) + 4.0f * (amul(C66_, C66_mul, I1) * guxy[I1]) - 3.0f * (c66 * guxy[I])) / (2.0f * cx);
        }
        fcxy[I] = f;

        // ---- row z ----
        f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(uz[I], uz[Ip], c55, c55p, Brxz[I], Blxz[Ip], cx);
            if (hasm) a += melas_d2_minus(uz[I], uz[Im], c55, c55m, Brxz[Im], Blxz[I], cx);
            f += a / cx; // note: melas_d2_* already carries one 1/h via denom (magnum.np a/dx_denom)

            f += melas_dmix(c55 * guxz[I], c55m * guxz[Im], c55p * guxz[Ip], c55, c55m, c55p, hasm, hasp, cx);

            f -= 3.0f * l111 * c55 * (m0x * gmzx[I] + m0z * gmxx[I]);
        }
        if (so && !PBCx && Nx > 2 && (ix == 0 || ix == Nx - 1)) {
            int s = (ix == 0) ? 1 : -1;
            int I1 = idx(ix + s, iy, iz);
            int I2 = idx(ix + 2 * s, iy, iz);
            float sgn = (ix == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c55 * guxz[I], c55m * guxz[Im], c55p * guxz[Ip], c55, c55m, c55p, hasm, hasp, cx);
            f += sgn * (-(amul(C55_, C55_mul, I2) * guxz[I2]) + 4.0f * (amul(C55_, C55_mul, I1) * guxz[I1]) - 3.0f * (c55 * guxz[I])) / (2.0f * cx);
        }
        fcxz[I] = f;
    }

    // ================= column y (outer derivative along y) =================
    {
        bool hasm = PBCy || (iy > 0);
        bool hasp = PBCy || (iy < Ny - 1);
        int Im = idx(ix, lclampy(iy - 1), iz);
        int Ip = idx(ix, hclampy(iy + 1), iz);

        float c12m = amul(C12_, C12_mul, Im), c12p = amul(C12_, C12_mul, Ip);
        float c22m = amul(C22_, C22_mul, Im), c22p = amul(C22_, C22_mul, Ip);
        float c23m = amul(C23_, C23_mul, Im), c23p = amul(C23_, C23_mul, Ip);
        float c44m = amul(C44_, C44_mul, Im), c44p = amul(C44_, C44_mul, Ip);
        float c66m = amul(C66_, C66_mul, Im), c66p = amul(C66_, C66_mul, Ip);

        // ---- row x ----
        float f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(ux[I], ux[Ip], c66, c66p, Bryx[I], Blyx[Ip], cy);
            if (hasm) a += melas_d2_minus(ux[I], ux[Im], c66, c66m, Bryx[Im], Blyx[I], cy);
            f += a / cy;

            f += melas_dmix(c66 * guyx[I], c66m * guyx[Im], c66p * guyx[Ip], c66, c66m, c66p, hasm, hasp, cy);

            f -= 3.0f * l111 * c66 * (m0x * gmyy[I] + m0y * gmxy[I]);
        }
        if (so && !PBCy && Ny > 2 && (iy == 0 || iy == Ny - 1)) {
            int s = (iy == 0) ? 1 : -1;
            int I1 = idx(ix, iy + s, iz);
            int I2 = idx(ix, iy + 2 * s, iz);
            float sgn = (iy == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c66 * guyx[I], c66m * guyx[Im], c66p * guyx[Ip], c66, c66m, c66p, hasm, hasp, cy);
            f += sgn * (-(amul(C66_, C66_mul, I2) * guyx[I2]) + 4.0f * (amul(C66_, C66_mul, I1) * guyx[I1]) - 3.0f * (c66 * guyx[I])) / (2.0f * cy);
        }
        fcyx[I] = f;

        // ---- row y ----
        f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(uy[I], uy[Ip], c22, c22p, Bryy[I], Blyy[Ip], cy);
            if (hasm) a += melas_d2_minus(uy[I], uy[Im], c22, c22m, Bryy[Im], Blyy[I], cy);
            f += a / cy;

            f += melas_dmix(c12 * guxx[I], c12m * guxx[Im], c12p * guxx[Ip], c12, c12m, c12p, hasm, hasp, cy);
            f += melas_dmix(c23 * guzz[I], c23m * guzz[Im], c23p * guzz[Ip], c23, c23m, c23p, hasm, hasp, cy);

            f -= 3.0f * l100 * (c12 * m0x * gmxy[I] + c22 * m0y * gmyy[I] + c23 * m0z * gmzy[I]);
        }
        if (so && !PBCy && Ny > 2 && (iy == 0 || iy == Ny - 1)) {
            int s = (iy == 0) ? 1 : -1;
            int I1 = idx(ix, iy + s, iz);
            int I2 = idx(ix, iy + 2 * s, iz);
            float sgn = (iy == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c12 * guxx[I], c12m * guxx[Im], c12p * guxx[Ip], c12, c12m, c12p, hasm, hasp, cy);
            f -= melas_dmix(c23 * guzz[I], c23m * guzz[Im], c23p * guzz[Ip], c23, c23m, c23p, hasm, hasp, cy);
            f += sgn * (-(amul(C12_, C12_mul, I2) * guxx[I2]) + 4.0f * (amul(C12_, C12_mul, I1) * guxx[I1]) - 3.0f * (c12 * guxx[I])) / (2.0f * cy);
            f += sgn * (-(amul(C23_, C23_mul, I2) * guzz[I2]) + 4.0f * (amul(C23_, C23_mul, I1) * guzz[I1]) - 3.0f * (c23 * guzz[I])) / (2.0f * cy);
        }
        fcyy[I] = f;

        // ---- row z ----
        f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(uz[I], uz[Ip], c44, c44p, Bryz[I], Blyz[Ip], cy);
            if (hasm) a += melas_d2_minus(uz[I], uz[Im], c44, c44m, Bryz[Im], Blyz[I], cy);
            f += a / cy;

            f += melas_dmix(c44 * guyz[I], c44m * guyz[Im], c44p * guyz[Ip], c44, c44m, c44p, hasm, hasp, cy);

            f -= 3.0f * l111 * c44 * (m0y * gmzy[I] + m0z * gmyy[I]);
        }
        if (so && !PBCy && Ny > 2 && (iy == 0 || iy == Ny - 1)) {
            int s = (iy == 0) ? 1 : -1;
            int I1 = idx(ix, iy + s, iz);
            int I2 = idx(ix, iy + 2 * s, iz);
            float sgn = (iy == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c44 * guyz[I], c44m * guyz[Im], c44p * guyz[Ip], c44, c44m, c44p, hasm, hasp, cy);
            f += sgn * (-(amul(C44_, C44_mul, I2) * guyz[I2]) + 4.0f * (amul(C44_, C44_mul, I1) * guyz[I1]) - 3.0f * (c44 * guyz[I])) / (2.0f * cy);
        }
        fcyz[I] = f;
    }

    // ================= column z (outer derivative along z) =================
    {
        bool hasm = PBCz || (iz > 0);
        bool hasp = PBCz || (iz < Nz - 1);
        int Im = idx(ix, iy, lclampz(iz - 1));
        int Ip = idx(ix, iy, hclampz(iz + 1));

        float c13m = amul(C13_, C13_mul, Im), c13p = amul(C13_, C13_mul, Ip);
        float c23m = amul(C23_, C23_mul, Im), c23p = amul(C23_, C23_mul, Ip);
        float c33m = amul(C33_, C33_mul, Im), c33p = amul(C33_, C33_mul, Ip);
        float c44m = amul(C44_, C44_mul, Im), c44p = amul(C44_, C44_mul, Ip);
        float c55m = amul(C55_, C55_mul, Im), c55p = amul(C55_, C55_mul, Ip);

        // ---- row x ----
        float f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(ux[I], ux[Ip], c55, c55p, Brzx[I], Blzx[Ip], cz);
            if (hasm) a += melas_d2_minus(ux[I], ux[Im], c55, c55m, Brzx[Im], Blzx[I], cz);
            f += a / cz;

            f += melas_dmix(c55 * guzx[I], c55m * guzx[Im], c55p * guzx[Ip], c55, c55m, c55p, hasm, hasp, cz);

            f -= 3.0f * l111 * c55 * (m0x * gmzz[I] + m0z * gmxz[I]);
        }
        if (so && !PBCz && Nz > 2 && (iz == 0 || iz == Nz - 1)) {
            int s = (iz == 0) ? 1 : -1;
            int I1 = idx(ix, iy, iz + s);
            int I2 = idx(ix, iy, iz + 2 * s);
            float sgn = (iz == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c55 * guzx[I], c55m * guzx[Im], c55p * guzx[Ip], c55, c55m, c55p, hasm, hasp, cz);
            f += sgn * (-(amul(C55_, C55_mul, I2) * guzx[I2]) + 4.0f * (amul(C55_, C55_mul, I1) * guzx[I1]) - 3.0f * (c55 * guzx[I])) / (2.0f * cz);
        }
        fczx[I] = f;

        // ---- row y ----
        f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(uy[I], uy[Ip], c44, c44p, Brzy[I], Blzy[Ip], cz);
            if (hasm) a += melas_d2_minus(uy[I], uy[Im], c44, c44m, Brzy[Im], Blzy[I], cz);
            f += a / cz;

            f += melas_dmix(c44 * guzy[I], c44m * guzy[Im], c44p * guzy[Ip], c44, c44m, c44p, hasm, hasp, cz);

            f -= 3.0f * l111 * c44 * (m0y * gmzz[I] + m0z * gmyz[I]);
        }
        if (so && !PBCz && Nz > 2 && (iz == 0 || iz == Nz - 1)) {
            int s = (iz == 0) ? 1 : -1;
            int I1 = idx(ix, iy, iz + s);
            int I2 = idx(ix, iy, iz + 2 * s);
            float sgn = (iz == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c44 * guzy[I], c44m * guzy[Im], c44p * guzy[Ip], c44, c44m, c44p, hasm, hasp, cz);
            f += sgn * (-(amul(C44_, C44_mul, I2) * guzy[I2]) + 4.0f * (amul(C44_, C44_mul, I1) * guzy[I1]) - 3.0f * (c44 * guzy[I])) / (2.0f * cz);
        }
        fczy[I] = f;

        // ---- row z ----
        f = 0.0f;
        {
            float a = 0.0f;
            if (hasp) a += melas_d2_plus(uz[I], uz[Ip], c33, c33p, Brzz[I], Blzz[Ip], cz);
            if (hasm) a += melas_d2_minus(uz[I], uz[Im], c33, c33m, Brzz[Im], Blzz[I], cz);
            f += a / cz;

            f += melas_dmix(c13 * guxx[I], c13m * guxx[Im], c13p * guxx[Ip], c13, c13m, c13p, hasm, hasp, cz);
            f += melas_dmix(c23 * guyy[I], c23m * guyy[Im], c23p * guyy[Ip], c23, c23m, c23p, hasm, hasp, cz);

            f -= 3.0f * l100 * (c13 * m0x * gmxz[I] + c23 * m0y * gmyz[I] + c33 * m0z * gmzz[I]);
        }
        if (so && !PBCz && Nz > 2 && (iz == 0 || iz == Nz - 1)) {
            int s = (iz == 0) ? 1 : -1;
            int I1 = idx(ix, iy, iz + s);
            int I2 = idx(ix, iy, iz + 2 * s);
            float sgn = (iz == 0) ? 1.0f : -1.0f;
            f -= melas_dmix(c13 * guxx[I], c13m * guxx[Im], c13p * guxx[Ip], c13, c13m, c13p, hasm, hasp, cz);
            f -= melas_dmix(c23 * guyy[I], c23m * guyy[Im], c23p * guyy[Ip], c23, c23m, c23p, hasm, hasp, cz);
            f += sgn * (-(amul(C13_, C13_mul, I2) * guxx[I2]) + 4.0f * (amul(C13_, C13_mul, I1) * guxx[I1]) - 3.0f * (c13 * guxx[I])) / (2.0f * cz);
            f += sgn * (-(amul(C23_, C23_mul, I2) * guyy[I2]) + 4.0f * (amul(C23_, C23_mul, I1) * guyy[I1]) - 3.0f * (c23 * guyy[I])) / (2.0f * cz);
        }
        fczz[I] = f;
    }
}
