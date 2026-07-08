#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Assembles the elastic right-hand side of the fully coupled magnetoelastic solver
// (magnum.np solver.dpd, final part):
//  1. Neumann traction boundary conditions on the non-periodic outer faces: the entire force
//     column fc<q> is replaced at boundary cells by the boundary estimate built from the
//     traction t (excitation, may be NULL -> homogeneous/natural BC) and the total stress
//     sigma_el = sigma - sigma_m (sd diag, so offdiag in (yz,xz,xy) order).
//     bn selects the boundary treatment (magnum.np boundary_nodes):
//       bn=3: 3-point/midpoint formula  g = -s (s2 hl^2 - t hr^2 + (hr^2-hl^2) s1)/(hr hl^2 + hl hr^2)
//             with hl = dq/2, hr = dq (uniform grid)
//       bn=2: g = s (t - s2)/(1.5 dq)
//       bn=1: g = s (t - s1)/dq
//     s = outward normal sign, s1 = stress at the boundary cell, s2 at the next inner cell.
//  2. f_el = fc_x + fc_y + fc_z (per row).
//  3. outMode = 0: dst = (f_el - eta*du)/rho, zero where rho = 0, masked by the elastic mask
//     and the Dirichlet (frozenDispLocNP) mask -> dst = d(du)/dt.
//     outMode = 1: dst = f_el (with BC overrides applied, unmasked) -> force diagnostic.
extern "C" __global__ void
MelasAssembleNP(float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
                float* __restrict__ fcxx, float* __restrict__ fcxy, float* __restrict__ fcxz,
                float* __restrict__ fcyx, float* __restrict__ fcyy, float* __restrict__ fcyz,
                float* __restrict__ fczx, float* __restrict__ fczy, float* __restrict__ fczz,
                float* __restrict__ sdx, float* __restrict__ sdy, float* __restrict__ sdz,
                float* __restrict__ sox, float* __restrict__ soy, float* __restrict__ soz,
                float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz,
                float* __restrict__ eta_, float eta_mul,
                float* __restrict__ rho_, float rho_mul,
                float* __restrict__ mask_, float mask_mul,
                float* __restrict__ frozen_, float frozen_mul,
                int bn, int outMode,
                int Nx, int Ny, int Nz,
                float cx, float cy, float cz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    float fx = fcxx[I], fy = fcxy[I], fz = fcxz[I];
    float gyx = fcyx[I], gyy = fcyy[I], gyz = fcyz[I];
    float gzx = fczx[I], gzy = fczy[I], gzz = fczz[I];

    float t0x = (tx == NULL) ? 0.0f : tx[I];
    float t0y = (ty == NULL) ? 0.0f : ty[I];
    float t0z = (tz == NULL) ? 0.0f : tz[I];

    // ---- x faces ----
    if (!PBCx && Nx > 1 && (ix == 0 || ix == Nx - 1)) {
        float s = (ix == 0) ? -1.0f : 1.0f;
        int Ii = (ix == 0) ? idx(1, iy, iz) : idx(Nx - 2, iy, iz);
        // rows: x uses sig_xx (sd[0]); y uses sig_xy (so[2]); z uses sig_xz (so[1])
        float s1x = sdx[I], s2x = sdx[Ii];
        float s1y = soz[I], s2y = soz[Ii];
        float s1z = soy[I], s2z = soy[Ii];
        float gx2, gy2, gz2;
        if (bn >= 3 && Nx > 1) {
            float hl = 0.5f * cx, hr = cx;
            float den = hr * hl * hl + hl * hr * hr;
            gx2 = -s * (s2x * hl * hl - t0x * hr * hr + (hr * hr - hl * hl) * s1x) / den;
            gy2 = -s * (s2y * hl * hl - t0y * hr * hr + (hr * hr - hl * hl) * s1y) / den;
            gz2 = -s * (s2z * hl * hl - t0z * hr * hr + (hr * hr - hl * hl) * s1z) / den;
        } else if (bn == 2 && Nx > 1) {
            float h = 1.5f * cx;
            gx2 = s * (t0x - s2x) / h;
            gy2 = s * (t0y - s2y) / h;
            gz2 = s * (t0z - s2z) / h;
        } else {
            gx2 = s * (t0x - s1x) / cx;
            gy2 = s * (t0y - s1y) / cx;
            gz2 = s * (t0z - s1z) / cx;
        }
        fx = gx2;
        fy = gy2;
        fz = gz2;
    }

    // ---- y faces ----
    if (!PBCy && Ny > 1 && (iy == 0 || iy == Ny - 1)) {
        float s = (iy == 0) ? -1.0f : 1.0f;
        int Ii = (iy == 0) ? idx(ix, 1, iz) : idx(ix, Ny - 2, iz);
        // rows: y uses sig_yy (sd[1]); x uses sig_xy (so[2]); z uses sig_yz (so[0])
        float s1y = sdy[I], s2y = sdy[Ii];
        float s1x = soz[I], s2x = soz[Ii];
        float s1z = sox[I], s2z = sox[Ii];
        float gx2, gy2, gz2;
        if (bn >= 3 && Ny > 1) {
            float hl = 0.5f * cy, hr = cy;
            float den = hr * hl * hl + hl * hr * hr;
            gx2 = -s * (s2x * hl * hl - t0x * hr * hr + (hr * hr - hl * hl) * s1x) / den;
            gy2 = -s * (s2y * hl * hl - t0y * hr * hr + (hr * hr - hl * hl) * s1y) / den;
            gz2 = -s * (s2z * hl * hl - t0z * hr * hr + (hr * hr - hl * hl) * s1z) / den;
        } else if (bn == 2 && Ny > 1) {
            float h = 1.5f * cy;
            gx2 = s * (t0x - s2x) / h;
            gy2 = s * (t0y - s2y) / h;
            gz2 = s * (t0z - s2z) / h;
        } else {
            gx2 = s * (t0x - s1x) / cy;
            gy2 = s * (t0y - s1y) / cy;
            gz2 = s * (t0z - s1z) / cy;
        }
        gyx = gx2;
        gyy = gy2;
        gyz = gz2;
    }

    // ---- z faces ----
    if (!PBCz && Nz > 1 && (iz == 0 || iz == Nz - 1)) {
        float s = (iz == 0) ? -1.0f : 1.0f;
        int Ii = (iz == 0) ? idx(ix, iy, 1) : idx(ix, iy, Nz - 2);
        // rows: z uses sig_zz (sd[2]); x uses sig_xz (so[1]); y uses sig_yz (so[0])
        float s1z = sdz[I], s2z = sdz[Ii];
        float s1x = soy[I], s2x = soy[Ii];
        float s1y = sox[I], s2y = sox[Ii];
        float gx2, gy2, gz2;
        if (bn >= 3 && Nz > 1) {
            float hl = 0.5f * cz, hr = cz;
            float den = hr * hl * hl + hl * hr * hr;
            gx2 = -s * (s2x * hl * hl - t0x * hr * hr + (hr * hr - hl * hl) * s1x) / den;
            gy2 = -s * (s2y * hl * hl - t0y * hr * hr + (hr * hr - hl * hl) * s1y) / den;
            gz2 = -s * (s2z * hl * hl - t0z * hr * hr + (hr * hr - hl * hl) * s1z) / den;
        } else if (bn == 2 && Nz > 1) {
            float h = 1.5f * cz;
            gx2 = s * (t0x - s2x) / h;
            gy2 = s * (t0y - s2y) / h;
            gz2 = s * (t0z - s2z) / h;
        } else {
            gx2 = s * (t0x - s1x) / cz;
            gy2 = s * (t0y - s1y) / cz;
            gz2 = s * (t0z - s1z) / cz;
        }
        gzx = gx2;
        gzy = gy2;
        gzz = gz2;
    }

    // sum all columns
    float felx = fx + gyx + gzx;
    float fely = fy + gyy + gzy;
    float felz = fz + gyz + gzz;

    if (outMode == 1) {
        dstx[I] = felx;
        dsty[I] = fely;
        dstz[I] = felz;
        return;
    }

    float eta = amul(eta_, eta_mul, I);
    float rho = amul(rho_, rho_mul, I);
    float mask = amul(mask_, mask_mul, I);
    float frozen = amul(frozen_, frozen_mul, I);

    float fac = ((mask != 0.0f) && (frozen == 0.0f) && (rho != 0.0f)) ? (1.0f / rho) : 0.0f;

    dstx[I] = (felx - eta * dux[I]) * fac;
    dsty[I] = (fely - eta * duy[I]) * fac;
    dstz[I] = (felz - eta * duz[I]) * fac;
}
