#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Jump-condition aware first derivative of a scalar field f along x, y and z
// (magnum.np: utils.py::first_derivative_with_jump_conditions(_and_pbc),
//  gradient_with_pbc incl. the optional second-order boundary stencil).
//
// Jump condition:  C * df/dq + B  continuous at cell interfaces.
// Uniform grid forward/backward differences (paper Eqs. 36/37):
//   g_fwd_i = (2 C_{i+1} (f_{i+1}-f_i) + dq (Bl_{i+1} - Br_i)) / (dq (C_i + C_{i+1}))
//   g_bkw_i = (2 C_{i-1} (f_i-f_{i-1}) + dq (Br_{i-1} - Bl_i)) / (dq (C_{i-1} + C_i))
// centered: g = (g_fwd + g_bkw)/2, one-sided at open boundaries.
// so != 0: overwrite window-boundary values with the 3-point one-sided stencil (Eqs. 39/40).
//
// The window [wx0,wx1)x[wy0,wy1)x[wz0,wz1) restricts the domain (used for the magnetic
// subdomain, magnum.np's slice_m). Outside the window the output is 0. Periodic wrap is only
// used along an axis if the window covers it fully and the corresponding PBC bit is set
// (magnum.np: slicing disables the pbc branch).
//
// B pointers may be NULL (treated as zero, e.g. for exchange gradients of m with C=Aex).
extern "C" __global__ void
MelasGradJumpNP(float* __restrict__ gx, float* __restrict__ gy, float* __restrict__ gz,
                float* __restrict__ f,
                float* __restrict__ Cx_, float Cx_mul,
                float* __restrict__ Cy_, float Cy_mul,
                float* __restrict__ Cz_, float Cz_mul,
                float* __restrict__ Blx, float* __restrict__ Brx,
                float* __restrict__ Bly, float* __restrict__ Bry,
                float* __restrict__ Blz, float* __restrict__ Brz,
                int wx0, int wx1, int wy0, int wy1, int wz0, int wz1,
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

    // outside the window: zero output
    if (ix < wx0 || ix >= wx1 || iy < wy0 || iy >= wy1 || iz < wz0 || iz >= wz1) {
        gx[I] = 0.0f;
        gy[I] = 0.0f;
        gz[I] = 0.0f;
        return;
    }

    float f0 = f[I];

    // ------------------- x direction -------------------
    {
        float g = 0.0f;
        int wn = wx1 - wx0;
        if (wn > 1) {
            float C0 = amul(Cx_, Cx_mul, I);
            float Blc = (Blx == NULL) ? 0.0f : Blx[I];
            float Brc = (Brx == NULL) ? 0.0f : Brx[I];
            if (wn == Nx && PBCx) {
                int Im = idx(MOD(ix - 1, Nx), iy, iz);
                int Ip = idx(MOD(ix + 1, Nx), iy, iz);
                float Cm = amul(Cx_, Cx_mul, Im);
                float Cp = amul(Cx_, Cx_mul, Ip);
                float Blp = (Blx == NULL) ? 0.0f : Blx[Ip];
                float Brm = (Brx == NULL) ? 0.0f : Brx[Im];
                float gf = melas_safediv(2.0f * Cp * (f[Ip] - f0) + cx * (Blp - Brc), cx * (C0 + Cp));
                float gb = melas_safediv(2.0f * Cm * (f0 - f[Im]) + cx * (Brm - Blc), cx * (Cm + C0));
                g = 0.5f * (gf + gb);
            } else {
                bool hasm = (ix > wx0);
                bool hasp = (ix < wx1 - 1);
                float gf = 0.0f, gb = 0.0f;
                if (hasp) {
                    int Ip = idx(ix + 1, iy, iz);
                    float Cp = amul(Cx_, Cx_mul, Ip);
                    float Blp = (Blx == NULL) ? 0.0f : Blx[Ip];
                    gf = melas_safediv(2.0f * Cp * (f[Ip] - f0) + cx * (Blp - Brc), cx * (C0 + Cp));
                }
                if (hasm) {
                    int Im = idx(ix - 1, iy, iz);
                    float Cm = amul(Cx_, Cx_mul, Im);
                    float Brm = (Brx == NULL) ? 0.0f : Brx[Im];
                    gb = melas_safediv(2.0f * Cm * (f0 - f[Im]) + cx * (Brm - Blc), cx * (Cm + C0));
                }
                if (hasp && hasm) {
                    g = 0.5f * (gf + gb);
                } else {
                    g = hasp ? gf : gb;
                }
                if (so && wn > 2) {
                    if (ix == wx0) {
                        float f1 = f[idx(ix + 1, iy, iz)];
                        float f2 = f[idx(ix + 2, iy, iz)];
                        g = (-f2 + 4.0f * f1 - 3.0f * f0) / (2.0f * cx);
                    } else if (ix == wx1 - 1) {
                        float f1 = f[idx(ix - 1, iy, iz)];
                        float f2 = f[idx(ix - 2, iy, iz)];
                        g = (3.0f * f0 - 4.0f * f1 + f2) / (2.0f * cx);
                    }
                }
            }
        }
        gx[I] = g;
    }

    // ------------------- y direction -------------------
    {
        float g = 0.0f;
        int wn = wy1 - wy0;
        if (wn > 1) {
            float C0 = amul(Cy_, Cy_mul, I);
            float Blc = (Bly == NULL) ? 0.0f : Bly[I];
            float Brc = (Bry == NULL) ? 0.0f : Bry[I];
        if (wn == Ny && PBCy) {
                int Im = idx(ix, MOD(iy - 1, Ny), iz);
                int Ip = idx(ix, MOD(iy + 1, Ny), iz);
                float Cm = amul(Cy_, Cy_mul, Im);
                float Cp = amul(Cy_, Cy_mul, Ip);
                float Blp = (Bly == NULL) ? 0.0f : Bly[Ip];
                float Brm = (Bry == NULL) ? 0.0f : Bry[Im];
                float gf = melas_safediv(2.0f * Cp * (f[Ip] - f0) + cy * (Blp - Brc), cy * (C0 + Cp));
                float gb = melas_safediv(2.0f * Cm * (f0 - f[Im]) + cy * (Brm - Blc), cy * (Cm + C0));
                g = 0.5f * (gf + gb);
            } else {
                bool hasm = (iy > wy0);
                bool hasp = (iy < wy1 - 1);
                float gf = 0.0f, gb = 0.0f;
                if (hasp) {
                    int Ip = idx(ix, iy + 1, iz);
                    float Cp = amul(Cy_, Cy_mul, Ip);
                    float Blp = (Bly == NULL) ? 0.0f : Bly[Ip];
                    gf = melas_safediv(2.0f * Cp * (f[Ip] - f0) + cy * (Blp - Brc), cy * (C0 + Cp));
                }
                if (hasm) {
                    int Im = idx(ix, iy - 1, iz);
                    float Cm = amul(Cy_, Cy_mul, Im);
                    float Brm = (Bry == NULL) ? 0.0f : Bry[Im];
                    gb = melas_safediv(2.0f * Cm * (f0 - f[Im]) + cy * (Brm - Blc), cy * (Cm + C0));
                }
                if (hasp && hasm) {
                    g = 0.5f * (gf + gb);
                } else {
                    g = hasp ? gf : gb;
                }
                if (so && wn > 2) {
                    if (iy == wy0) {
                        float f1 = f[idx(ix, iy + 1, iz)];
                        float f2 = f[idx(ix, iy + 2, iz)];
                        g = (-f2 + 4.0f * f1 - 3.0f * f0) / (2.0f * cy);
                    } else if (iy == wy1 - 1) {
                        float f1 = f[idx(ix, iy - 1, iz)];
                        float f2 = f[idx(ix, iy - 2, iz)];
                        g = (3.0f * f0 - 4.0f * f1 + f2) / (2.0f * cy);
                    }
                }
            }
        }
        gy[I] = g;
    }

    // ------------------- z direction -------------------
    {
        float g = 0.0f;
        int wn = wz1 - wz0;
        if (wn > 1) {
            float C0 = amul(Cz_, Cz_mul, I);
            float Blc = (Blz == NULL) ? 0.0f : Blz[I];
            float Brc = (Brz == NULL) ? 0.0f : Brz[I];
            if (wn == Nz && PBCz) {
                int Im = idx(ix, iy, MOD(iz - 1, Nz));
                int Ip = idx(ix, iy, MOD(iz + 1, Nz));
                float Cm = amul(Cz_, Cz_mul, Im);
                float Cp = amul(Cz_, Cz_mul, Ip);
                float Blp = (Blz == NULL) ? 0.0f : Blz[Ip];
                float Brm = (Brz == NULL) ? 0.0f : Brz[Im];
                float gf = melas_safediv(2.0f * Cp * (f[Ip] - f0) + cz * (Blp - Brc), cz * (C0 + Cp));
                float gb = melas_safediv(2.0f * Cm * (f0 - f[Im]) + cz * (Brm - Blc), cz * (Cm + C0));
                g = 0.5f * (gf + gb);
            } else {
                bool hasm = (iz > wz0);
                bool hasp = (iz < wz1 - 1);
                float gf = 0.0f, gb = 0.0f;
                if (hasp) {
                    int Ip = idx(ix, iy, iz + 1);
                    float Cp = amul(Cz_, Cz_mul, Ip);
                    float Blp = (Blz == NULL) ? 0.0f : Blz[Ip];
                    gf = melas_safediv(2.0f * Cp * (f[Ip] - f0) + cz * (Blp - Brc), cz * (C0 + Cp));
                }
                if (hasm) {
                    int Im = idx(ix, iy, iz - 1);
                    float Cm = amul(Cz_, Cz_mul, Im);
                    float Brm = (Brz == NULL) ? 0.0f : Brz[Im];
                    gb = melas_safediv(2.0f * Cm * (f0 - f[Im]) + cz * (Brm - Blc), cz * (Cm + C0));
                }
                if (hasp && hasm) {
                    g = 0.5f * (gf + gb);
                } else {
                    g = hasp ? gf : gb;
                }
                if (so && wn > 2) {
                    if (iz == wz0) {
                        float f1 = f[idx(ix, iy, iz + 1)];
                        float f2 = f[idx(ix, iy, iz + 2)];
                        g = (-f2 + 4.0f * f1 - 3.0f * f0) / (2.0f * cz);
                    } else if (iz == wz1 - 1) {
                        float f1 = f[idx(ix, iy, iz - 1)];
                        float f2 = f[idx(ix, iy, iz - 2)];
                        g = (3.0f * f0 - 4.0f * f1 + f2) / (2.0f * cz);
                    }
                }
            }
        }
        gz[I] = g;
    }
}
