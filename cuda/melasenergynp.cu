#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Energy densities of the fully coupled magnetoelastic solver (per cell, J/m3).
// mode = 0: elastic potential energy density   1/2 (eps-eps_m):C:(eps-eps_m)   (magnum.np U_el)
// mode = 1: magnetoelastic energy density      (1/2 eps_m - eps):C:eps_m      (MagnetoElasticField.E)
// mode = 2: total strain energy density        1/2 eps:C:eps                  (magnum.np U)
// The Voigt dot product with engineering shear strains equals the tensor Frobenius product.
// gu<c> = jump-aware gradient of u_<c> with components (d_x, d_y, d_z).
extern "C" __global__ void
MelasEnergyNP(float* __restrict__ dst,
              float* __restrict__ guxx, float* __restrict__ guxy, float* __restrict__ guxz,
              float* __restrict__ guyx, float* __restrict__ guyy, float* __restrict__ guyz,
              float* __restrict__ guzx, float* __restrict__ guzy, float* __restrict__ guzz,
              float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
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
              int mode,
              int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    float eps[6];
    eps[0] = guxx[I];
    eps[1] = guyy[I];
    eps[2] = guzz[I];
    eps[3] = guyz[I] + guzy[I];
    eps[4] = guxz[I] + guzx[I];
    eps[5] = guxy[I] + guyx[I];

    float em[6];
    melas_eps_m(em, mx[I], my[I], mz[I],
                amul(l100_, l100_mul, I), amul(l111_, l111_mul, I));

    float c11 = amul(C11_, C11_mul, I), c12 = amul(C12_, C12_mul, I), c13 = amul(C13_, C13_mul, I);
    float c22 = amul(C22_, C22_mul, I), c23 = amul(C23_, C23_mul, I), c33 = amul(C33_, C33_mul, I);
    float c44 = amul(C44_, C44_mul, I), c55 = amul(C55_, C55_mul, I), c66 = amul(C66_, C66_mul, I);

    float e = 0.0f;
    float sig[6];

    if (mode == 0) {
        float el[6];
        for (int c = 0; c < 6; c++) {
            el[c] = eps[c] - em[c];
        }
        melas_sigma(sig, el, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        for (int c = 0; c < 6; c++) {
            e += 0.5f * el[c] * sig[c];
        }
    } else if (mode == 1) {
        melas_sigma(sig, em, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        for (int c = 0; c < 6; c++) {
            e += (0.5f * em[c] - eps[c]) * sig[c];
        }
    } else {
        melas_sigma(sig, eps, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        for (int c = 0; c < 6; c++) {
            e += 0.5f * eps[c] * sig[c];
        }
    }

    dst[I] = e;
}
