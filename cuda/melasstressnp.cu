#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Elastic stress sigma_el = C:(eps - eps_m) in Voigt notation from jump-aware displacement
// gradients and the magnetization (magnum.np solver.dpd: sig_ii/sig_ij minus sig_m).
//
// Outputs: sd = (sig_xx, sig_yy, sig_zz), so = (sig_yz, sig_xz, sig_xy) ("inverse" order like
// magnum.np's sig_ij).
// gu<c> = jump-aware gradient of u_<c> with components (d_x, d_y, d_z).
extern "C" __global__ void
MelasStressNP(float* __restrict__ sdx, float* __restrict__ sdy, float* __restrict__ sdz,
              float* __restrict__ sox, float* __restrict__ soy, float* __restrict__ soz,
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
              int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    // total strain in Voigt notation (engineering shear)
    float eps[6];
    eps[0] = guxx[I];
    eps[1] = guyy[I];
    eps[2] = guzz[I];
    eps[3] = guyz[I] + guzy[I]; // d_z u_y + d_y u_z
    eps[4] = guxz[I] + guzx[I]; // d_z u_x + d_x u_z
    eps[5] = guxy[I] + guyx[I]; // d_y u_x + d_x u_y

    // magnetostrictive strain
    float em[6];
    melas_eps_m(em, mx[I], my[I], mz[I],
                amul(l100_, l100_mul, I), amul(l111_, l111_mul, I));

    float el[6];
    for (int c = 0; c < 6; c++) {
        el[c] = eps[c] - em[c];
    }

    float sig[6];
    melas_sigma(sig, el,
                amul(C11_, C11_mul, I), amul(C12_, C12_mul, I), amul(C13_, C13_mul, I),
                amul(C22_, C22_mul, I), amul(C23_, C23_mul, I), amul(C33_, C33_mul, I),
                amul(C44_, C44_mul, I), amul(C55_, C55_mul, I), amul(C66_, C66_mul, I));

    sdx[I] = sig[0];
    sdy[I] = sig[1];
    sdz[I] = sig[2];
    sox[I] = sig[3]; // yz
    soy[I] = sig[4]; // xz
    soz[I] = sig[5]; // xy
}
