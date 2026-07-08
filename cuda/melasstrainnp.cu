#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Strain diagnostic in Voigt notation from jump-aware displacement gradients.
// mode = 0: total strain eps; mode = 1: elastic strain eps - eps_m; mode = 2: eps_m.
// Outputs: nrm = (eps_xx, eps_yy, eps_zz), shr = (eps_yz, eps_xz, eps_xy) in
// ENGINEERING shear convention (2*eps_ij), matching magnum.np.
extern "C" __global__ void
MelasStrainNP(float* __restrict__ nrmx, float* __restrict__ nrmy, float* __restrict__ nrmz,
              float* __restrict__ shrx, float* __restrict__ shry, float* __restrict__ shrz,
              float* __restrict__ guxx, float* __restrict__ guxy, float* __restrict__ guxz,
              float* __restrict__ guyx, float* __restrict__ guyy, float* __restrict__ guyz,
              float* __restrict__ guzx, float* __restrict__ guzy, float* __restrict__ guzz,
              float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
              float* __restrict__ l100_, float l100_mul,
              float* __restrict__ l111_, float l111_mul,
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

    float out[6];
    for (int c = 0; c < 6; c++) {
        if (mode == 0) {
            out[c] = eps[c];
        } else if (mode == 1) {
            out[c] = eps[c] - em[c];
        } else {
            out[c] = em[c];
        }
    }

    nrmx[I] = out[0];
    nrmy[I] = out[1];
    nrmz[I] = out[2];
    shrx[I] = out[3];
    shry[I] = out[4];
    shrz[I] = out[5];
}
