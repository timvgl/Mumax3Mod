#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Gilbert dissipation density diagnostic (paper Eqs. 60/67):
//   dE/dt = -mu0 Ms alpha gamma/(1+alpha^2) |m x H|^2
// With B = mu0 H [T] this kernel stores the positive factor
//   dst = Ms alpha/(1+alpha^2) |m x B|^2
// The Go side multiplies by gamma/mu0 and the cell volume and applies the minus sign.
extern "C" __global__ void
MelasGilbertNP(float* __restrict__ dst,
               float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
               float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
               float* __restrict__ alpha_, float alpha_mul,
               float* __restrict__ Ms_, float Ms_mul,
               int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    float alpha = amul(alpha_, alpha_mul, I);
    float Ms = amul(Ms_, Ms_mul, I);

    float m0x = mx[I], m0y = my[I], m0z = mz[I];
    float b0x = bx[I], b0y = by[I], b0z = bz[I];

    float cxp = m0y * b0z - m0z * b0y;
    float cyp = m0z * b0x - m0x * b0z;
    float czp = m0x * b0y - m0y * b0x;

    dst[I] = Ms * alpha / (1.0f + alpha * alpha) * (cxp * cxp + cyp * cyp + czp * czp);
}
