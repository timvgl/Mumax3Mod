#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Thermodynamically consistent (nonlinear) magnetoelastic effective field
// (magnum.np field_terms/magnetoelastic.py::MagnetoElasticField.h):
//
//   H_i = -1/(mu0 Ms) (sigma - sigma_m) : d eps_m / d m_i
//   H_x = -1/(mu0 Ms) [ sig_el[0] (-3 l100 m_x) + sig_el[5] (-3 l111 m_y) + sig_el[4] (-3 l111 m_z) ]
//   H_y = -1/(mu0 Ms) [ sig_el[1] (-3 l100 m_y) + sig_el[5] (-3 l111 m_x) + sig_el[3] (-3 l111 m_z) ]
//   H_z = -1/(mu0 Ms) [ sig_el[2] (-3 l100 m_z) + sig_el[4] (-3 l111 m_x) + sig_el[3] (-3 l111 m_y) ]
//
// mumax stores the effective field as B = mu0 H [T], so the mu0 cancels:
//   B_i = 3/Ms (l100 sig_el[ii] m_i + l111 (...)).
// sd = (sig_xx, sig_yy, sig_zz), so = (sig_yz, sig_xz, sig_xy) of sigma_el = sigma - sigma_m.
// Cells with Ms = 0 give zero field (nan_to_num equivalent). The field is ADDED to dst.
extern "C" __global__ void
MelasHFieldNP(float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
              float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
              float* __restrict__ sdx, float* __restrict__ sdy, float* __restrict__ sdz,
              float* __restrict__ sox, float* __restrict__ soy, float* __restrict__ soz,
              float* __restrict__ l100_, float l100_mul,
              float* __restrict__ l111_, float l111_mul,
              float* __restrict__ Ms_, float Ms_mul,
              int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    float invMs = inv_Msat(Ms_, Ms_mul, I);
    float l100 = amul(l100_, l100_mul, I);
    float l111 = amul(l111_, l111_mul, I);

    float m0x = mx[I], m0y = my[I], m0z = mz[I];
    float syz = sox[I], sxz = soy[I], sxy = soz[I];

    dstx[I] += 3.0f * invMs * (l100 * sdx[I] * m0x + l111 * (sxy * m0y + sxz * m0z));
    dsty[I] += 3.0f * invMs * (l100 * sdy[I] * m0y + l111 * (sxy * m0x + syz * m0z));
    dstz[I] += 3.0f * invMs * (l100 * sdz[I] * m0z + l111 * (sxz * m0x + syz * m0y));
}
