#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Magnetic-stress part of the B jump parameters (magnum.np strain.py::_get_sigM_jump_conditions).
//
// Interface magnetization values from exchange-jump-aware gradients (paper Eqs. 44-47):
//   m^-_i = m_i - dq/2 * Dm_i ,  m^+_i = m_i + dq/2 * Dm_i
// From m^+/m^-: eps_m -> sig_m = C:eps_m, and per derivative direction q the B entries are
// minus the sig_m Voigt components:
//   q=x: (Bx,By,Bz) = -(sig_m[0], sig_m[5], sig_m[4])
//   q=y: (Bx,By,Bz) = -(sig_m[5], sig_m[1], sig_m[3])
//   q=z: (Bx,By,Bz) = -(sig_m[4], sig_m[3], sig_m[2])
// Outputs are 3-component per direction: Bl<q>/Br<q> with component index = displacement comp.
//
// gm<c> = exchange-jump-aware gradient of m_<c>, components (d_x, d_y, d_z).
// Limitation (paper Sec. II.B): interface m values derive from the exchange boundary integrals
// only; additional boundary field terms (e.g. DMI) are not accounted for.
extern "C" __global__ void
MelasBSigMNP(float* __restrict__ Blxx, float* __restrict__ Blxy, float* __restrict__ Blxz,
             float* __restrict__ Brxx, float* __restrict__ Brxy, float* __restrict__ Brxz,
             float* __restrict__ Blyx, float* __restrict__ Blyy, float* __restrict__ Blyz,
             float* __restrict__ Bryx, float* __restrict__ Bryy, float* __restrict__ Bryz,
             float* __restrict__ Blzx, float* __restrict__ Blzy, float* __restrict__ Blzz,
             float* __restrict__ Brzx, float* __restrict__ Brzy, float* __restrict__ Brzz,
             float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
             float* __restrict__ gmxx, float* __restrict__ gmxy, float* __restrict__ gmxz,
             float* __restrict__ gmyx, float* __restrict__ gmyy, float* __restrict__ gmyz,
             float* __restrict__ gmzx, float* __restrict__ gmzy, float* __restrict__ gmzz,
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
             float cx, float cy, float cz) {

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

    float em[6], sg[6];

    // ---- x direction: dmdx = (d_x m_x, d_x m_y, d_x m_z) ----
    {
        float hx = 0.5f * cx;
        float dx_mx = gmxx[I], dx_my = gmyx[I], dx_mz = gmzx[I];

        // left interface value m^-
        melas_eps_m(em, m0x - hx * dx_mx, m0y - hx * dx_my, m0z - hx * dx_mz, l100, l111);
        melas_sigma(sg, em, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        Blxx[I] = -sg[0];
        Blxy[I] = -sg[5];
        Blxz[I] = -sg[4];

        // right interface value m^+
        melas_eps_m(em, m0x + hx * dx_mx, m0y + hx * dx_my, m0z + hx * dx_mz, l100, l111);
        melas_sigma(sg, em, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        Brxx[I] = -sg[0];
        Brxy[I] = -sg[5];
        Brxz[I] = -sg[4];
    }

    // ---- y direction: dmdy = (d_y m_x, d_y m_y, d_y m_z) ----
    {
        float hy = 0.5f * cy;
        float dy_mx = gmxy[I], dy_my = gmyy[I], dy_mz = gmzy[I];

        melas_eps_m(em, m0x - hy * dy_mx, m0y - hy * dy_my, m0z - hy * dy_mz, l100, l111);
        melas_sigma(sg, em, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        Blyx[I] = -sg[5];
        Blyy[I] = -sg[1];
        Blyz[I] = -sg[3];

        melas_eps_m(em, m0x + hy * dy_mx, m0y + hy * dy_my, m0z + hy * dy_mz, l100, l111);
        melas_sigma(sg, em, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        Bryx[I] = -sg[5];
        Bryy[I] = -sg[1];
        Bryz[I] = -sg[3];
    }

    // ---- z direction: dmdz = (d_z m_x, d_z m_y, d_z m_z) ----
    {
        float hz = 0.5f * cz;
        float dz_mx = gmxz[I], dz_my = gmyz[I], dz_mz = gmzz[I];

        melas_eps_m(em, m0x - hz * dz_mx, m0y - hz * dz_my, m0z - hz * dz_mz, l100, l111);
        melas_sigma(sg, em, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        Blzx[I] = -sg[4];
        Blzy[I] = -sg[3];
        Blzz[I] = -sg[2];

        melas_eps_m(em, m0x + hz * dz_mx, m0y + hz * dz_my, m0z + hz * dz_mz, l100, l111);
        melas_sigma(sg, em, c11, c12, c13, c22, c23, c33, c44, c55, c66);
        Brzx[I] = -sg[4];
        Brzy[I] = -sg[3];
        Brzz[I] = -sg[2];
    }
}
