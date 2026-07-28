#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// AC-modified exchange stencil.
//
// Ordinary MuMax exchange is the finite-difference Laplacian
//     sum_j w_ij A_ij (m_j - m_i)
// followed by multiplication with 1/Ms.
//
// The Aharonov-Casher contribution is included as a Peierls-like
// phase on each neighbor contribution. Since MuMax stores a real
// vector magnetization, the complex phase exp(i theta_ij) is represented
// as a rotation of the neighbor magnetization around the quantization
// axis q.
//
// Convention used here:
//     k_AC,ij = acMul * eta_ij * (q x E_ij)
//     theta_ij = k_AC,ij . (r_j - r_i)
//
// with
//     acMul = gamma / c^2
//
// and
//     eta_ij = 0.5 * (dot(m_i,q) + dot(m_j,q)).
//
// eta_ij implements the projected magnon magnetic moment
//     mu_magnon(m,q) = -gamma*hbar*dot(m,q)*q.
//
// If E = 0, theta_ij = 0 and this kernel reproduces the old exchange
// stencil exactly, except for possible round-off from cosf(0), sinf(0).

inline __device__ float3 rotate_axis(float3 v, float3 q, float theta) {
    float c = cosf(theta);
    float s = sinf(theta);
    return v*c + cross(q, v)*s + q*dot(q, v)*(1.0f - c);
}

inline __device__ float3 bond_E(float* __restrict__ Ex,
                                float* __restrict__ Ey,
                                float* __restrict__ Ez,
                                int I, int J) {
    return make_float3(
        0.5f * (Ex[I] + Ex[J]),
        0.5f * (Ey[I] + Ey[J]),
        0.5f * (Ez[I] + Ez[J])
    );
}

inline __device__ float3 ac_neighbor(float3 mi,
                                     float3 mj,
                                     float3 q,
                                     float3 Eij,
                                     float3 delta,
                                     float acMul,
                                     bool do_rotate) {
    if (!do_rotate) {
        return mj;
    }

    // Projection factor for the nonlinear/local magnon moment:
    // mu_magnon = -gamma*hbar*(m.q)*q.
    // Use the average projection on the bond.
    float eta = 0.5f * (dot(mi, q) + dot(mj, q));

    // k_AC = (gamma/c^2) * eta * (q x E)
    float3 kAC = (acMul * eta) * cross(q, Eij);
    float theta = dot(kAC, delta);

    return rotate_axis(mj, q, theta);
}

extern "C" __global__ void
addexchange_ac(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
               float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
               float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
               float* __restrict__ Ms_, float Ms_mul,
               float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
               float wx, float wy, float wz,
               float dx, float dy, float dz,
               int Nx, int Ny, int Nz,
               float* __restrict__ qx_, float qx_mul,
               float* __restrict__ qy_, float qy_mul,
               float* __restrict__ qz_, float qz_mul,
               float acMul,
               uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    float3 q = normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, I));

    // If q is zero, the AC part vanishes and this reduces to exchange.
    // This also avoids undefined rotations.
    bool q_ok = !is0(q);

    uint8_t r0 = regions[I];
    float3 B = make_float3(0.0f, 0.0f, 0.0f);

    int i_;
    float3 m_;
    float3 E_;
    float3 delta;
    float3 mrot;
    float a__;
    bool do_rotate;

    // left neighbor
    i_ = idx(lclampx(ix-1), iy, iz);
    m_ = make_float3(mx[i_], my[i_], mz[i_]);
    do_rotate = q_ok && !(is0(m_) || i_ == I);
    m_ = (is0(m_) ? m0 : m_);
    E_ = bond_E(Ex, Ey, Ez, I, i_);
    delta = make_float3(-dx, 0.0f, 0.0f);
    mrot = ac_neighbor(m0, m_, q, E_, delta, acMul, do_rotate);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ * (mrot - m0);

    // right neighbor
    i_ = idx(hclampx(ix+1), iy, iz);
    m_ = make_float3(mx[i_], my[i_], mz[i_]);
    do_rotate = q_ok && !(is0(m_) || i_ == I);
    m_ = (is0(m_) ? m0 : m_);
    E_ = bond_E(Ex, Ey, Ez, I, i_);
    delta = make_float3(dx, 0.0f, 0.0f);
    mrot = ac_neighbor(m0, m_, q, E_, delta, acMul, do_rotate);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ * (mrot - m0);

    // back neighbor
    i_ = idx(ix, lclampy(iy-1), iz);
    m_ = make_float3(mx[i_], my[i_], mz[i_]);
    do_rotate = q_ok && !(is0(m_) || i_ == I);
    m_ = (is0(m_) ? m0 : m_);
    E_ = bond_E(Ex, Ey, Ez, I, i_);
    delta = make_float3(0.0f, -dy, 0.0f);
    mrot = ac_neighbor(m0, m_, q, E_, delta, acMul, do_rotate);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ * (mrot - m0);

    // front neighbor
    i_ = idx(ix, hclampy(iy+1), iz);
    m_ = make_float3(mx[i_], my[i_], mz[i_]);
    do_rotate = q_ok && !(is0(m_) || i_ == I);
    m_ = (is0(m_) ? m0 : m_);
    E_ = bond_E(Ex, Ey, Ez, I, i_);
    delta = make_float3(0.0f, dy, 0.0f);
    mrot = ac_neighbor(m0, m_, q, E_, delta, acMul, do_rotate);
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ * (mrot - m0);

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_ = idx(ix, iy, lclampz(iz-1));
        m_ = make_float3(mx[i_], my[i_], mz[i_]);
        do_rotate = q_ok && !(is0(m_) || i_ == I);
        m_ = (is0(m_) ? m0 : m_);
        E_ = bond_E(Ex, Ey, Ez, I, i_);
        delta = make_float3(0.0f, 0.0f, -dz);
        mrot = ac_neighbor(m0, m_, q, E_, delta, acMul, do_rotate);
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ * (mrot - m0);

        // top neighbor
        i_ = idx(ix, iy, hclampz(iz+1));
        m_ = make_float3(mx[i_], my[i_], mz[i_]);
        do_rotate = q_ok && !(is0(m_) || i_ == I);
        m_ = (is0(m_) ? m0 : m_);
        E_ = bond_E(Ex, Ey, Ez, I, i_);
        delta = make_float3(0.0f, 0.0f, dz);
        mrot = ac_neighbor(m0, m_, q, E_, delta, acMul, do_rotate);
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ * (mrot - m0);
    }

    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Bx[I] += B.x * invMs;
    By[I] += B.y * invMs;
    Bz[I] += B.z * invMs;
}
