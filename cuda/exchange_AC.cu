#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// Aharonov-Casher (AC) modified exchange stencil.
//
// Ordinary MuMax exchange is the finite-difference Laplacian
//     B_i = (1/Ms) sum_j w_ij A_ij (m_j - m_i)
// which discretizes B_ex = (2A/Ms) laplacian(m).
//
// The AC effect shifts the spin-wave dispersion by a minimal coupling of the
// *magnon field* to an effective (wavevector-valued) gauge field
//     k_AC = (gamma/c^2) * (q x E),          [ acMul = gamma/c^2 ]
// giving omega = omega0 + D |k - k_AC|^2. In real space this is the
// gauge-covariant Laplacian (D = grad - i k_AC):
//     (D^2 psi)_i = sum_j w_ij ( exp(-i k_AC . delta_ij) psi_j - psi_i ),
//     delta_ij = r_j - r_i.
//
// The magnon field psi = m_perp,1 + i m_perp,2 is the transverse magnetization
// in the frame (e1, e2, q). Multiplying psi by a U(1) phase exp(i theta) is a
// right-hand rotation of the real vector m about the quantization axis q by
// theta (the q-component is left invariant). Hence the neighbour hopping picks
// up a rotation about q:
//     m_j -> R(q, theta_ij) m_j,   theta_ij = - k_AC . delta_ij.
// Only the neighbour is rotated; the central -m_i term stays bare (that is what
// makes this the covariant Laplacian). See ac_exchange_corrected.md.
//
// Notes on correctness:
//   * eta = 1 (linear magnon moment mu = -gamma*hbar*q). A projection factor
//     (m.q) would scale k_AC with the magnon *density*, which is not the linear
//     AC effect, so it is not used. The sign of the moment in a reversed domain
//     is carried by the q field itself.
//   * q and E are averaged on the bond so that theta_ji = -theta_ij exactly
//     (Hermitian / energy conserving) even for a non-uniform axis.
//   * In a collinear state (m = q) rotation about q fixes q, so R(q,.)m_j - m_i
//     = 0: the AC term adds no spurious static field, only a spin-wave shift.
//   * If E = 0 (or q = 0) this reduces to the ordinary exchange stencil, up to
//     cosf(0)/sinf(0) round-off.

inline __device__ float3 rotate_axis(float3 v, float3 q, float theta) {
    // Rodrigues rotation of v about unit axis q by angle theta (right-hand rule).
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

// Rotated neighbour magnetization implementing the AC covariant hopping.
// qb: bond quantization axis (unit), Eij: bond E, delta: r_j - r_i.
inline __device__ float3 ac_rotate(float3 mj, float3 qb, float3 Eij,
                                   float3 delta, float acMul) {
    // k_AC = acMul * (q x E), with acMul = gamma/c^2 (linear moment, eta = 1).
    float3 kAC = (acMul) * cross(qb, Eij);
    // Covariant Laplacian: neighbour carries phase exp(-i k_AC . delta),
    // realized as a rotation of the transverse (magnon) part of m about q.
    float theta = -dot(kAC, delta);
    return rotate_axis(mj, qb, theta);
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

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    // central quantization axis (unit). If unset (q == 0) the AC part vanishes
    // per bond and this reduces to ordinary exchange.
    float3 q0 = normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, I));

    uint8_t r0 = regions[I];
    float3 B = make_float3(0.0f, 0.0f, 0.0f);

    int i_;         // neighbor index
    float3 m_;      // neighbor mag
    float3 qb;      // bond quantization axis
    float3 E_;      // bond E
    float3 delta;   // r_j - r_i
    float3 mrot;    // rotated neighbor
    float a__;      // inter-cell exchange stiffness
    bool present;   // real (non-empty, non-self-clamped) neighbor

    // left neighbor
    i_      = idx(lclampx(ix-1), iy, iz);            // clamps or wraps per PBC
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    qb      = normalized(q0 + normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, i_)));
    E_      = bond_E(Ex, Ey, Ez, I, i_);
    delta   = make_float3(-dx, 0.0f, 0.0f);
    m_      = (is0(m_) ? m0 : m_);                   // Neumann fill for empty neighbor
    mrot    = (present && !is0(qb)) ? ac_rotate(m_, qb, E_, delta, acMul) : m_;
    a__     = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ * (mrot - m0);

    // right neighbor
    i_      = idx(hclampx(ix+1), iy, iz);
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    qb      = normalized(q0 + normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, i_)));
    E_      = bond_E(Ex, Ey, Ez, I, i_);
    delta   = make_float3(dx, 0.0f, 0.0f);
    m_      = (is0(m_) ? m0 : m_);
    mrot    = (present && !is0(qb)) ? ac_rotate(m_, qb, E_, delta, acMul) : m_;
    a__     = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ * (mrot - m0);

    // back neighbor
    i_      = idx(ix, lclampy(iy-1), iz);
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    qb      = normalized(q0 + normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, i_)));
    E_      = bond_E(Ex, Ey, Ez, I, i_);
    delta   = make_float3(0.0f, -dy, 0.0f);
    m_      = (is0(m_) ? m0 : m_);
    mrot    = (present && !is0(qb)) ? ac_rotate(m_, qb, E_, delta, acMul) : m_;
    a__     = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ * (mrot - m0);

    // front neighbor
    i_      = idx(ix, hclampy(iy+1), iz);
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    qb      = normalized(q0 + normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, i_)));
    E_      = bond_E(Ex, Ey, Ez, I, i_);
    delta   = make_float3(0.0f, dy, 0.0f);
    m_      = (is0(m_) ? m0 : m_);
    mrot    = (present && !is0(qb)) ? ac_rotate(m_, qb, E_, delta, acMul) : m_;
    a__     = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ * (mrot - m0);

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_      = idx(ix, iy, lclampz(iz-1));
        m_      = make_float3(mx[i_], my[i_], mz[i_]);
        present = !(is0(m_) || i_ == I);
        qb      = normalized(q0 + normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, i_)));
        E_      = bond_E(Ex, Ey, Ez, I, i_);
        delta   = make_float3(0.0f, 0.0f, -dz);
        m_      = (is0(m_) ? m0 : m_);
        mrot    = (present && !is0(qb)) ? ac_rotate(m_, qb, E_, delta, acMul) : m_;
        a__     = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ * (mrot - m0);

        // top neighbor
        i_      = idx(ix, iy, hclampz(iz+1));
        m_      = make_float3(mx[i_], my[i_], mz[i_]);
        present = !(is0(m_) || i_ == I);
        qb      = normalized(q0 + normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, i_)));
        E_      = bond_E(Ex, Ey, Ez, I, i_);
        delta   = make_float3(0.0f, 0.0f, dz);
        m_      = (is0(m_) ? m0 : m_);
        mrot    = (present && !is0(qb)) ? ac_rotate(m_, qb, E_, delta, acMul) : m_;
        a__     = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ * (mrot - m0);
    }

    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Bx[I] += B.x * invMs;
    By[I] += B.y * invMs;
    Bz[I] += B.z * invMs;
}
