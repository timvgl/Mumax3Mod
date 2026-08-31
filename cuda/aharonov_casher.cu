#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// =====================================================================================
//  Aharonov-Casher (AC) effect as a GENERAL micromagnetic effective-field contribution.
//
//  Derivation : ac_gauge_v5_general_effective_field_DERIVATION.md
//  Numerics   : ac_gauge_v5_general_effective_field_NUMERICS.md
//  Reference  : V. Basso, P. Ansalone, A. Di Pietro, Physica B 671 (2023) 415422.
//
//  PHYSICS ------------------------------------------------------------------------
//  A magnetic moment transported with velocity v through an electric field E sees the
//  motional magnetic field B' = -(v x E)/c^2 and therefore precesses at a rate that is
//  independent of |v| per unit length travelled. That fixed rotation-per-length is an
//  SO(3) (spin-rotation) gauge connection, one spin-space vector per lattice direction:
//
//      A_alpha = xi * (gamma/c^2) * (e_alpha x E)                            [1/m]
//
//  with xi a dimensionless screening/enhancement factor (Basso's alpha ~ 1.2e-2 for the
//  AC effect in YIG; beta ~ 1e4 for the electric-field-induced DMI - the SAME operator).
//  Minimal coupling promotes every gradient of m to the covariant derivative
//
//      D_alpha m = d_alpha m + A_alpha x m
//
//  so the exchange energy becomes A_ex |D m|^2. Subtracting ordinary exchange leaves the
//  AC energy density
//
//      u_AC = 2 A_ex sum_a A_a . (m x d_a m)      <- DMI-type, linear in E
//           +   A_ex sum_a |A_a x m|^2            <- hard axis along E, quadratic in E
//
//  whose variational derivative is the effective field computed here:
//
//      B_AC = (2 A_ex / Ms) * ( D^2 - laplacian ) m
//
//  Linearising about any local axis q recovers the magnon picture automatically, with
//  k_AC = xi*(gamma/c^2)*(q x E) and the exchange parabola shifted to k = k_AC.
//
//  SCOPE: this is the complete *local* content of the AC effect. It shifts the exchange
//  sector of the dispersion exactly; the dipolar (magnetostatic) sector is NOT shifted,
//  because the magnetostatic magnon Hamiltonian is not U(1)_q invariant and admits no
//  local gauge representation (see DERIVATION section 7). For dipolar-dominated
//  Damon-Eshbach waves the observable is therefore a nonreciprocal TILT, not a rigid
//  shift of the minimum - this is physics, not a missing term.
//
//  DISCRETISATION ------------------------------------------------------------------
//  Lattice-gauge (Peierls) form: the neighbour is parallel-transported back to the
//  central cell with the SO(3) holonomy of the bond,
//
//      U_ij = R( v_ij ),   v_ij = sum_a A_a (delta_ij)_a = delta_ij x G,
//      G    = xi * (gamma/c^2) * E ,   delta_ij = r_j - r_i
//
//  and the AC field is the covariant hopping MINUS the plain hopping:
//
//      B_i = (1/Ms_i) sum_j w_ij A_ij ( U_ij - 1 ) m_j
//
//  This is exactly -1/(Ms V) dE/dm_i of the discrete energy
//      eps_i = -(1/2) sum_j w_ij A_ij  m_i . (U_ij - 1) m_j = -(1/2) M_i . B_i,
//  so mumax's standard "-0.5 * M . B" energy-density adder is exact, not approximate.
//  U_ji = U_ij^T on every bond (v_ji = -v_ij), so the interaction is Hermitian and
//  energy conserving even for spatially varying E and xi.
//
//  NUMERICAL STABILITY: |v| is typically 1e-8 rad or smaller, so forming R(v)m and then
//  subtracting m would annihilate every significant bit in float32. (U-1)m is therefore
//  evaluated in the cancellation-free form
//      (R(v)-1) m = sinc(t) (v x m) + vers(t) [ v (v.m) - |v|^2 m ],   t = |v|
//  with sinc(t) = sin t / t and vers(t) = (1-cos t)/t^2, both O(1) and analytic at t=0.
// =====================================================================================

// sin(t)/t and (1-cos(t))/t^2 given t^2, evaluated without cancellation.
// Series switch at t = 0.1 rad: there the 3-term series is accurate to ~1e-10, while the
// closed form of (1-cos t)/t^2 has already lost ~4 digits in float32.
inline __device__ void ac_sincvers(float t2, float *sinc, float *vers) {
    if (t2 < 1e-2f) {
        *sinc = 1.0f - t2 * (1.0f / 6.0f) + t2 * t2 * (1.0f / 120.0f);
        *vers = 0.5f - t2 * (1.0f / 24.0f) + t2 * t2 * (1.0f / 720.0f);
    } else {
        float t = sqrtf(t2);
        *sinc = sinf(t) / t;
        *vers = (1.0f - cosf(t)) / t2;
    }
}

// ( R(v) - 1 ) m : Rodrigues rotation about v by |v|, minus the identity.
// Exact to machine precision for arbitrarily small |v| (no catastrophic cancellation).
inline __device__ float3 ac_rot_minus_id(float3 v, float3 m) {
    float t2 = dot(v, v);
    if (t2 == 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float sinc, vers;
    ac_sincvers(t2, &sinc, &vers);
    return sinc * cross(v, m) + vers * (dot(v, m) * v - t2 * m);
}

// Contribution of one bond i->j to the raw (pre-1/Ms) AC field of cell i.
//
//   present : j is a real neighbour (inside the geometry and not the clamped self)
//   m0, mj  : central and neighbour magnetisation
//   delta   : r_j - r_i
//
// Absent neighbours use the covariant Neumann boundary condition D_n m = 0, i.e. the
// ghost spin is the parallel transport of the central one, m_ghost = R(-v) m0, giving
// (U-1) m_ghost = m0 - R(-v) m0. This is the condition that makes the surface term of
// the energy variation vanish - the gauge-covariant analogue of mumax's free-boundary
// exchange. With openBC the bond is simply dropped instead.
inline __device__ float3 ac_bond(bool present, float3 m0, float3 mj, float3 delta,
                                 int I, int i_,
                                 float *__restrict__ Ex, float *__restrict__ Ey, float *__restrict__ Ez,
                                 float *__restrict__ xi_, float xi_mul,
                                 float *__restrict__ qx_, float qx_mul,
                                 float *__restrict__ qy_, float qy_mul,
                                 float *__restrict__ qz_, float qz_mul,
                                 float acConst, uint8_t projectAxis, uint8_t openBC) {

    float3 zero = make_float3(0.0f, 0.0f, 0.0f);

    if (!present && openBC) {
        return zero;
    }

    // Bond-symmetric gauge seed G = xi * (gamma/c^2) * E.
    // Symmetric averaging guarantees v_ji = -v_ij exactly, hence U_ji = U_ij^T (Hermitian).
    // For an absent neighbour the central-cell values are used (the neighbour cell carries
    // no material and its E/xi are not meaningful for the boundary bond).
    float3 Eb;
    float xib;
    if (present) {
        Eb = 0.5f * (make_float3(Ex[I], Ey[I], Ez[I]) + make_float3(Ex[i_], Ey[i_], Ez[i_]));
        xib = 0.5f * (amul(xi_, xi_mul, I) + amul(xi_, xi_mul, i_));
    } else {
        Eb = make_float3(Ex[I], Ey[I], Ez[I]);
        xib = amul(xi_, xi_mul, I);
    }

    // Bond rotation vector.
    //   v = sum_a A_a delta_a = xi*(gamma/c^2) * sum_a (e_a x E) delta_a
    //     = xi*(gamma/c^2) * (delta x E) = delta x G.
    // Note the order: cross(G, delta) would implement the opposite connection and
    // flip the sign of the whole linear (DMI-type) term.
    float3 v = cross(delta, (xib * acConst) * Eb);

    // Optional U(1) reduction: keep only the rotation about the magnon quantization axis.
    // Then v.q = k_AC . delta with k_AC = xi*(gamma/c^2)*(q x E) - the linear spin-wave
    // theory of ac_exchange_v2_corrected_note.md, reproduced exactly.
    if (projectAxis) {
        float3 qi = normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, I));
        float3 qb = present
                        ? normalized(qi + normalized(vmul(qx_, qy_, qz_, qx_mul, qy_mul, qz_mul, i_)))
                        : qi;
        if (is0(qb)) {
            return zero;
        }
        v = dot(v, qb) * qb;
    }

    if (present) {
        return ac_rot_minus_id(v, mj);          // (U - 1) m_j
    }
    return -ac_rot_minus_id(-v, m0);            // m0 - R(-v) m0
}

// Adds the Aharonov-Casher effective field (Tesla) to B.
//
//   B{x,y,z}   : effective field, added to (not overwritten)
//   m{x,y,z}   : normalised magnetisation
//   E{x,y,z}   : electric field, V/m
//   xi_, xi_mul: dimensionless AC coupling (screening/enhancement), region-wise
//   q{x,y,z}_  : magnon quantization axis, only read when projectAxis != 0
//   Ms_, Ms_mul: saturation magnetisation
//   aLUT2d     : inter-region exchange stiffness A_ij (same table as plain exchange)
//   w{x,y,z}   : 2/d^2 Laplacian weights
//   d{x,y,z}   : cell sizes; they set the bond length entering the AC phase
//   acConst    : gamma/c^2 = 1.9577e-6 m^-1 per V/m
//   projectAxis: 0 = full SO(3) connection (general); 1 = U(1) projection onto q
//   openBC     : 1 = drop boundary bonds; 0 = covariant Neumann D_n m = 0
extern "C" __global__ void
addaharonovcasher(float *__restrict__ Bx, float *__restrict__ By, float *__restrict__ Bz,
                  float *__restrict__ mx, float *__restrict__ my, float *__restrict__ mz,
                  float *__restrict__ Ex, float *__restrict__ Ey, float *__restrict__ Ez,
                  float *__restrict__ xi_, float xi_mul,
                  float *__restrict__ qx_, float qx_mul,
                  float *__restrict__ qy_, float qy_mul,
                  float *__restrict__ qz_, float qz_mul,
                  float *__restrict__ Ms_, float Ms_mul,
                  float *__restrict__ aLUT2d, uint8_t *__restrict__ regions,
                  float wx, float wy, float wz,
                  float dx, float dy, float dz,
                  int Nx, int Ny, int Nz,
                  float acConst, uint8_t projectAxis, uint8_t openBC, uint8_t PBC) {

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

    uint8_t r0 = regions[I];
    float3 B = make_float3(0.0f, 0.0f, 0.0f);

    int i_;        // neighbour index
    float3 m_;     // neighbour magnetisation
    float3 delta;  // r_j - r_i
    float a__;     // inter-cell exchange stiffness
    bool present;  // real neighbour?

    // ---- x bonds -------------------------------------------------------------------
    i_      = idx(lclampx(ix - 1), iy, iz);
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    delta   = make_float3(-dx, 0.0f, 0.0f);
    a__     = aLUT2d[symidx(r0, present ? regions[i_] : r0)];
    B += wx * a__ * ac_bond(present, m0, m_, delta, I, i_, Ex, Ey, Ez, xi_, xi_mul,
                            qx_, qx_mul, qy_, qy_mul, qz_, qz_mul, acConst, projectAxis, openBC);

    i_      = idx(hclampx(ix + 1), iy, iz);
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    delta   = make_float3(dx, 0.0f, 0.0f);
    a__     = aLUT2d[symidx(r0, present ? regions[i_] : r0)];
    B += wx * a__ * ac_bond(present, m0, m_, delta, I, i_, Ex, Ey, Ez, xi_, xi_mul,
                            qx_, qx_mul, qy_, qy_mul, qz_, qz_mul, acConst, projectAxis, openBC);

    // ---- y bonds -------------------------------------------------------------------
    i_      = idx(ix, lclampy(iy - 1), iz);
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    delta   = make_float3(0.0f, -dy, 0.0f);
    a__     = aLUT2d[symidx(r0, present ? regions[i_] : r0)];
    B += wy * a__ * ac_bond(present, m0, m_, delta, I, i_, Ex, Ey, Ez, xi_, xi_mul,
                            qx_, qx_mul, qy_, qy_mul, qz_, qz_mul, acConst, projectAxis, openBC);

    i_      = idx(ix, hclampy(iy + 1), iz);
    m_      = make_float3(mx[i_], my[i_], mz[i_]);
    present = !(is0(m_) || i_ == I);
    delta   = make_float3(0.0f, dy, 0.0f);
    a__     = aLUT2d[symidx(r0, present ? regions[i_] : r0)];
    B += wy * a__ * ac_bond(present, m0, m_, delta, I, i_, Ex, Ey, Ez, xi_, xi_mul,
                            qx_, qx_mul, qy_, qy_mul, qz_, qz_mul, acConst, projectAxis, openBC);

    // ---- z bonds (3D only, matching the plain exchange stencil) ---------------------
    if (Nz != 1) {
        i_      = idx(ix, iy, lclampz(iz - 1));
        m_      = make_float3(mx[i_], my[i_], mz[i_]);
        present = !(is0(m_) || i_ == I);
        delta   = make_float3(0.0f, 0.0f, -dz);
        a__     = aLUT2d[symidx(r0, present ? regions[i_] : r0)];
        B += wz * a__ * ac_bond(present, m0, m_, delta, I, i_, Ex, Ey, Ez, xi_, xi_mul,
                                qx_, qx_mul, qy_, qy_mul, qz_, qz_mul, acConst, projectAxis, openBC);

        i_      = idx(ix, iy, hclampz(iz + 1));
        m_      = make_float3(mx[i_], my[i_], mz[i_]);
        present = !(is0(m_) || i_ == I);
        delta   = make_float3(0.0f, 0.0f, dz);
        a__     = aLUT2d[symidx(r0, present ? regions[i_] : r0)];
        B += wz * a__ * ac_bond(present, m0, m_, delta, I, i_, Ex, Ey, Ez, xi_, xi_mul,
                                qx_, qx_mul, qy_, qy_mul, qz_, qz_mul, acConst, projectAxis, openBC);
    }

    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Bx[I] += B.x * invMs;
    By[I] += B.y * invMs;
    Bz[I] += B.z * invMs;
}
