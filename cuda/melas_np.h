#ifndef _MELAS_NP_H_
#define _MELAS_NP_H_

// Device helpers for the magnum.np-style fully coupled magnetoelastic solver (NP solver).
//
// Conventions (match magnum.np / PRApplied 25, 034050 (2026)):
//  - Voigt order [xx, yy, zz, yz, xz, xy] with engineering shear (eps[3] = dyuz+dzuy, ...).
//  - Stiffness matrix restricted to the block-diagonal form of Eq. (24):
//    C11 C12 C13 / C12 C22 C23 / C13 C23 C33 / C44 / C55 / C66.
//  - All divisions that can hit C=0 (airbox) are guarded like magnum.np's nan_to_num.

#include "amul.h"

// nan_to_num(posinf=0, neginf=0) equivalent: a/b with 0 when b == 0.
inline __device__ float melas_safediv(float a, float b) {
    return (b == 0.0f) ? 0.0f : (a / b);
}

// magnetostrictive strain in Voigt notation from unit magnetization
// eps_m[0..2] = 3/2 l100 (m_i^2 - 1/3);  eps_m[3..5] = 3 l111 (m_y m_z, m_x m_z, m_x m_y)
inline __device__ void melas_eps_m(float* em, float mx, float my, float mz, float l100, float l111) {
    em[0] = 1.5f * l100 * (mx * mx - (1.0f / 3.0f));
    em[1] = 1.5f * l100 * (my * my - (1.0f / 3.0f));
    em[2] = 1.5f * l100 * (mz * mz - (1.0f / 3.0f));
    em[3] = 3.0f * l111 * my * mz;
    em[4] = 3.0f * l111 * mx * mz;
    em[5] = 3.0f * l111 * mx * my;
}

// sig = C : eps for the block-diagonal Voigt stiffness (Eq. 24)
inline __device__ void melas_sigma(float* sig, const float* eps,
                                   float c11, float c12, float c13,
                                   float c22, float c23, float c33,
                                   float c44, float c55, float c66) {
    sig[0] = c11 * eps[0] + c12 * eps[1] + c13 * eps[2];
    sig[1] = c12 * eps[0] + c22 * eps[1] + c23 * eps[2];
    sig[2] = c13 * eps[0] + c23 * eps[1] + c33 * eps[2];
    sig[3] = c44 * eps[3];
    sig[4] = c55 * eps[4];
    sig[5] = c66 * eps[5];
}

// harmonic mean of gradient g weighted by stiffness C over a cell pair (i, j):
// mean = (C_i g_i + C_j g_j) / (C_i + C_j), guarded for C=0 (magnum.np _get_B_jump_conditions)
inline __device__ float melas_harmonic(float gi, float Ci, float gj, float Cj) {
    return melas_safediv(Ci * gi + Cj * gj, Ci + Cj);
}

// Contribution of the (i, i+1) pair to the jump-aware second derivative d/dq (C du/dq)
// (magnum.np solver._2nd_derivative_*, uniform grid):
//   Cavg (u_{i+1}-u_i) + h C_i jump  with Cavg = 2 C_i C_{i+1}/denom,
//   jump = -(Br_i - Bl_{i+1})/denom, denom = h (C_i + C_{i+1}).
// The total second derivative is (plus_part + minus_part)/h^2, dropping missing sides
// at open boundaries (homogeneous Neumann is built in).
inline __device__ float melas_d2_plus(float u0, float up, float C0, float Cp,
                                      float Br0, float Blp, float h) {
    float denom = h * (C0 + Cp);
    return melas_safediv(2.0f * C0 * Cp, denom) * (up - u0)
         + h * C0 * melas_safediv(-(Br0 - Blp), denom);
}

// Contribution of the (i-1, i) pair; jump = -(Br_{i-1} - Bl_i)/denom.
inline __device__ float melas_d2_minus(float u0, float um, float C0, float Cm,
                                       float Brm, float Bl0, float h) {
    float denom = h * (Cm + C0);
    return -melas_safediv(2.0f * Cm * C0, denom) * (u0 - um)
         + h * C0 * melas_safediv(-(Brm - Bl0), denom);
}

// Jump-aware first derivative of a product field q = C*g along one direction with jump
// coefficient C and B = 0 (magnum.np solver._mixed_derivative via gradient_with_pbc).
// q0/qm/qp and C0/Cm/Cp are the product and coefficient at the cell and its neighbors.
inline __device__ float melas_dmix(float q0, float qm, float qp,
                                   float C0, float Cm, float Cp,
                                   bool hasm, bool hasp, float h) {
    float gf = 0.0f, gb = 0.0f;
    if (hasp) {
        gf = melas_safediv(2.0f * Cp * (qp - q0), h * (C0 + Cp));
    }
    if (hasm) {
        gb = melas_safediv(2.0f * Cm * (q0 - qm), h * (Cm + C0));
    }
    if (hasm && hasp) {
        return 0.5f * (gf + gb);
    }
    return hasp ? gf : gb;
}

#endif
