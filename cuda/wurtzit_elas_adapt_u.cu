#include "stencil.h"
#include "amul.h"
#include <stdint.h>
#include "deriv_axis.cuh"

// === PASTE-POINT A: one-sided 4th-order derivative helpers (x/y) ===========
__device__ inline void d1_forward4_coeff_sum_x(const float* u, int i,int j,int k,
                                               int Nx,int Ny,int Nz, uint8_t PBC,
                                               float h, bool leftBoundary,
                                               float& coeff_u0, float& sum_nb)
{
    if (leftBoundary) {
        int I1=ID3(1,j,k,Nx,Ny,Nz,PBC), I2=ID3(2,j,k,Nx,Ny,Nz,PBC),
            I3=ID3(3,j,k,Nx,Ny,Nz,PBC), I4=ID3(4,j,k,Nx,Ny,Nz,PBC);
        coeff_u0 = -25.f/(12.f*h);
        sum_nb   = ( 48.f*u[I1] - 36.f*u[I2] + 16.f*u[I3] - 3.f*u[I4])/(12.f*h);
    } else {
        int I1=ID3(Nx-2,j,k,Nx,Ny,Nz,PBC), I2=ID3(Nx-3,j,k,Nx,Ny,Nz,PBC),
            I3=ID3(Nx-4,j,k,Nx,Ny,Nz,PBC), I4=ID3(Nx-5,j,k,Nx,Ny,Nz,PBC);
        coeff_u0 = +25.f/(12.f*h);
        sum_nb   = (-48.f*u[I1] + 36.f*u[I2] - 16.f*u[I3] + 3.f*u[I4])/(12.f*h);
    }
}

__device__ inline void d1_forward4_coeff_sum_y(const float* u, int i,int j,int k,
                                               int Nx,int Ny,int Nz, uint8_t PBC,
                                               float h, bool bottomBoundary,
                                               float& coeff_u0, float& sum_nb)
{
    if (bottomBoundary) {
        int J1=ID3(i,1,k,Nx,Ny,Nz,PBC), J2=ID3(i,2,k,Nx,Ny,Nz,PBC),
            J3=ID3(i,3,k,Nx,Ny,Nz,PBC), J4=ID3(i,4,k,Nx,Ny,Nz,PBC);
        coeff_u0 = -25.f/(12.f*h);
        sum_nb   = ( 48.f*u[J1] - 36.f*u[J2] + 16.f*u[J3] - 3.f*u[J4])/(12.f*h);
    } else {
        int J1=ID3(i,Ny-2,k,Nx,Ny,Nz,PBC), J2=ID3(i,Ny-3,k,Nx,Ny,Nz,PBC),
            J3=ID3(i,Ny-4,k,Nx,Ny,Nz,PBC), J4=ID3(i,Ny-5,k,Nx,Ny,Nz,PBC);
        coeff_u0 = +25.f/(12.f*h);
        sum_nb   = (-48.f*u[J1] + 36.f*u[J2] - 16.f*u[J3] + 3.f*u[J4])/(12.f*h);
    }
}
// =========================================================================== 


extern "C" __global__ void
adaptUNeumannBndry( float *ux, float *uy, float *uz,
                    float *mx, float *my, float *mz,
                    float* __restrict__  C11_, float  C11_mul,
                    float* __restrict__  C12_, float  C12_mul,
                    float* __restrict__  C13_, float  C13_mul,
                    float* __restrict__  C33_, float  C33_mul,
                    float* __restrict__  C44_, float  C44_mul,
                    float* __restrict__  B1_,  float  B1_mul,
                    float* __restrict__  B2_,  float  B2_mul,
                    float dx, float dy, float dz,
                    int Nx, int Ny, int Nz,
                    uint8_t PBC, bool cubic)
{
    int idx1D = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    long long Fx0   = (long long)Ny * (long long)Nz;          // x = 0
    long long Fx1   = Fx0;                                    // x = Nx-1
    long long Fy0   = (long long)Nx * (long long)Nz;          // y = 0
    long long Fy1   = Fy0;                                    // y = Ny-1
    long long Fz0   = (long long)Nx * (long long)Ny;          // z = 0
    long long Fz1   = Fz0;                                    // z = Nz-1
    long long TOTAL = Fx0 + Fx1 + Fy0 + Fy1 + Fz0 + Fz1;

    if ((long long)idx1D >= TOTAL) return;

    int face = 0;
    long long t = idx1D;
    if (t < Fx0) { face = 0; }
    else { t -= Fx0;
        if (t < Fx1) { face = 1; }
        else { t -= Fx1;
            if (t < Fy0) { face = 2; }
            else { t -= Fy0;
                if (t < Fy1) { face = 3; }
                else { t -= Fy1;
                    if (t < Fz0) { face = 4; }
                    else { t -= Fz0; face = 5; }
                }
            }
        }
    }

    int i=0, j=0, k=0;
    if (face == 0) { j = (int)( t % Ny ); k = (int)( t / Ny ); i = 0; }
    else if (face == 1) { j = (int)( t % Ny ); k = (int)( t / Ny ); i = Nx - 1; }
    else if (face == 2) { i = (int)( t % Nx ); k = (int)( t / Nx ); j = 0; }
    else if (face == 3) { i = (int)( t % Nx ); k = (int)( t / Nx ); j = Ny - 1; }
    else if (face == 4) { i = (int)( t % Nx ); j = (int)( t / Nx ); k = 0; }
    else {                i = (int)( t % Nx ); j = (int)( t / Nx ); k = Nz - 1; }

    if (i < 0 || i >= Nx || j < 0 || j >= Ny || k < 0 || k >= Nz) return;

    // ownership masks to avoid double-writes:
    // - x faces: own everything on their face (including edges/corners)
    // - y faces: skip points touching any x boundary
    // - z faces: skip points touching any x or y boundary
    if ((face==2 || face==3) && (i==0 || i==Nx-1)) return;
    if ((face==4 || face==5) && (i==0 || i==Nx-1 || j==0 || j==Ny-1)) return;

    int I = idx(i,j,k);

    float c11 = amul(C11_, C11_mul, I);
    float c12 = amul(C12_, C12_mul, I);
    float c13 = amul(C13_, C13_mul, I);
    float c33 = amul(C33_, C33_mul, I);
    float c44 = amul(C44_, C44_mul, I);
    if (cubic) { c13 = c12; c33 = c11; }
    float c66 = 0.5f * (c11 - c12);

    float B1 = amul(B1_, B1_mul, I);
    float B2 = amul(B2_, B2_mul, I);
    float mx_i = mx[I], my_i = my[I], mz_i = mz[I];
    float sm_v[6];
    sm_v[0] = B1 * mx_i*mx_i;          // xx
    sm_v[1] = B1 * my_i*my_i;          // yy
    sm_v[2] = B1 * mz_i*mz_i;          // zz
    sm_v[3] = 2.f * B2 * (my_i*mz_i);  // yz
    sm_v[4] = 2.f * B2 * (mz_i*mx_i);  // zx
    sm_v[5] = 2.f * B2 * (mx_i*my_i);  // xy

    // base derivatives at (i,j,k) using your robust deriv_axis
    float dUx_dx = deriv_axis(ux, i,j,k, Nx,Ny,Nz, PBC, dx, 0, 0, 0);
    float dUy_dy = deriv_axis(uy, i,j,k, Nx,Ny,Nz, PBC, dy, 1, 0, 0);
    float dUz_dz = (Nz > 1 ? deriv_axis(uz, i,j,k, Nx,Ny,Nz, PBC, dz, 2, 0, 0) : 0.0f);
    float dUy_dx = deriv_axis(uy, i,j,k, Nx,Ny,Nz, PBC, dx, 0, 0, 0);
    float dUx_dy = deriv_axis(ux, i,j,k, Nx,Ny,Nz, PBC, dy, 1, 0, 0);
    float dUz_dx = deriv_axis(uz, i,j,k, Nx,Ny,Nz, PBC, dx, 0, 0, 0);
    float dUx_dz = (Nz > 1 ? deriv_axis(ux, i,j,k, Nx,Ny,Nz, PBC, dz, 2, 0, 0) : 0.0f);
    float dUz_dy = (Nz > 1 ? deriv_axis(uz, i,j,k, Nx,Ny,Nz, PBC, dy, 1, 0, 0) : 0.0f);
    float dUy_dz = (Nz > 1 ? deriv_axis(uy, i,j,k, Nx,Ny,Nz, PBC, dz, 2, 0, 0) : 0.0f);

    // === PASTE-POINT B: Nz==1-Block (2D) inkl. Flächen, Kanten, Ecken ========
    if (Nz == 1) {
        const bool at_x0 = (i==0), at_x1 = (i==Nx-1);
        const bool at_y0 = (j==0), at_y1 = (j==Ny-1);
        const bool on_x  = (at_x0||at_x1);
        const bool on_y  = (at_y0||at_y1);

        // --- Ecke/Kante (gleichzeitig x- und y-Rand) -> 2x2-System für ux,uy ---
        if (on_x && on_y) {
            float Ax, bx_ux; d1_forward4_coeff_sum_x(ux,i,j,k,Nx,Ny,Nz,PBC,dx,at_x0,Ax,bx_ux);
            float Ay, by_uy; d1_forward4_coeff_sum_y(uy,i,j,k,Nx,Ny,Nz,PBC,dy,at_y0,Ay,by_uy);

            // σ_xx = c11*(Ax*ux0 + bx_ux) + c12*(∂y uy) ≈ sm_v[0]
            // σ_yy = c12*(Ax*ux0 + bx_ux) + c11*(Ay*uy0 + by_uy) ≈ sm_v[1]
            float rhs_xx = sm_v[0] - c11*bx_ux - c12*dUy_dy;
            float rhs_yy = sm_v[1] - c12*bx_ux - c11*by_uy;

            float a = c11*Ax, b = c12*Ay;
            float c = c12*Ax, d = c11*Ay;
            float det = a*d - b*c + 1e-30f;

            float ux0 = ( rhs_xx*d - b*rhs_yy ) / det;
            float uy0 = ( a*rhs_yy - c*rhs_xx ) / det;

            ux[I] = ux0;
            uy[I] = uy0;
            return;
        }

        // --- reine x-Flächen (ohne y-Rand) ---
        if (on_x && !on_y) {
            float Ax, bx_ux; d1_forward4_coeff_sum_x(ux,i,j,k,Nx,Ny,Nz,PBC,dx,at_x0,Ax,bx_ux);
            float rhs_xx = sm_v[0] - c11*bx_ux - c12*dUy_dy;
            ux[I] = rhs_xx / (c11*Ax);

            // σ_xy = c66*(∂x uy + ∂y ux) -> solve uy
            float Ax_uy, bx_uy; d1_forward4_coeff_sum_x(uy,i,j,k,Nx,Ny,Nz,PBC,dx,at_x0,Ax_uy,bx_uy);
            float Ay_ux = deriv_axis(ux,i,j,k,Nx,Ny,Nz,PBC,dy,1,0,0);
            float rhs_xy = sm_v[5] - c66*(Ay_ux + bx_uy);
            uy[I] = rhs_xy / (c66*Ax_uy);
            return;
        }

        // --- reine y-Flächen (ohne x-Rand) ---
        if (on_y && !on_x) {
            float Ay, by_uy; d1_forward4_coeff_sum_y(uy,i,j,k,Nx,Ny,Nz,PBC,dy,at_y0,Ay,by_uy);
            float rhs_yy = sm_v[1] - c11*by_uy - c12*dUx_dx;
            uy[I] = rhs_yy / (c11*Ay);

            float Ay_ux, by_ux; d1_forward4_coeff_sum_y(ux,i,j,k,Nx,Ny,Nz,PBC,dy,at_y0,Ay_ux,by_ux);
            float Ax_uy = deriv_axis(uy,i,j,k,Nx,Ny,Nz,PBC,dx,0,0,0);
            float rhs_xy = sm_v[5] - c66*(Ax_uy + by_ux);
            ux[I] = rhs_xy / (c66*Ay_ux);
            return;
        }

        // Innenpunkte (kein Rand): nichts zu tun
        return;
    }
    // === Ende PASTE-POINT B (2D) =============================================

    // === PASTE-POINT C: Original 3D-Logik für Nz>1 (dein bestehender Code) ===
    // -> Hier kannst du deinen existierenden 3D-Block lassen (x/y/z faces + edges),
    //    identisch wie von dir schon implementiert.
    // ==========================================================================
}
