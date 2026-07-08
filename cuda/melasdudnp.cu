#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// du/dt right-hand side of the fully coupled magnetoelastic solver
// (magnum.np solver.dud: dud = pd/rho * mask, with the state variable being the velocity
// du here: dst = du masked by the elastic mask and the Dirichlet mask).
extern "C" __global__ void
MelasDudNP(float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
           float* __restrict__ dux, float* __restrict__ duy, float* __restrict__ duz,
           float* __restrict__ mask_, float mask_mul,
           float* __restrict__ frozen_, float frozen_mul,
           int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    float mask = amul(mask_, mask_mul, I);
    float frozen = amul(frozen_, frozen_mul, I);
    float fac = ((mask != 0.0f) && (frozen == 0.0f)) ? 1.0f : 0.0f;

    dstx[I] = fac * dux[I];
    dsty[I] = fac * duy[I];
    dstz[I] = fac * duz[I];
}
