#include <stdint.h>
#include "stencil.h"
#include "amul.h"
#include "melas_np.h"

// Applies space- and time-dependent Dirichlet displacement values (NP solver):
// u = val where the Dirichlet mask (frozenDispLoc) is nonzero
// (magnum.np: state.ud[bc.mask] = bc.condition(state) with a callable condition).
extern "C" __global__ void
MelasDirichletNP(float* __restrict__ ux, float* __restrict__ uy, float* __restrict__ uz,
                 float* __restrict__ valx, float* __restrict__ valy, float* __restrict__ valz,
                 float* __restrict__ frozen_, float frozen_mul,
                 int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);

    if (amul(frozen_, frozen_mul, I) != 0.0f) {
        ux[I] = valx[I];
        uy[I] = valy[I];
        uz[I] = valz[I];
    }
}
