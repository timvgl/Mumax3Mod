#include "reduce_idx.h"
#include "atomicf_idx.h"
#include "float3.h"

#define load_vecnorm2(ix, iy, iz) \
	pow2(z[ix, iy, iz])

extern "C" __global__ void
reducemaxvecCellZCompIndex(float* __restrict__ x, float* __restrict__ y, float* __restrict__ z, int* __restrict__ dst, float initVal, int n) {
    reduce3D(load_vecnorm2, fmax, atomicMaxIndex)

}
