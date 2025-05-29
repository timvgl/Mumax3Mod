#include "reduce.h"
#include "atomicf.h"
#include "float3.h"
#include "min.h"

#define load(i)  \
	        a[i]

extern "C" __global__ void minGovaluate(float* __restrict__ a,
    float* __restrict__ dst, float initVal, int n) {
        reduce(load, mymin, atomicFmin)
}
