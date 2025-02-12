extern "C" __global__ void absGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = fabsf(value[idx]);
    }
}
#################
extern "C" __global__ void acosGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = acosf(value[idx]);
    }
}
#################
extern "C" __global__ void acoshGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = acoshf(value[idx]);
    }
}
#################
extern "C" __global__ void asinGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = asinf(value[idx]);
    }
}
#################
extern "C" __global__ void asinhGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = asinhf(value[idx]);
    }
}
#################
extern "C" __global__ void atanGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = atanf(value[idx]);
    }
}
#################
extern "C" __global__ void atanhGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = atanhf(value[idx]);
    }
}
#################
extern "C" __global__ void cbrtGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = cbrtf(value[idx]);
    }
}
#################
extern "C" __global__ void ceilGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = ceilf(value[idx]);
    }
}
#################
extern "C" __global__ void cosGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = cosf(value[idx]);
    }
}
#################
extern "C" __global__ void coshGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = coshf(value[idx]);
    }
}
#################
extern "C" __global__ void erfGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = erff(value[idx]);
    }
}
#################
extern "C" __global__ void erfcGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = erfcf(value[idx]);
    }
}
#################
extern "C" __global__ void expGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = expf(value[idx]);
    }
}
#################
extern "C" __global__ void exp2Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = exp2f(value[idx]);
    }
}
#################
extern "C" __global__ void expm1Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = expm1f(value[idx]);
    }
}
#################
extern "C" __global__ void floorGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = floorf(value[idx]);
    }
}
#################
extern "C" __global__ void gammaGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Use tgammaf to compute the Gamma function.
        value[idx] = tgammaf(value[idx]);
    }
}
#################
extern "C" __global__ void j0Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = j0f(value[idx]);
    }
}
#################
extern "C" __global__ void j1Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = j1f(value[idx]);
    }
}
#################
extern "C" __global__ void logGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = logf(value[idx]);
    }
}
#################
extern "C" __global__ void log10Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = log10f(value[idx]);
    }
}
#################
extern "C" __global__ void log1pGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = log1pf(value[idx]);
    }
}
#################
extern "C" __global__ void log2Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = log2f(value[idx]);
    }
}
#################
extern "C" __global__ void logbGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = logbf(value[idx]);
    }
}
#################
extern "C" __global__ void sinGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = sinf(value[idx]);
    }
}
#################
extern "C" __global__ void sinhGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = sinhf(value[idx]);
    }
}
#################
extern "C" __global__ void sqrtGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = sqrtf(value[idx]);
    }
}
#################
extern "C" __global__ void tanGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = tanf(value[idx]);
    }
}
#################
extern "C" __global__ void tanhGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = tanhf(value[idx]);
    }
}
#################
extern "C" __global__ void truncGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = truncf(value[idx]);
    }
}
#################
extern "C" __global__ void y0Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = y0f(value[idx]);
    }
}
#################
extern "C" __global__ void y1Govaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        value[idx] = y1f(value[idx]);
    }
}
#################
extern "C" __global__ void ilogbGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        // ilogbf returns an int; convert the result to float.
        value[idx] = (float) ilogbf(value[idx]);
    }
}
#################
extern "C" __global__ void heavisideGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = value[idx];
        if (x < 0.0f)
            value[idx] = 0.0f;
        else if (x > 0.0f)
            value[idx] = 1.0f;
        else
            value[idx] = 0.5f;
    }
}
#################
extern "C" __global__ void normGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Assuming norm(x) is defined as the absolute value.
        value[idx] = fabsf(value[idx]);
    }
}
#################
extern "C" __global__ void sincGovaluate(float* __restrict__ value, int N) {
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = value[idx];
        // sinc(x) = sin(x)/x for x != 0, and 1 for x == 0.
        value[idx] = (fabsf(x) < 1e-7f) ? 1.0f : sinf(x) / x;
    }
}
#################
#include "reduce.h"
#include "atomicf.h"
#include "float3.h"

#define returner(i)  \
	        a[i]

extern "C" __global__ void maxGovaluate(float* __restrict__ a,
    float* __restrict__ dst, float initVal, int n) {
    reduce(returner, fmax, atomicFmax)
}
#################
#include "reduce.h"
#include "atomicf.h"
#include "float3.h"

#define returner(i)  \
	        a[i]

extern "C" __global__ void minGovaluate(float* __restrict__ a,
    float* __restrict__ dst, float initVal, int n) {
    reduce(returner, fmax, atomicFmin)
}

