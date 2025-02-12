#include <stdint.h>
#include "amul.h"

// TODO: this could act on x,y,z, so that we need to call it only once.
extern "C" __global__ void
settemperature_elastic(float* __restrict__  F,      float* __restrict__ noise,
                float* __restrict__ eta_, float eta_mul,
                float* __restrict__ temp_, float temp_mul,
                float deltaT, float cellvolume,
                int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float temp = amul(temp_, temp_mul, i);
        float eta = amul(eta_, eta_mul, i);
        F[i] = noise[i] * sqrtf(2. * eta * temp * 1.380649e-23)/(deltaT*cellvolume);
    }
}

