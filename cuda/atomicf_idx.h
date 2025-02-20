#include <stdio.h>
#define DSIZE 5000
#define nTPB 256

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

typedef union  {
  float floats[2];                 // floats[0] = lowest
  int intsX[2];                    // ints[1] = lowIdx
  int intsY[2]; 
  int intsZ[2]; 
  int idx;                         // for atomic update
} atomicsIdx;

__device__ atomicsIdx test;

inline __device__ int atomicMinIndex(int *address, float initVal, int index3Dx, int index3Dy, int index3Dz)
{
    atomicsIdx loc, loctest;
    loc.floats[0] = initVal;
    loc.intsX[1] = index3Dx;
    loc.intsY[1] = index3Dy;
    loc.intsZ[1] = index3Dz;
    loctest.idx = *address;
    while (loctest.floats[0] > initVal) 
      loctest.idx = atomicCAS(address, loctest.idx,  loc.idx);
    return loctest.idx;
}

inline __device__ int atomicMaxIndex(int *address, float initVal, int index3Dx, int index3Dy, int index3Dz)
{
    atomicsIdx loc, loctest;
    loc.floats[0] = initVal;
    loc.intsX[1] = index3Dx;
    loc.intsY[1] = index3Dy;
    loc.intsZ[1] = index3Dz;
    loctest.idx = *address;
    while (loctest.floats[0] < initVal) 
      loctest.idx = atomicCAS(address, loctest.idx,  loc.idx);
    return loctest.idx;
}