#ifndef _REDUCE3D_H_
#define _REDUCE3D_H_


// Block size for reduce kernels.
#define REDUCE_BLOCKSIZE 512

// This macro expands to a reduce kernel with arbitrary reduce operation.
// Ugly, perhaps, but arguably nicer than some 1000+ line C++ template.
// load(i): loads element i, possibly pre-processing the data
// op(a, b): reduce operation. e.g. sum
// atomicOp(a, b): atomic reduce operation in global mem.
#define reduce3D(load, op, atomicOp)                                                        \
    __shared__ float sdata[REDUCE_BLOCKSIZE];                                               \
    int tid = threadIdx.x +                                                                 \
              threadIdx.y * blockDim.x +                                                    \
              threadIdx.z * blockDim.x * blockDim.y;                                        \
                                                                                            \
    int3 index3D;                                                                           \
    index3D.x = blockIdx.x * blockDim.x + threadIdx.x;                                      \
    index3D.y = blockIdx.y * blockDim.y + threadIdx.y;                                      \
    index3D.z = blockIdx.z * blockDim.z + threadIdx.z;                                      \
                                                                                            \
    int linearIndex = index3D.x +                                                           \
                      index3D.y * gridDim.x * blockDim.x +                                  \
                      index3D.z * gridDim.y * gridDim.x * blockDim.x;                       \
                                                                                            \
    float mine = initVal;                                                                   \
    int stride = gridDim.x * blockDim.x *                                                   \
                 gridDim.y * blockDim.y *                                                   \
                 gridDim.z * blockDim.z;                                                    \
    while (linearIndex < n) {                                                               \
        mine = op(mine, load(index3D.x, index3D.y, index3D.z));                             \
        linearIndex += stride;                                                              \
        index3D.x = linearIndex % (gridDim.x * blockDim.x);                                 \
        index3D.y = (linearIndex / (gridDim.x * blockDim.x)) % (gridDim.y * blockDim.y);    \
        index3D.z = linearIndex / (gridDim.x * blockDim.x * gridDim.y * blockDim.y);        \
    }                                                                                       \
    sdata[tid] = mine;                                                                      \
    __syncthreads();                                                                        \
                                                                                            \
    for (unsigned int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 32; s >>= 1) {      \
        if (tid < s) {                                                                      \
            sdata[tid] = op(sdata[tid], sdata[tid + s]);                                    \
        }                                                                                   \
        __syncthreads();                                                                    \
    }                                                                                       \
                                                                                            \
    if (tid < 32) {                                                                         \
        volatile float* smem = sdata;                                                       \
        smem[tid] = op(smem[tid], smem[tid + 32]);                                          \
        smem[tid] = op(smem[tid], smem[tid + 16]);                                          \
        smem[tid] = op(smem[tid], smem[tid +  8]);                                          \
        smem[tid] = op(smem[tid], smem[tid +  4]);                                          \
        smem[tid] = op(smem[tid], smem[tid +  2]);                                          \
        smem[tid] = op(smem[tid], smem[tid +  1]);                                          \
    }                                                                                       \
                                                                                            \
    if (tid == 0) { atomicOp(dst, sdata[0], index3D.x, index3D.y, index3D.z); }             \

#endif
