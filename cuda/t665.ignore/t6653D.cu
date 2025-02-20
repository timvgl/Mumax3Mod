#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#define DSIZEX 64
#define DSIZEY 64
#define DSIZEZ 64
#define DSIZE DSIZEX*DSIZEY*DSIZEZ
#define nTPBX 8
#define nTPBY 8
#define nTPBZ 8
#define nTPB 8
#define FLOAT_MIN -1.0f
#define MAX_KERNEL_BLOCKS 30
#define MAX_BLOCKS ((DSIZE/nTPB)+1)

unsigned long long dtime_usec(unsigned long long prev) {
    #define USECPSEC 1000000ULL
    timeval tv1;
    gettimeofday(&tv1, 0);
    return ((tv1.tv_sec * USECPSEC) + tv1.tv_usec) - prev;
}

__device__ volatile float blk_vals[MAX_BLOCKS];
__device__ volatile float blk_idxs[MAX_BLOCKS];
__device__ int   blk_num = 0;



template <typename T>
__global__ void max_idx_kernel(const T* data, const int dsize, int gridSize, int* result) {
    __shared__ volatile T vals[nTPB];
    __shared__ volatile int idxs[nTPB * 3]; // Store x, y, z indices consecutively
    __shared__ volatile int last_block;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    last_block = 0;

    T my_val = FLOAT_MIN;
    int my_idx = -1;

    // Sweep from global memory
    while (idx < dsize) {
        if (data[idx] > my_val) {
            my_val = data[idx];
            my_idx = idx;
        }
        idx += blockDim.x * gridDim.x;
    }

    // Calculate 3D indices from 1D index
    int z = my_idx / (gridSize * gridSize);
    int y = (my_idx % (gridSize * gridSize)) / gridSize;
    int x = my_idx % gridSize;

    // Populate shared memory
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x * 3] = x;
    idxs[threadIdx.x * 3 + 1] = y;
    idxs[threadIdx.x * 3 + 2] = z;

    __syncthreads();

    // Sweep in shared memory
    for (int i = (nTPB >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
                vals[threadIdx.x] = vals[threadIdx.x + i];
                idxs[threadIdx.x * 3] = idxs[(threadIdx.x + i) * 3];
                idxs[threadIdx.x * 3 + 1] = idxs[(threadIdx.x + i) * 3 + 1];
                idxs[threadIdx.x * 3 + 2] = idxs[(threadIdx.x + i) * 3 + 2];
            }
        }
        __syncthreads();
    }

    // Perform block-level reduction
    if (threadIdx.x == 0) {
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x * 3] = idxs[0];
        blk_idxs[blockIdx.x * 3 + 1] = idxs[1];
        blk_idxs[blockIdx.x * 3 + 2] = idxs[2];
        if (atomicAdd(&blk_num, 1) == gridDim.x - 1) {
            last_block = 1;
        }
    }

    __syncthreads();

    if (last_block) {
        idx = threadIdx.x;
        my_val = FLOAT_MIN;
        int my_idxX = -1, my_idxY = -1, my_idxZ = -1;
        while (idx < gridDim.x) {
            if (blk_vals[idx] > my_val) {
                my_val = blk_vals[idx];
                my_idxX = blk_idxs[idx * 3];
                my_idxY = blk_idxs[idx * 3 + 1];
                my_idxZ = blk_idxs[idx * 3 + 2];
            }
            idx += blockDim.x;
        }

        // Populate shared memory
        vals[threadIdx.x] = my_val;
        idxs[threadIdx.x * 3] = my_idxX;
        idxs[threadIdx.x * 3 + 1] = my_idxY;
        idxs[threadIdx.x * 3 + 2] = my_idxZ;

        __syncthreads();

        // Sweep in shared memory
        for (int i = (nTPB >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x * 3] = idxs[(threadIdx.x + i) * 3];
                    idxs[threadIdx.x * 3 + 1] = idxs[(threadIdx.x + i) * 3 + 1];
                    idxs[threadIdx.x * 3 + 2] = idxs[(threadIdx.x + i) * 3 + 2];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            result[0] = idxs[0];
            result[1] = idxs[1];
            result[2] = idxs[2];
        }
    }
}

int main() {
    float* d_vector, * h_vector;
    h_vector = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++) {
        h_vector[i] = rand() / (float)RAND_MAX;
    }
    h_vector[65] = 10; // Create definite max element

    cudaMalloc(&d_vector, DSIZE * sizeof(float));
    cudaMemcpy(d_vector, h_vector, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    int max_index[3] = { 0, 0, 0 };
    int* d_max_index;
    cudaMalloc(&d_max_index, 3 * sizeof(int));

    int gridSize = ceil(pow((double)DSIZE, 1.0 / 3.0));

    unsigned long long dtime = dtime_usec(0);
    max_idx_kernel<<<MAX_KERNEL_BLOCKS, nTPB>>>(d_vector, DSIZE, gridSize, d_max_index);
    cudaMemcpy(max_index, d_max_index, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    dtime = dtime_usec(dtime);

    std::cout << "Kernel time: " << dtime / (float)USECPSEC << " seconds" << std::endl;
    std::cout << "Max index: [" << max_index[0] << ", " << max_index[1] << ", " << max_index[2] << "]" << std::endl;

    delete[] h_vector;
}
