#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#define DSIZEX 64
#define DSIZEY 32
#define DSIZEZ 2
#define DSIZE DSIZEX*DSIZEY*DSIZEZ
#define nTPBX 32
#define nTPBY 32
#define nTPBZ 32
#define nTPB 32
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
__device__ int   blk_numX = 0;
__device__ int   blk_numY = 0;
__device__ int   blk_numZ = 0;




__global__ void max_idx_kernel(float* data,
                int gridSizeX, int gridSizeY, int gridSizeZ,
                int* resultX, int* resultY, int* resultZ,
                float* resultVal_n1X, float* resultVal_p1X,
                            float* resultVal, 
                float* resultVal_n1Y, float* resultVal_p1Y) {
    __shared__ volatile float vals[nTPB];
    __shared__ volatile int idxs[nTPB * 3]; // Store x, y, z indices consecutively
    __shared__ volatile int last_block;
    const int dsize = gridSizeX*gridSizeY*gridSizeZ;
    int idx = threadIdx.x + blockDim.x * blockIdx.x + threadIdx.y + blockDim.y * blockIdx.y + threadIdx.z + blockDim.z * blockIdx.z;
    last_block = 0;

    float my_val = FLOAT_MIN;
    int my_idx = -1;

    // Sweep from global memory
    while (idx < dsize) {
        if (fabsf(data[idx]) > fabsf(my_val)) {
            my_val = data[idx];
            my_idx = idx;
        }
        idx += blockDim.x * gridDim.x;
    }
    while (idx < dsize) {
        if (fabsf(data[idx]) > fabsf(my_val)) {
            my_val = data[idx];
            my_idx = idx;
        }
        idx += blockDim.y * gridDim.y;
    }
    while (idx < dsize) {
        if (fabsf(data[idx]) > fabsf(my_val)) {
            my_val = data[idx];
            my_idx = idx;
        }
        idx += blockDim.z * gridDim.z;
    }

    // Calculate 3D indices from 1D index
    int x = my_idx % gridSizeX;
    int y = (my_idx / gridSizeX) % gridSizeY;
    int z = my_idx / (gridSizeX * gridSizeY);
    
    // Populate shared memory
    vals[threadIdx.x + threadIdx.y + threadIdx.z] = my_val;
    idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3] = x;
    idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 1] = y;
    idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 2] = z;

    __syncthreads();

    // Sweep in shared memory in x
    for (int i = (nTPB >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            if (fabsf(vals[(threadIdx.x + threadIdx.y + threadIdx.z)]) < fabsf(vals[(threadIdx.x + threadIdx.y + threadIdx.z) + i])) {
                vals[(threadIdx.x + threadIdx.y + threadIdx.z)] = vals[(threadIdx.x + threadIdx.y + threadIdx.z) + i];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z)* 3] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 1] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3 + 1];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 2] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3 + 2];
            }
        }
        __syncthreads();
    }
    // Sweep in shared memory in y
    for (int i = (nTPB >> 1); i > 0; i >>= 1) {
        if (threadIdx.y < i) {
            if (fabsf(vals[(threadIdx.x + threadIdx.y + threadIdx.z)]) < fabsf(vals[(threadIdx.x + threadIdx.y + threadIdx.z) + i])) {
                vals[(threadIdx.x + threadIdx.y + threadIdx.z)] = vals[(threadIdx.x + threadIdx.y + threadIdx.z) + i];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z)* 3] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 1] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3 + 1];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 2] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3 + 2];
            }
        }
        __syncthreads();
    }
    // Sweep in shared memory in z
    for (int i = (nTPB >> 1); i > 0; i >>= 1) {
        if (threadIdx.z < i) {
            if (fabsf(vals[(threadIdx.x + threadIdx.y + threadIdx.z)]) < fabsf(vals[(threadIdx.x + threadIdx.y + threadIdx.z) + i])) {
                vals[(threadIdx.x + threadIdx.y + threadIdx.z)] = vals[(threadIdx.x + threadIdx.y + threadIdx.z) + i];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z)* 3] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 1] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3 + 1];
                idxs[(threadIdx.x + threadIdx.y + threadIdx.z) * 3 + 2] = idxs[((threadIdx.x + threadIdx.y + threadIdx.z) + i) * 3 + 2];
            }
        }
        __syncthreads();
    }

    // Perform block-level reduction x
    if (threadIdx.x == 0) {
        blk_vals[blockIdx.x + blockIdx.y + blockIdx.z] = vals[0];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z) * 3] = idxs[0];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z)* 3 + 1] = idxs[1];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z) * 3 + 2] = idxs[2];
        if (atomicAdd(&blk_numX, 1) == gridDim.x - 1) {
            last_block = 1;
        }
    }

    __syncthreads();

    // Perform block-level reduction x
    if (threadIdx.y == 0) {
        blk_vals[blockIdx.x + blockIdx.y + blockIdx.z] = vals[0];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z) * 3] = idxs[0];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z)* 3 + 1] = idxs[1];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z) * 3 + 2] = idxs[2];
        if (atomicAdd(&blk_numY, 1) == gridDim.y - 1) {
            last_block = 1;
        }
    }

    __syncthreads();

    // Perform block-level reduction z
    if (threadIdx.z == 0) {
        blk_vals[blockIdx.x + blockIdx.y + blockIdx.z] = vals[0];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z) * 3] = idxs[0];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z)* 3 + 1] = idxs[1];
        blk_idxs[(blockIdx.x + blockIdx.y + blockIdx.z) * 3 + 2] = idxs[2];
        if (atomicAdd(&blk_numZ, 1) == gridDim.z - 1) {
            last_block = 1;
        }
    }

    __syncthreads();

    if (last_block) {
        idx = threadIdx.x;
        my_val = FLOAT_MIN;
        int my_idxX = -1, my_idxY = -1, my_idxZ = -1;
        while (idx < gridDim.x) {
            if (fabsf(blk_vals[idx]) > fabsf(my_val)) {
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
                if (fabsf(vals[threadIdx.x]) < fabsf(vals[threadIdx.x + i])) {
                    vals[threadIdx.x] = vals[threadIdx.x + i];
                    idxs[threadIdx.x * 3] = idxs[(threadIdx.x + i) * 3];
                    idxs[threadIdx.x * 3 + 1] = idxs[(threadIdx.x + i) * 3 + 1];
                    idxs[threadIdx.x * 3 + 2] = idxs[(threadIdx.x + i) * 3 + 2];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *resultX = idxs[0];
            *resultY = idxs[1];
            *resultZ = idxs[2];
            *resultVal = vals[0];
            if (idxs[0]-1 >= 0) {
                *resultVal_n1X = data[(idxs[0] - 1) + idxs[1] * gridSizeX + idxs[2] * gridSizeX * gridSizeY];
            } else {
                *resultVal_n1X = 0.0f;
            }
            if (idxs[0]+1 >= 0) {
                *resultVal_p1X = data[(idxs[0] + 1) + idxs[1] * gridSizeX + idxs[2] * gridSizeX * gridSizeY];
            } else {
                *resultVal_p1X = 0.0f;
            }
            if (idxs[1]-1 >= 0) {
                *resultVal_n1Y = data[idxs[0] + (idxs[1] - 1) * gridSizeX + idxs[2] * gridSizeX * gridSizeY];
            } else {
                *resultVal_n1Y = 0.0f;
            }
            if (idxs[1]+1 >= 0) {
                *resultVal_p1Y = data[idxs[0] + (idxs[1] + 1) * gridSizeX + idxs[2] * gridSizeX * gridSizeY];
            } else {
                *resultVal_p1Y = 0.0f;
            }
            
        }
    }
}
int main() {
    float* d_vector, * h_vector;
    h_vector = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++) {
        h_vector[i] = rand() / (float)RAND_MAX;
    }
    h_vector[64 + 32] = 10; // Create definite max element

    cudaMalloc(&d_vector, DSIZE * sizeof(float));
    cudaMemcpy(d_vector, h_vector, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    int max_indexX = 0;
    int max_indexY = 0;
    int max_indexZ = 0;
    int* d_max_indexX;
    int* d_max_indexY;
    int* d_max_indexZ;
    cudaMalloc(&d_max_indexX, sizeof(int));
    cudaMalloc(&d_max_indexY, sizeof(int));
    cudaMalloc(&d_max_indexZ, sizeof(int));

    
    float resultVal_n1XC = 0.0f;
    float resultVal_p1XC = 0.0f;
    float resultValC = 0.0f;
    float resultVal_n1YC = 0.0f;
    float resultVal_p1YC = 0.0f;

    float* resultVal_n1X;
    float* resultVal_p1X;
    float* resultVal;
    float* resultVal_n1Y;
    float* resultVal_p1Y;

    cudaMalloc(&resultVal_n1X, sizeof(float));
    cudaMalloc(&resultVal_p1X, sizeof(float));
    cudaMalloc(&resultVal, sizeof(float));
    cudaMalloc(&resultVal_n1Y, sizeof(float));
    cudaMalloc(&resultVal_p1Y, sizeof(float));

    int gridSizeX = DSIZEX;
    int gridSizeY = DSIZEY;
    int gridSizeZ = DSIZEZ;

    unsigned long long dtime = dtime_usec(0);
    max_idx_kernel<<<MAX_KERNEL_BLOCKS, nTPB>>>(d_vector, gridSizeX, gridSizeY, gridSizeZ, d_max_indexX, d_max_indexY, d_max_indexZ, resultVal_n1X, resultVal_p1X, resultVal, resultVal_n1Y, resultVal_p1Y);
    cudaMemcpy(&max_indexX, d_max_indexX, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_indexY, d_max_indexY, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_indexZ, d_max_indexZ, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(&resultVal_n1XC, resultVal_n1X, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultVal_p1XC, resultVal_p1X, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultValC, resultVal, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultVal_n1YC, resultVal_n1Y, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultVal_p1YC, resultVal_p1Y, sizeof(int), cudaMemcpyDeviceToHost);

    dtime = dtime_usec(dtime);

    std::cout << "Kernel time: " << dtime / (float)USECPSEC << " seconds" << std::endl;
    std::cout << "Max index: [" << max_indexX << ", " << max_indexY << ", " << max_indexZ << "]" << std::endl;
    std::cout << "[" << resultVal_n1XC << ", " << resultVal_p1XC << "]" << std::endl;
    std::cout << "[   " << resultValC << "   ]" << std::endl;
    std::cout << "[" << resultVal_n1YC << ", " << resultVal_p1YC << "]" << std::endl;

    delete[] h_vector;
}