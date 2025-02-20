#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#define FLOAT_MIN 0.0f
#define MAX_KERNEL_BLOCKS 30

/*
unsigned long long dtime_usec(unsigned long long prev) {
    #define USECPSEC 1000000ULL
    timeval tv1;
    gettimeofday(&tv1, 0);
    return ((tv1.tv_sec * USECPSEC) + tv1.tv_usec) - prev;
}
*/

extern "C" __global__ void
max_idx_kernel( float* __restrict__ data,
                float* __restrict__ vals, int* __restrict__  idxs,
                float* __restrict__ blk_vals, int* __restrict__  blk_idxs,
                int* blk_num, 
                int gridSizeX, int gridSizeY, int gridSizeZ,
                int* __restrict__ resultX, int* __restrict__ resultY, int* __restrict__ resultZ,
                float* __restrict__ resultVal_n1X, float* __restrict__ resultVal_p1X,
                            float* __restrict__ resultVal, 
                float* __restrict__ resultVal_n1Y, float* __restrict__ resultVal_p1Y) {
    const int nTPB = blockDim.x;
    //__shared__ volatile float vals[nTPB];
    //__shared__ volatile int idxs[nTPB * 3]; // Store x, y, z indices consecutively
    //__shared__ volatile int last_block;
    
    const int dsize = gridSizeX*gridSizeY*gridSizeZ;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int last_block = 0;

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
    // Calculate 3D indices from 1D index
    int x = my_idx % gridSizeX;
    int y = (my_idx / gridSizeX) % gridSizeY;
    int z = my_idx / (gridSizeX * gridSizeY);

    // Populate shared memory
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x * 3] = x;
    idxs[threadIdx.x * 3 + 1] = y;
    idxs[threadIdx.x * 3 + 2] = z;
    
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
    // Perform block-level reduction
    if (threadIdx.x == 0) {
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x * 3] = idxs[0];
        blk_idxs[blockIdx.x * 3 + 1] = idxs[1];
        blk_idxs[blockIdx.x * 3 + 2] = idxs[2];
        if (atomicAdd(blk_num, 1) == gridDim.x - 1) {
            last_block = 1;
        }
    }
    //printf("block: %i ", last_block);
    __syncthreads();
    if (last_block == 1) {
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
            printf("gridSizex %i ", gridSizeX);
            printf("gridSizey %i ", gridSizeY);
            printf("gridSizez %i ", gridSizeZ);
            printf("value %f ", vals[0]);
            *resultX = idxs[0];
            printf("%i ", idxs[0]);
            *resultY = idxs[1];
            printf("%i ", idxs[1]);
            *resultZ = idxs[2];
            printf("%i ", idxs[2]);
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
/*
extern "C" __global__ void
reducemaxvecCellZCompIndex() {
    int MAX_BLOCKS ((Nx*Ny*Nz/nTPB)+1);
    max_idx_kernel(data, Nx, Ny, Nz, dst);
}
*/


/*
int main() {
    float* d_vector, * h_vector;
    h_vector = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++) {
        h_vector[i] = rand() / (float)RAND_MAX;
    }
    h_vector[64 + 32] = 10; // Create definite max element

    cudaMalloc(&d_vector, DSIZE * sizeof(float));
    cudaMemcpy(d_vector, h_vector, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    int max_index[3] = { 0, 0, 0 };
    int* d_max_index;
    cudaMalloc(&d_max_index, 3 * sizeof(int));

    int gridSizeX = DSIZEX;
    int gridSizeY = DSIZEY;
    int gridSizeZ = DSIZEZ;

    unsigned long long dtime = dtime_usec(0);
    max_idx_kernel<<<MAX_KERNEL_BLOCKS, nTPB>>>(d_vector, gridSizeX, gridSizeY, gridSizeZ, d_max_index);
    cudaMemcpy(max_index, d_max_index, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    dtime = dtime_usec(dtime);

    std::cout << "Kernel time: " << dtime / (float)USECPSEC << " seconds" << std::endl;
    std::cout << "Max index: [" << max_index[0] << ", " << max_index[1] << ", " << max_index[2] << "]" << std::endl;

    delete[] h_vector;
}
*/