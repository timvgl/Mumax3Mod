#include <cublas_v2.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>

#define DSIZEX 512 
#define DSIZEY 512 / 2
#define DSIZEZ 512 / 4

// nTPB should be a power-of-2
#define nTPBX 256 / 16
#define nTPBY 256 / 16
#define nTPBZ 256 / 16
#define MAX_KERNEL_BLOCKS 30
#define MAX_BLOCKSX ((DSIZEX/nTPBX)+1)
#define MAX_BLOCKSY ((DSIZEY/nTPBY)+1)
#define MAX_BLOCKSZ ((DSIZEZ/nTPBZ)+1)
#define MIN(a,b) ((a>b)?b:a)
#define FLOAT_MIN -1.0f

#include <time.h>
#include <sys/time.h>

unsigned long long dtime_usec(unsigned long long prev){
#define USECPSEC 1000000ULL
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}


__device__ volatile float blk_vals[MAX_BLOCKSX][MAX_BLOCKSY][MAX_BLOCKSZ];
__device__ volatile int   blk_idxsX[MAX_BLOCKSX][MAX_BLOCKSY][MAX_BLOCKSZ];
__device__ volatile int   blk_idxsY[MAX_BLOCKSX][MAX_BLOCKSY][MAX_BLOCKSZ];
__device__ volatile int   blk_idxsZ[MAX_BLOCKSX][MAX_BLOCKSY][MAX_BLOCKSZ];
__device__ int   blk_numX = 0;
__device__ int   blk_numY = 0;
__device__ int   blk_numZ = 0;

template <typename T>
__global__ void max_idx_kernel(const T *data, const int dsizeX, const int dsizeY, const int dsizeZ, int *resultX, int *resultY, int *resultZ){

  __shared__ volatile T   vals[nTPBX][nTPBY][nTPB];
  __shared__ volatile int idxsX[nTPBX][nTPBY][nTPB];
  __shared__ volatile int idxsY[nTPBX][nTPBY][nTPB];
  __shared__ volatile int idxsZ[nTPBX][nTPBY][nTPB];
  __shared__ volatile int last_block;
  int idxX = threadIdx.x+blockDim.x*blockIdx.x;
  int idxY = threadIdx.y+blockDim.y*blockIdx.y;
  int idxZ = threadIdx.z+blockDim.z*blockIdx.z;
  last_block = 0;
  T   my_val = FLOAT_MIN;
  int my_idxX = -1;
  int my_idxY = -1;
  int my_idxZ = -1;
  // sweep from global memory
  while (idxX < dsizeX){
    while (idxY < dsizeY){ 
      while (idxZ < dsizeZ){
        if (data[idxX, idxY, idxZ] > my_val) {
          my_val = data[idxX, idxY, idxZ];
          my_idxX = idxX;
          my_idxY = idxY;
          my_idxZ = idxZ;
        }
        idxZ += blockDim.z*gridDim.z;
      }
      idxY += blockDim.y*gridDim.y;
    }
    idxX += blockDim.x*gridDim.x;
  }
  // populate shared memory
  vals[threadIdx.x][threadIdx.y][threadIdx.z] = my_val;
  idxsX[threadIdx.x][threadIdx.y][threadIdx.z] = my_idxX;
  idxsY[threadIdx.x][threadIdx.y][threadIdx.z] = my_idxY;
  idxsZ[threadIdx.x][threadIdx.y][threadIdx.z] = my_idxZ;
  __syncthreads();
  // sweep in shared memory
  for (int i = (nTPBX>>1); i > 0; i>>=1){
    for (int j = (nTPBY>>1); j > 0; j>>=1){
      for (int k = (nTPB>>1); k > 0; k>>=1){
        if (threadIdx.x < i && threadIdx.y < j && threadIdx.z < k)
          if (vals[threadIdx.x][threadIdx.y][threadIdx.z] < vals[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k]) {
            vals[threadIdx.x][threadIdx.y][threadIdx.z] = vals[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k]; 
            idxsX[threadIdx.x][threadIdx.y][threadIdx.z] = idxsX[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
            idxsY[threadIdx.x][threadIdx.y][threadIdx.z] = idxsY[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
            idxsZ[threadIdx.x][threadIdx.y][threadIdx.z] = idxsZ[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
            }
        __syncthreads();
      }
    }
  }
  // perform block-level reduction
  if (!threadIdx.x && !threadIdx.y && !threadIdx.z){
    blk_vals[blockIdx.x][blockIdx.y][blockIdx.z] = vals[0][0][0];
    blk_idxsX[blockIdx.x][blockIdx.y][blockIdx.z] = idxsX[0][0][0];
    blk_idxsY[blockIdx.x][blockIdx.y][blockIdx.z] = idxsY[0][0][0];
    blk_idxsZ[blockIdx.x][blockIdx.y][blockIdx.z] = idxsZ[0][0][0];
    if (atomicAdd(&blk_numX, 1) == gridDim.x - 1 && atomicAdd(&blk_numY, 1) == gridDim.y - 1 && atomicAdd(&blk_numZ, 1) == gridDim.z - 1) // then I am the last block
      last_block = 1;
    }
  __syncthreads();
  if (last_block){
    idxX = threadIdx.x;
    idxY = threadIdx.y;
    idxZ = threadIdx.z;
    my_val = FLOAT_MIN;
    my_idxX = -1;
    my_idxY = -1;
    my_idxZ = -1;
    while (idxX < gridDim.x){
      while (idxY < gridDim.y){
        while (idxZ < gridDim.z){
          if (blk_vals[idxX][idxY][idxZ] > my_val) {
            my_val = blk_vals[idxX][idxY][idxZ];
            my_idxX = blk_idxsX[idxX][idxY][idxZ];
            my_idxY = blk_idxsY[idxX][idxY][idxZ];
            my_idxZ = blk_idxsZ[idxX][idxY][idxZ];
          }
          idxZ += blockDim.z;
        }
        idxY += blockDim.y;
      }
      idxX += blockDim.x;
    }
  // populate shared memory
    vals[threadIdx.x][threadIdx.y][threadIdx.z] = my_val;
    idxsX[threadIdx.x][threadIdx.y][threadIdx.z] = my_idxX;
    idxsY[threadIdx.x][threadIdx.y][threadIdx.z] = my_idxY;
    idxsZ[threadIdx.x][threadIdx.y][threadIdx.z] = my_idxZ;
    __syncthreads();
  // sweep in shared memory
    for (int i = (nTPBX>>1); i > 0; i>>=1){
      for (int j = (nTPBY>>1); j > 0; j>>=1){
        for (int k = (nTPB>>1); k > 0; k>>=1){
          if (threadIdx.x < i && threadIdx.y < j && threadIdx.z < k)
            if (vals[threadIdx.x][threadIdx.y][threadIdx.z] < vals[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k]) {
              vals[threadIdx.x][threadIdx.y][threadIdx.z] = vals[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
              idxsX[threadIdx.x][threadIdx.y][threadIdx.z] = idxsX[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
              idxsY[threadIdx.x][threadIdx.y][threadIdx.z] = idxsY[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
              idxsZ[threadIdx.x][threadIdx.y][threadIdx.z] = idxsZ[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
            }
          __syncthreads();
        }
      }
    }
    if (!threadIdx.x && !threadIdx.y && !threadIdx.z)
      *resultX = idxsX[0][0][0];
      *resultY = idxsY[0][0][0];
      *resultZ = idxsZ[0][0][0];
    }
}

int main(){
  float *d_vector, *h_vector;
  h_vector = new float[DSIZEX, DSIZEY, DSIZEZ];
  for (int i = 0; i < DSIZEX; i++) {
    for (int k = 0; i < DSIZEY; i++) {
      for (int j = 0; i < DSIZEZ; i++) {
        h_vector[i, k, j] = rand()/(float)RAND_MAX;
      }
    }
  }
  h_vector[10, 50, 3] = 10;  // create definite max element
  std::cout << "data generated" << std::endl;
  cudaMalloc(&d_vector, DSIZEX*DSIZEY*DSIZEZ*sizeof(float));
  cudaMemcpy(d_vector, h_vector, DSIZEX*DSIZEY*DSIZEZ*sizeof(float), cudaMemcpyHostToDevice);
  int max_indexX = 0;
  int max_indexY = 0;
  int max_indexZ = 0;
  int *d_max_indexX;
  int *d_max_indexY;
  int *d_max_indexZ;
  cudaMalloc(&d_max_indexX, sizeof(int));
  cudaMalloc(&d_max_indexY, sizeof(int));
  cudaMalloc(&d_max_indexZ, sizeof(int));
  unsigned long long dtime = dtime_usec(0);
  max_idx_kernel<<<MIN(MAX_KERNEL_BLOCKS, ((DSIZEX+nTPBX-1)/nTPBX)), nTPBX>>>(d_vector, DSIZEX, DSIZEY, DSIZEZ, d_max_indexX, d_max_indexY, d_max_indexZ);
  cudaMemcpy(&max_indexX, d_max_indexX, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_indexY, d_max_indexY, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_indexZ, d_max_indexZ, sizeof(int), cudaMemcpyDeviceToHost);
  dtime = dtime_usec(dtime);
  std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: X" << max_indexX << std::endl;
  std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: Y" << max_indexY << std::endl;
  std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: Z" << max_indexZ << std::endl;
  return 0;
}