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
#define nTPBX 256 / 8
#define nTPBY 256 / 8
#define MAX_KERNEL_BLOCKS 30
#define MAX_BLOCKSX ((DSIZEX/nTPBX)+1)
#define MAX_BLOCKSY ((DSIZEY/nTPBY)+1)
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


__device__ volatile float blk_vals[MAX_BLOCKSX][MAX_BLOCKSY];
__device__ volatile int   blk_idxsX[MAX_BLOCKSX][MAX_BLOCKSY];
__device__ volatile int   blk_idxsY[MAX_BLOCKSX][MAX_BLOCKSY];
__device__ int   blk_numX = 0;
__device__ int   blk_numY = 0;

template <typename T>
__global__ void max_idx_kernel(const T *data, const int dsizeX, const int dsizeY, const int dsizeZ, int *resultX, int *resultY, int *resultZ){

  __shared__ volatile T   vals[nTPBX][nTPBY];
  __shared__ volatile int idxsX[nTPBX][nTPBY];
  __shared__ volatile int idxsY[nTPBX][nTPBY];
  __shared__ volatile int last_block;
  int idxX = threadIdx.x+blockDim.x*blockIdx.x;
  int idxY = threadIdx.y+blockDim.y*blockIdx.y;
  last_block = 0;
  T   my_val = FLOAT_MIN;
  int my_idxX = -1;
  int my_idxY = -1;
  // sweep from global memory
  while (idxX < dsizeX){
    while (idxY < dsizeY){ 
      if (data[idxX, idxY] > my_val) {
        my_val = data[idxX, idxY];
        my_idxX = idxX;
        my_idxY = idxY;
      }
      idxY += blockDim.y*gridDim.y;
    }
    idxX += blockDim.x*gridDim.x;
  }
  // populate shared memory
  vals[threadIdx.x][threadIdx.y] = my_val;
  idxsX[threadIdx.x][threadIdx.y] = my_idxX;
  idxsY[threadIdx.x][threadIdx.y] = my_idxY;
  __syncthreads();
  // sweep in shared memory
  for (int i = (nTPBX>>1); i > 0; i>>=1){
    for (int j = (nTPBY>>1); j > 0; j>>=1){
      if (threadIdx.x < i && threadIdx.y < j) {
        if (vals[threadIdx.x][threadIdx.y] < vals[threadIdx.x + i][threadIdx.y + j]) {
          vals[threadIdx.x][threadIdx.y] = vals[threadIdx.x + i][threadIdx.y + j]; 
          idxsX[threadIdx.x][threadIdx.y] = idxsX[threadIdx.x + i][threadIdx.y + j];
          idxsY[threadIdx.x][threadIdx.y] = idxsY[threadIdx.x + i][threadIdx.y + j];
        }
      }
      __syncthreads();
    }
  }
  // perform block-level reduction
  if (!threadIdx.x && !threadIdx.y){
    blk_vals[blockIdx.x][blockIdx.y] = vals[0][0];
    blk_idxsX[blockIdx.x][blockIdx.y] = idxsX[0][0];
    blk_idxsY[blockIdx.x][blockIdx.y] = idxsY[0][0];
    if (atomicAdd(&blk_numX, 1) == gridDim.x - 1 && atomicAdd(&blk_numY, 1) == gridDim.y - 1) { // then I am the last block
      last_block = 1;
    }
  }
  __syncthreads();
  if (last_block == 1){
    idxX = threadIdx.x;
    idxY = threadIdx.y;
    my_val = FLOAT_MIN;
    my_idxX = -1;
    my_idxY = -1;
    while (idxX < gridDim.x){
      while (idxY < gridDim.y){
        if (blk_vals[idxX][idxY] > my_val) {
          my_val = blk_vals[idxX][idxY];
          my_idxX = blk_idxsX[idxX][idxY];
          my_idxY = blk_idxsY[idxX][idxY];
        }
        idxY += blockDim.y;
      }
      idxX += blockDim.x;
    }
    // populate shared memory
    vals[threadIdx.x][threadIdx.y] = my_val;
    idxsX[threadIdx.x][threadIdx.y] = my_idxX;
    idxsY[threadIdx.x][threadIdx.y] = my_idxY;
    __syncthreads();
    // sweep in shared memory
      for (int i = (nTPBX>>1); i > 0; i>>=1){
        for (int j = (nTPBY>>1); j > 0; j>>=1){
          if (threadIdx.x < i && threadIdx.y < j)
            if (vals[threadIdx.x][threadIdx.y] < vals[threadIdx.x + i][threadIdx.y + j]) {
              vals[threadIdx.x][threadIdx.y] = vals[threadIdx.x + i][threadIdx.y + j];
              idxsX[threadIdx.x][threadIdx.y] = idxsX[threadIdx.x + i][threadIdx.y + j];
              idxsY[threadIdx.x][threadIdx.y] = idxsY[threadIdx.x + i][threadIdx.y + j];
            }
          __syncthreads();
        }
      }
      if (!threadIdx.x && !threadIdx.y) {
        *resultX = idxsX[0][0];
        *resultY = idxsY[0][0];
      }
  }
}

int main(){
  float *d_vector, *h_vector;
  h_vector = new float[DSIZEX, DSIZEY];
  for (int i = 0; i < DSIZEX; i++) {
    for (int k = 0; i < DSIZEY; i++) {
      h_vector[i, k] = rand()/(float)RAND_MAX;
    }
  }
  h_vector[10, 50] = 10;  // create definite max element
  std::cout << "data generated" << std::endl;
  cudaMalloc(&d_vector, DSIZEX*DSIZEY*sizeof(float));
  cudaMemcpy(d_vector, h_vector, DSIZEX*DSIZEY*sizeof(float), cudaMemcpyHostToDevice);
  std::cout << "data uploaded to gpu" << std::endl;
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
  std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: X " << max_indexX << std::endl;
  std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: Y " << max_indexY << std::endl;
  std::cout << "kernel time: " << dtime/(float)USECPSEC << " max index: Z " << max_indexZ << std::endl;
  return 0;
}