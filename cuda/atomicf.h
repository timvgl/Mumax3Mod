#ifndef _ATOMICF_H_
#define _ATOMICF_H_

// Atomic max of abs value.
inline __device__ void atomicFmaxabs(float* a, float b){
	b = fabs(b);
	atomicMax((int*)(a), *((int*)(&b)));
}

// Atomic min of abs value.
inline __device__ void atomicFminabs(float* a, float b){
	b = fabs(b);
	atomicMin((int*)(a), *((int*)(&b)));
}

// Atomic max of value.
inline __device__ void atomicFmax(float* a, float b){
	atomicMax((int*)(a), *((int*)(&b)));
}

// Atomic min of value.
inline __device__ void atomicFmin(float* a, float b){
	atomicMin((int*)(a), *((int*)(&b)));
}


#endif
