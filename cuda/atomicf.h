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
__device__ inline int floatToOrderedInt(float f) {
    int x = __float_as_int(f);
    return (x >= 0) ?  x : ~x;
}

__device__ inline float orderedIntToFloat(int x) {
    x = (x >= 0) ? x : ~x;
    return __int_as_float(x);
}

__device__ inline void atomicFmin(float* addr, float val) {
    int* int_addr    = reinterpret_cast<int*>(addr);
    int  rawOld      = *int_addr;                  // aktueller Int-Bits-Wert
    int  ordNew      = floatToOrderedInt(val);     // geordnetes Int des neuen Werts
    int  rawNew      = __float_as_int(val);        // rohe Bits des neuen Werts
    int  assumed;

    do {
        assumed = rawOld;
        if (ordNew >= floatToOrderedInt(__int_as_float(assumed)))
            break;
        rawOld = atomicCAS(int_addr, assumed, rawNew);
    } while (assumed != rawOld);
}

// Atomic max of value.
inline __device__ void atomicFmax(float* a, float b){
	atomicMax((int*)(a), *((int*)(&b)));
}


#endif
