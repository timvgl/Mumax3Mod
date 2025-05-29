#ifndef _MIN_H_
#define _MIN_H_

inline __device__ float mymin(float a, float b) {
    return a < b ? a : b;
}

#endif // _MIN_H_
