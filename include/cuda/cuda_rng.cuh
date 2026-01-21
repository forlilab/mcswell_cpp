#pragma once
#include <curand_kernel.h>

__device__ inline float randu(curandStatePhilox4_32_10_t* st) {
    return curand_uniform(st); // (0,1]
}

__device__ inline float rand_range(curandStatePhilox4_32_10_t* st, float a, float b) {
    return a + (b - a) * randu(st);
}

__device__ inline int rand_int(curandStatePhilox4_32_10_t* st, int n) {
    // uniform int in [0, n-1]
    return (int)(curand(st) % (unsigned)n);
}
