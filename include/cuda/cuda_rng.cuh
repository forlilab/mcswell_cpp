//  Copyright (c) 2026 Scripps Research, Forli Lab.
//  All rights reserved.
//
//  Author: Niccolo Bruciaferri
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the Free Software
//  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

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
