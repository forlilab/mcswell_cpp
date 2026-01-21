#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>
#include <type_traits>

#include "water_model.cuh"
#include "consts.cuh"

template <typename T>
__host__ __device__ __forceinline__ T tsqrt(T x) {
    #if defined(__CUDA_ARCH__)
        if constexpr (std::is_same<T, float>::value) return sqrtf(x);
        else return sqrt(x);
    #else
        using std::sqrt;
        return (T)sqrt((double)x);
    #endif
}

template <typename T>
__host__ __device__ __forceinline__ T trsqr(T x) { return x * x; }

template <typename T>
__host__ __device__ __forceinline__ T lennard_jones_rmin_half(
    T epsilon_1,
    T epsilon_2,
    T dist,
    T rmin_half1,
    T rmin_half2
) {
    const T rmin = rmin_half1 + rmin_half2;
    const T eps = tsqrt(epsilon_1 * epsilon_2);
    dist = (dist < (T)1e-8 ? (T)1e-8 : dist); 
    const T inv = rmin / dist;
    const T inv2 = inv * inv;
    const T inv6 = inv2 * inv2 * inv2;
    const T inv12 = inv6 * inv6;
    return eps * (inv12 - static_cast<T>(2.0) * inv6);
}

template <typename T>
__host__ __device__ __forceinline__ T coulomb_energy(T q1, T q2, T r) {
    const T k_e = static_cast<T>(332.0636);
    return k_e * (q1 * q2) / r;
}

__device__ __forceinline__
float eval_energy_with_receptor(
    const float* __restrict__ receptor_atoms,   // [n_receptor*7]
    const float* __restrict__ target_water,     // [12] compact
    uint32_t n_receptor_atoms,
    float* __restrict__ sh_partials             // [blockDim.x]
){
    const int tid = (int)threadIdx.x;
    const int stride = (int)blockDim.x;

    float partial = 0.0f;

    for (uint32_t r_idx = (uint32_t)tid; r_idx < n_receptor_atoms; r_idx += (uint32_t)stride) {
        const uint32_t rb = r_idx * ATOM_FEATURES;

        const float r_x         = receptor_atoms[rb + 0];
        const float r_y         = receptor_atoms[rb + 1];
        const float r_z         = receptor_atoms[rb + 2];
        const float r_q         = receptor_atoms[rb + 3];
        const float r_eps       = receptor_atoms[rb + 4];
        const float r_rmin_half = receptor_atoms[rb + 5];

        #pragma unroll
        for (int t_ai = 0; t_ai < ATOMS_PER_W; ++t_ai) {
            const int tb = t_ai * W_STRIDE;
            const float t_x = target_water[tb + 0];
            const float t_y = target_water[tb + 1];
            const float t_z = target_water[tb + 2];

            const bool t_is_O = (t_ai == 0);
            const float t_q         = t_is_O ? water_model::Q_O : water_model::Q_H;
            const float t_eps       = t_is_O ? water_model::EPS_O : 0.0f;
            const float t_rmin_half = t_is_O ? water_model::RMINH_O : 0.0f;

            const float dx = t_x - r_x;
            const float dy = t_y - r_y;
            const float dz = t_z - r_z;
            const float r2 = dx*dx + dy*dy + dz*dz;

            const float dist = sqrtf(r2);
            const float r_val = fmaxf(dist, 1e-8f);

            // LJ only if both are non-hydrogen (your convention: eps==0 => H)
            if (t_eps != 0.0f && r_eps != 0.0f) {
                partial += lennard_jones_rmin_half(t_eps, r_eps, r_val, t_rmin_half, r_rmin_half);
            }

            partial += coulomb_energy(t_q, r_q, r_val);
        }
    }

    // block reduction
    sh_partials[tid] = partial;
    __syncthreads();

    for (int step = stride >> 1; step > 0; step >>= 1) {
        if (tid < step) sh_partials[tid] += sh_partials[tid + step];
        __syncthreads();
    }
    return sh_partials[0];
}

__device__ __forceinline__
float eval_energy_with_waters(
    const float* __restrict__ waters_compact,  // [n_mu * target_n_waters * 12] (water-major)
    const float* __restrict__ target_water,    // [12] compact
    uint32_t sim_id,
    uint32_t active_waters,
    uint32_t target_n_waters,
    int skip_w_idx,
    float* __restrict__ sh_partials            // [blockDim.x]
){
    const int tid = (int)threadIdx.x;
    const int stride = (int)blockDim.x;

    float partial = 0.0f;

    // base in floats for this sim
    const uint32_t sim_base = sim_id * target_n_waters * (uint32_t)WATER_SIZE;

    for (uint32_t w_idx = (uint32_t)tid; w_idx < active_waters; w_idx += (uint32_t)stride) {
        
        if (skip_w_idx >= 0 && (int)w_idx == skip_w_idx) continue;

        const uint32_t wb = sim_base + w_idx * (uint32_t)WATER_SIZE;

        // Load water atoms (x,y,z,res) for O/H1/H2
        const float wOx = waters_compact[wb + 0];
        const float wOy = waters_compact[wb + 1];
        const float wOz = waters_compact[wb + 2];
        const uint32_t wRes = (uint32_t)waters_compact[wb + 3];

        const float wH1x = waters_compact[wb + 4];
        const float wH1y = waters_compact[wb + 5];
        const float wH1z = waters_compact[wb + 6];

        const float wH2x = waters_compact[wb + 8];
        const float wH2y = waters_compact[wb + 9];
        const float wH2z = waters_compact[wb + 10];

        // Interact each atom of this water with each atom of target water, skipping self by resnum
        // Atom 0: Oxygen
        {
            const float w_x = wOx, w_y = wOy, w_z = wOz;
            const float w_q = water_model::Q_O;
            const float w_eps = water_model::EPS_O;
            const float w_rminh = water_model::RMINH_O;

#pragma unroll
            for (int t_ai=0; t_ai<ATOMS_PER_W; ++t_ai) {
                const int tb = t_ai * W_STRIDE;
                const float t_x = target_water[tb + 0];
                const float t_y = target_water[tb + 1];
                const float t_z = target_water[tb + 2];
                const uint32_t t_res = (uint32_t)target_water[tb + 3];

                const bool t_is_O = (t_ai == 0);
                const float t_q = t_is_O ? water_model::Q_O : water_model::Q_H;
                const float t_eps = t_is_O ? water_model::EPS_O : 0.0f;
                const float t_rminh = t_is_O ? water_model::RMINH_O : 0.0f;

                const float dx=t_x-w_x, dy=t_y-w_y, dz=t_z-w_z;
                const float r2=dx*dx+dy*dy+dz*dz;
                const float r = fmaxf(sqrtf(r2), 1e-8f);

                // if (t_res == wRes) continue;

                if (t_eps != 0.0f) { // w is O so nonzero
                    partial += lennard_jones_rmin_half(t_eps, w_eps, r, t_rminh, w_rminh);
                }
                partial += coulomb_energy(t_q, w_q, r);
            }
        }

        // Atom 1: H1
        {
            const float w_x = wH1x, w_y = wH1y, w_z = wH1z;
            const float w_q = water_model::Q_H;

#pragma unroll
            for (int t_ai=0; t_ai<ATOMS_PER_W; ++t_ai) {
                const int tb = t_ai * W_STRIDE;
                const float t_x = target_water[tb + 0];
                const float t_y = target_water[tb + 1];
                const float t_z = target_water[tb + 2];
                const uint32_t t_res = (uint32_t)target_water[tb + 3];

                const bool t_is_O = (t_ai == 0);
                const float t_q = t_is_O ? water_model::Q_O : water_model::Q_H;

                const float dx=t_x-w_x, dy=t_y-w_y, dz=t_z-w_z;
                const float r2=dx*dx+dy*dy+dz*dz;
                const float r = fmaxf(sqrtf(r2), 1e-8f);

                // if (t_res == wRes) continue;

                // LJ skipped (H eps=0)
                partial += coulomb_energy(t_q, w_q, r);
            }
        }

        // Atom 2: H2
        {
            const float w_x = wH2x, w_y = wH2y, w_z = wH2z;
            const float w_q = water_model::Q_H;

#pragma unroll
            for (int t_ai=0; t_ai<ATOMS_PER_W; ++t_ai) {
                const int tb = t_ai * W_STRIDE;
                const float t_x = target_water[tb + 0];
                const float t_y = target_water[tb + 1];
                const float t_z = target_water[tb + 2];
                const uint32_t t_res = (uint32_t)target_water[tb + 3];

                const bool t_is_O = (t_ai == 0);
                const float t_q = t_is_O ? water_model::Q_O : water_model::Q_H;

                const float dx=t_x-w_x, dy=t_y-w_y, dz=t_z-w_z;
                const float r2=dx*dx+dy*dy+dz*dz;
                const float r = fmaxf(sqrtf(r2), 1e-8f);

                // if (t_res == wRes) continue;

                partial += coulomb_energy(t_q, w_q, r);
            }
        }
    }

    sh_partials[tid] = partial;
    __syncthreads();

    for (int step = stride >> 1; step > 0; step >>= 1) {
        if (tid < step) sh_partials[tid] += sh_partials[tid + step];
        __syncthreads();
    }
    return sh_partials[0];
}