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

template <typename T>
__host__ __device__ __forceinline__ T coulomb_rf_energy(T q1, T q2, T r) {
    const T k_e = static_cast<T>(332.0636);
    // Reaction Field coefficients: (eps - 1) / (2 eps + 1)
    const T kappa = (EPSILON_RF - static_cast<T>(1)) /
                    (static_cast<T>(2) * EPSILON_RF * static_cast<T>(1));
    const T inv_r = static_cast<T>(1) / r;
    const T inv_rc = static_cast<T>(1) / RF_CUTOFF;

    // r^2 / r_cut^3
    const T rc2 = RF_CUTOFF * RF_CUTOFF;
    const T inv_rc3 = static_cast<T>(1) / (rc2 * RF_CUTOFF);
    const T r2_over_rc3 = (r * r) * inv_rc3;

    // E = ke*q1*q2 [1/r + kapppa*r^2/rc^3 - (1+kappa)/rc]
    const T bracket = inv_r + kappa * r2_over_rc3 - (static_cast<T>(1) + kappa) * inv_rc;
    return k_e * (q1 * q2) * bracket;
}

// __device__ __forceinline__
// float eval_energy_with_receptor(
//     const float* __restrict__ receptor_atoms,   // [n_receptor*7]
//     const float* __restrict__ target_water,     // [12] compact
//     uint32_t n_receptor_atoms,
//     float* __restrict__ sh_partials             // [blockDim.x]
// ){
//     const int tid = (int)threadIdx.x;
//     const int stride = (int)blockDim.x;

//     float partial = 0.0f;

//     for (uint32_t r_idx = (uint32_t)tid; r_idx < n_receptor_atoms; r_idx += (uint32_t)stride) {
//         const uint32_t rb = r_idx * ATOM_FEATURES;

//         const float r_x         = receptor_atoms[rb + 0];
//         const float r_y         = receptor_atoms[rb + 1];
//         const float r_z         = receptor_atoms[rb + 2];
//         const float r_q         = receptor_atoms[rb + 3];
//         const float r_eps       = receptor_atoms[rb + 4];
//         const float r_rmin_half = receptor_atoms[rb + 5];

//         #pragma unroll
//         for (int t_ai = 0; t_ai < ATOMS_PER_W; ++t_ai) {
//             const int tb = t_ai * W_STRIDE;
//             const float t_x = target_water[tb + 0];
//             const float t_y = target_water[tb + 1];
//             const float t_z = target_water[tb + 2];

//             const bool t_is_O = (t_ai == 0);
//             const float t_q         = t_is_O ? water_model::Q_O : water_model::Q_H;
//             const float t_eps       = t_is_O ? water_model::EPS_O : 0.0f;
//             const float t_rmin_half = t_is_O ? water_model::RMINH_O : 0.0f;

//             const float dx = t_x - r_x;
//             const float dy = t_y - r_y;
//             const float dz = t_z - r_z;
//             const float r2 = dx*dx + dy*dy + dz*dz;

//             const float dist = sqrtf(r2);
//             const float r_val = fmaxf(dist, 1e-8f);

//             // Trying to see if a distance cutoff works
//             // LJ only if both are non-hydrogen (your convention: eps==0 => H)
//             if (r_val <= RF_CUTOFF) {
//                 if (t_eps != 0.0f && r_eps != 0.0f) {
//                     partial += lennard_jones_rmin_half(t_eps, r_eps, r_val, t_rmin_half, r_rmin_half);
//                 }
//                 // partial += coulomb_rf_energy(t_q, r_q, r_val);
//             }
            

//             partial += coulomb_energy(t_q, r_q, r_val);
//         }
//     }

//     // block reduction
//     sh_partials[tid] = partial;
//     __syncthreads();

//     for (int step = stride >> 1; step > 0; step >>= 1) {
//         if (tid < step) sh_partials[tid] += sh_partials[tid + step];
//         __syncthreads();
//     }
//     return sh_partials[0];
// }

// __device__ __forceinline__
// float eval_energy_with_waters(
//     const float* __restrict__ waters_compact,  // [n_mu * target_n_waters * 12] (water-major)
//     const float* __restrict__ target_water,    // [12] compact
//     uint32_t sim_id,
//     uint32_t active_waters,
//     uint32_t target_n_waters,
//     int skip_w_idx,
//     float* __restrict__ sh_partials            // [blockDim.x]
// ){
//     const int tid = (int)threadIdx.x;
//     const int stride = (int)blockDim.x;

//     float partial = 0.0f;

//     // base in floats for this sim
//     const uint32_t sim_base = sim_id * target_n_waters * (uint32_t)WATER_SIZE;

//     for (uint32_t w_idx = (uint32_t)tid; w_idx < active_waters; w_idx += (uint32_t)stride) {
        
//         if (skip_w_idx >= 0 && (int)w_idx == skip_w_idx) continue;

//         const uint32_t wb = sim_base + w_idx * (uint32_t)WATER_SIZE;

//         // Load water atoms (x,y,z,res) for O/H1/H2
//         const float wOx = waters_compact[wb + 0];
//         const float wOy = waters_compact[wb + 1];
//         const float wOz = waters_compact[wb + 2];
//         const uint32_t wRes = (uint32_t)waters_compact[wb + 3];

//         const float wH1x = waters_compact[wb + 4];
//         const float wH1y = waters_compact[wb + 5];
//         const float wH1z = waters_compact[wb + 6];

//         const float wH2x = waters_compact[wb + 8];
//         const float wH2y = waters_compact[wb + 9];
//         const float wH2z = waters_compact[wb + 10];

//         // Interact each atom of this water with each atom of target water
//         // Atom 0: Oxygen
//         {
//             const float w_x = wOx, w_y = wOy, w_z = wOz;
//             const float w_q = water_model::Q_O;
//             const float w_eps = water_model::EPS_O;
//             const float w_rminh = water_model::RMINH_O;

// #pragma unroll
//             for (int t_ai=0; t_ai<ATOMS_PER_W; ++t_ai) {
//                 const int tb = t_ai * W_STRIDE;
//                 const float t_x = target_water[tb + 0];
//                 const float t_y = target_water[tb + 1];
//                 const float t_z = target_water[tb + 2];
//                 const uint32_t t_res = (uint32_t)target_water[tb + 3];

//                 const bool t_is_O = (t_ai == 0);
//                 const float t_q = t_is_O ? water_model::Q_O : water_model::Q_H;
//                 const float t_eps = t_is_O ? water_model::EPS_O : 0.0f;
//                 const float t_rminh = t_is_O ? water_model::RMINH_O : 0.0f;

//                 const float dx=t_x-w_x, dy=t_y-w_y, dz=t_z-w_z;
//                 const float r2=dx*dx+dy*dy+dz*dz;
//                 const float r = fmaxf(sqrtf(r2), 1e-8f);

//                 if (r <= RF_CUTOFF) {
//                     if (t_eps != 0.0f) { // w is O so nonzero
//                         partial += lennard_jones_rmin_half(t_eps, w_eps, r, t_rminh, w_rminh);
//                     }
//                     // partial += coulomb_rf_energy(t_q, w_q, r);
//                 }
//                 partial += coulomb_energy(t_q, w_q, r);
//                 // }
//             }
//         }

//         // Atom 1: H1
//         {
//             const float w_x = wH1x, w_y = wH1y, w_z = wH1z;
//             const float w_q = water_model::Q_H;

// #pragma unroll
//             for (int t_ai=0; t_ai<ATOMS_PER_W; ++t_ai) {
//                 const int tb = t_ai * W_STRIDE;
//                 const float t_x = target_water[tb + 0];
//                 const float t_y = target_water[tb + 1];
//                 const float t_z = target_water[tb + 2];
//                 const uint32_t t_res = (uint32_t)target_water[tb + 3];

//                 const bool t_is_O = (t_ai == 0);
//                 const float t_q = t_is_O ? water_model::Q_O : water_model::Q_H;

//                 const float dx=t_x-w_x, dy=t_y-w_y, dz=t_z-w_z;
//                 const float r2=dx*dx+dy*dy+dz*dz;
//                 const float r = fmaxf(sqrtf(r2), 1e-8f);

//                 // LJ skipped (H eps=0)
//                 // if (r <= RF_CUTOFF) {
//                 //     partial += coulomb_rf_energy(t_q, w_q, r);
//                 // }
//                 partial += coulomb_energy(t_q, w_q, r);
//             }
//         }

//         // Atom 2: H2
//         {
//             const float w_x = wH2x, w_y = wH2y, w_z = wH2z;
//             const float w_q = water_model::Q_H;

// #pragma unroll
//             for (int t_ai=0; t_ai<ATOMS_PER_W; ++t_ai) {
//                 const int tb = t_ai * W_STRIDE;
//                 const float t_x = target_water[tb + 0];
//                 const float t_y = target_water[tb + 1];
//                 const float t_z = target_water[tb + 2];
//                 const uint32_t t_res = (uint32_t)target_water[tb + 3];

//                 const bool t_is_O = (t_ai == 0);
//                 const float t_q = t_is_O ? water_model::Q_O : water_model::Q_H;

//                 const float dx=t_x-w_x, dy=t_y-w_y, dz=t_z-w_z;
//                 const float r2=dx*dx+dy*dy+dz*dz;
//                 const float r = fmaxf(sqrtf(r2), 1e-8f);

//                 // if (r <= RF_CUTOFF) {
//                 //     partial += coulomb_rf_energy(t_q, w_q, r);
//                 // }
//                 partial += coulomb_energy(t_q, w_q, r);
//             }
//         }
//     }

//     sh_partials[tid] = partial;
//     __syncthreads();

//     for (int step = stride >> 1; step > 0; step >>= 1) {
//         if (tid < step) sh_partials[tid] += sh_partials[tid + step];
//         __syncthreads();
//     }
//     return sh_partials[0];
// }

__device__ __forceinline__
float eval_energy_with_receptor(
    const float* __restrict__ receptor_atoms,   // [n_receptor*7]
    const float* __restrict__ target_water,     // [12] compact (O/H1/H2)
    uint32_t n_receptor_atoms,
    float* __restrict__ sh_partials             // [blockDim.x]
){
    const int tid    = (int)threadIdx.x;
    const int stride = (int)blockDim.x;

    float partial = 0.0f;

    // --- load target O/H/H once ---
    const float tOx  = target_water[0];
    const float tOy  = target_water[1];
    const float tOz  = target_water[2];

    const float tH1x = target_water[4];
    const float tH1y = target_water[5];
    const float tH1z = target_water[6];

    const float tH2x = target_water[8];
    const float tH2y = target_water[9];
    const float tH2z = target_water[10];

#if defined(TIP4P)
    // --- compute M once ---
    const float3u tM = tip4p_m_site(
        tOx, tOy, tOz,
        tH1x, tH1y, tH1z,
        tH2x, tH2y, tH2z,
        (float)water_model::dOM
    );
#endif

    for (uint32_t r_idx = (uint32_t)tid; r_idx < n_receptor_atoms; r_idx += (uint32_t)stride) {
        const uint32_t rb = r_idx * ATOM_FEATURES;

        const float r_x         = receptor_atoms[rb + 0];
        const float r_y         = receptor_atoms[rb + 1];
        const float r_z         = receptor_atoms[rb + 2];
        const float r_q         = receptor_atoms[rb + 3];
        const float r_eps       = receptor_atoms[rb + 4];
        const float r_rmin_half = receptor_atoms[rb + 5];

        // -----------------------
        // Lennard-Jones: O only
        // -----------------------
        {
            const float dx = tOx - r_x;
            const float dy = tOy - r_y;
            const float dz = tOz - r_z;
            const float r2 = dx*dx + dy*dy + dz*dz;
            const float r  = fmaxf(sqrtf(r2), 1e-8f);

            if (r <= RF_CUTOFF) {
                // LJ only if receptor is non-H (your eps==0 convention)
                if (r_eps != 0.0f) {
                    const float t_eps= water_model::EPS_O;
                    const float t_r_min_half = water_model::RMINH_O;
                    partial += lennard_jones_rmin_half(
                        t_eps, r_eps, r,
                        t_r_min_half, r_rmin_half
                    );
                }
                // If you want reaction-field Coulomb, put it here consistently.
                // partial += coulomb_rf_energy(...);
            }
        }

        // -----------------------
        // Coulomb: model-dependent
        // -----------------------
#if defined(TIP4P)
        // TIP4P charges are on H/H/M (O is uncharged)
        {
            // H1
            {
                const float dx = tH1x - r_x;
                const float dy = tH1y - r_y;
                const float dz = tH1z - r_z;
                const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                const float t_q_h1 = water_model::Q_H;
                partial += coulomb_energy(t_q_h1, r_q, r);
            }
            // H2
            {
                const float dx = tH2x - r_x;
                const float dy = tH2y - r_y;
                const float dz = tH2z - r_z;
                const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                const float t_q_h2 = water_model::Q_H;
                partial += coulomb_energy(t_q_h2, r_q, r);
            }
            // M
            {
                const float dx = tM.x - r_x;
                const float dy = tM.y - r_y;
                const float dz = tM.z - r_z;
                const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                const float t_m = water_model::Q_M;
                partial += coulomb_energy(t_m, r_q, r);
            }
        }
#else
        // TIP3P charges are on O/H/H
        {
            // O
            {
                const float dx = tOx - r_x;
                const float dy = tOy - r_y;
                const float dz = tOz - r_z;
                const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                const float t_q_o = water_model::Q_O;
                partial += coulomb_energy(t_q_o, r_q, r);
            }
            // H1
            {
                const float dx = tH1x - r_x;
                const float dy = tH1y - r_y;
                const float dz = tH1z - r_z;
                const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                const float t_q_h1 = water_model::Q_H;
                partial += coulomb_energy(t_q_h1, r_q, r);
            }
            // H2
            {
                const float dx = tH2x - r_x;
                const float dy = tH2y - r_y;
                const float dz = tH2z - r_z;
                const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                const float t_q_h2 = water_model::Q_H;
                partial += coulomb_energy(t_q_h2, r_q, r);
            }
        }
#endif
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

    // ---- Load target O/H/H once ----
    const float tOx  = target_water[0];
    const float tOy  = target_water[1];
    const float tOz  = target_water[2];

    const float tH1x = target_water[4];
    const float tH1y = target_water[5];
    const float tH1z = target_water[6];

    const float tH2x = target_water[8];
    const float tH2y = target_water[9];
    const float tH2z = target_water[10];

#if defined(TIP4P)
    const float3u tM = tip4p_m_site(
        tOx, tOy, tOz,
        tH1x, tH1y, tH1z,
        tH2x, tH2y, tH2z,
        (double)water_model::dOM
    );
#endif

    float partial = 0.0f;

    // base in floats for this sim
    const uint32_t sim_base = sim_id * target_n_waters * (uint32_t)WATER_SIZE;

    for (uint32_t w_idx = (uint32_t)tid; w_idx < active_waters; w_idx += (uint32_t)stride) {

        if (skip_w_idx >= 0 && (int)w_idx == skip_w_idx) continue;

        const uint32_t wb = sim_base + w_idx * (uint32_t)WATER_SIZE;

        // Load other water O/H/H
        const float wOx  = waters_compact[wb + 0];
        const float wOy  = waters_compact[wb + 1];
        const float wOz  = waters_compact[wb + 2];

        const float wH1x = waters_compact[wb + 4];
        const float wH1y = waters_compact[wb + 5];
        const float wH1z = waters_compact[wb + 6];

        const float wH2x = waters_compact[wb + 8];
        const float wH2y = waters_compact[wb + 9];
        const float wH2z = waters_compact[wb + 10];

#if defined(TIP4P)
        const float3u wM = tip4p_m_site(
            wOx, wOy, wOz,
            wH1x, wH1y, wH1z,
            wH2x, wH2y, wH2z,
            (double)water_model::dOM
        );
#endif

        // -----------------------------
        // LJ: oxygen-oxygen only
        // -----------------------------
        {
            const float dx = tOx - wOx;
            const float dy = tOy - wOy;
            const float dz = tOz - wOz;
            const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);

            if (r <= RF_CUTOFF) {
                const float eps_o = water_model::EPS_O;
                const float r_min_half = water_model::RMINH_O;
                partial += lennard_jones_rmin_half(
                   eps_o, eps_o, r,
                   r_min_half, r_min_half
                );
                // optional RF here if you use it for coulomb below
            }
        }

        // -----------------------------
        // Coulomb: 3x3 between charge sites
        // TIP3P: (O,H1,H2) ; TIP4P: (H1,H2,M)
        // -----------------------------
#if defined(TIP4P)
        {
            const float tcx[3] = { tH1x, tH2x, tM.x };
            const float tcy[3] = { tH1y, tH2y, tM.y };
            const float tcz[3] = { tH1z, tH2z, tM.z };
            const float tq [3] = { water_model::Q_H, water_model::Q_H, water_model::Q_M };

            const float wcx[3] = { wH1x, wH2x, wM.x };
            const float wcy[3] = { wH1y, wH2y, wM.y };
            const float wcz[3] = { wH1z, wH2z, wM.z };
            const float wq [3] = { water_model::Q_H, water_model::Q_H, water_model::Q_M };

#pragma unroll
            for (int i = 0; i < 3; ++i) {
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const float dx = tcx[i] - wcx[j];
                    const float dy = tcy[i] - wcy[j];
                    const float dz = tcz[i] - wcz[j];
                    const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                    partial += coulomb_energy(tq[i], wq[j], r);
                }
            }
        }
#else
        {
            const float tcx[3] = { tOx,  tH1x, tH2x };
            const float tcy[3] = { tOy,  tH1y, tH2y };
            const float tcz[3] = { tOz,  tH1z, tH2z };
            const float tq [3] = { water_model::Q_O, water_model::Q_H, water_model::Q_H };

            const float wcx[3] = { wOx,  wH1x, wH2x };
            const float wcy[3] = { wOy,  wH1y, wH2y };
            const float wcz[3] = { wOz,  wH1z, wH2z };
            const float wq [3] = { water_model::Q_O, water_model::Q_H, water_model::Q_H };

#pragma unroll
            for (int i = 0; i < 3; ++i) {
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const float dx = tcx[i] - wcx[j];
                    const float dy = tcy[i] - wcy[j];
                    const float dz = tcz[i] - wcz[j];
                    const float r  = fmaxf(sqrtf(dx*dx + dy*dy + dz*dz), 1e-8f);
                    partial += coulomb_energy(tq[i], wq[j], r);
                }
            }
        }
#endif
    }

    sh_partials[tid] = partial;
    __syncthreads();

    for (int step = stride >> 1; step > 0; step >>= 1) {
        if (tid < step) sh_partials[tid] += sh_partials[tid + step];
        __syncthreads();
    }
    return sh_partials[0];
}