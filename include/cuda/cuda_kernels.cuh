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
#include <cstdint>

#include "cuda/titration_state.cuh"

#ifdef __CUDACC__
// Only visible to NVCC compilation units (.cu)
#include <curand_kernel.h>

// Device kernels can use the real type here
__global__ void init_rng(curandStatePhilox4_32_10_t* states, std::uint64_t seed, int n_mu);

// The d_diag_* arrays collect per-(mu, window) sampling diagnostics; see
// include/analysis/sampler_diagnostics.hpp for their layout and meaning.
// Pass n_diag_windows == 0 (or null pointers) to switch the instrumentation
// off. Collecting it consumes no random numbers, so an instrumented run
// reproduces an uninstrumented one with the same seed exactly.
//
// The kernel is resumable: see include/cuda/titration_state.cuh and the
// comment on its definition.
__global__ void run_gcmc_titration_kernel(
    const float* d_boundaries_xyz, int n_points,
    const float* d_receptor, int n_receptor_atoms,
    float* d_waters,
    std::uint32_t* d_wat_num,
    const float* d_scalars,
    const float* d_mu_values, int n_mu,
    float distance_cutoff,
    std::uint32_t gcmc_steps,
    std::uint32_t step_end,                      // run no further than this step in this launch
    std::uint32_t equil_steps,
    std::uint32_t n_snapshots,
    std::uint32_t water_capacity,                // per-mu stride of d_waters, in waters
    bool grow_on_full,                           // stop at water_capacity instead of refusing insertions
    curandStatePhilox4_32_10_t* states,
    TitrationCarry carry,
    SnapshotArena arena,
    unsigned long long* d_diag_proposed,         // [n_mu * n_diag_windows * 3]
    unsigned long long* d_diag_accepted,         // [n_mu * n_diag_windows * 3]
    unsigned long long* d_diag_capacity_blocked, // [n_mu * n_diag_windows]
    unsigned long long* d_diag_n_sum,            // [n_mu * n_diag_windows]
    std::uint32_t* d_diag_n_end,                 // [n_mu * n_diag_windows]
    float* d_diag_energy_end,                    // [n_mu * n_diag_windows]
    std::uint32_t n_diag_windows,
    std::uint32_t steps_per_diag_window,
    std::uint32_t diag_energy_stride,          // evaluate the energy every Nth window
    // Dense occupancy mask over the insertion lattice. When non-null,
    // displacement moves that would take a water out of the sampled region are
    // rejected, which makes insertion and deletion reciprocal. Null leaves
    // displacement unconstrained (the historical behaviour).
    const unsigned char* d_cavity_mask,
    int mask_nx, int mask_ny, int mask_nz,
    float mask_ox, float mask_oy, float mask_oz,
    float mask_spacing
);

#endif // __CUDACC__
