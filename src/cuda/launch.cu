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

#include <curand_kernel.h>

#include "cuda/launch.cuh"
#include "cuda/cuda_kernels.cuh"

std::size_t dummy_rng_state_bytes() {
    return sizeof(curandStatePhilox4_32_10_t);
}

void init_rng_launch(void* d_states, std::uint64_t seed, int n_mu) {
    auto* states = static_cast<curandStatePhilox4_32_10_t*>(d_states);
    int threads = 256;
    int blocks  = (n_mu + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(states, seed, n_mu);
}

// ----------------------------
// NEW: main titration launcher
// ----------------------------
void run_gcmc_titration_launch(
    const float* d_boundaries_xyz,
    int n_points,

    const float* d_receptor,
    int n_receptor_atoms,

    float* d_waters,
    std::uint32_t* d_wat_num,

    const float* d_scalars,
    float* d_snapshots,

    const float* d_mu_values,
    int n_mu,

    float distance_cutoff,
    std::uint32_t gcmc_steps,
    std::uint32_t equil_steps,
    std::uint32_t n_snapshots,
    std::uint32_t target_n_waters,

    void* d_states
) {
    auto* states = static_cast<curandStatePhilox4_32_10_t*>(d_states);
    int blocks  = n_mu;
    int threads = 512;

    run_gcmc_titration_kernel<<<blocks, threads>>>(
        d_boundaries_xyz, n_points,
        d_receptor, n_receptor_atoms,
        d_waters, d_wat_num,
        d_scalars,
        d_snapshots,
        d_mu_values, n_mu,
        distance_cutoff,
        gcmc_steps,
        equil_steps,
        n_snapshots,
        target_n_waters,
        states
    );
}
