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

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <cstdio>
#include <math_constants.h>

#include "cuda/cuda_kernels.cuh"
#include "cuda/consts.cuh"
#include "cuda/cuda_rng.cuh"
#include "cuda/water_ops.cuh"
#include "cuda/water_model.cuh"
#include "cuda/energy.cuh"

static constexpr int BLOCK_THREADS = 512;

// min(1, pref * exp(-beta * delta_e)), with a non-finite result rejecting.
// Insertion puts the oxygen exactly on a lattice point, so it can land on a
// water still sitting there: the oxygen-oxygen distance is then 0, the LJ
// terms overflow to inf - inf, and delta_e is NaN. fminf returns its non-NaN
// argument, so a bare fminf(NaN, 1) would accept that overlap with
// probability 1. `p >= 0` is false for NaN.
__device__ __forceinline__ float metropolis_probability(float pref, float delta_e) {
    const float p = pref * expf(-(float)BETA * delta_e);
    return (p >= 0.0f) ? fminf(p, 1.0f) : 0.0f;
}


__global__ void init_rng(curandStatePhilox4_32_10_t* states, std::uint64_t seed, int n_mu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_mu) {
        // sequence = tid so each mu has independent stream
        curand_init((unsigned long long)seed, (unsigned long long)tid, 0ULL, &states[tid]);
    }
}

// Assumes:
// - gridDim.x == n_mu (one sim per block)
// - blockDim.x == 512 (power-of-two for reduction)
// - boundaries_xyz is a cloud: [boundaries_max*3] (x,y,z per point)
// - water_atoms is compact water-major: [n_mu * water_capacity * 12]
// - rng_states has one Philox state per mu: [n_mu]
//
// scalars layout (same as your Rust):
//   scalars[0] = last_resnum (unused here, kept for parity)
//   scalars[1] = volume
//   scalars[2] = steps
//   scalars[3] = initial water capacity (unused here, kept for parity)
//
// Each block resumes at carry.step[mu] and runs up to step_end (the host runs
// the titration in chunks), or until it has to stop for the host (see
// include/cuda/titration_state.cuh): with
// grow_on_full set, reaching water_capacity stops the block so the buffer can
// be grown, instead of refusing the insertion. Stopping happens before the
// step draws anything from the RNG, so a run that is stopped and resumed any
// number of times follows exactly the same chain as one uninterrupted launch.
__global__ void run_gcmc_titration_kernel(
    const float* d_boundaries_xyz, int n_points,
    const float* d_receptor, int n_receptor_atoms,
    float* d_waters,
    std::uint32_t* d_wat_num,
    const float* d_scalars,
    const float* d_mu_values, int n_mu,
    float distance_cutoff,
    std::uint32_t gcmc_steps,
    std::uint32_t step_end,
    std::uint32_t equil_steps,
    std::uint32_t n_snapshots,
    std::uint32_t water_capacity,
    bool grow_on_full,
    float max_rotation_rad,
    curandStatePhilox4_32_10_t* states,
    TitrationCarry carry,
    SnapshotArena arena,
    unsigned long long* d_diag_proposed,
    unsigned long long* d_diag_accepted,
    unsigned long long* d_diag_capacity_blocked,
    unsigned long long* d_diag_n_sum,
    std::uint32_t* d_diag_n_end,
    float* d_diag_energy_end,
    std::uint32_t n_diag_windows,
    std::uint32_t steps_per_diag_window,
    std::uint32_t diag_energy_stride,
    const unsigned char* d_cavity_mask,
    int mask_nx, int mask_ny, int mask_nz,
    float mask_ox, float mask_oy, float mask_oz,
    float mask_spacing
){
    const float* __restrict__ boundaries_xyz   = d_boundaries_xyz;
    const uint32_t            boundaries_max   = (uint32_t)n_points;

    const float* __restrict__ receptor_atoms   = d_receptor;
    const uint32_t            n_receptor_atoms_u = (uint32_t)n_receptor_atoms;

    float* __restrict__ water_atoms            = d_waters;
    curandStatePhilox4_32_10_t* __restrict__ rng_states = states;
    uint32_t* __restrict__ num_waters          = d_wat_num;

    const float* __restrict__ scalars          = d_scalars;
    const float* __restrict__ mu_array         = d_mu_values;

    (void)n_mu;             // one block per mu, so not needed inside
    (void)distance_cutoff;  // currently unused (ok to ignore)

    const uint32_t mu_id = (uint32_t)blockIdx.x;
    const int tid = (int)threadIdx.x;

    // ------------ unpack scalars ------------
    const uint32_t last_resnum      = (uint32_t)scalars[0]; (void)last_resnum;
    const float    volume           = scalars[1];
    const uint32_t steps            = (uint32_t)gcmc_steps;
    const uint32_t water_capacity_local = (uint32_t)water_capacity;
    const uint32_t n_moves          = 3;
    const float chemical_potential = mu_array[mu_id];
    const uint32_t equil_steps_local = (uint32_t)equil_steps;
    const uint32_t steps_between_snaps =
        (steps > equil_steps_local && n_snapshots > 0) ? (steps - equil_steps_local) / n_snapshots : 0;
    // B = mu*beta + log(volume/standard_volume)
    const float volume_var = volume / (float)STANDARD_VOLUME;
    const float B = chemical_potential * (float)BETA + logf(volume_var);
    // water buffer base (float index) for this mu
    const size_t mu_waters_base = (size_t)mu_id * water_capacity_local * (size_t)WATER_SIZE;
    // this mu's slice of the snapshot staging buffer
    float* __restrict__ snap_out = arena.data + (size_t)mu_id * arena.floats_per_mu;
    const uint32_t step_begin = carry.step[mu_id];
    const uint32_t step_stop  = (step_end < steps) ? step_end : steps;

    // ------------ shared state ------------
    __shared__ float sh_trial[WATER_SIZE];
    __shared__ float sh_old[WATER_SIZE];
    // Staging buffer for the diagnostics energy sweep. eval_energy_with_waters
    // marks both the water buffer and the target water __restrict__, so the
    // target must not alias the buffer -- passing &water_atoms[k] directly is
    // undefined behaviour and in practice miscompiles into garbage energies.
    __shared__ float sh_diag[WATER_SIZE];
    __shared__ float sh_partials[512]; // blockDim.x must be 512
    __shared__ uint32_t sh_active;
    __shared__ uint32_t sh_move_type;
    __shared__ uint32_t sh_tgt_idx;
    __shared__ float    sh_u;
    __shared__ uint32_t sh_snap_id;
    __shared__ uint32_t sh_exit;

    // ------------ RNG state (thread 0) ------------
    curandStatePhilox4_32_10_t st;
    unsigned long long snap_used = 0ull;  // floats in snap_out (thread 0)
    if (tid == 0) {
        st = rng_states[mu_id];
        sh_active = num_waters[mu_id];
        sh_snap_id = carry.snap_id[mu_id];
        sh_exit = TITRATION_DONE;
        snap_used = arena.used[mu_id];
    }
    __syncthreads();

    // ------------ sampling diagnostics ------------
    // Per-window counters live in thread 0's registers and are flushed to
    // global memory at each window boundary. None of this draws from `st`, so
    // the Markov chain is bit-for-bit identical with diagnostics on or off.
    const bool diag_on = (n_diag_windows > 0u) && (steps_per_diag_window > 0u) &&
                         (d_diag_proposed != nullptr);
    unsigned long long w_proposed[3] = {0ull, 0ull, 0ull};
    unsigned long long w_accepted[3] = {0ull, 0ull, 0ull};
    unsigned long long w_capacity_blocked = 0ull;
    unsigned long long w_n_sum = 0ull;
    // A window can straddle a relaunch, so its partial counts are carried.
    if (tid == 0 && diag_on) {
        const unsigned long long* c = carry.diag + (size_t)mu_id * DIAG_CARRY_SLOTS;
#pragma unroll
        for (int m = 0; m < 3; ++m) { w_proposed[m] = c[m]; w_accepted[m] = c[3 + m]; }
        w_capacity_blocked = c[6];
        w_n_sum = c[7];
    }

    float trial_water[WATER_SIZE];

    uint32_t step = step_begin;
    for (; step < step_stop; ++step) {
        if (tid == 0) {
            const uint32_t active = sh_active;

            // ------------ stop for the host? ------------
            // Decided before this step draws any random number, so resuming
            // here later replays the step exactly. A step can add at most one
            // water, so a snapshot taken at its end needs room for active + 1.
            const bool snap_step =
                step >= equil_steps_local && steps_between_snaps > 0 &&
                ((step - equil_steps_local) % steps_between_snaps) == 0 && sh_snap_id < n_snapshots;
            if (grow_on_full && active >= water_capacity_local) {
                sh_exit = TITRATION_WATER_BUFFER_FULL;
            } else if (snap_step &&
                       snap_used + (unsigned long long)(active + 1u) * WATER_SIZE > arena.floats_per_mu) {
                sh_exit = TITRATION_SNAPSHOT_BUFFER_FULL;
            }
        }
        // Every thread already waits at the barrier after the proposal, so
        // the stop decision rides on it instead of costing a barrier of its own.
        if (tid == 0 && sh_exit == TITRATION_DONE) {
            const uint32_t active = sh_active;

            const uint32_t rnd_pos_idx = (uint32_t)rand_int(&st, (int)boundaries_max);
            const uint32_t bi = rnd_pos_idx * 3u;
            const float ins_x = boundaries_xyz[bi + 0];
            const float ins_y = boundaries_xyz[bi + 1];
            const float ins_z = boundaries_xyz[bi + 2];

            const float dx = rand_range(&st, -0.3f, 0.3f);
            const float dy = rand_range(&st, -0.3f, 0.3f);
            const float dz = rand_range(&st, -0.3f, 0.3f);

            const float ax = randu(&st);
            const float ay = randu(&st);
            const float az = randu(&st);

            // Displacement rotation angle, uniform in +-max_rotation_rad. The
            // proposal stays symmetric (the reverse move is the same axis with
            // the opposite angle) for any cap, so the cap only tunes the
            // acceptance rate: a full +-180 deg reorientation inside a
            // hydrogen-bonded network is almost always rejected.
            const float angle_rad = rand_range(&st, -max_rotation_rad, max_rotation_rad);
            const float u = randu(&st);

            const uint32_t tgt_idx = (active > 0) ? (uint32_t)rand_int(&st, (int)active) : 0u;

            int move_type = (active == 0) ? 0 : rand_int(&st, n_moves);

            sh_move_type = move_type;
            sh_tgt_idx   = tgt_idx;
            sh_u         = u;

            if (move_type == 0) {
                const uint32_t possible_resnum = sh_active + 1u;

                float w[WATER_SIZE];
                water_model::create_water_std(0u, w);

                // Same three uniforms the displacement branch uses as its
                // rotation axis, so the random stream is unchanged.
                randomize_water_uniform(ax, ay, az, w);
                propose_insertion_xyz(ins_x, ins_y, ins_z, (float)possible_resnum, w);

#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) { sh_trial[i]=w[i]; trial_water[i]=w[i]; }
#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) sh_old[i]=0.0f;

            } else if (move_type == 1) {
                const size_t src = mu_waters_base + (size_t)tgt_idx * WATER_SIZE;
                load_water_from_array(water_atoms, src, trial_water);
#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) sh_trial[i]=trial_water[i];
#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) sh_old[i]=0.0f;

            } else {
                const size_t src = mu_waters_base + (size_t)tgt_idx * WATER_SIZE;

                float oldw[WATER_SIZE];
                load_water_from_array(water_atoms, src, oldw);

                float neww[WATER_SIZE];

                const bool ok = propose_perturbation_cloud(
                    oldw,
                    boundaries_max,
                    &st,
                    d_cavity_mask,
                    mask_nx, mask_ny, mask_nz,
                    mask_ox, mask_oy, mask_oz,
                    mask_spacing,
                    dx, dy, dz,
                    ax, ay, az,
                    angle_rad,
                    neww
                );

                if (!ok) {
                    sh_u = 2.0f;
#pragma unroll
                    for (int i=0;i<WATER_SIZE;++i) { sh_old[i]=oldw[i]; sh_trial[i]=oldw[i]; trial_water[i]=oldw[i]; }
                } else {
#pragma unroll
                    for (int i=0;i<WATER_SIZE;++i) { sh_old[i]=oldw[i]; sh_trial[i]=neww[i]; trial_water[i]=neww[i]; }
                }
            }
        }

        __syncthreads();

        // Block-uniform: sh_exit is only written by thread 0 before the
        // barrier above, so every thread leaves together.
        if (sh_exit != TITRATION_DONE) break;

        const uint32_t active = sh_active;
        const uint32_t move_type = sh_move_type;
        int skip_w_idx = -1;
        if (move_type == 2 || move_type == 1) skip_w_idx = (int)sh_tgt_idx;

        const float e_rec_new = eval_energy_with_receptor(
            receptor_atoms, sh_trial, n_receptor_atoms_u, sh_partials
        );
        const float e_wat_new = eval_energy_with_waters(
            water_atoms, sh_trial, mu_id, active, water_capacity_local, skip_w_idx, sh_partials
        );

        float e_rec_old = 0.0f;
        float e_wat_old = 0.0f;

        if (move_type == 2) {
            e_rec_old = eval_energy_with_receptor(
                receptor_atoms, sh_old, n_receptor_atoms_u, sh_partials
            );
            e_wat_old = eval_energy_with_waters(
                water_atoms, sh_old, mu_id, active, water_capacity_local, skip_w_idx, sh_partials
            );
        }

        __syncthreads();

        if (tid == 0) {
            uint32_t active_local = sh_active;
            const float u = sh_u;

            if (diag_on) w_proposed[move_type] += 1ull;

            // Move selection is not uniform: with no water present the step
            // always proposes an insertion, otherwise each move type is picked
            // with probability 1/n_moves. Detailed balance between N = 0 and
            // N = 1 needs that asymmetry in the acceptance ratio: insertion
            // into an empty region is proposed n_moves times more often than
            // the deletion that empties it, so the former is scaled by
            // 1/n_moves and the latter by n_moves. Without it P(N>=1)/P(N=0)
            // comes out n_moves (3x) too large.
            if (move_type == 0) {
                const float new_energy = e_rec_new + e_wat_new;
                const float delta_e = new_energy;

                float pref = (1.0f / (float)(active_local + 1u)) * expf(B);
                if (active_local == 0u) pref /= (float)n_moves;
                const float p_acc = metropolis_probability(pref, delta_e);

                // With grow_on_full the block never gets here at capacity
                // (it stops at the top of the step instead), so this guard
                // only bites once the run's max_waters limit is reached.
                if (u < p_acc && active_local < water_capacity_local) {
                    const size_t dst = mu_waters_base + (size_t)active_local * WATER_SIZE;
                    copy_water_to_array(trial_water, water_atoms, dst);
                    active_local += 1u;
                    num_waters[mu_id] += 1u;
                    if (diag_on) w_accepted[0] += 1ull;
                } else if (diag_on && u < p_acc) {
                    // Metropolis said yes, the max_waters limit said no.
                    // Tracked apart from genuine rejections: a run where this is
                    // not negligible is limit-bound, not badly converged.
                    w_capacity_blocked += 1ull;
                }

            } else if (move_type == 1 && active_local > 0) {
                const float contrib = e_rec_new + e_wat_new;
                const float delta_e = -contrib;

                float pref = (float)active_local * expf(-B);
                if (active_local == 1u) pref *= (float)n_moves;  // see the insertion branch
                const float p_acc = metropolis_probability(pref, delta_e);
                if (u < p_acc) {
                    const uint32_t del_idx = sh_tgt_idx;
                    const size_t dst  = mu_waters_base + (size_t)del_idx * WATER_SIZE;
                    const size_t last = mu_waters_base + (size_t)(active_local - 1u) * WATER_SIZE;

                    if (del_idx + 1u < active_local) {
                        move_water_in_array(water_atoms, last, dst);
                    }
                    clear_water_in_array(water_atoms, last);
                    active_local -= 1u;
                    num_waters[mu_id] -= 1u;
                    if (diag_on) w_accepted[1] += 1ull;
                }

            } else if (move_type == 2 && active_local > 0) {
                const float oldE = e_rec_old + e_wat_old;
                const float newE = e_rec_new + e_wat_new;
                const float delta_e = newE - oldE;

                const float p_acc = metropolis_probability(1.0f, delta_e);

                if (u < p_acc) {
                    const uint32_t tgt = sh_tgt_idx;
                    const size_t dst = mu_waters_base + (size_t)tgt * WATER_SIZE;
                    copy_water_to_array(trial_water, water_atoms, dst);
                    if (diag_on) w_accepted[2] += 1ull;
                }
            }

            sh_active = active_local;
            if (diag_on) w_n_sum += (unsigned long long)active_local;

            // `step >= equil`, not `>`: with a strict `>` the first snapshot
            // offset is 1 instead of 0, the offsets 0, S, ..., (n-1)S no longer
            // all fit in the production phase, and the last snapshot slot is
            // left at its memset zeros -- which post-processing then reads back
            // as a real frame containing no waters. That one phantom empty
            // frame biased every site occupancy low by 1/n_snapshots and
            // dominated the reported N_region_std.
            if (step >= equil_steps_local && steps_between_snaps > 0) {
                if (((step - equil_steps_local) % steps_between_snaps) == 0 && sh_snap_id < n_snapshots) {
                    // Appended unpadded: the frame is exactly active_local
                    // waters, and its size goes to arena.counts. The stop
                    // check at the top of the step guaranteed the room.
                    const size_t n_floats = (size_t)active_local * WATER_SIZE;
                    const float* src = water_atoms + mu_waters_base;
                    float* dst = snap_out + snap_used;
                    for (size_t i = 0; i < n_floats; ++i) dst[i] = src[i];
                    snap_used += n_floats;
                    arena.counts[(size_t)mu_id * n_snapshots + sh_snap_id] = active_local;

                    sh_snap_id += 1u;
                }
            }
        }

        __syncthreads();

        // ------------ diagnostics flush ------------
        // Block-uniform: `step`, `steps_per_diag_window` and `steps` are the
        // same for every thread, so the whole block enters together and the
        // energy reductions below (which carry their own __syncthreads) are
        // safe.
        if (diag_on) {
            const bool flush = (((step + 1u) % steps_per_diag_window) == 0u) || ((step + 1u) == steps);
            if (flush) {
                uint32_t w_idx = step / steps_per_diag_window;
                if (w_idx >= n_diag_windows) w_idx = n_diag_windows - 1u;

                // Total interaction energy of the water subsystem at the end of
                // the window: every water against the receptor, plus every
                // water-water pair once (each pair is visited twice by the loop,
                // hence the 1/2). The constant receptor-receptor term is left
                // out, so this is a relative trace, not an absolute energy.
                //
                // This is the only part of the instrumentation that costs real
                // work -- O(n_waters * n_receptor_atoms) per evaluation -- so
                // diag_energy_stride decouples it from the counter windows. A
                // fine-grained occupancy trace (short windows, for measuring
                // correlation times in steps) can then skip most of them; the
                // skipped windows get NaN rather than a stale value.
                const uint32_t n_active = sh_active;
                const bool want_energy = (diag_energy_stride <= 1u) ||
                                         ((w_idx % diag_energy_stride) == 0u) ||
                                         ((step + 1u) == steps);
                float e_total = CUDART_NAN_F;
                if (want_energy) {
                    e_total = 0.0f;
                    for (uint32_t k = 0; k < n_active; ++k) {
                        const size_t src = mu_waters_base + (size_t)k * WATER_SIZE;
                        if (tid < WATER_SIZE) sh_diag[tid] = water_atoms[src + tid];
                        __syncthreads();

                        e_total += eval_energy_with_receptor(
                            receptor_atoms, sh_diag, n_receptor_atoms_u, sh_partials
                        );
                        e_total += 0.5f * eval_energy_with_waters(
                            water_atoms, sh_diag, mu_id, n_active,
                            water_capacity_local, (int)k, sh_partials
                        );
                    }
                }

                if (tid == 0) {
                    const uint32_t flat = mu_id * n_diag_windows + w_idx;
                    const uint32_t flat3 = flat * 3u;
#pragma unroll
                    for (int m = 0; m < 3; ++m) {
                        d_diag_proposed[flat3 + m] = w_proposed[m];
                        d_diag_accepted[flat3 + m] = w_accepted[m];
                        w_proposed[m] = 0ull;
                        w_accepted[m] = 0ull;
                    }
                    d_diag_capacity_blocked[flat] = w_capacity_blocked;
                    d_diag_n_sum[flat] = w_n_sum;
                    d_diag_n_end[flat] = n_active;
                    d_diag_energy_end[flat] = e_total;
                    w_capacity_blocked = 0ull;
                    w_n_sum = 0ull;
                }

                __syncthreads();
            }
        }
    }

    // Save everything the next launch needs to resume at `step` (== steps
    // when the chain ran to completion).
    if (tid == 0) {
        rng_states[mu_id] = st;
        num_waters[mu_id] = sh_active;
        carry.step[mu_id] = step;
        carry.snap_id[mu_id] = sh_snap_id;
        carry.exit_reason[mu_id] = sh_exit;
        arena.used[mu_id] = snap_used;
        if (diag_on) {
            unsigned long long* c = carry.diag + (size_t)mu_id * DIAG_CARRY_SLOTS;
#pragma unroll
            for (int m = 0; m < 3; ++m) { c[m] = w_proposed[m]; c[3 + m] = w_accepted[m]; }
            c[6] = w_capacity_blocked;
            c[7] = w_n_sum;
        }
    }
}
