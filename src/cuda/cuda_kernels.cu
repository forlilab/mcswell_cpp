#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <cstdio>
#include <math_constants.h>

#include "cuda_kernels.cuh"
#include "consts.cuh"
#include "cuda_rng.cuh"
#include "water_ops.cuh"
#include "water_model.cuh"
#include "energy.cuh"

static constexpr int BLOCK_THREADS = 512;


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
// - water_atoms is compact water-major: [n_mu * target_n_waters * 12]
// - rng_states has one Philox state per mu: [n_mu]
//
// scalars layout (same as your Rust):
//   scalars[0] = last_resnum (unused here, kept for parity)
//   scalars[1] = volume
//   scalars[2] = steps
//   scalars[3] = target_n_waters
//
__global__ void run_gcmc_titration_kernel(
    const float* d_boundaries_xyz, int n_points,
    const float* d_receptor, int n_receptor_atoms,
    float* d_waters,
    std::uint32_t* d_wat_num,
    const float* d_scalars,
    float* d_snapshots,
    const float* d_mu_values, int n_mu,
    float distance_cutoff,
    std::uint32_t gcmc_steps,
    std::uint32_t equil_steps,
    std::uint32_t n_snapshots,
    std::uint32_t target_n_waters,
    curandStatePhilox4_32_10_t* states
){
    const float* __restrict__ boundaries_xyz   = d_boundaries_xyz;
    const uint32_t            boundaries_max   = (uint32_t)n_points;

    const float* __restrict__ receptor_atoms   = d_receptor;
    const uint32_t            n_receptor_atoms_u = (uint32_t)n_receptor_atoms;

    float* __restrict__ water_atoms            = d_waters;
    curandStatePhilox4_32_10_t* __restrict__ rng_states = states;
    uint32_t* __restrict__ num_waters          = d_wat_num;

    const float* __restrict__ scalars          = d_scalars;
    float* __restrict__ snapshots              = d_snapshots;
    const float* __restrict__ mu_array         = d_mu_values;

    (void)n_mu;             // one block per mu, so not needed inside
    (void)distance_cutoff;  // currently unused (ok to ignore)

    const uint32_t mu_id = (uint32_t)blockIdx.x;
    const int tid = (int)threadIdx.x;

    // ------------ unpack scalars ------------
    const uint32_t last_resnum      = (uint32_t)scalars[0]; (void)last_resnum;
    const float    volume           = scalars[1];
    const uint32_t steps            = (uint32_t)gcmc_steps;
    const uint32_t target_n_waters_local = (uint32_t)target_n_waters;
    const uint32_t n_moves          = 3;
    const float chemical_potential = mu_array[mu_id];
    const uint32_t equil_steps_local = (uint32_t)equil_steps;
    const uint32_t steps_between_snaps =
        (steps > equil_steps_local && n_snapshots > 0) ? (steps - equil_steps_local) / n_snapshots : 0;
    // B = mu*beta + log(volume/standard_volume)
    const float volume_var = volume / (float)STANDARD_VOLUME;
    const float B = chemical_potential * (float)BETA + logf(volume_var);
    // water buffer base (float index) for this mu
    const uint32_t mu_waters_base = mu_id * target_n_waters_local * (uint32_t)WATER_SIZE;

    // ------------ shared state ------------
    __shared__ float sh_trial[WATER_SIZE];
    __shared__ float sh_old[WATER_SIZE];
    __shared__ float sh_partials[512]; // blockDim.x must be 512
    __shared__ uint32_t sh_active;
    __shared__ uint32_t sh_move_type;
    __shared__ uint32_t sh_tgt_idx;
    __shared__ float    sh_u;
    __shared__ uint32_t sh_snap_id;

    // ------------ RNG state (thread 0) ------------
    curandStatePhilox4_32_10_t st;
    if (tid == 0) {
        st = rng_states[mu_id];
        sh_active = num_waters[mu_id];
        sh_snap_id = 0;
    }
    __syncthreads();

    float trial_water[WATER_SIZE];

    for (uint32_t step = 0; step < steps; ++step) {
        if (tid == 0) {
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

            const float angle_rad = rand_range(&st, -180.0f, 180.0f) * (CUDART_PI_F / 180.0f);
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

                randomize_water_axis_angle(ax, ay, az, angle_rad, w);
                propose_insertion_xyz(ins_x, ins_y, ins_z, (float)possible_resnum, w);

#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) { sh_trial[i]=w[i]; trial_water[i]=w[i]; }
#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) sh_old[i]=0.0f;

            } else if (move_type == 1) {
                const uint32_t src = mu_waters_base + tgt_idx * (uint32_t)WATER_SIZE;
                load_water_from_array(water_atoms, src, trial_water);
#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) sh_trial[i]=trial_water[i];
#pragma unroll
                for (int i=0;i<WATER_SIZE;++i) sh_old[i]=0.0f;

            } else {
                const uint32_t src = mu_waters_base + tgt_idx * (uint32_t)WATER_SIZE;

                float oldw[WATER_SIZE];
                load_water_from_array(water_atoms, src, oldw);

                float neww[WATER_SIZE];

                const bool ok = propose_perturbation_cloud(
                    oldw,
                    boundaries_xyz, boundaries_max,
                    &st,
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

        const uint32_t active = sh_active;
        const uint32_t move_type = sh_move_type;
        int skip_w_idx = -1;
        if (move_type == 2 || move_type == 1) skip_w_idx = (int)sh_tgt_idx;

        const float e_rec_new = eval_energy_with_receptor(
            receptor_atoms, sh_trial, n_receptor_atoms_u, sh_partials
        );
        const float e_wat_new = eval_energy_with_waters(
            water_atoms, sh_trial, mu_id, active, target_n_waters_local, skip_w_idx, sh_partials
        );

        float e_rec_old = 0.0f;
        float e_wat_old = 0.0f;

        if (move_type == 2) {
            e_rec_old = eval_energy_with_receptor(
                receptor_atoms, sh_old, n_receptor_atoms_u, sh_partials
            );
            e_wat_old = eval_energy_with_waters(
                water_atoms, sh_old, mu_id, active, target_n_waters_local, skip_w_idx, sh_partials
            );
        }

        __syncthreads();

        if (tid == 0) {
            uint32_t active_local = sh_active;
            const float u = sh_u;

            if (move_type == 0) {
                const float new_energy = e_rec_new + e_wat_new;
                const float delta_e = new_energy;

                const float pref = (1.0f / (float)(active_local + 1u)) * expf(B);
                const float p_acc = fminf(pref * expf(-(float)BETA * delta_e), 1.0f);

                if (u < p_acc && active_local < target_n_waters_local) {
                    const uint32_t dst = mu_waters_base + active_local * (uint32_t)WATER_SIZE;
                    copy_water_to_array(trial_water, water_atoms, dst);
                    active_local += 1u;
                    num_waters[mu_id] += 1u;
                }

            } else if (move_type == 1 && active_local > 0) {
                const float contrib = e_rec_new + e_wat_new;
                const float delta_e = -contrib;

                const float pref = (float)active_local * expf(-B);
                const float p_acc = fminf(pref * expf(-(float)BETA * delta_e), 1.0f);
                if (u < p_acc) {
                    const uint32_t del_idx = sh_tgt_idx;
                    const uint32_t dst  = mu_waters_base + del_idx * (uint32_t)WATER_SIZE;
                    const uint32_t last = mu_waters_base + (active_local - 1u) * (uint32_t)WATER_SIZE;

                    if (del_idx + 1u < active_local) {
                        move_water_in_array(water_atoms, last, dst);
                    }
                    clear_water_in_array(water_atoms, last);
                    active_local -= 1u;
                    num_waters[mu_id] -= 1u;
                }

            } else if (move_type == 2 && active_local > 0) {
                const float oldE = e_rec_old + e_wat_old;
                const float newE = e_rec_new + e_wat_new;
                const float delta_e = newE - oldE;

                const float p_acc = fminf(expf(-(float)BETA * delta_e), 1.0f);

                if (u < p_acc) {
                    const uint32_t tgt = sh_tgt_idx;
                    const uint32_t dst = mu_waters_base + tgt * (uint32_t)WATER_SIZE;
                    copy_water_to_array(trial_water, water_atoms, dst);
                }
            }

            sh_active = active_local;

            if (step > equil_steps_local && steps_between_snaps > 0) {
                if (((step - equil_steps_local) % steps_between_snaps) == 0 && sh_snap_id < n_snapshots) {

                    const uint32_t base =
                        mu_id * n_snapshots * target_n_waters_local * (uint32_t)WATER_SIZE +
                        sh_snap_id * target_n_waters_local * (uint32_t)WATER_SIZE;

                    uint32_t out_idx = base;

                    for (uint32_t w = 0; w < active_local; ++w) {
                        const uint32_t src = mu_waters_base + w * (uint32_t)WATER_SIZE;
#pragma unroll
                        for (int i = 0; i < WATER_SIZE; ++i) snapshots[out_idx++] = water_atoms[src + i];
                    }

                    for (uint32_t w = active_local; w < target_n_waters_local; ++w) {
#pragma unroll
                        for (int i = 0; i < WATER_SIZE; ++i) snapshots[out_idx++] = 0.0f;
                    }

                    sh_snap_id += 1u;
                }
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        rng_states[mu_id] = st;
        num_waters[mu_id] = sh_active;
    }
}
