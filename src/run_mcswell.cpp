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

#include <thread>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "snapshots_io.hpp"
#include "atom.hpp"
#include "run_mcswell.hpp"
#include "cuda/launch.cuh"
#include "cuda/consts.cuh"

namespace {
inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("[CUDA] ") + what + ": " + cudaGetErrorString(e));
    }
}

} // namespace

// - build receptor_atoms_buffer (flattened float)
// - upload boundaries (insertion_points_xyz)
// - allocate waters buffer and wat_num buffer
// - allocate the snapshot staging buffer and the per-mu resume state
// - allocate mu array + scalars array
// - create RNG states (1 per mu)
// - run the CUDA kernel (1 block per mu, 512 threads) chunk by chunk,
//   draining snapshots and growing the water buffer between launches
// - assemble the snapshots and return
TitrationResult run_mcswell_gpu_titration(
    const std::vector<Atom>& receptor_points,
    const float* insertion_points_xyz,   // length = 3*n_points
    std::size_t n_points,
    float distance_cutoff,               // kept for API; pass into kernel if/when needed
    float spacing,
    std::size_t gcmc_steps,
    std::size_t equilibration_steps,
    const std::string& save_path,        // only used if dump_debug_pdbs is set
    const std::vector<float>& mu_values,
    std::size_t n_snapshots,
    std::uint64_t seed,
    bool dump_debug_pdbs,
    std::size_t n_diagnostic_windows,
    std::size_t diagnostic_energy_stride,
    const InsertionGrid* cavity_grid,
    bool constrain_displacement,
    std::size_t max_waters,
    std::size_t snapshot_buffer_bytes
) {
    // -------------------------
    // Input checks
    // -------------------------
    if (spacing <= 0.0f) throw std::invalid_argument("spacing must be > 0");
    if (!insertion_points_xyz) throw std::invalid_argument("insertion_points_xyz is null");
    if (n_points == 0) throw std::invalid_argument("n_points must be > 0");
    if (mu_values.empty()) throw std::invalid_argument("mu_values is empty");
    if (n_snapshots == 0) throw std::invalid_argument("n_snapshots must be > 0");
    if (receptor_points.empty()) throw std::invalid_argument("receptor_points is empty");
    if (equilibration_steps > gcmc_steps) {
        throw std::invalid_argument("equilibration_steps must be <= gcmc_steps");
    }
    // The kernel's snapshot stride is (gcmc_steps - equilibration_steps) /
    // n_snapshots in integer arithmetic. Too short a production phase makes
    // that 0, which silently skips every snapshot and hands the analyses
    // n_snapshots all-zero frames -- i.e. a titration curve pinned at N = 0
    // rather than an error.
    if (gcmc_steps - equilibration_steps < n_snapshots) {
        throw std::invalid_argument(
            "production phase (gcmc_steps - equilibration_steps = " +
            std::to_string(gcmc_steps - equilibration_steps) +
            ") must be at least n_snapshots (" + std::to_string(n_snapshots) +
            ") so that every snapshot slot gets a frame");
    }

    const int n_mu = static_cast<int>(mu_values.size());

    // -------------------------
    // Receptor_atoms_buffer
    // -------------------------
    // Rust pushes 7 floats per atom:
    //   x, y, z, charge, epsilon, rmin_half, residue_number
    std::vector<float> h_receptor;
    h_receptor.reserve(receptor_points.size() * ATOM_FEATURES);

    for (const auto& a : receptor_points) {
        h_receptor.push_back(static_cast<float>(a.coords[0]));
        h_receptor.push_back(static_cast<float>(a.coords[1]));
        h_receptor.push_back(static_cast<float>(a.coords[2]));
        h_receptor.push_back(static_cast<float>(a.charge));
        h_receptor.push_back(static_cast<float>(a.epsilon));
        h_receptor.push_back(static_cast<float>(a.rmin_half));
        h_receptor.push_back(static_cast<float>(a.residue_number));
    }

    const std::size_t n_receptor_atoms = receptor_points.size();
    const double voxel_volume = static_cast<double>(spacing) * spacing * spacing;
    const double total_volume = voxel_volume * static_cast<double>(n_points);

    // Water capacity per mu window. The bulk-density count is only the
    // starting size: unless max_waters says otherwise, the buffer doubles
    // whenever a window fills it, so it never limits the chain.
    std::size_t water_capacity = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::floor(total_volume * BULK_WATER_DENSITY)));
    if (max_waters > 0) water_capacity = std::min(water_capacity, max_waters);
    const std::size_t initial_water_capacity = water_capacity;
    // The kernel indexes waters with 32-bit counts.
    const std::size_t water_capacity_limit = std::numeric_limits<std::uint32_t>::max() / WATER_SIZE;

    // The run goes to the GPU in chunks of steps. Between chunks the host can
    // grow the water buffer for a window that filled it, and a window that
    // stopped for that mid-chunk only idles until the end of the chunk, so
    // more chunks bound that idle time more tightly. One launch per 2% of the
    // run makes the launch overhead negligible.
    constexpr std::size_t TARGET_CHUNKS = 50;
    const std::size_t chunk_steps = std::max<std::size_t>(1, (gcmc_steps + TARGET_CHUNKS - 1) / TARGET_CHUNKS);

    // Snapshot staging buffer: enough for one chunk's snapshots at the
    // current capacity when snapshot_buffer_bytes allows, and never less than
    // one frame (a window can always make progress after a drain). A window
    // that would overflow it mid-chunk stops and is resumed after a drain.
    const std::size_t steps_between_snaps = (gcmc_steps - equilibration_steps) / n_snapshots;
    const std::size_t snaps_per_chunk = std::min(n_snapshots, chunk_steps / steps_between_snaps + 1);
    const auto arena_floats_for = [&](std::size_t capacity) {
        const std::size_t frame = (capacity + 1) * WATER_SIZE;  // a step adds at most one water
        const std::size_t budget = snapshot_buffer_bytes / sizeof(float) / static_cast<std::size_t>(n_mu);
        return std::max(frame, std::min(budget, snaps_per_chunk * frame));
    };
    std::size_t arena_floats_per_mu = arena_floats_for(water_capacity);

    // Diagnostics windowing. Windows tile the whole run, equilibration
    // included, so the equilibration/production split stays a host-side slice.
    // steps_per_window is rounded up so the windows cover every step, then the
    // window count is recomputed from it; for the usual round step counts this
    // leaves steps_per_window dividing equilibration_steps, so the split falls
    // exactly on a window boundary.
    std::size_t n_diag_windows = 0;
    std::uint64_t steps_per_diag_window = 0;
    if (n_diagnostic_windows > 0 && gcmc_steps > 0) {
        n_diag_windows = std::min(n_diagnostic_windows, gcmc_steps);
        steps_per_diag_window = (gcmc_steps + n_diag_windows - 1) / n_diag_windows;
        n_diag_windows = static_cast<std::size_t>(
            (gcmc_steps + steps_per_diag_window - 1) / steps_per_diag_window);
    }
    const std::size_t diag_len = static_cast<std::size_t>(n_mu) * n_diag_windows;
    const std::size_t diag_len3 = diag_len * mcswell::analysis::N_MOVE_TYPES;

    std::vector<std::uint32_t> h_wat_num(static_cast<std::size_t>(n_mu), 0u);

    float h_scalars[4] = {
        static_cast<float>(receptor_points.back().residue_number), // same “last element” debug
        static_cast<float>(total_volume),                          // volume
        static_cast<float>(gcmc_steps),                             // num_steps
        static_cast<float>(initial_water_capacity),                 // initial water capacity
    };

    // -------------------------
    // Debug prints (optional)
    // -------------------------
    

    std::cout << "[run_mcswell_gpu_titration]\n"
              << "  Water model: " << WATER_MODEL_NAME << "\n"
              << "  receptor_atoms: " << n_receptor_atoms << "\n"
              << "  n_points: " << n_points << "\n"
              << "  spacing: " << spacing << "\n"
              << "  total_volume: " << total_volume << "\n"
              << "  initial water capacity: " << initial_water_capacity << " per window ("
              << (max_waters > 0 ? "max_waters " + std::to_string(max_waters) : std::string("grows on demand"))
              << ")\n"
              << "  n_mu: " << n_mu << "\n"
              << "  gcmc_steps: " << gcmc_steps << " (" << chunk_steps << " per launch)\n"
              << "  equilibration_steps: " << equilibration_steps << "\n"
              << "  n_snapshots: " << n_snapshots << "\n"
              << "  snapshot staging: " << arena_floats_per_mu * sizeof(float) * n_mu / (1 << 20)
              << " MiB\n"
              << "  diagnostic_windows: " << n_diag_windows
              << " (" << steps_per_diag_window << " steps each, energy every "
              << std::max<std::size_t>(diagnostic_energy_stride, 1) << ")\n"
              << "  save_path: " << save_path << "\n";

    // -------------------------
    // Device allocations
    // -------------------------
    float* d_boundaries = nullptr;   // insertion points, flattened float[3*n_points]
    float* d_receptor   = nullptr;   // receptor_atoms_buffer
    float* d_waters     = nullptr;   // waters_buffer
    std::uint32_t* d_wat_num = nullptr; // wat_num buffer
    float* d_scalars    = nullptr;   // scalars[4]
    float* d_mu         = nullptr;   // mu_values[n_mu]
    void*  d_states     = nullptr;   // RNG states (opaque bytes; 1 per mu)

    // boundaries
    cuda_check(cudaMalloc(&d_boundaries, n_points * 3 * sizeof(float)), "cudaMalloc d_boundaries");
    cuda_check(cudaMemcpy(d_boundaries, insertion_points_xyz,
                          n_points * 3 * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy boundaries");

    // receptor
    cuda_check(cudaMalloc(&d_receptor, h_receptor.size() * sizeof(float)), "cudaMalloc d_receptor");
    cuda_check(cudaMemcpy(d_receptor, h_receptor.data(),
                          h_receptor.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy receptor");

    // waters (zero)
    const std::size_t waters_bytes = static_cast<std::size_t>(n_mu) * water_capacity * WATER_SIZE * sizeof(float);
    cuda_check(cudaMalloc(&d_waters, waters_bytes), "cudaMalloc d_waters");
    cuda_check(cudaMemset(d_waters, 0, waters_bytes), "cudaMemset d_waters");

    // wat_num (zero)
    cuda_check(cudaMalloc(&d_wat_num, static_cast<std::size_t>(n_mu) * sizeof(std::uint32_t)),
               "cudaMalloc d_wat_num");
    cuda_check(cudaMemcpy(d_wat_num, h_wat_num.data(),
                          static_cast<std::size_t>(n_mu) * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy wat_num");

    // scalars
    cuda_check(cudaMalloc(&d_scalars, 4 * sizeof(float)), "cudaMalloc d_scalars");
    cuda_check(cudaMemcpy(d_scalars, h_scalars, 4 * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy scalars");

    // mu array
    cuda_check(cudaMalloc(&d_mu, static_cast<std::size_t>(n_mu) * sizeof(float)), "cudaMalloc d_mu");
    cuda_check(cudaMemcpy(d_mu, mu_values.data(),
                          static_cast<std::size_t>(n_mu) * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy mu_values");

    // per-mu resume state and snapshot staging (see cuda/titration_state.cuh)
    const auto alloc_zeroed = [&](auto** p, std::size_t n, const char* what) {
        cuda_check(cudaMalloc(p, n * sizeof(**p)), what);
        cuda_check(cudaMemset(*p, 0, n * sizeof(**p)), what);
    };
    TitrationCarry carry{};
    SnapshotArena arena{};
    alloc_zeroed(&carry.step, static_cast<std::size_t>(n_mu), "cudaMalloc carry.step");
    alloc_zeroed(&carry.snap_id, static_cast<std::size_t>(n_mu), "cudaMalloc carry.snap_id");
    alloc_zeroed(&carry.exit_reason, static_cast<std::size_t>(n_mu), "cudaMalloc carry.exit_reason");
    if (diag_len > 0) {
        alloc_zeroed(&carry.diag, static_cast<std::size_t>(n_mu) * DIAG_CARRY_SLOTS, "cudaMalloc carry.diag");
    }
    arena.floats_per_mu = arena_floats_per_mu;
    cuda_check(cudaMalloc(&arena.data, static_cast<std::size_t>(n_mu) * arena_floats_per_mu * sizeof(float)),
               "cudaMalloc snapshot staging buffer");
    alloc_zeroed(&arena.used, static_cast<std::size_t>(n_mu), "cudaMalloc arena.used");
    alloc_zeroed(&arena.counts, static_cast<std::size_t>(n_mu) * n_snapshots, "cudaMalloc arena.counts");

    // rng states (opaque; launch layer tells you bytes per state)
    const std::size_t state_bytes = dummy_rng_state_bytes(); // provided by your launch layer
    cuda_check(cudaMalloc(&d_states, static_cast<std::size_t>(n_mu) * state_bytes), "cudaMalloc d_states");

    // diagnostics (zero; all null when disabled, which the kernel reads as
    // "don't instrument")
    unsigned long long* d_diag_proposed = nullptr;
    unsigned long long* d_diag_accepted = nullptr;
    unsigned long long* d_diag_capacity_blocked = nullptr;
    unsigned long long* d_diag_n_sum = nullptr;
    std::uint32_t* d_diag_n_end = nullptr;
    float* d_diag_energy_end = nullptr;

    if (diag_len > 0) {
        const auto alloc_zeroed_u64 = [&](unsigned long long** p, std::size_t n, const char* what) {
            cuda_check(cudaMalloc(p, n * sizeof(unsigned long long)), what);
            cuda_check(cudaMemset(*p, 0, n * sizeof(unsigned long long)), what);
        };
        alloc_zeroed_u64(&d_diag_proposed, diag_len3, "cudaMalloc d_diag_proposed");
        alloc_zeroed_u64(&d_diag_accepted, diag_len3, "cudaMalloc d_diag_accepted");
        alloc_zeroed_u64(&d_diag_capacity_blocked, diag_len, "cudaMalloc d_diag_capacity_blocked");
        alloc_zeroed_u64(&d_diag_n_sum, diag_len, "cudaMalloc d_diag_n_sum");

        cuda_check(cudaMalloc(&d_diag_n_end, diag_len * sizeof(std::uint32_t)), "cudaMalloc d_diag_n_end");
        cuda_check(cudaMemset(d_diag_n_end, 0, diag_len * sizeof(std::uint32_t)), "cudaMemset d_diag_n_end");
        cuda_check(cudaMalloc(&d_diag_energy_end, diag_len * sizeof(float)), "cudaMalloc d_diag_energy_end");
        cuda_check(cudaMemset(d_diag_energy_end, 0, diag_len * sizeof(float)), "cudaMemset d_diag_energy_end");
    }

    // cavity mask (optional): one byte per lattice cell, 1 where a kept
    // insertion point sits. Only uploaded when the displacement constraint is
    // actually requested.
    unsigned char* d_cavity_mask = nullptr;
    int mask_nx = 0, mask_ny = 0, mask_nz = 0;
    float mask_ox = 0.0f, mask_oy = 0.0f, mask_oz = 0.0f, mask_spacing = spacing;
    if (constrain_displacement) {
        if (cavity_grid == nullptr || cavity_grid->mask.empty()) {
            throw std::invalid_argument(
                "constrain_displacement requires a cavity grid with a populated mask");
        }
        mask_nx = static_cast<int>(cavity_grid->dims[0]);
        mask_ny = static_cast<int>(cavity_grid->dims[1]);
        mask_nz = static_cast<int>(cavity_grid->dims[2]);
        mask_ox = static_cast<float>(cavity_grid->origin[0]);
        mask_oy = static_cast<float>(cavity_grid->origin[1]);
        mask_oz = static_cast<float>(cavity_grid->origin[2]);
        mask_spacing = static_cast<float>(cavity_grid->spacing);
        const std::size_t mask_bytes = cavity_grid->mask.size() * sizeof(unsigned char);
        cuda_check(cudaMalloc(&d_cavity_mask, mask_bytes), "cudaMalloc d_cavity_mask");
        cuda_check(cudaMemcpy(d_cavity_mask, cavity_grid->mask.data(), mask_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy cavity_mask");
        std::cout << "  displacement constrained to the sampled region ("
                  << mask_nx << "x" << mask_ny << "x" << mask_nz << " lattice)\n";
    }

    // -------------------------
    // Init random seeds (seed_tensor) + kernel launch
    // -------------------------
    // CUDA: initialize n_mu RNG states with a master seed.
    init_rng_launch(d_states, seed, n_mu);
    cuda_check(cudaGetLastError(), "init_rng_launch");
    cuda_check(cudaDeviceSynchronize(), "init_rng sync");

    // -------------------------
    // Titration: chunked, resumable launches
    // -------------------------
    // Each launch runs every window up to step_end, unless a window stops
    // early because its water buffer or snapshot staging is full. Between
    // launches the staged snapshots are drained to the host and, if a window
    // filled the water buffer, the buffer is doubled. A launch in which
    // every window reaches step_end completes the chunk.
    std::vector<std::vector<float>> h_frames(static_cast<std::size_t>(n_mu));
    std::vector<std::uint32_t> h_exit(static_cast<std::size_t>(n_mu));
    std::vector<unsigned long long> h_used(static_cast<std::size_t>(n_mu));

    const auto drain_snapshots = [&]() {
        cuda_check(cudaMemcpy(h_used.data(), arena.used, h_used.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy arena.used back");
        for (std::size_t mu = 0; mu < h_used.size(); ++mu) {
            if (h_used[mu] == 0) continue;
            auto& dst = h_frames[mu];
            const std::size_t old_size = dst.size();
            dst.resize(old_size + h_used[mu]);
            cuda_check(cudaMemcpy(dst.data() + old_size, arena.data + mu * arena.floats_per_mu,
                                  h_used[mu] * sizeof(float), cudaMemcpyDeviceToHost),
                       "cudaMemcpy staged snapshots back");
        }
        cuda_check(cudaMemset(arena.used, 0, h_used.size() * sizeof(unsigned long long)),
                   "cudaMemset arena.used");
    };

    // Doubles the per-window water buffer (up to max_waters), keeping each
    // window's waters at the start of its new, larger slice.
    const auto grow_water_buffer = [&]() {
        std::size_t new_capacity = water_capacity * 2;
        if (max_waters > 0) new_capacity = std::min(new_capacity, max_waters);
        if (new_capacity > water_capacity_limit) {
            throw std::runtime_error("water buffer cannot grow past " + std::to_string(water_capacity_limit) +
                                     " waters per window");
        }
        const std::size_t old_pitch = water_capacity * WATER_SIZE * sizeof(float);
        const std::size_t new_pitch = new_capacity * WATER_SIZE * sizeof(float);
        float* d_bigger = nullptr;
        const cudaError_t e = cudaMalloc(&d_bigger, static_cast<std::size_t>(n_mu) * new_pitch);
        if (e != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA] out of GPU memory growing the water buffer to ") +
                std::to_string(new_capacity) + " waters per window (" + cudaGetErrorString(e) +
                "); set max_waters to cap the number of waters");
        }
        cuda_check(cudaMemset(d_bigger, 0, static_cast<std::size_t>(n_mu) * new_pitch), "cudaMemset grown d_waters");
        cuda_check(cudaMemcpy2D(d_bigger, new_pitch, d_waters, old_pitch, old_pitch, static_cast<std::size_t>(n_mu),
                                cudaMemcpyDeviceToDevice),
                   "cudaMemcpy2D grow d_waters");
        cudaFree(d_waters);
        d_waters = d_bigger;
        water_capacity = new_capacity;

        // Staging is empty right after a drain, so it can simply be replaced.
        const std::size_t needed = arena_floats_for(water_capacity);
        if (needed > arena.floats_per_mu) {
            cudaFree(arena.data);
            arena.data = nullptr;
            cuda_check(cudaMalloc(&arena.data, static_cast<std::size_t>(n_mu) * needed * sizeof(float)),
                       "cudaMalloc grown snapshot staging buffer");
            arena.floats_per_mu = needed;
        }
        std::cout << "  water buffer grown to " << water_capacity << " waters per window\n";
    };

    std::size_t n_launches = 0;
    for (std::size_t step_end = 0; step_end < gcmc_steps;) {
        step_end = std::min(gcmc_steps, step_end + chunk_steps);
        for (;;) {
            const bool grow_on_full = (max_waters == 0) || (water_capacity < max_waters);
            run_gcmc_titration_launch(
                /*boundaries_xyz*/ d_boundaries,
                /*n_points*/       static_cast<int>(n_points),
                /*receptor*/       d_receptor,
                /*n_receptor*/     static_cast<int>(n_receptor_atoms),
                /*waters*/         d_waters,
                /*wat_num*/        d_wat_num,
                /*scalars*/        d_scalars,
                /*mu_values*/      d_mu,
                /*n_mu*/           n_mu,
                /*distance_cutoff*/distance_cutoff,
                /*gcmc_steps*/     static_cast<std::uint32_t>(gcmc_steps),
                /*step_end*/       static_cast<std::uint32_t>(step_end),
                /*equil_steps*/    static_cast<std::uint32_t>(equilibration_steps),
                /*n_snapshots*/    static_cast<std::uint32_t>(n_snapshots),
                /*water_capacity*/ static_cast<std::uint32_t>(water_capacity),
                /*grow_on_full*/   grow_on_full,
                /*rng_states*/     d_states,
                /*carry*/          carry,
                /*arena*/          arena,
                /*diag_proposed*/  d_diag_proposed,
                /*diag_accepted*/  d_diag_accepted,
                /*diag_cap_block*/ d_diag_capacity_blocked,
                /*diag_n_sum*/     d_diag_n_sum,
                /*diag_n_end*/     d_diag_n_end,
                /*diag_energy*/    d_diag_energy_end,
                /*n_diag_windows*/ static_cast<std::uint32_t>(n_diag_windows),
                /*steps_per_win*/  static_cast<std::uint32_t>(steps_per_diag_window),
                /*energy_stride*/  static_cast<std::uint32_t>(std::max<std::size_t>(diagnostic_energy_stride, 1)),
                /*cavity_mask*/    d_cavity_mask,
                mask_nx, mask_ny, mask_nz,
                mask_ox, mask_oy, mask_oz,
                mask_spacing
            );
            cuda_check(cudaGetLastError(), "run_gcmc_titration_launch");
            cuda_check(cudaDeviceSynchronize(), "run_gcmc_titration sync");
            ++n_launches;

            drain_snapshots();
            cuda_check(cudaMemcpy(h_exit.data(), carry.exit_reason, h_exit.size() * sizeof(std::uint32_t),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy exit reasons back");
            const auto stopped_for = [&](std::uint32_t reason) {
                return std::find(h_exit.begin(), h_exit.end(), reason) != h_exit.end();
            };
            if (stopped_for(TITRATION_WATER_BUFFER_FULL)) {
                grow_water_buffer();
            } else if (!stopped_for(TITRATION_SNAPSHOT_BUFFER_FULL)) {
                break;  // every window reached step_end
            }
        }
    }

    // -------------------------
    // Assemble snapshots
    // -------------------------
    std::vector<std::uint32_t> h_counts(static_cast<std::size_t>(n_mu) * n_snapshots);
    cuda_check(cudaMemcpy(h_counts.data(), arena.counts, h_counts.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy snapshot counts back");

    mcswell::WaterSnapshots snapshots;
    snapshots.n_mu = static_cast<std::size_t>(n_mu);
    snapshots.n_snapshots = n_snapshots;
    snapshots.offsets.resize(h_counts.size() + 1, 0);
    for (std::size_t f = 0; f < h_counts.size(); ++f) {
        snapshots.offsets[f + 1] = snapshots.offsets[f] + h_counts[f];
    }
    snapshots.waters.reserve(snapshots.offsets.back() * WATER_SIZE);
    for (std::size_t mu = 0; mu < h_frames.size(); ++mu) {
        const std::size_t expected =
            (snapshots.offsets[(mu + 1) * n_snapshots] - snapshots.offsets[mu * n_snapshots]) * WATER_SIZE;
        if (h_frames[mu].size() != expected) {
            throw std::runtime_error("snapshot bookkeeping mismatch for mu window " + std::to_string(mu));
        }
        snapshots.waters.insert(snapshots.waters.end(), h_frames[mu].begin(), h_frames[mu].end());
        std::vector<float>().swap(h_frames[mu]);
    }

    std::cout << "  titration finished in " << n_launches << " launches; water capacity "
              << water_capacity << " per window\n";

    // -------------------------
    // Read back diagnostics
    // -------------------------
    mcswell::analysis::SamplerDiagnostics diagnostics;
    if (diag_len > 0) {
        diagnostics.n_mu = static_cast<std::size_t>(n_mu);
        diagnostics.n_windows = n_diag_windows;
        diagnostics.steps_per_window = steps_per_diag_window;
        diagnostics.total_steps = static_cast<std::uint64_t>(gcmc_steps);
        diagnostics.equilibration_steps = static_cast<std::uint64_t>(equilibration_steps);

        diagnostics.proposed.resize(diag_len3);
        diagnostics.accepted.resize(diag_len3);
        diagnostics.capacity_blocked.resize(diag_len);
        diagnostics.n_waters_sum.resize(diag_len);
        diagnostics.n_waters_end.resize(diag_len);
        diagnostics.energy_end.resize(diag_len);

        const auto copy_back = [&](void* dst, const void* src, std::size_t bytes, const char* what) {
            cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), what);
        };
        copy_back(diagnostics.proposed.data(), d_diag_proposed,
                  diag_len3 * sizeof(unsigned long long), "cudaMemcpy diag_proposed back");
        copy_back(diagnostics.accepted.data(), d_diag_accepted,
                  diag_len3 * sizeof(unsigned long long), "cudaMemcpy diag_accepted back");
        copy_back(diagnostics.capacity_blocked.data(), d_diag_capacity_blocked,
                  diag_len * sizeof(unsigned long long), "cudaMemcpy diag_capacity_blocked back");
        copy_back(diagnostics.n_waters_sum.data(), d_diag_n_sum,
                  diag_len * sizeof(unsigned long long), "cudaMemcpy diag_n_sum back");
        copy_back(diagnostics.n_waters_end.data(), d_diag_n_end,
                  diag_len * sizeof(std::uint32_t), "cudaMemcpy diag_n_end back");
        copy_back(diagnostics.energy_end.data(), d_diag_energy_end,
                  diag_len * sizeof(float), "cudaMemcpy diag_energy_end back");
    }

    if (dump_debug_pdbs) {
        mcswell::save_pdbs_by_mu_snapshot(snapshots, save_path);
    }

    // -------------------------
    // Cleanup
    // -------------------------
    cudaFree(d_cavity_mask);
    cudaFree(d_diag_energy_end);
    cudaFree(d_diag_n_end);
    cudaFree(d_diag_n_sum);
    cudaFree(d_diag_capacity_blocked);
    cudaFree(d_diag_accepted);
    cudaFree(d_diag_proposed);
    cudaFree(d_states);
    cudaFree(arena.counts);
    cudaFree(arena.used);
    cudaFree(arena.data);
    cudaFree(carry.diag);
    cudaFree(carry.exit_reason);
    cudaFree(carry.snap_id);
    cudaFree(carry.step);
    cudaFree(d_mu);
    cudaFree(d_scalars);
    cudaFree(d_wat_num);
    cudaFree(d_waters);
    cudaFree(d_receptor);
    cudaFree(d_boundaries);

    TitrationResult result;
    result.snapshots = std::move(snapshots);
    result.max_waters = max_waters;
    result.water_capacity = water_capacity;
    result.diagnostics = std::move(diagnostics);
    return result;
}
