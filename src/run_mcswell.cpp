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
// - allocate snapshots buffer
// - allocate mu array + scalars array
// - create RNG states (1 per mu)
// - launch the CUDA kernel (1 block per mu, 512 threads)
// - read back snapshots and return
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
    bool constrain_displacement
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
    const std::size_t target_n_waters =
        static_cast<std::size_t>(std::floor(total_volume * BULK_WATER_DENSITY));

    if (target_n_waters == 0) {
        throw std::runtime_error("target_n_waters computed as 0 (check spacing/n_points/units).");
    }

    const std::size_t waters_len =
        static_cast<std::size_t>(n_mu) * target_n_waters * WATER_SIZE;

    const std::size_t snapshots_len =
        static_cast<std::size_t>(n_mu) * n_snapshots * target_n_waters * WATER_SIZE;

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
        static_cast<float>(target_n_waters),                        // target_n_waters
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
              << "  target_n_waters: " << target_n_waters << "\n"
              << "  n_mu: " << n_mu << "\n"
              << "  gcmc_steps: " << gcmc_steps << "\n"
              << "  equilibration_steps: " << equilibration_steps << "\n"
              << "  n_snapshots: " << n_snapshots << "\n"
              << "  snapshots_len: " << snapshots_len << "\n"
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
    float* d_snapshots  = nullptr;   // snapshots_buffer
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
    cuda_check(cudaMalloc(&d_waters, waters_len * sizeof(float)), "cudaMalloc d_waters");
    cuda_check(cudaMemset(d_waters, 0, waters_len * sizeof(float)), "cudaMemset d_waters");

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

    // snapshots (zero)
    cuda_check(cudaMalloc(&d_snapshots, snapshots_len * sizeof(float)), "cudaMalloc d_snapshots");
    cuda_check(cudaMemset(d_snapshots, 0, snapshots_len * sizeof(float)), "cudaMemset d_snapshots");

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

    run_gcmc_titration_launch(
        /*boundaries_xyz*/ d_boundaries,
        /*n_points*/       static_cast<int>(n_points),
        /*receptor*/       d_receptor,
        /*n_receptor*/     static_cast<int>(n_receptor_atoms),
        /*waters*/         d_waters,
        /*wat_num*/        d_wat_num,
        /*scalars*/        d_scalars,
        /*snapshots*/      d_snapshots,
        /*mu_values*/      d_mu,
        /*n_mu*/           n_mu,
        /*distance_cutoff*/distance_cutoff,
        /*gcmc_steps*/     static_cast<std::uint32_t>(gcmc_steps),
        /*equil_steps*/    static_cast<std::uint32_t>(equilibration_steps),
        /*n_snapshots*/    static_cast<std::uint32_t>(n_snapshots),
        /*target_n_waters*/static_cast<std::uint32_t>(target_n_waters),
        /*rng_states*/     d_states,
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

    // -------------------------
    // Read back snapshots_handle
    // -------------------------
    std::vector<float> h_snapshots(snapshots_len);
    cuda_check(cudaMemcpy(h_snapshots.data(), d_snapshots,
                          snapshots_len * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy snapshots back");

    cuda_check(cudaDeviceSynchronize(), "sync before inspection");

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
        mcswell::save_pdbs_by_mu_snapshot(
            h_snapshots,
            (std::size_t)n_mu,
            (std::size_t)n_snapshots,
            (std::size_t)target_n_waters,
            save_path
        );
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
    cudaFree(d_snapshots);
    cudaFree(d_mu);
    cudaFree(d_scalars);
    cudaFree(d_wat_num);
    cudaFree(d_waters);
    cudaFree(d_receptor);
    cudaFree(d_boundaries);

    TitrationResult result;
    result.snapshots = std::move(h_snapshots);
    result.target_n_waters = static_cast<std::size_t>(target_n_waters);
    result.diagnostics = std::move(diagnostics);
    return result;
}
