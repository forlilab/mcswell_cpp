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
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "analysis/sampler_diagnostics.hpp"
#include "insertion_points.hpp"

struct Atom;

// Default number of diagnostic windows the titration is split into. 1000 is
// fine-grained enough to resolve the approach to equilibrium on the usual
// 10^7-10^8 step runs while costing one total-energy evaluation per window
// (well under 1% of the run), and the resulting trace is what the convergence
// analysis in python/convergence.py consumes.
inline constexpr std::size_t DEFAULT_DIAGNOSTIC_WINDOWS = 1000;

// Evaluate the per-window total energy every Nth window. The energy is the
// only part of the instrumentation that costs real work, so raising this lets
// a high-resolution occupancy trace (short windows, used to measure
// correlation times in steps rather than in windows) stay cheap.
inline constexpr std::size_t DEFAULT_DIAGNOSTIC_ENERGY_STRIDE = 1;

// Host copy of the titration kernel's full snapshot buffer, plus the
// sizing the caller needs to index into it (n_mu/n_snapshots are already
// known to the caller from its own inputs; target_n_waters is derived
// on the C++ side from n_points/spacing/BULK_WATER_DENSITY and previously
// had to be reconstructed independently on the Python side to size the
// water-buffer capacity for post-processing).
struct TitrationResult {
    std::vector<float> snapshots;
    std::size_t target_n_waters = 0;

    // Per-(mu, window) acceptance/occupancy/energy trace. Empty when the run
    // was launched with n_diagnostic_windows == 0.
    mcswell::analysis::SamplerDiagnostics diagnostics;
};

// Layout of `snapshots` is whatever the kernel produces; keep it
// consistent with mcswell::analysis::extract_oxygen_frames() and
// include/cuda/consts.cuh's water-buffer offsets.
//
// `dump_debug_pdbs`: if true, also writes one PDB per (mu, snapshot) under
// save_path/frames/ for visual inspection (PyMOL/VMD). Off by default --
// the in-memory `snapshots` buffer is sufficient for post-processing, so
// a full run no longer needs to touch disk for anything but the final,
// small result artifacts.
TitrationResult run_mcswell_gpu_titration(
    const std::vector<Atom>& receptor_points,
    const float* insertion_points_xyz,   // length = 3*n_points
    std::size_t n_points,
    float distance_cutoff,
    float spacing,
    std::size_t gcmc_steps,
    std::size_t equilibration_steps,
    const std::string& save_path,
    const std::vector<float>& mu_values,
    std::size_t n_snapshots,
    std::uint64_t seed = 12345ULL,
    bool dump_debug_pdbs = false,
    std::size_t n_diagnostic_windows = DEFAULT_DIAGNOSTIC_WINDOWS,
    std::size_t diagnostic_energy_stride = DEFAULT_DIAGNOSTIC_ENERGY_STRIDE,
    // Occupancy mask over the insertion lattice. When supplied and
    // `constrain_displacement` is set, a displacement that would move a water
    // out of the sampled region is rejected, so insertion and deletion act on
    // the same region. Pass nullptr (or leave the flag off) for the historical
    // unconstrained behaviour.
    const InsertionGrid* cavity_grid = nullptr,
    bool constrain_displacement = false
);
