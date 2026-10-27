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
#include "water_snapshots.hpp"

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

// GPU memory for staging snapshots before they are copied to the host. This
// bounds the device-side cost of snapshots regardless of run length or water
// count; a smaller value only means more frequent copies.
inline constexpr std::size_t DEFAULT_SNAPSHOT_BUFFER_BYTES = std::size_t(256) << 20;

struct TitrationResult {
    // Every snapshot of every mu window, unpadded.
    mcswell::WaterSnapshots snapshots;

    // Hard limit the run enforced on the water count per window, 0 if none.
    // Only a run with a limit can have insertions refused for lack of room
    // (diagnostics.capacity_blocked); the analyses' capacity filter keys off it.
    std::size_t max_waters = 0;

    // Size the per-window water buffer ended at (it starts at the bulk-density
    // estimate floor(V * 0.0334) and doubles whenever a window fills it).
    std::size_t water_capacity = 0;

    // Per-(mu, window) acceptance/occupancy/energy trace. Empty when the run
    // was launched with n_diagnostic_windows == 0.
    mcswell::analysis::SamplerDiagnostics diagnostics;
};

// There is no fixed limit on the number of waters by default: the water
// buffer starts at the bulk-density estimate and grows on demand, and
// snapshots are staged in a fixed-size GPU buffer (snapshot_buffer_bytes) and
// streamed to host memory. Growing the buffer does not perturb the chain: a
// run reproduces, bit for bit, the old fixed-buffer code for as long as that
// code would not have refused an insertion.
//
// `max_waters` > 0 restores a hard limit: insertions beyond it are refused
// (and counted in diagnostics.capacity_blocked), as the fixed buffer did.
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
    bool constrain_displacement = false,
    std::size_t max_waters = 0,
    std::size_t snapshot_buffer_bytes = DEFAULT_SNAPSHOT_BUFFER_BYTES
);
