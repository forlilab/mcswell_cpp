//  Copyright (c) 2026 Scripps Research, Forli Lab.
//  All rights reserved.
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
#include <string>
#include <vector>

namespace mcswell::analysis {

// Oxygen coordinates of every active water in one GCMC snapshot, in the
// order they appear in the GPU water buffer. An "inactive" water slot (an
// (x,y,z) == (0,0,0) never-written slot, the same convention the GPU kernel
// zero-initializes the water buffer with) is skipped, exactly like the old
// PDB-based pipeline skipped it when writing snap_#####.pdb.
struct WaterFrame {
    std::vector<std::array<double, 3>> oxygens;
};

// All snapshots sampled at a single chemical-potential (mu) window.
struct MuWindowFrames {
    std::vector<WaterFrame> frames; // size == n_snapshots
};

// Extracts oxygen-only point clouds directly from the flat host snapshot
// buffer produced by run_mcswell_gpu_titration(), replacing the old
// write-PDB-then-reparse-PDB round trip. `snapshots_flat` must have length
// n_mu * n_snapshots * max_n_waters * WATER_SIZE, laid out exactly as
// snapshots_io.hpp::reconstruct_snapshots_by_mu expects (mu-major, then
// snapshot, then water, using the OX/OY/OZ/... offsets from
// include/cuda/consts.cuh).
std::vector<MuWindowFrames> extract_oxygen_frames(
    const std::vector<float>& snapshots_flat,
    std::size_t n_mu,
    std::size_t n_snapshots,
    std::size_t max_n_waters
);

// Per-mu-window water-count statistics and the capacity-safety verdict.
// Ports scan_capacity_and_collect_density_points() from both
// estimate_free_energies.py and estimate_free_energies_GCI.py (the two
// were near-duplicates; this is the single, shared implementation).
struct CapacityDiagnostics {
    std::size_t mu_index = 0;
    double mu_kcal_mol = 0.0;
    std::size_t n_frames = 0;
    double n_region_mean = 0.0;
    double n_region_std = 0.0;
    std::size_t n_region_max_observed = 0;
    std::size_t water_capacity = 0;
    double capacity_mean_fraction = 0.0;
    double capacity_hit_fraction = 0.0;
    bool valid_for_analysis = false;
    std::string capacity_filter_reason;
};

struct CapacityScanResult {
    // Oxygen coordinates pooled from every snapshot in every capacity-safe
    // (valid_for_analysis) mu window, for density/site detection.
    std::vector<std::array<double, 3>> density_points;
    // One entry per mu window, in the same order as `frames_by_mu`.
    std::vector<CapacityDiagnostics> diagnostics;
    std::vector<bool> valid_mask;
};

// A window "triggers" capacity truncation if either:
//   1) more than `max_capacity_hit_fraction` of its frames have a water
//      count >= `water_capacity`, or
//   2) its mean water count is >= `max_mean_capacity_fraction * water_capacity`.
// Once triggered, that window AND every later one in `frames_by_mu` order
// (i.e. every higher-mu window, since callers pass windows sorted by mu)
// are excluded -- a contiguous low-mu prefix is kept, matching the Python
// "first_capacity_limited_window" / "after_first_capacity_limited_window"
// behavior. If `capacity_filter` is false, every window is valid.
CapacityScanResult scan_capacity_and_collect_density_points(
    const std::vector<MuWindowFrames>& frames_by_mu,
    const std::vector<double>& mu_values,
    std::size_t water_capacity,
    bool capacity_filter,
    double max_capacity_hit_fraction,
    double max_mean_capacity_fraction
);

} // namespace mcswell::analysis
