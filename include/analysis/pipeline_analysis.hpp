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

#include "analysis/frame_extraction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mcswell::analysis {

// Configuration for the post-simulation analyses (mirrors the [gci]
// section and related knobs in the MCSwell TOML config / CLI flags).
// mu_bulk is a required, already-resolved value: the water-model-dependent
// default lookup (utils.resolve_mu_bulk in the Python driver) stays in
// Python, since it belongs next to the CLI/config layer, not the numerics.
//
// Deliberately CUDA-free: this whole struct and both analyses below
// operate only on already-extracted MuWindowFrames, so they build and are
// unit-testable with a plain C++ compiler, without a GPU. Only the top
// level (include/pipeline.hpp, which also runs the GPU titration itself)
// needs CUDA.
struct PostAnalysisConfig {
    double temperature = 300.0;
    double mu_bulk = 0.0;

    // Density-grid / site detection (shared by both analyses).
    std::array<double, 3> box_center{};
    std::array<double, 3> box_halfsize{};
    double grid_spacing = 0.5;
    double grid_sigma = 1.4;
    double peak_percentile = 90.0;
    double peak_merge_cutoff = 1.4;

    // Capacity-safety filtering (shared).
    bool capacity_filter = true;
    double bulk_water_density = 0.0334;
    double max_capacity_hit_fraction = 0.01;
    double max_mean_capacity_fraction = 0.90;

    // Binomial-fit specific.
    double assignment_cutoff = 2.4;

    // GCI-fit specific.
    double local_radius = 1.4;
    std::string local_volume_mode = "sampler"; // "sampler" | "standard" | "sphere"
    int region_max_terms = 12;
    int local_max_terms = 4;
    int random_starts = 64; // see the longer justification on LogisticSumFitOptions::random_starts
    std::uint64_t fit_seed = 20260812ULL;
};

// Capacity-safe bounded independent-site binomial MLE analysis. Writes
// capacity_diagnostics.csv, sites_all.csv, titration.csv,
// titration_diagnostics.csv, gO.dx, sites.pdb, sites.csv and
// binomial_metadata.json under save_path/gci/<peak_percentile>_binomial/,
// and returns that directory.
std::string run_binomial_analysis(
    const std::vector<MuWindowFrames>& frames_by_mu, const std::vector<double>& mu_values,
    const PostAnalysisConfig& cfg, std::size_t n_insertion_points, double sampler_volume,
    const std::string& save_path);

// ProtoMS-style GCI (whole-region + local logistic-sum) analysis. Writes
// region_titration.csv, local_titration.csv, region_gci_pmf.csv, gO.dx,
// sites.pdb, sites.csv/local_gci_sites.csv and gci_metadata.json under
// save_path/gci/<peak_percentile>_protoms_gci/, and returns that directory.
std::string run_gci_analysis(
    const std::vector<MuWindowFrames>& frames_by_mu, const std::vector<double>& mu_values,
    const PostAnalysisConfig& cfg, std::size_t n_insertion_points, double sampler_volume,
    const std::string& save_path);

} // namespace mcswell::analysis
