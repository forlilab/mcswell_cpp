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

#include "analysis/pipeline_analysis.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Atom;

namespace mcswell {

struct PipelineResult {
    std::size_t n_insertion_points = 0;
    double sampler_volume = 0.0;
    std::string binomial_output_dir;
    std::string gci_output_dir;
};

// Runs the GCMC titration (run_mcswell_gpu_titration) and, without ever
// writing a per-snapshot PDB to disk, feeds its in-memory result directly
// into the requested post-simulation analyses, writing only the final
// small result artifacts under save_path (sites.csv, sites.pdb, titration
// CSVs, gO.dx, metadata JSON -- the same outputs the old two-script Python
// pipeline produced from disk). The GPU titration and Atom dependency are
// the only reason this lives outside the CUDA-free mcswell_analysis
// library; the analyses themselves (run_binomial_analysis/run_gci_analysis
// in analysis/pipeline_analysis.hpp) do not.
PipelineResult run_mcswell_full_pipeline(
    const std::vector<Atom>& receptor_points, const float* insertion_points_xyz, std::size_t n_points,
    float distance_cutoff, float spacing, std::size_t gcmc_steps, std::size_t equilibration_steps,
    const std::string& save_path, const std::vector<float>& mu_values, std::size_t n_snapshots,
    const analysis::PostAnalysisConfig& analysis_cfg, bool run_binomial = true, bool run_gci = true,
    std::uint64_t seed = 12345ULL, bool dump_debug_pdbs = false);

} // namespace mcswell
