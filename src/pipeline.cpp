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

#include "pipeline.hpp"

#include "analysis/frame_extraction.hpp"
#include "run_mcswell.hpp"

#include <filesystem>

namespace mcswell {

PipelineResult run_mcswell_full_pipeline(
    const std::vector<Atom>& receptor_points, const float* insertion_points_xyz, std::size_t n_points,
    float distance_cutoff, float spacing, std::size_t gcmc_steps, std::size_t equilibration_steps,
    const std::string& save_path, const std::vector<float>& mu_values_f, std::size_t n_snapshots,
    const analysis::PostAnalysisConfig& analysis_cfg, bool run_binomial, bool run_gci, std::uint64_t seed,
    bool dump_debug_pdbs) {
    std::filesystem::create_directories(save_path);

    const auto titration = run_mcswell_gpu_titration(
        receptor_points, insertion_points_xyz, n_points, distance_cutoff, spacing, gcmc_steps, equilibration_steps,
        save_path, mu_values_f, n_snapshots, seed, dump_debug_pdbs);

    const std::size_t n_mu = mu_values_f.size();
    const auto frames_by_mu =
        analysis::extract_oxygen_frames(titration.snapshots, n_mu, n_snapshots, titration.target_n_waters);

    const std::vector<double> mu_values(mu_values_f.begin(), mu_values_f.end());

    const double voxel_volume = static_cast<double>(spacing) * spacing * spacing;
    const double sampler_volume = voxel_volume * static_cast<double>(n_points);

    PipelineResult result;
    result.n_insertion_points = n_points;
    result.sampler_volume = sampler_volume;

    if (run_binomial) {
        result.binomial_output_dir = analysis::run_binomial_analysis(
            frames_by_mu, mu_values, analysis_cfg, n_points, sampler_volume, save_path);
    }
    if (run_gci) {
        result.gci_output_dir =
            analysis::run_gci_analysis(frames_by_mu, mu_values, analysis_cfg, n_points, sampler_volume, save_path);
    }

    return result;
}

} // namespace mcswell
