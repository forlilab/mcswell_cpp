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

// Integration-style tests for run_binomial_analysis/run_gci_analysis:
// synthetic frames_by_mu engineered to have a clean, deterministic,
// multi-step titration (no GPU/RNG involved), exercising the orchestration
// glue (capacity scan -> site detection -> fits -> CSV/PDB/JSON writers)
// end to end rather than testing any one algorithm in isolation (those
// already have dedicated, scipy-cross-validated unit tests elsewhere).

#include "analysis/pipeline_analysis.hpp"
#include "harness.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

using mcswell::analysis::MuWindowFrames;
using mcswell::analysis::PostAnalysisConfig;
using mcswell::analysis::run_binomial_analysis;
using mcswell::analysis::run_gci_analysis;
using mcswell::analysis::WaterFrame;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One water cluster tightly packed near the origin (all within one 0.5 A
// grid voxel), `count` waters per frame, identical across all n_snapshots
// frames in the window -- deterministic N_region_mean == count, std == 0.
MuWindowFrames make_window(int count, int n_snapshots) {
    MuWindowFrames w;
    for (int f = 0; f < n_snapshots; ++f) {
        WaterFrame frame;
        for (int k = 0; k < count; ++k) {
            frame.oxygens.push_back({0.01 + 0.01 * k, 0.0, 0.0});
        }
        w.frames.push_back(std::move(frame));
    }
    return w;
}

std::pair<std::vector<MuWindowFrames>, std::vector<double>> build_synthetic_titration() {
    // Clean 0 -> 1 -> 2 -> 3 step profile across 10 mu windows, noiseless.
    const std::vector<int> profile = {0, 0, 0, 1, 1, 2, 2, 3, 3, 3};
    const int n_snapshots = 20;

    std::vector<MuWindowFrames> frames_by_mu;
    std::vector<double> mu_values;
    for (std::size_t j = 0; j < profile.size(); ++j) {
        frames_by_mu.push_back(make_window(profile[j], n_snapshots));
        mu_values.push_back(-9.0 + static_cast<double>(j));
    }
    return {frames_by_mu, mu_values};
}

PostAnalysisConfig make_config() {
    PostAnalysisConfig cfg;
    cfg.temperature = 300.0;
    cfg.mu_bulk = -6.0;
    cfg.box_center = {0.0, 0.0, 0.0};
    cfg.box_halfsize = {5.0, 5.0, 5.0};
    cfg.grid_spacing = 0.5;
    cfg.grid_sigma = 1.0;
    cfg.peak_percentile = 90.0;
    cfg.peak_merge_cutoff = 1.4;
    cfg.capacity_filter = true;
    cfg.bulk_water_density = 0.0334; // capacity = floor(1000*0.0334) = 33, well above the max count of 3
    cfg.assignment_cutoff = 2.4;
    cfg.local_radius = 1.4;
    cfg.random_starts = 16; // lighter than the production default (64) to keep this test fast
    cfg.fit_seed = 1;
    return cfg;
}

void test_binomial_analysis_end_to_end() {
    auto [frames_by_mu, mu_values] = build_synthetic_titration();
    const auto cfg = make_config();
    const std::string save_path = "/tmp/mcswell_test_pipeline_binomial";
    std::filesystem::remove_all(save_path);

    const auto out_dir = run_binomial_analysis(frames_by_mu, mu_values, cfg, /*n_insertion_points=*/2960,
                                                /*sampler_volume=*/1000.0, save_path);

    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/sites.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/sites_all.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/sites.pdb"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/gO.dx"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/binomial_metadata.json"));

    const auto sites_all = read_file(out_dir + "/sites_all.csv");
    // Header plus at least one detected site row.
    MCSWELL_CHECK(sites_all.find("source_site_id,x,y,z,epsilon_midpoint") == 0);
    MCSWELL_CHECK(sites_all.find("\n0,") != std::string::npos);
    // The clean 0->1 step at mu=-6 should give a finite, well-defined fit
    // (not one of the "all_empty"/"all_occupied"/"root_not_bracketed" escapes).
    MCSWELL_CHECK(sites_all.find(",ok,") != std::string::npos);

    const auto meta = read_file(out_dir + "/binomial_metadata.json");
    MCSWELL_CHECK(meta.find("\"n_detected_sites\"") != std::string::npos);

    std::filesystem::remove_all(save_path);
}

void test_gci_analysis_end_to_end() {
    auto [frames_by_mu, mu_values] = build_synthetic_titration();
    const auto cfg = make_config();
    const std::string save_path = "/tmp/mcswell_test_pipeline_gci";
    std::filesystem::remove_all(save_path);

    const auto out_dir = run_gci_analysis(frames_by_mu, mu_values, cfg, /*n_insertion_points=*/2960,
                                           /*sampler_volume=*/1000.0, save_path);

    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/region_titration.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/local_titration.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/region_gci_pmf.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/region_gci_model.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/sites.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/local_gci_sites.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/local_gci_models.csv"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/sites.pdb"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/gO.dx"));
    MCSWELL_CHECK(std::filesystem::exists(out_dir + "/gci_metadata.json"));

    // The clean 0->1->2->3 region profile should recover exactly those
    // asymptotes -- this is the region_model.n_min/n_max fed straight from
    // fit_logistic_sum, so this is really checking the wiring (did the
    // right B/N arrays reach the fit) more than the fit itself.
    const auto pmf = read_file(out_dir + "/region_gci_pmf.csv");
    MCSWELL_CHECK(pmf.find("\n0,") != std::string::npos); // N=0 row present
    MCSWELL_CHECK(pmf.find("\n3,") != std::string::npos); // N=3 row present

    const auto model_csv = read_file(out_dir + "/region_gci_model.csv");
    MCSWELL_CHECK(model_csv.find("term,amplitude,slope,center,n_min,n_max") == 0);
    MCSWELL_CHECK(model_csv.find("\n0,") != std::string::npos); // at least one term row

    const auto meta = read_file(out_dir + "/gci_metadata.json");
    MCSWELL_CHECK(meta.find("\"region_fit_N_min\": 0") != std::string::npos);
    MCSWELL_CHECK(meta.find("\"region_fit_N_max\": 3") != std::string::npos);

    std::filesystem::remove_all(save_path);
}

} // namespace

void run_pipeline_analysis_tests() {
    test_binomial_analysis_end_to_end();
    test_gci_analysis_end_to_end();
}
