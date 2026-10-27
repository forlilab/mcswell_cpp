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

#include "analysis/frame_extraction.hpp"
#include "cuda/consts.cuh"
#include "harness.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

using mcswell::analysis::CapacityScanResult;
using mcswell::analysis::extract_oxygen_frames;
using mcswell::analysis::MuWindowFrames;
using mcswell::analysis::scan_capacity_and_collect_density_points;
using mcswell::analysis::WaterFrame;

namespace {

// Builds a WaterSnapshots from per-frame oxygen lists, in frame order
// (mu-major, then snapshot). Hydrogens are left at zero; only the oxygens
// matter to the extraction.
mcswell::WaterSnapshots make_snapshots(
    std::size_t n_mu, std::size_t n_snapshots,
    const std::vector<std::vector<std::array<float, 3>>>& frames) {
    const auto ws = static_cast<std::size_t>(WATER_SIZE);
    mcswell::WaterSnapshots s;
    s.n_mu = n_mu;
    s.n_snapshots = n_snapshots;
    s.offsets.push_back(0);
    for (const auto& frame : frames) {
        for (const auto& o : frame) {
            const std::size_t base = s.waters.size();
            s.waters.resize(base + ws, 0.0f);
            s.waters[base + OX] = o[0];
            s.waters[base + OY] = o[1];
            s.waters[base + OZ] = o[2];
        }
        s.offsets.push_back(s.offsets.back() + frame.size());
    }
    return s;
}

void test_extract_oxygen_frames() {
    const std::size_t n_mu = 2, n_snapshots = 2;
    const auto snaps = make_snapshots(n_mu, n_snapshots, {
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},    // mu=0, snap=0
        {},                                          // mu=0, snap=1: no waters
        {{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}}, // mu=1, snap=0
        {{0.5f, 0.5f, 0.5f}},                        // mu=1, snap=1
    });

    auto frames = extract_oxygen_frames(snaps);

    MCSWELL_CHECK(frames.size() == n_mu);
    MCSWELL_CHECK(frames[0].frames.size() == n_snapshots);
    MCSWELL_CHECK(frames[1].frames.size() == n_snapshots);

    const auto& mu0_snap0 = frames[0].frames[0].oxygens;
    MCSWELL_CHECK(mu0_snap0.size() == 2);
    MCSWELL_CHECK_NEAR(mu0_snap0[0][0], 1.0, 1e-6);
    MCSWELL_CHECK_NEAR(mu0_snap0[1][2], 6.0, 1e-6);

    MCSWELL_CHECK(frames[0].frames[1].oxygens.empty());

    const auto& mu1_snap0 = frames[1].frames[0].oxygens;
    MCSWELL_CHECK(mu1_snap0.size() == 2);
    MCSWELL_CHECK_NEAR(mu1_snap0[1][1], 11.0, 1e-6);

    const auto& mu1_snap1 = frames[1].frames[1].oxygens;
    MCSWELL_CHECK(mu1_snap1.size() == 1);
    MCSWELL_CHECK_NEAR(mu1_snap1[0][0], 0.5, 1e-6);
}

void test_extract_keeps_water_at_origin() {
    // The old padded buffer used an all-zero oxygen to mark an empty slot,
    // so a real water at exactly the origin was silently dropped. Frame
    // sizes now come from the offsets alone.
    const auto snaps = make_snapshots(1, 1, {{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}});
    auto frames = extract_oxygen_frames(snaps);
    MCSWELL_CHECK(frames[0].frames[0].oxygens.size() == 2);
    MCSWELL_CHECK_NEAR(frames[0].frames[0].oxygens[0][0], 0.0, 1e-12);
}

void test_extract_rejects_malformed_snapshots() {
    auto snaps = make_snapshots(1, 2, {{{1.0f, 2.0f, 3.0f}}, {}});
    snaps.waters.pop_back();  // one float short of offsets.back() * WATER_SIZE
    bool threw = false;
    try {
        extract_oxygen_frames(snaps);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    MCSWELL_CHECK(threw);
}

MuWindowFrames make_window(std::vector<std::size_t> counts_per_frame) {
    MuWindowFrames w;
    for (std::size_t c : counts_per_frame) {
        WaterFrame f;
        for (std::size_t i = 0; i < c; ++i) {
            f.oxygens.push_back({static_cast<double>(i), 0.0, 0.0});
        }
        w.frames.push_back(std::move(f));
    }
    return w;
}

void test_capacity_scan_contiguous_prefix() {
    // mu0: well below capacity -> valid, "capacity_safe".
    // mu1: every frame at/over capacity -> first excluded window.
    // mu2: below capacity on its own merits, but excluded anyway because
    //      it comes after the first capacity-limited window.
    std::vector<MuWindowFrames> frames_by_mu = {
        make_window({1, 2}),
        make_window({5, 5}),
        make_window({1, 1}),
    };
    std::vector<double> mu_values = {-10.0, -5.0, 0.0};

    auto result = scan_capacity_and_collect_density_points(
        frames_by_mu, mu_values, /*water_capacity=*/5,
        /*capacity_filter=*/true,
        /*max_capacity_hit_fraction=*/0.01,
        /*max_mean_capacity_fraction=*/0.90);

    MCSWELL_CHECK(result.valid_mask.size() == 3);
    MCSWELL_CHECK(result.valid_mask[0]);
    MCSWELL_CHECK(!result.valid_mask[1]);
    MCSWELL_CHECK(!result.valid_mask[2]);

    MCSWELL_CHECK(result.diagnostics[0].capacity_filter_reason == "capacity_safe");
    MCSWELL_CHECK(result.diagnostics[1].capacity_filter_reason == "first_capacity_limited_window");
    MCSWELL_CHECK(result.diagnostics[2].capacity_filter_reason == "after_first_capacity_limited_window");

    // Only mu0's 1 + 2 = 3 oxygens should be pooled.
    MCSWELL_CHECK(result.density_points.size() == 3);

    MCSWELL_CHECK_NEAR(result.diagnostics[0].n_region_mean, 1.5, 1e-9);
    MCSWELL_CHECK(result.diagnostics[0].n_region_max_observed == 2);
    MCSWELL_CHECK_NEAR(result.diagnostics[1].capacity_hit_fraction, 1.0, 1e-9);
}

void test_capacity_scan_filter_disabled() {
    std::vector<MuWindowFrames> frames_by_mu = {make_window({1}), make_window({5}), make_window({1})};
    std::vector<double> mu_values = {-10.0, -5.0, 0.0};

    auto result = scan_capacity_and_collect_density_points(
        frames_by_mu, mu_values, /*water_capacity=*/5,
        /*capacity_filter=*/false,
        /*max_capacity_hit_fraction=*/0.01,
        /*max_mean_capacity_fraction=*/0.90);

    MCSWELL_CHECK(result.valid_mask[0] && result.valid_mask[1] && result.valid_mask[2]);
    MCSWELL_CHECK(result.density_points.size() == 1 + 5 + 1);
    for (const auto& d : result.diagnostics) {
        MCSWELL_CHECK(d.capacity_filter_reason == "capacity_filter_disabled");
    }
}

void test_capacity_scan_no_limit() {
    // water_capacity == 0: the sampler had no hard limit, so even a window
    // full of waters cannot have been truncated and stays valid.
    std::vector<MuWindowFrames> frames_by_mu = {make_window({1}), make_window({50, 60}), make_window({1})};
    std::vector<double> mu_values = {-10.0, -5.0, 0.0};

    auto result = scan_capacity_and_collect_density_points(
        frames_by_mu, mu_values, /*water_capacity=*/0,
        /*capacity_filter=*/true,
        /*max_capacity_hit_fraction=*/0.01,
        /*max_mean_capacity_fraction=*/0.90);

    MCSWELL_CHECK(result.valid_mask[0] && result.valid_mask[1] && result.valid_mask[2]);
    MCSWELL_CHECK(result.density_points.size() == 1 + 50 + 60 + 1);
    for (const auto& d : result.diagnostics) {
        MCSWELL_CHECK(d.capacity_filter_reason == "no_capacity_limit");
        MCSWELL_CHECK(d.water_capacity == 0);
        MCSWELL_CHECK(std::isnan(d.capacity_mean_fraction));
        MCSWELL_CHECK(std::isnan(d.capacity_hit_fraction));
    }
    MCSWELL_CHECK(result.diagnostics[1].n_region_max_observed == 60);
}

void test_capacity_scan_validation_errors() {
    std::vector<MuWindowFrames> frames_by_mu = {make_window({1})};
    std::vector<double> mismatched = {-10.0, -5.0};
    bool threw = false;
    try {
        scan_capacity_and_collect_density_points(frames_by_mu, mismatched, 5, true, 0.01, 0.9);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    MCSWELL_CHECK(threw);
}

} // namespace

void run_frame_extraction_tests() {
    test_extract_oxygen_frames();
    test_extract_keeps_water_at_origin();
    test_extract_rejects_malformed_snapshots();
    test_capacity_scan_contiguous_prefix();
    test_capacity_scan_filter_disabled();
    test_capacity_scan_no_limit();
    test_capacity_scan_validation_errors();
}
