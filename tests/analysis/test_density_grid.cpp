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

// Ground-truth values in this file (gaussian_filter, percentile,
// maximum_filter-based peak detection) were generated with the real
// scipy/numpy in the `mcswell` environment, not hand-derived -- see the
// generator scripts referenced in the port's development notes. This
// catches boundary-condition/normalization bugs that hand arithmetic
// would likely get subtly wrong.

#include "analysis/density_grid.hpp"
#include "harness.hpp"

#include <algorithm>

using mcswell::analysis::compute_density_grid;
using mcswell::analysis::DensityGrid;
using mcswell::analysis::find_peaks;
using mcswell::analysis::gaussian_smooth;
using mcswell::analysis::make_edges;
using mcswell::analysis::merge_close_sites;
using mcswell::analysis::Peak;

namespace {

void test_make_edges_evenly_divisible() {
    auto edges = make_edges({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 0.5);
    for (int d = 0; d < 3; ++d) {
        MCSWELL_CHECK(edges[static_cast<std::size_t>(d)].size() == 5);
    }
    const std::vector<double> expected = {-1.0, -0.5, 0.0, 0.5, 1.0};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        MCSWELL_CHECK_NEAR(edges[0][i], expected[i], 1e-12);
    }
}

void test_make_edges_not_evenly_divisible() {
    // 2*halfsize/spacing = 2/0.3 = 6.666..., ceil -> 7 bins, 8 edge points,
    // so the true bin width (2/7) differs from the nominal spacing (0.3).
    auto edges = make_edges({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 0.3);
    MCSWELL_CHECK(edges[0].size() == 8);
    MCSWELL_CHECK_NEAR(edges[0].front(), -1.0, 1e-12);
    MCSWELL_CHECK_NEAR(edges[0].back(), 1.0, 1e-12);
    const double true_bin_width = edges[0][1] - edges[0][0];
    MCSWELL_CHECK_NEAR(true_bin_width, 2.0 / 7.0, 1e-12);
}

void test_compute_density_grid_binning() {
    auto edges = make_edges({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 0.5); // [-1,-0.5,0,0.5,1]

    std::vector<std::array<double, 3>> points = {
        {-0.9, -0.9, -0.9},  // -> bin (0,0,0)
        {0.9, 0.9, 0.9},     // -> bin (3,3,3) (last bin, right-open normally, but < 1.0)
        {1.0, 1.0, 1.0},     // exactly on the outer edge -> still bin (3,3,3) (last bin closed both ends)
        {0.0, 0.0, 0.0},     // exactly on an *internal* edge -> half-open means bin (2,2,2), not (1,1,1)
        {2.0, 0.0, 0.0},     // out of range on x -> dropped entirely
    };

    auto grid = compute_density_grid(points, edges);
    MCSWELL_CHECK(grid.shape[0] == 4 && grid.shape[1] == 4 && grid.shape[2] == 4);

    double total = 0.0;
    for (double v : grid.values) total += v;
    MCSWELL_CHECK_NEAR(total, 4.0, 1e-12); // one point dropped

    MCSWELL_CHECK_NEAR(grid.values[grid.index(0, 0, 0)], 1.0, 1e-12);
    MCSWELL_CHECK_NEAR(grid.values[grid.index(3, 3, 3)], 2.0, 1e-12);
    MCSWELL_CHECK_NEAR(grid.values[grid.index(2, 2, 2)], 1.0, 1e-12);
    MCSWELL_CHECK_NEAR(grid.values[grid.index(1, 1, 1)], 0.0, 1e-12);
}

void test_gaussian_smooth_1d_reflect() {
    // shape (5,1,1), single impulse at index 2, sigma=0.6. ny=nz=1 makes
    // the y/z passes no-ops (reflect_index(_, 1) == 0), isolating the
    // reflect-boundary behavior of the axis-0 pass.
    DensityGrid grid;
    grid.shape = {5, 1, 1};
    grid.values = {0.0, 0.0, 1.0, 0.0, 0.0};
    grid.edges[0] = {0, 1, 2, 3, 4, 5};
    grid.edges[1] = {0, 1};
    grid.edges[2] = {0, 1};

    auto smoothed = gaussian_smooth(grid, 0.6);

    // scipy.ndimage.gaussian_filter(a, sigma=0.6, mode='reflect') on the
    // same impulse array.
    const std::vector<double> expected = {
        0.0025662686485205368, 0.16552456665899618, 0.6638183293849667,
        0.16552456665899618, 0.0025662686485205368};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        MCSWELL_CHECK_NEAR(smoothed.values[smoothed.index(i, 0, 0)], expected[i], 1e-9);
    }
}

void test_find_peaks_matches_scipy() {
    // 4x4x4 grid on a unit-spacing box centered at the origin: edges =
    // [-2,-1,0,1,2] per axis, so bin(i,j,k)'s center is (i-1.5, j-1.5, k-1.5).
    auto edges = make_edges({0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}, 1.0);

    DensityGrid grid;
    grid.shape = {4, 4, 4};
    grid.edges = edges;
    grid.values = {
        0.0, 6.0, 5.0, 0.0, 9.0, 2.0, 2.0, 0.0, 3.0, 0.0, 3.0, 8.0, 4.0, 9.0, 4.0, 2.0,
        7.0, 8.0, 8.0, 8.0, 0.0, 5.0, 2.0, 2.0, 2.0, 8.0, 7.0, 2.0, 4.0, 7.0, 0.0, 6.0,
        4.0, 9.0, 7.0, 2.0, 8.0, 7.0, 2.0, 5.0, 7.0, 2.0, 2.0, 0.0, 0.0, 4.0, 0.0, 5.0,
        4.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 7.0, 2.0, 9.0, 0.0, 6.0, 5.0, 9.0, 6.0, 8.0,
    };

    auto peaks = find_peaks(grid, /*percentile=*/90.0, /*neighborhood=*/1);

    // Ground truth from a scipy implementation of the same algorithm
    // (percentile threshold + maximum_filter(footprint=ones(3,3,3),
    // mode='nearest') + peak_mask), grid indices (i,j,k) -> center/score.
    std::vector<Peak> expected;
    auto add = [&](double i, double j, double k, double score) {
        Peak p;
        p.center = {i - 1.5, j - 1.5, k - 1.5};
        p.score = score;
        expected.push_back(p);
    };
    add(0, 1, 0, 9.0);
    add(0, 3, 1, 9.0);
    add(2, 0, 1, 9.0);
    add(3, 2, 1, 9.0);
    add(3, 3, 1, 9.0);

    MCSWELL_CHECK(peaks.size() == expected.size());
    for (const auto& e : expected) {
        bool found = false;
        for (const auto& p : peaks) {
            if (std::abs(p.center[0] - e.center[0]) < 1e-9 &&
                std::abs(p.center[1] - e.center[1]) < 1e-9 &&
                std::abs(p.center[2] - e.center[2]) < 1e-9 &&
                std::abs(p.score - e.score) < 1e-9) {
                found = true;
                break;
            }
        }
        MCSWELL_CHECK(found);
    }
}

void test_merge_close_sites() {
    std::vector<Peak> peaks;
    // Two peaks close together (should merge into their weighted centroid)
    // and one far-away singleton (should pass through unchanged).
    peaks.push_back({{0.0, 0.0, 0.0}, 1.0});
    peaks.push_back({{1.0, 0.0, 0.0}, 3.0}); // weight 3x the first
    peaks.push_back({{100.0, 0.0, 0.0}, 5.0});

    auto merged = merge_close_sites(peaks, /*cutoff=*/1.5);
    MCSWELL_CHECK(merged.size() == 2);

    bool found_pair = false, found_singleton = false;
    for (const auto& m : merged) {
        if (std::abs(m.score - 5.0) < 1e-9 && std::abs(m.center[0] - 100.0) < 1e-9) {
            found_singleton = true;
        }
        if (std::abs(m.score - 4.0) < 1e-9) {
            // Weighted centroid: (0*1 + 1*3) / 4 = 0.75
            MCSWELL_CHECK_NEAR(m.center[0], 0.75, 1e-9);
            found_pair = true;
        }
    }
    MCSWELL_CHECK(found_pair);
    MCSWELL_CHECK(found_singleton);
}

} // namespace

void run_density_grid_tests() {
    test_make_edges_evenly_divisible();
    test_make_edges_not_evenly_divisible();
    test_compute_density_grid_binning();
    test_gaussian_smooth_1d_reflect();
    test_find_peaks_matches_scipy();
    test_merge_close_sites();
}
