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

// Ground-truth values in this file (gaussian_filter, percentile, and the
// hydration-site finder, from WaterKit's own _hydration_sites loop) were
// generated with the real scipy/numpy in the `mcswell` environment, not
// hand-derived. This
// catches boundary-condition/normalization bugs that hand arithmetic
// would likely get subtly wrong.

#include "analysis/density_grid.hpp"
#include "harness.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using mcswell::analysis::compute_density_grid;
using mcswell::analysis::DensityGrid;
using mcswell::analysis::find_hydration_sites;
using mcswell::analysis::gaussian_smooth;
using mcswell::analysis::make_edges;
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

void test_find_hydration_sites_matches_waterkit() {
    // 6x6x6 grid, 0.5 A voxels, centres -1.25 ... 1.25 per axis. Random
    // distinct densities (about 20% zero), 60th percentile, 1.2 A separation.
    // Expected sites from WaterKit's _hydration_sites loop (cdist-based) on
    // the same voxel centres and values.
    auto edges = make_edges({0.0, 0.0, 0.0}, {1.5, 1.5, 1.5}, 0.5);
    DensityGrid grid;
    grid.shape = {6, 6, 6};
    grid.edges = edges;
    grid.values = {
        62.51, 89.721, 0.0, 22.521, 30.017, 0.0, 0.527, 82.123, 79.707, 0.0, 30.303, 27.843,
        25.487, 44.508, 50.455, 55.35, 99.55, 79.266, 0.0, 0.0, 21.531, 16.021, 61.254, 4.394,
        3.568, 51.489, 46.621, 91.717, 0.0, 51.412, 0.0, 24.751, 1.179, 19.24, 69.203, 20.061,
        36.954, 0.373, 83.005, 15.446, 0.0, 88.033, 50.979, 84.715, 0.0, 74.177, 0.0, 0.0,
        0.0, 0.0, 36.126, 59.818, 5.925, 38.763, 32.304, 0.0, 81.634, 37.945, 97.875, 58.999,
        0.0, 63.8, 67.645, 15.079, 44.031, 23.956, 0.0, 9.67, 96.783, 0.0, 67.177, 30.042,
        87.408, 66.221, 0.0, 0.0, 94.495, 90.392, 56.972, 14.546, 19.246, 92.791, 55.233, 18.055,
        88.406, 0.0, 56.969, 37.629, 41.096, 23.949, 3.806, 87.622, 0.0, 54.764, 32.216, 75.132,
        2.52, 37.219, 3.035, 12.289, 96.715, 65.776, 0.0, 52.374, 87.281, 0.0, 59.029, 68.368,
        35.541, 0.0, 0.0, 90.918, 0.0, 0.0, 0.518, 75.298, 81.053, 0.0, 41.89, 81.526,
        1.427, 0.0, 79.302, 0.0, 72.585, 22.642, 19.852, 36.313, 0.0, 34.606, 94.812, 57.333,
        34.007, 27.152, 95.204, 0.0, 0.0, 51.552, 0.0, 0.0, 74.277, 58.065, 0.0, 87.819,
        41.165, 92.276, 6.872, 43.0, 51.951, 95.094, 25.1, 80.604, 67.647, 0.0, 0.0, 0.0,
        33.268, 39.828, 20.291, 5.07, 21.291, 91.546, 84.017, 11.241, 0.0, 47.92, 59.468, 0.0,
        30.666, 96.135, 46.584, 62.81, 63.523, 18.389, 6.187, 41.152, 76.403, 81.522, 72.999, 0.0,
        91.335, 0.0, 87.769, 52.33, 91.564, 4.665, 0.0, 2.022, 0.0, 24.857, 18.75, 56.706,
        3.899, 59.039, 16.601, 67.787, 2.108, 31.057, 93.834, 53.84, 81.159, 65.803, 61.075, 19.125,
        57.439, 3.969, 0.0, 96.007, 0.0, 5.071, 33.866, 31.8, 11.272, 62.661, 79.746, 0.0
    };

    auto sites = find_hydration_sites(grid, /*percentile=*/60.0, /*min_separation=*/1.2);

    std::vector<Peak> expected;
    auto add = [&](double x, double y, double z, double score) { expected.push_back({{x, y, z}, score}); };
    add(-1.25, -0.25, 0.75, 99.55);
    add(-0.75, 1.25, -0.25, 96.783);
    add(-0.25, 0.75, 0.75, 96.715);
    add(0.75, 0.75, -0.75, 96.135);
    add(0.75, -1.25, 1.25, 95.094);
    add(-0.25, -0.75, 0.25, 92.791);
    add(0.75, -1.25, -0.75, 92.276);
    add(-1.25, -1.25, -0.75, 89.721);
    add(-0.25, -0.25, -1.25, 88.406);
    add(-0.75, -1.25, 1.25, 88.033);
    add(1.25, 1.25, 0.75, 79.746);
    add(1.25, -0.25, 0.25, 67.787);

    // Same sites, same (decreasing-score) order.
    MCSWELL_CHECK(sites.size() == expected.size());
    for (std::size_t i = 0; i < std::min(sites.size(), expected.size()); ++i) {
        for (std::size_t d = 0; d < 3; ++d) MCSWELL_CHECK_NEAR(sites[i].center[d], expected[i].center[d], 1e-9);
        MCSWELL_CHECK_NEAR(sites[i].score, expected[i].score, 1e-9);
    }
}

void test_find_hydration_sites_separation() {
    // 1D row of 13 voxels (0.5 A), densities rising to a maximum at index 6.
    // With a 1.0 A separation the exclusion is inclusive: voxels exactly
    // 1.0 A from a site are dropped, so sites sit 1.5 A apart.
    DensityGrid grid;
    grid.shape = {13, 1, 1};
    for (int i = 0; i <= 13; ++i) grid.edges[0].push_back(0.5 * i);
    grid.edges[1] = {0.0, 0.5};
    grid.edges[2] = {0.0, 0.5};
    for (int i = 0; i < 13; ++i) grid.values.push_back(10.0 - std::abs(i - 6));

    auto sites = find_hydration_sites(grid, /*percentile=*/0.0, /*min_separation=*/1.0);
    // Site voxel indices 6, then 3 and 9 (equal density: lower index first), then 0 and 12.
    const std::vector<double> expected_x = {3.25, 1.75, 4.75, 0.25, 6.25};
    MCSWELL_CHECK(sites.size() == expected_x.size());
    for (std::size_t i = 0; i < std::min(sites.size(), expected_x.size()); ++i) {
        MCSWELL_CHECK_NEAR(sites[i].center[0], expected_x[i], 1e-12);
    }
    for (std::size_t a = 0; a < sites.size(); ++a) {
        for (std::size_t b = a + 1; b < sites.size(); ++b) {
            MCSWELL_CHECK(std::abs(sites[a].center[0] - sites[b].center[0]) > 1.0);
        }
    }

    // A percentile threshold drops the low-density tails before the search.
    auto top = find_hydration_sites(grid, /*percentile=*/50.0, /*min_separation=*/1.0);
    MCSWELL_CHECK(top.size() == 3); // indices 6, 3, 9 (densities 10, 7, 7); 0 and 12 (4) are below threshold
}

void test_find_hydration_sites_rejects_bad_input() {
    DensityGrid grid;
    grid.shape = {2, 1, 1};
    grid.edges[0] = {0.0, 0.5, 1.0};
    grid.edges[1] = {0.0, 0.5};
    grid.edges[2] = {0.0, 0.5};
    grid.values = {0.0, 0.0};
    bool threw = false;
    try { find_hydration_sites(grid, 90.0, 2.5); } catch (const std::runtime_error&) { threw = true; }
    MCSWELL_CHECK(threw); // no density at all
    grid.values = {1.0, 2.0};
    threw = false;
    try { find_hydration_sites(grid, 90.0, 0.0); } catch (const std::invalid_argument&) { threw = true; }
    MCSWELL_CHECK(threw);
}

} // namespace

void run_density_grid_tests() {
    test_make_edges_evenly_divisible();
    test_make_edges_not_evenly_divisible();
    test_compute_density_grid_binning();
    test_gaussian_smooth_1d_reflect();
    test_find_hydration_sites_matches_waterkit();
    test_find_hydration_sites_separation();
    test_find_hydration_sites_rejects_bad_input();
}
