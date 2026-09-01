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
#include <limits>
#include <string>
#include <vector>

namespace mcswell::analysis {

// Bounded independent-site grand-canonical occupancy isotherm:
//   p(mu) = 1 / (1 + exp(-beta * (mu - epsilon))), beta = 1 / (k_B * T).
double independent_site_occupancy(double mu, double epsilon, double temperature);

struct BinomialFitResult {
    double epsilon = std::numeric_limits<double>::quiet_NaN();
    double epsilon_se = std::numeric_limits<double>::quiet_NaN();
    std::size_t n_mu_points = 0;
    std::size_t total_occupied = 0;
    std::size_t total_frames = 0;
    double fit_rmse_p = std::numeric_limits<double>::quiet_NaN();
    double log_likelihood = std::numeric_limits<double>::quiet_NaN();
    // "ok", "no_frames", "all_empty", "all_occupied", or "root_not_bracketed".
    std::string status;
};

// Binomial maximum-likelihood fit of the isotherm's midpoint chemical
// potential (epsilon) given, at each sampled mu_i, k_i occupied frames out
// of n_i observed. Windows with n_i == 0 are dropped before fitting.
BinomialFitResult fit_epsilon_binomial(
    const std::vector<double>& mu_values,
    const std::vector<std::size_t>& occ_counts,
    const std::vector<std::size_t>& n_frames,
    double temperature
);

struct OccupancyResult {
    // Row-major (site-major): p[site * n_mu + mu_idx].
    std::vector<double> p;
    std::vector<std::size_t> occ_counts;
    std::vector<std::size_t> n_frames; // size n_mu
    std::size_t n_sites = 0;
    std::size_t n_mu = 0;
};

// Per-frame binary site occupancy: a site counts as occupied in a frame if
// its nearest water oxygen (within `cutoff`) is closer than to any other
// site (at most one water assigned per site per frame -- the closest one).
// Operates on every window in `frames_by_mu`, not just capacity-safe ones,
// matching the old pipeline's "report every sampled window" behavior.
OccupancyResult occupancy_probabilities_by_mu(
    const std::vector<MuWindowFrames>& frames_by_mu,
    const std::vector<std::array<double, 3>>& site_centers,
    double cutoff
);

} // namespace mcswell::analysis
