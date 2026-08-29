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

#include "analysis/binomial_fit.hpp"
#include "harness.hpp"

#include <cmath>

using mcswell::analysis::BinomialFitResult;
using mcswell::analysis::fit_epsilon_binomial;
using mcswell::analysis::independent_site_occupancy;
using mcswell::analysis::MuWindowFrames;
using mcswell::analysis::occupancy_probabilities_by_mu;
using mcswell::analysis::WaterFrame;

namespace {

void test_fit_recovers_known_epsilon() {
    const double true_epsilon = -15.0;
    const double temperature = 300.0;
    const std::size_t n_per_window = 100000; // large N minimizes rounding-induced bias

    std::vector<double> mu_values;
    std::vector<std::size_t> occ_counts;
    std::vector<std::size_t> n_frames;

    for (double mu = -40.0; mu <= 0.0 + 1e-9; mu += 2.0) {
        const double p = independent_site_occupancy(mu, true_epsilon, temperature);
        mu_values.push_back(mu);
        occ_counts.push_back(static_cast<std::size_t>(std::lround(p * static_cast<double>(n_per_window))));
        n_frames.push_back(n_per_window);
    }

    auto fit = fit_epsilon_binomial(mu_values, occ_counts, n_frames, temperature);
    MCSWELL_CHECK(fit.status == "ok");
    MCSWELL_CHECK_NEAR(fit.epsilon, true_epsilon, 1e-2);
    MCSWELL_CHECK(fit.epsilon_se > 0.0);
    MCSWELL_CHECK(fit.fit_rmse_p < 1e-3);
}

void test_fit_all_empty_and_all_occupied() {
    std::vector<double> mu_values = {-10.0, -5.0, 0.0};

    auto empty = fit_epsilon_binomial(mu_values, {0, 0, 0}, {10, 10, 10}, 300.0);
    MCSWELL_CHECK(empty.status == "all_empty");
    MCSWELL_CHECK(std::isnan(empty.epsilon));

    auto full = fit_epsilon_binomial(mu_values, {10, 10, 10}, {10, 10, 10}, 300.0);
    MCSWELL_CHECK(full.status == "all_occupied");
    MCSWELL_CHECK(std::isnan(full.epsilon));
}

void test_fit_no_frames() {
    std::vector<double> mu_values = {-10.0, -5.0};
    auto fit = fit_epsilon_binomial(mu_values, {0, 0}, {0, 0}, 300.0);
    MCSWELL_CHECK(fit.status == "no_frames");
}

MuWindowFrames make_frames(std::vector<std::vector<std::array<double, 3>>> frames) {
    MuWindowFrames w;
    for (auto& oxy : frames) {
        WaterFrame f;
        f.oxygens = std::move(oxy);
        w.frames.push_back(std::move(f));
    }
    return w;
}

void test_occupancy_probabilities() {
    // Two sites, far apart.
    std::vector<std::array<double, 3>> sites = {{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}};

    // mu window 0: 2 frames.
    //   frame0: one water near site0 (occupies it), nothing near site1.
    //   frame1: two waters both near site0 (at different distances) -- only
    //           the closer one should count, still 1 occupied frame for
    //           site0, and site1 unoccupied.
    // mu window 1: 1 frame with a water far from both sites (cutoff miss).
    std::vector<MuWindowFrames> frames_by_mu;
    frames_by_mu.push_back(make_frames({
        {{0.1, 0.0, 0.0}},
        {{0.3, 0.0, 0.0}, {0.05, 0.0, 0.0}},
    }));
    frames_by_mu.push_back(make_frames({
        {{50.0, 0.0, 0.0}},
    }));

    auto result = occupancy_probabilities_by_mu(frames_by_mu, sites, /*cutoff=*/1.0);

    MCSWELL_CHECK(result.n_sites == 2);
    MCSWELL_CHECK(result.n_mu == 2);
    MCSWELL_CHECK(result.n_frames[0] == 2);
    MCSWELL_CHECK(result.n_frames[1] == 1);

    // site0 (row 0), mu0 (col 0): occupied in both frames -> occ=2, p=1.0.
    MCSWELL_CHECK(result.occ_counts[0 * result.n_mu + 0] == 2);
    MCSWELL_CHECK_NEAR(result.p[0 * result.n_mu + 0], 1.0, 1e-12);

    // site1 (row 1), mu0: never occupied.
    MCSWELL_CHECK(result.occ_counts[1 * result.n_mu + 0] == 0);
    MCSWELL_CHECK_NEAR(result.p[1 * result.n_mu + 0], 0.0, 1e-12);

    // mu1: water at distance 50 from both sites, cutoff=1.0 -> no site occupied.
    MCSWELL_CHECK(result.occ_counts[0 * result.n_mu + 1] == 0);
    MCSWELL_CHECK(result.occ_counts[1 * result.n_mu + 1] == 0);
}

} // namespace

void run_binomial_fit_tests() {
    test_fit_recovers_known_epsilon();
    test_fit_all_empty_and_all_occupied();
    test_fit_no_frames();
    test_occupancy_probabilities();
}
