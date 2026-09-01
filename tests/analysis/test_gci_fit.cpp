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

#include "analysis/gci_fit.hpp"
#include "harness.hpp"

#include <cmath>

using mcswell::analysis::first_water_score_from_pmf;
using mcswell::analysis::fit_logistic_sum;
using mcswell::analysis::gci_pmf;
using mcswell::analysis::LogisticSumFitOptions;
using mcswell::analysis::LogisticSumModel;
using mcswell::analysis::MuWindowFrames;
using mcswell::analysis::WaterFrame;

namespace {

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

void test_fit_recovers_single_logistic() {
    const double true_slope = 1.5, true_center = -10.0;
    std::vector<double> B, N;
    for (double b = -20.0; b <= 0.0; b += 1.0) {
        B.push_back(b);
        N.push_back(sigmoid(true_slope * (b - true_center))); // n_min=0, n_max=1, amplitude=1
    }

    LogisticSumFitOptions opts;
    opts.seed = 1;
    auto model = fit_logistic_sum(B, N, opts);

    MCSWELL_CHECK_NEAR(model.n_min, 0.0, 1e-9);
    MCSWELL_CHECK_NEAR(model.n_max, 1.0, 1e-9);
    MCSWELL_CHECK(model.amplitudes.size() == 1);
    MCSWELL_CHECK_NEAR(model.amplitudes[0], 1.0, 1e-2);
    MCSWELL_CHECK_NEAR(model.slopes[0], true_slope, 5e-2);
    MCSWELL_CHECK_NEAR(model.centers[0], true_center, 5e-2);
    MCSWELL_CHECK(model.rmse < 1e-3);
}

void test_fit_recovers_two_step_logistic() {
    const double c1 = -15.0, c2 = -5.0, slope = 1.0;
    std::vector<double> B, N;
    for (double b = -30.0; b <= 10.0; b += 1.0) {
        B.push_back(b);
        N.push_back(sigmoid(slope * (b - c1)) + sigmoid(slope * (b - c2))); // n_min=0, n_max=2
    }

    LogisticSumFitOptions opts;
    opts.n_steps = 2;
    opts.seed = 7;
    auto model = fit_logistic_sum(B, N, opts);

    MCSWELL_CHECK_NEAR(model.n_min, 0.0, 1e-9);
    MCSWELL_CHECK_NEAR(model.n_max, 2.0, 1e-9);
    MCSWELL_CHECK(model.amplitudes.size() == 2);
    // Sorted ascending by center.
    MCSWELL_CHECK_NEAR(model.centers[0], c1, 0.1);
    MCSWELL_CHECK_NEAR(model.centers[1], c2, 0.1);
    MCSWELL_CHECK(model.rmse < 1e-2);
}

void test_gci_pmf_closed_form_single_step() {
    // Hand-built model (not fit): n_min=0, n_max=1, single term. For the
    // n=1 (== n_max) row, dimensionless_transfer_from_nmin takes the
    // closed-form branch (sum(amplitude*center)), independent of slope --
    // a good closed-form check that doesn't depend on the root finder.
    LogisticSumModel model;
    model.n_min = 0.0;
    model.n_max = 1.0;
    model.amplitudes = {1.0};
    model.slopes = {2.0};
    model.centers = {5.0};

    const double kT = 0.6;
    const double mu_bulk = -6.0;
    const double volume = 29.914;
    const double standard_volume = 29.914; // ln(V/V0) == 0 -> no volume correction

    std::vector<double> B_obs = {0.0, 10.0}; // brackets, unused by the n=0/n=1 closed-form branches
    auto pmf = gci_pmf(model, B_obs, kT, volume, standard_volume, mu_bulk);

    MCSWELL_CHECK(pmf.size() == 2);
    MCSWELL_CHECK_NEAR(pmf[0].delta_g_insert_kcal_mol, 0.0, 1e-12);
    MCSWELL_CHECK_NEAR(pmf[0].delta_g_bind_network_kcal_mol, 0.0, 1e-12);
    MCSWELL_CHECK_NEAR(pmf[1].beta_dF_raw, 5.0, 1e-9); // == center, per the closed-form branch
    MCSWELL_CHECK_NEAR(pmf[1].delta_g_insert_kcal_mol, 5.0 * kT, 1e-9);
    MCSWELL_CHECK_NEAR(pmf[1].delta_g_bind_network_kcal_mol, 5.0 * kT - mu_bulk, 1e-9);

    auto score = first_water_score_from_pmf(pmf);
    MCSWELL_CHECK(score.status == "ok");
    MCSWELL_CHECK_NEAR(score.delta_g_insert, 5.0 * kT, 1e-9);
    MCSWELL_CHECK_NEAR(score.delta_g_bind, 5.0 * kT - mu_bulk, 1e-9);
}

void test_first_water_score_missing_states() {
    LogisticSumModel model;
    model.n_min = 2.0;
    model.n_max = 3.0;
    model.amplitudes = {1.0};
    model.slopes = {1.0};
    model.centers = {0.0};

    std::vector<double> B_obs = {-5.0, 5.0};
    auto pmf = gci_pmf(model, B_obs, 0.6, 29.914, 29.914, -6.0);
    auto score = first_water_score_from_pmf(pmf); // doesn't span N=0..1
    MCSWELL_CHECK(score.status == "does_not_span_0_to_1");
    MCSWELL_CHECK(std::isnan(score.delta_g_insert));
}

void test_count_region_and_local_by_mu() {
    std::vector<std::array<double, 3>> sites = {{0.0, 0.0, 0.0}, {20.0, 0.0, 0.0}};

    MuWindowFrames w0;
    WaterFrame f0a;
    f0a.oxygens = {{0.1, 0.0, 0.0}, {0.2, 0.0, 0.0}, {30.0, 0.0, 0.0}}; // 2 near site0, 1 far from both
    WaterFrame f0b;
    f0b.oxygens = {{0.1, 0.0, 0.0}};
    w0.frames = {f0a, f0b};

    std::vector<MuWindowFrames> frames_by_mu = {w0};

    auto result = mcswell::analysis::count_region_and_local_by_mu(frames_by_mu, sites, /*local_radius=*/1.0);

    MCSWELL_CHECK(result.n_frames[0] == 2);
    MCSWELL_CHECK_NEAR(result.region_mean[0], 2.0, 1e-9); // (3+1)/2

    // site0: frame0 has 2 within radius, frame1 has 1 -> mean 1.5.
    MCSWELL_CHECK_NEAR(result.local_mean[0 * 1 + 0], 1.5, 1e-9);
    // site1: never has anything nearby.
    MCSWELL_CHECK_NEAR(result.local_mean[1 * 1 + 0], 0.0, 1e-9);
}

} // namespace

void run_gci_fit_tests() {
    test_fit_recovers_single_logistic();
    test_fit_recovers_two_step_logistic();
    test_gci_pmf_closed_form_single_step();
    test_first_water_score_missing_states();
    test_count_region_and_local_by_mu();
}
