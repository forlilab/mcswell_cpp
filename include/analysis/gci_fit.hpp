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
#include <limits>
#include <string>
#include <vector>

namespace mcswell::analysis {

// Monotone sum-of-logistics titration model:
//   N(B) = n_min + sum_j amplitude_j * sigmoid(slope_j * (B - center_j))
// with integer asymptotes n_min/n_max (the common GCI practice of
// evaluating integer occupancy states) and sum(amplitude_j) == n_max-n_min.
struct LogisticSumModel {
    double n_min = 0.0;
    double n_max = 0.0;
    std::vector<double> amplitudes;
    std::vector<double> slopes;
    std::vector<double> centers;
    double rmse = 0.0;
    double sse = 0.0;

    double delta_n() const { return n_max - n_min; }
    double predict(double B) const;

    // Integral_{-inf}^{B} [N(b) - n_min] db.
    double excess_integral_from_minus_inf(double B) const;

    // Finite inverse for n_min < target_n < n_max, bracket-expanding from
    // [B_lo, B_hi].
    double inverse(double target_n, double B_lo, double B_hi) const;

    // ProtoMS-style GCI transfer term relative to n_min:
    //   Delta(beta F)_raw = DeltaN * B_N - integral_{-inf}^{B_N} [N(B)-n_min] dB
    double dimensionless_transfer_from_nmin(double target_n, double B_lo, double B_hi) const;
};

struct LogisticSumFitOptions {
    int n_steps = -1; // < 1 means "unset": auto = max(1, round(n_max - n_min))
    int max_terms = 12;
    int random_starts = 8;
    std::uint64_t seed = 20260812ULL;
};

// Fits the monotone sum-of-logistics model to (B, N) titration data via a
// reparameterized, multi-start Levenberg-Marquardt search (see
// levenberg_marquardt.hpp for why this isn't scipy's bounded trust-region
// solver). Requires at least 4 finite points spanning >1 integer occupancy.
LogisticSumModel fit_logistic_sum(
    const std::vector<double>& B, const std::vector<double>& N,
    const LogisticSumFitOptions& options = {});

struct GciPmfRow {
    int n = 0;
    double delta_n_from_min = 0.0;
    double beta_dF_raw = 0.0;
    double delta_g_insert_kcal_mol = 0.0;
    double delta_g_bind_network_kcal_mol = 0.0;
    double delta_g_bind_incremental_kcal_mol = std::numeric_limits<double>::quiet_NaN();
};

// Integer-state insertion and bulk-referenced binding PMF, following the
// ProtoMS-style volume correction:
//   beta*DeltaG_insert(N) = beta*DeltaF_raw(N) - DeltaN * ln(V_analysis/V0)
//   DeltaG_bind(N) = DeltaG_insert(N) - DeltaN * mu_bulk
std::vector<GciPmfRow> gci_pmf(
    const LogisticSumModel& model, const std::vector<double>& B_obs, double kT,
    double analysis_volume, double standard_volume, double mu_bulk);

struct FirstWaterScore {
    double delta_g_insert = std::numeric_limits<double>::quiet_NaN();
    double delta_g_bind = std::numeric_limits<double>::quiet_NaN();
    std::string status; // "ok" or "does_not_span_0_to_1"
};

// Insertion/binding dG for adding the first water (N: 0 -> 1).
FirstWaterScore first_water_score_from_pmf(const std::vector<GciPmfRow>& rows);

struct RegionLocalCounts {
    std::vector<double> region_mean; // size n_mu
    std::vector<double> region_std;  // size n_mu
    // Row-major (site-major): local_mean[site * n_mu + mu_idx].
    std::vector<double> local_mean;
    std::vector<double> local_std;
    std::vector<std::size_t> n_frames; // size n_mu
};

// Whole-region GCMC water counts and local (within `local_radius` of each
// site) water counts, per mu window. Counts are NOT capped at 1 (unlike
// the binary binomial-fit occupancy).
RegionLocalCounts count_region_and_local_by_mu(
    const std::vector<MuWindowFrames>& frames_by_mu,
    const std::vector<std::array<double, 3>>& site_centers, double local_radius);

} // namespace mcswell::analysis
