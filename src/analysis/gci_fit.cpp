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

#include "analysis/kdtree.hpp"
#include "analysis/levenberg_marquardt.hpp"
#include "analysis/root_finding.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mcswell::analysis {

namespace {

std::vector<double> softmax(const std::vector<double>& z) {
    const double mx = *std::max_element(z.begin(), z.end());
    std::vector<double> e(z.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        e[i] = std::exp(z[i] - mx);
        sum += e[i];
    }
    for (double& v : e) v /= sum;
    return e;
}

// log(1 + exp(z)), numerically stable for large |z|.
double log1p_exp(double z) {
    if (z > 0.0) return z + std::log1p(std::exp(-z));
    return std::log1p(std::exp(z));
}

double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double sample_std(const std::vector<double>& v, double m) {
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

} // namespace

double LogisticSumModel::predict(double B) const {
    double val = n_min;
    for (std::size_t j = 0; j < amplitudes.size(); ++j) {
        const double z = slopes[j] * (B - centers[j]);
        val += amplitudes[j] * (1.0 / (1.0 + std::exp(-z)));
    }
    return val;
}

double LogisticSumModel::excess_integral_from_minus_inf(double B) const {
    double s = 0.0;
    for (std::size_t j = 0; j < amplitudes.size(); ++j) {
        const double z = slopes[j] * (B - centers[j]);
        s += (amplitudes[j] / slopes[j]) * log1p_exp(z);
    }
    return s;
}

double LogisticSumModel::inverse(double target_n, double B_lo, double B_hi) const {
    if (!(n_min < target_n && target_n < n_max)) {
        throw std::invalid_argument("inverse target must be strictly inside model asymptotes");
    }
    auto f = [&](double b) { return predict(b) - target_n; };

    double lo = B_lo, hi = B_hi;
    for (int i = 0; i < 20; ++i) {
        const double flo = f(lo);
        const double fhi = f(hi);
        if (flo <= 0.0 && fhi >= 0.0) {
            return brentq(f, lo, hi, 2e-12, 8.881784197001252e-16, 300);
        }
        const double span = std::max(10.0, hi - lo);
        if (flo > 0.0) lo -= span;
        if (fhi < 0.0) hi += span;
    }
    throw std::runtime_error("Could not bracket inverse for the requested N");
}

double LogisticSumModel::dimensionless_transfer_from_nmin(double target_n, double B_lo, double B_hi) const {
    if (target_n < n_min - 1e-8 || target_n > n_max + 1e-8) {
        throw std::invalid_argument("target N outside model range");
    }
    const double dn = target_n - n_min;
    if (std::fabs(dn) < 1e-12) return 0.0;

    if (std::fabs(target_n - n_max) < 1e-10) {
        double s = 0.0;
        for (std::size_t j = 0; j < amplitudes.size(); ++j) s += amplitudes[j] * centers[j];
        return s;
    }

    const double Bn = inverse(target_n, B_lo, B_hi);
    return dn * Bn - excess_integral_from_minus_inf(Bn);
}

LogisticSumModel fit_logistic_sum(
    const std::vector<double>& B_in, const std::vector<double>& N_in, const LogisticSumFitOptions& options) {
    std::vector<double> B, N;
    for (std::size_t i = 0; i < B_in.size(); ++i) {
        if (std::isfinite(B_in[i]) && std::isfinite(N_in[i])) {
            B.push_back(B_in[i]);
            N.push_back(N_in[i]);
        }
    }
    if (B.size() < 4) {
        throw std::runtime_error("Need at least four finite titration points for GCI fitting.");
    }

    std::vector<std::size_t> order(B.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return B[a] < B[b]; });

    std::vector<double> Bs(B.size()), Ns(N.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        Bs[i] = B[order[i]];
        Ns[i] = N[order[i]];
    }

    const double n_min = std::round(*std::min_element(Ns.begin(), Ns.end()));
    const double n_max = std::round(*std::max_element(Ns.begin(), Ns.end()));
    if (n_max <= n_min) {
        throw std::runtime_error("Titration does not span distinct integer occupancies.");
    }

    const double delta = n_max - n_min;
    const int n_steps = options.n_steps >= 1 ? options.n_steps : std::max(1, static_cast<int>(std::lround(delta)));
    const int m = std::max(1, std::min(n_steps, options.max_terms));

    const double bmin = Bs.front();
    const double bmax = Bs.back();
    const double bspan = std::max(bmax - bmin, 1.0);

    std::vector<double> centers0(static_cast<std::size_t>(m));
    for (int j = 0; j < m; ++j) {
        const double target = n_min + delta * (j + 0.5) / m;
        std::size_t best_idx = 0;
        double best_diff = std::fabs(Ns[0] - target);
        for (std::size_t i = 1; i < Ns.size(); ++i) {
            const double d = std::fabs(Ns[i] - target);
            if (d < best_diff) {
                best_diff = d;
                best_idx = i;
            }
        }
        centers0[static_cast<std::size_t>(j)] = Bs[best_idx];
    }

    // theta layout: m==1 -> [log_slope, center] (2 params);
    // m>1 -> [m-1 logits, m log_slopes, m centers] (3m-1 params).
    const std::size_t param_count = (m == 1) ? 2 : static_cast<std::size_t>(3 * m - 1);
    const std::size_t center_start = param_count - static_cast<std::size_t>(m);
    const std::size_t slope_start = (m == 1) ? 0 : static_cast<std::size_t>(m - 1);

    auto unpack = [&](const std::vector<double>& theta, std::vector<double>& amps, std::vector<double>& slopes,
                       std::vector<double>& centers) {
        amps.assign(static_cast<std::size_t>(m), 0.0);
        slopes.assign(static_cast<std::size_t>(m), 0.0);
        centers.assign(static_cast<std::size_t>(m), 0.0);

        if (m == 1) {
            amps[0] = delta;
        } else {
            std::vector<double> logits(static_cast<std::size_t>(m));
            logits[0] = 0.0;
            for (int j = 1; j < m; ++j) logits[static_cast<std::size_t>(j)] = theta[static_cast<std::size_t>(j - 1)];
            const auto sm = softmax(logits);
            for (int j = 0; j < m; ++j) amps[static_cast<std::size_t>(j)] = delta * sm[static_cast<std::size_t>(j)];
        }
        for (int j = 0; j < m; ++j) {
            slopes[static_cast<std::size_t>(j)] = std::exp(theta[slope_start + static_cast<std::size_t>(j)]);
        }
        for (int j = 0; j < m; ++j) {
            centers[static_cast<std::size_t>(j)] = theta[center_start + static_cast<std::size_t>(j)];
        }
    };

    auto residual = [&](const std::vector<double>& theta, std::vector<double>& out) {
        std::vector<double> amps, slopes, centers;
        unpack(theta, amps, slopes, centers);
        out.resize(Bs.size());
        for (std::size_t i = 0; i < Bs.size(); ++i) {
            double pred = n_min;
            for (int j = 0; j < m; ++j) {
                const double z = slopes[static_cast<std::size_t>(j)] * (Bs[i] - centers[static_cast<std::size_t>(j)]);
                pred += amps[static_cast<std::size_t>(j)] * (1.0 / (1.0 + std::exp(-z)));
            }
            out[i] = pred - Ns[i];
        }
    };

    std::vector<double> theta0(param_count, 0.0);
    for (int j = 0; j < m; ++j) theta0[center_start + static_cast<std::size_t>(j)] = centers0[static_cast<std::size_t>(j)];
    // slope block left at 0.0 -> exp(0) == 1.0 initial slope.

    std::vector<double> lower(param_count), upper(param_count);
    {
        std::size_t pos = 0;
        if (m > 1) {
            for (int j = 0; j < m - 1; ++j) {
                lower[pos] = -6.0;
                upper[pos] = 6.0;
                ++pos;
            }
        }
        for (int j = 0; j < m; ++j) {
            (void)j;
            lower[pos] = std::log(0.02);
            upper[pos] = std::log(20.0);
            ++pos;
        }
        for (int j = 0; j < m; ++j) {
            (void)j;
            lower[pos] = bmin - bspan;
            upper[pos] = bmax + bspan;
            ++pos;
        }
    }

    std::mt19937_64 rng(options.seed);
    std::normal_distribution<double> logit_jitter(0.0, 0.6);
    std::normal_distribution<double> slope_jitter(0.0, 0.35);
    std::normal_distribution<double> center_jitter(0.0, 0.12 * bspan);

    double best_sse = std::numeric_limits<double>::infinity();
    std::vector<double> best_theta = theta0;

    const int starts = std::max(1, options.random_starts);
    for (int s = 0; s < starts; ++s) {
        std::vector<double> t0 = theta0;
        if (s > 0) {
            if (m > 1) {
                for (int j = 0; j < m - 1; ++j) t0[static_cast<std::size_t>(j)] += logit_jitter(rng);
            }
            for (int j = 0; j < m; ++j) t0[slope_start + static_cast<std::size_t>(j)] += slope_jitter(rng);
            for (int j = 0; j < m; ++j) t0[center_start + static_cast<std::size_t>(j)] += center_jitter(rng);

            for (std::size_t idx = 0; idx < t0.size(); ++idx) {
                t0[idx] = std::min(std::max(t0[idx], lower[idx] + 1e-8), upper[idx] - 1e-8);
            }
        }

        const auto lm_result = levenberg_marquardt(residual, t0, Bs.size());
        if (lm_result.cost < best_sse) {
            best_sse = lm_result.cost;
            best_theta = lm_result.params;
        }
    }

    std::vector<double> amps, slopes, centers;
    unpack(best_theta, amps, slopes, centers);

    std::vector<std::size_t> idx_order(static_cast<std::size_t>(m));
    std::iota(idx_order.begin(), idx_order.end(), 0);
    std::sort(idx_order.begin(), idx_order.end(), [&](std::size_t a, std::size_t b) { return centers[a] < centers[b]; });

    LogisticSumModel model;
    model.n_min = n_min;
    model.n_max = n_max;
    model.amplitudes.resize(static_cast<std::size_t>(m));
    model.slopes.resize(static_cast<std::size_t>(m));
    model.centers.resize(static_cast<std::size_t>(m));
    for (int j = 0; j < m; ++j) {
        model.amplitudes[static_cast<std::size_t>(j)] = amps[idx_order[static_cast<std::size_t>(j)]];
        model.slopes[static_cast<std::size_t>(j)] = slopes[idx_order[static_cast<std::size_t>(j)]];
        model.centers[static_cast<std::size_t>(j)] = centers[idx_order[static_cast<std::size_t>(j)]];
    }
    model.sse = best_sse;
    model.rmse = std::sqrt(best_sse / static_cast<double>(Bs.size()));
    return model;
}

std::vector<GciPmfRow> gci_pmf(
    const LogisticSumModel& model, const std::vector<double>& B_obs, double kT, double analysis_volume,
    double standard_volume, double mu_bulk) {
    const int n0 = static_cast<int>(std::lround(model.n_min));
    const int n1 = static_cast<int>(std::lround(model.n_max));

    const double blo = *std::min_element(B_obs.begin(), B_obs.end());
    const double bhi = *std::max_element(B_obs.begin(), B_obs.end());
    const double volume_log = std::log(analysis_volume / standard_volume);

    std::vector<GciPmfRow> rows;
    rows.reserve(static_cast<std::size_t>(n1 - n0 + 1));

    bool has_prev = false;
    double prev_bind = 0.0;

    for (int n = n0; n <= n1; ++n) {
        const double raw_dimless = model.dimensionless_transfer_from_nmin(static_cast<double>(n), blo, bhi);
        const double dn = static_cast<double>(n - n0);
        const double corrected_dimless = raw_dimless - dn * volume_log;
        const double dg_insert = corrected_dimless * kT;
        const double dg_bind = dg_insert - dn * mu_bulk;

        GciPmfRow row;
        row.n = n;
        row.delta_n_from_min = dn;
        row.beta_dF_raw = raw_dimless;
        row.delta_g_insert_kcal_mol = dg_insert;
        row.delta_g_bind_network_kcal_mol = dg_bind;
        row.delta_g_bind_incremental_kcal_mol = has_prev ? (dg_bind - prev_bind) : std::numeric_limits<double>::quiet_NaN();
        rows.push_back(row);

        prev_bind = dg_bind;
        has_prev = true;
    }
    return rows;
}

FirstWaterScore first_water_score_from_pmf(const std::vector<GciPmfRow>& rows) {
    const GciPmfRow* r0 = nullptr;
    const GciPmfRow* r1 = nullptr;
    for (const auto& r : rows) {
        if (r.n == 0) r0 = &r;
        if (r.n == 1) r1 = &r;
    }

    FirstWaterScore result;
    if (!r0 || !r1) {
        result.status = "does_not_span_0_to_1";
        return result;
    }
    result.delta_g_insert = r1->delta_g_insert_kcal_mol - r0->delta_g_insert_kcal_mol;
    result.delta_g_bind = r1->delta_g_bind_network_kcal_mol - r0->delta_g_bind_network_kcal_mol;
    result.status = "ok";
    return result;
}

RegionLocalCounts count_region_and_local_by_mu(
    const std::vector<MuWindowFrames>& frames_by_mu, const std::vector<std::array<double, 3>>& site_centers,
    double local_radius) {
    const std::size_t n_mu = frames_by_mu.size();
    const std::size_t n_sites = site_centers.size();

    RegionLocalCounts result;
    result.region_mean.assign(n_mu, std::numeric_limits<double>::quiet_NaN());
    result.region_std.assign(n_mu, std::numeric_limits<double>::quiet_NaN());
    result.local_mean.assign(n_sites * n_mu, std::numeric_limits<double>::quiet_NaN());
    result.local_std.assign(n_sites * n_mu, std::numeric_limits<double>::quiet_NaN());
    result.n_frames.assign(n_mu, 0);

    for (std::size_t mi = 0; mi < n_mu; ++mi) {
        const auto& frames = frames_by_mu[mi].frames;
        result.n_frames[mi] = frames.size();
        if (frames.empty()) continue;

        std::vector<double> region_counts(frames.size());
        std::vector<std::vector<double>> local_counts(n_sites, std::vector<double>(frames.size(), 0.0));

        for (std::size_t fi = 0; fi < frames.size(); ++fi) {
            const auto& xyz = frames[fi].oxygens;
            region_counts[fi] = static_cast<double>(xyz.size());

            if (n_sites > 0 && !xyz.empty()) {
                KDTree3D water_tree(xyz);
                const auto counts = water_tree.count_within_radius_batch(site_centers, local_radius);
                for (std::size_t s = 0; s < n_sites; ++s) local_counts[s][fi] = static_cast<double>(counts[s]);
            }
        }

        const double mean_region = mean(region_counts);
        result.region_mean[mi] = mean_region;
        result.region_std[mi] = frames.size() > 1 ? sample_std(region_counts, mean_region) : 0.0;

        for (std::size_t s = 0; s < n_sites; ++s) {
            const double mean_local = mean(local_counts[s]);
            result.local_mean[s * n_mu + mi] = mean_local;
            result.local_std[s * n_mu + mi] = frames.size() > 1 ? sample_std(local_counts[s], mean_local) : 0.0;
        }
    }

    return result;
}

} // namespace mcswell::analysis
