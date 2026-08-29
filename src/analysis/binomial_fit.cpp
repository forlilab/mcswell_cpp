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

#include "analysis/kdtree.hpp"
#include "analysis/root_finding.hpp"
#include "cuda/consts.cuh"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace mcswell::analysis {

namespace {
// Numerical bracket for the root search, in units of k_B*T -- +-100 kT is
// ~60 kcal/mol at 300K, intentionally generous (matches
// estimate_free_energies.py::FIT_BRACKET_KT).
constexpr double FIT_BRACKET_KT = 100.0;
} // namespace

double independent_site_occupancy(double mu, double epsilon, double temperature) {
    const double beta = 1.0 / (static_cast<double>(BOLTZMANN_K) * temperature);
    return 1.0 / (1.0 + std::exp(-beta * (mu - epsilon)));
}

BinomialFitResult fit_epsilon_binomial(
    const std::vector<double>& mu_values,
    const std::vector<std::size_t>& occ_counts,
    const std::vector<std::size_t>& n_frames,
    double temperature
) {
    if (mu_values.size() != occ_counts.size() || mu_values.size() != n_frames.size()) {
        throw std::invalid_argument("mu_values, occ_counts and n_frames must have the same length");
    }

    BinomialFitResult result;

    std::vector<double> mu, k, n;
    for (std::size_t i = 0; i < mu_values.size(); ++i) {
        if (n_frames[i] > 0) {
            mu.push_back(mu_values[i]);
            k.push_back(static_cast<double>(occ_counts[i]));
            n.push_back(static_cast<double>(n_frames[i]));
        }
    }

    if (mu.empty()) {
        result.status = "no_frames";
        return result;
    }

    double total_occupied = 0.0, total_frames = 0.0;
    for (std::size_t i = 0; i < k.size(); ++i) {
        total_occupied += k[i];
        total_frames += n[i];
    }
    result.n_mu_points = mu.size();
    result.total_occupied = static_cast<std::size_t>(total_occupied);
    result.total_frames = static_cast<std::size_t>(total_frames);

    if (total_occupied == 0.0) {
        result.status = "all_empty";
        return result;
    }
    if (total_occupied == total_frames) {
        result.status = "all_occupied";
        return result;
    }

    const double beta = 1.0 / (static_cast<double>(BOLTZMANN_K) * temperature);

    // d(log L)/d(epsilon) = beta * sum_i [k_i - N_i * p_i]; MLE satisfies
    // sum_i N_i p_i = sum_i k_i.
    auto score = [&](double epsilon) {
        double s = 0.0;
        for (std::size_t i = 0; i < mu.size(); ++i) {
            s += n[i] * independent_site_occupancy(mu[i], epsilon, temperature) - k[i];
        }
        return s;
    };

    const double pad = FIT_BRACKET_KT * static_cast<double>(BOLTZMANN_K) * temperature;
    const double lo = *std::min_element(mu.begin(), mu.end()) - pad;
    const double hi = *std::max_element(mu.begin(), mu.end()) + pad;

    const double f_lo = score(lo);
    const double f_hi = score(hi);
    if (!(f_lo > 0.0 && f_hi < 0.0)) {
        result.status = "root_not_bracketed";
        return result;
    }

    const double epsilon_hat = brentq(score, lo, hi);

    std::vector<double> p_hat(mu.size());
    for (std::size_t i = 0; i < mu.size(); ++i) {
        p_hat[i] = independent_site_occupancy(mu[i], epsilon_hat, temperature);
    }

    double fisher = 0.0;
    for (std::size_t i = 0; i < mu.size(); ++i) {
        fisher += n[i] * p_hat[i] * (1.0 - p_hat[i]);
    }
    fisher *= beta * beta;
    const double epsilon_se = fisher > 0.0 ? std::sqrt(1.0 / fisher) : std::numeric_limits<double>::quiet_NaN();

    double sse = 0.0;
    for (std::size_t i = 0; i < mu.size(); ++i) {
        const double p_obs = k[i] / n[i];
        const double d = p_obs - p_hat[i];
        sse += d * d;
    }
    const double fit_rmse_p = std::sqrt(sse / static_cast<double>(mu.size()));

    const double tiny = std::numeric_limits<double>::min();
    const double eps_mach = std::numeric_limits<double>::epsilon();
    double log_likelihood = 0.0;
    for (std::size_t i = 0; i < mu.size(); ++i) {
        const double p_clip = std::min(std::max(p_hat[i], tiny), 1.0 - eps_mach);
        log_likelihood += k[i] * std::log(p_clip) + (n[i] - k[i]) * std::log1p(-p_clip);
    }

    result.epsilon = epsilon_hat;
    result.epsilon_se = epsilon_se;
    result.fit_rmse_p = fit_rmse_p;
    result.log_likelihood = log_likelihood;
    result.status = "ok";
    return result;
}

OccupancyResult occupancy_probabilities_by_mu(
    const std::vector<MuWindowFrames>& frames_by_mu,
    const std::vector<std::array<double, 3>>& site_centers,
    double cutoff
) {
    const std::size_t n_sites = site_centers.size();
    const std::size_t n_mu = frames_by_mu.size();

    OccupancyResult result;
    result.n_sites = n_sites;
    result.n_mu = n_mu;
    result.p.assign(n_sites * n_mu, 0.0);
    result.occ_counts.assign(n_sites * n_mu, 0);
    result.n_frames.assign(n_mu, 0);

    if (n_sites == 0) return result;

    KDTree3D tree(site_centers);

    for (std::size_t j = 0; j < n_mu; ++j) {
        const auto& frames = frames_by_mu[j].frames;
        result.n_frames[j] = frames.size();

        std::vector<std::size_t> occ_this_mu(n_sites, 0);

        for (const auto& frame : frames) {
            if (frame.oxygens.empty()) continue;

            // At most one water per site per frame: keep the closest.
            std::map<std::size_t, double> best_dist_by_site;
            auto hits = tree.nearest_batch(frame.oxygens, cutoff);
            for (const auto& hit : hits) {
                if (!hit.found) continue;
                auto it = best_dist_by_site.find(hit.index);
                if (it == best_dist_by_site.end() || hit.distance < it->second) {
                    best_dist_by_site[hit.index] = hit.distance;
                }
            }
            for (const auto& [site_idx, dist] : best_dist_by_site) {
                (void)dist;
                occ_this_mu[site_idx] += 1;
            }
        }

        for (std::size_t s = 0; s < n_sites; ++s) {
            result.occ_counts[s * n_mu + j] = occ_this_mu[s];
            if (result.n_frames[j] > 0) {
                result.p[s * n_mu + j] =
                    static_cast<double>(occ_this_mu[s]) / static_cast<double>(result.n_frames[j]);
            }
        }
    }

    return result;
}

} // namespace mcswell::analysis
