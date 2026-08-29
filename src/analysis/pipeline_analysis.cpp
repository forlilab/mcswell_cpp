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

#include "analysis/pipeline_analysis.hpp"

#include "analysis/binomial_fit.hpp"
#include "analysis/density_grid.hpp"
#include "analysis/gci_fit.hpp"
#include "analysis/writers.hpp"
#include "cuda/consts.cuh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mcswell::analysis {

namespace {

namespace fs = std::filesystem;

std::string mu_column_label(std::size_t mu_idx, double mu_value) {
    std::ostringstream ss;
    ss << "p_muidx_" << std::setw(3) << std::setfill('0') << mu_idx << "_mu_" << std::setfill(' ')
       << std::defaultfloat << std::setprecision(6) << mu_value;
    return ss.str();
}

std::vector<std::string> capacity_diagnostics_header(const std::string& valid_col_name) {
    return {
        "mu_index",      "mu_kcal_mol",      "n_frames",       "N_region_mean",
        "N_region_std",  "N_region_max_observed", "water_capacity", "capacity_mean_fraction",
        "capacity_hit_fraction", valid_col_name, "capacity_filter_reason",
    };
}

std::vector<std::string> capacity_diagnostics_row(const CapacityDiagnostics& d) {
    return {
        std::to_string(d.mu_index),
        format_csv_double(d.mu_kcal_mol),
        std::to_string(d.n_frames),
        format_csv_double(d.n_region_mean),
        format_csv_double(d.n_region_std),
        std::to_string(d.n_region_max_observed),
        std::to_string(d.water_capacity),
        format_csv_double(d.capacity_mean_fraction),
        format_csv_double(d.capacity_hit_fraction),
        d.valid_for_analysis ? "True" : "False",
        d.capacity_filter_reason,
    };
}

std::vector<double> select_by_mask(const std::vector<double>& values, const std::vector<bool>& mask) {
    std::vector<double> out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (mask[i]) out.push_back(values[i]);
    }
    return out;
}

std::vector<std::size_t> select_by_mask(const std::vector<std::size_t>& values, const std::vector<bool>& mask) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (mask[i]) out.push_back(values[i]);
    }
    return out;
}

std::size_t argmin_abs_diff(const std::vector<double>& values, double target) {
    std::size_t best = 0;
    double best_diff = std::fabs(values[0] - target);
    for (std::size_t i = 1; i < values.size(); ++i) {
        const double d = std::fabs(values[i] - target);
        if (d < best_diff) {
            best_diff = d;
            best = i;
        }
    }
    return best;
}

std::vector<Peak> detect_sites(
    const std::vector<std::array<double, 3>>& density_points, const PostAnalysisConfig& cfg) {
    const auto edges = make_edges(cfg.box_center, cfg.box_halfsize, cfg.grid_spacing);
    const auto grid = compute_density_grid(density_points, edges);
    const auto smoothed = gaussian_smooth(grid, cfg.grid_sigma);
    const auto raw_peaks = find_peaks(smoothed, cfg.peak_percentile, 1);
    return merge_close_sites(raw_peaks, cfg.peak_merge_cutoff);
}

void write_density_map(
    const std::vector<std::array<double, 3>>& density_points, const PostAnalysisConfig& cfg, const std::string& path) {
    const auto edges = make_edges(cfg.box_center, cfg.box_halfsize, cfg.grid_spacing);
    const auto grid = compute_density_grid(density_points, edges);
    const auto smoothed = gaussian_smooth(grid, cfg.grid_sigma);
    density_to_dx(smoothed, path);
}

} // namespace

// ===========================================================================
// Binomial (independent-site occupancy) analysis
// ===========================================================================

std::string run_binomial_analysis(
    const std::vector<MuWindowFrames>& frames_by_mu, const std::vector<double>& mu_values,
    const PostAnalysisConfig& cfg, std::size_t n_insertion_points, double sampler_volume,
    const std::string& save_path) {
    const std::string out_dir = save_path + "/gci/" + std::to_string(static_cast<int>(cfg.peak_percentile)) +
                                 ".0_binomial";
    fs::create_directories(out_dir);

    const auto water_capacity =
        static_cast<std::size_t>(std::floor(sampler_volume * cfg.bulk_water_density));
    if (water_capacity == 0) throw std::runtime_error("Reconstructed water-buffer capacity is zero.");

    auto capacity = scan_capacity_and_collect_density_points(
        frames_by_mu, mu_values, water_capacity, cfg.capacity_filter, cfg.max_capacity_hit_fraction,
        cfg.max_mean_capacity_fraction);

    const std::size_t n_valid = std::count(capacity.valid_mask.begin(), capacity.valid_mask.end(), true);
    if (n_valid < 4) {
        throw std::runtime_error(
            "Only " + std::to_string(n_valid) + " chemical-potential windows remain after capacity "
            "filtering; too few for a meaningful titration fit.");
    }

    const std::size_t mu_eq_idx = argmin_abs_diff(mu_values, cfg.mu_bulk);
    if (!capacity.valid_mask[mu_eq_idx]) {
        throw std::runtime_error(
            "The sampled window nearest bulk mu is already capacity-limited. Increase the MCSwell "
            "water-buffer capacity and rerun.");
    }

    write_csv_table(out_dir + "/capacity_diagnostics.csv", capacity_diagnostics_header("valid_for_binomial"), [&] {
        std::vector<std::vector<std::string>> rows;
        for (const auto& d : capacity.diagnostics) rows.push_back(capacity_diagnostics_row(d));
        return rows;
    }());

    const auto peaks = detect_sites(capacity.density_points, cfg);
    std::vector<std::array<double, 3>> site_centers;
    std::vector<double> peak_scores;
    for (const auto& p : peaks) {
        site_centers.push_back(p.center);
        peak_scores.push_back(p.score);
    }
    const std::size_t n_sites = site_centers.size();

    const auto occ = occupancy_probabilities_by_mu(frames_by_mu, site_centers, cfg.assignment_cutoff);
    const std::size_t n_mu = mu_values.size();

    std::vector<double> epsilon(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> epsilon_se(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> delta_g_bind(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<std::size_t> n_mu_points(n_sites, 0);
    std::vector<double> fit_rmse_p(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<std::size_t> total_occupied(n_sites, 0), total_frames(n_sites, 0);
    std::vector<std::string> status(n_sites);

    const auto mu_fit_values = select_by_mask(mu_values, capacity.valid_mask);

    for (std::size_t s = 0; s < n_sites; ++s) {
        std::vector<std::size_t> row_occ(n_mu), row_n(n_mu);
        for (std::size_t j = 0; j < n_mu; ++j) {
            row_occ[j] = occ.occ_counts[s * n_mu + j];
            row_n[j] = occ.n_frames[j];
        }
        const auto occ_fit = select_by_mask(row_occ, capacity.valid_mask);
        const auto n_fit = select_by_mask(row_n, capacity.valid_mask);

        const auto fit = fit_epsilon_binomial(mu_fit_values, occ_fit, n_fit, cfg.temperature);
        epsilon[s] = fit.epsilon;
        epsilon_se[s] = fit.epsilon_se;
        n_mu_points[s] = fit.n_mu_points;
        fit_rmse_p[s] = fit.fit_rmse_p;
        total_occupied[s] = fit.total_occupied;
        total_frames[s] = fit.total_frames;
        status[s] = fit.status;
        if (std::isfinite(epsilon[s])) delta_g_bind[s] = epsilon[s] - cfg.mu_bulk;
    }

    std::vector<double> p_mu_eq(n_sites);
    for (std::size_t s = 0; s < n_sites; ++s) p_mu_eq[s] = occ.p[s * n_mu + mu_eq_idx];

    // sites_all.csv
    {
        std::vector<std::string> headers = {
            "source_site_id", "x", "y", "z", "epsilon_midpoint", "epsilon_fit_se_binomial", "deltaG_bind",
            "deltaG_bind_se_binomial", "n_mu_points", "p_mu_eq", "fit_rmse_p", "total_occupied_frames",
            "total_frame_observations", "fit_status", "density_peak_score",
        };
        std::vector<std::vector<std::string>> rows;
        for (std::size_t s = 0; s < n_sites; ++s) {
            rows.push_back({
                std::to_string(s),
                format_csv_double(site_centers[s][0]),
                format_csv_double(site_centers[s][1]),
                format_csv_double(site_centers[s][2]),
                format_csv_double(epsilon[s]),
                format_csv_double(epsilon_se[s]),
                format_csv_double(delta_g_bind[s]),
                format_csv_double(epsilon_se[s]),
                std::to_string(n_mu_points[s]),
                format_csv_double(p_mu_eq[s]),
                format_csv_double(fit_rmse_p[s]),
                std::to_string(total_occupied[s]),
                std::to_string(total_frames[s]),
                status[s],
                format_csv_double(peak_scores[s]),
            });
        }
        write_csv_table(out_dir + "/sites_all.csv", headers, rows);
    }

    // titration.csv (wide format, every sampled mu, every detected site)
    {
        std::vector<std::string> headers = {"source_site_id"};
        for (std::size_t j = 0; j < n_mu; ++j) headers.push_back(mu_column_label(j, mu_values[j]));

        std::vector<std::vector<std::string>> rows;
        for (std::size_t s = 0; s < n_sites; ++s) {
            std::vector<std::string> row = {std::to_string(s)};
            for (std::size_t j = 0; j < n_mu; ++j) row.push_back(format_csv_double(occ.p[s * n_mu + j]));
            rows.push_back(std::move(row));
        }
        write_csv_table(out_dir + "/titration.csv", headers, rows);
    }

    // titration_diagnostics.csv
    {
        std::vector<std::string> headers = {
            "source_site_id", "mu_index", "mu_kcal_mol", "n_frames", "n_occupied", "p_observed", "p_fitted",
            "valid_for_binomial", "N_region_mean", "water_capacity", "capacity_mean_fraction",
            "capacity_hit_fraction", "capacity_filter_reason",
        };
        std::vector<std::vector<std::string>> rows;
        for (std::size_t s = 0; s < n_sites; ++s) {
            for (std::size_t j = 0; j < n_mu; ++j) {
                const auto& cap = capacity.diagnostics[j];
                const double p_fit = std::isfinite(epsilon[s])
                    ? independent_site_occupancy(mu_values[j], epsilon[s], cfg.temperature)
                    : std::numeric_limits<double>::quiet_NaN();
                const double p_obs = occ.n_frames[j] > 0 ? occ.p[s * n_mu + j] : std::numeric_limits<double>::quiet_NaN();
                rows.push_back({
                    std::to_string(s),
                    std::to_string(j),
                    format_csv_double(mu_values[j]),
                    std::to_string(occ.n_frames[j]),
                    std::to_string(occ.occ_counts[s * n_mu + j]),
                    format_csv_double(p_obs),
                    format_csv_double(p_fit),
                    capacity.valid_mask[j] ? "True" : "False",
                    format_csv_double(cap.n_region_mean),
                    std::to_string(cap.water_capacity),
                    format_csv_double(cap.capacity_mean_fraction),
                    format_csv_double(cap.capacity_hit_fraction),
                    cap.capacity_filter_reason,
                });
            }
        }
        write_csv_table(out_dir + "/titration_diagnostics.csv", headers, rows);
    }

    write_density_map(capacity.density_points, cfg, out_dir + "/gO.dx");

    // Usable-site outputs (finite delta_g_bind and status == "ok").
    std::vector<std::size_t> usable_ids;
    for (std::size_t s = 0; s < n_sites; ++s) {
        if (std::isfinite(delta_g_bind[s]) && status[s] == "ok") usable_ids.push_back(s);
    }

    {
        std::vector<std::array<double, 3>> centers_usable;
        std::vector<double> occ_col, bfac_col;
        for (auto s : usable_ids) {
            centers_usable.push_back(site_centers[s]);
            occ_col.push_back(p_mu_eq[s]);
            bfac_col.push_back(delta_g_bind[s]);
        }
        write_sites_pdb(
            out_dir + "/sites.pdb", centers_usable, occ_col, bfac_col,
            {"Independent-site grand-canonical occupancy fit; B-factor = DeltaG_bind = epsilon - mu_bulk (kcal/mol)"});
    }

    {
        std::vector<std::string> headers = {
            "site", "source_site_id", "x", "y", "z", "epsilon_midpoint", "epsilon_fit_se_binomial", "deltaG_bind",
            "deltaG_bind_se_binomial", "n_mu_points", "p_mu_eq", "fit_rmse_p", "total_occupied_frames",
            "total_frame_observations", "epsilon_mean", "density_peak_score",
        };
        std::vector<std::vector<std::string>> rows;
        for (std::size_t row_idx = 0; row_idx < usable_ids.size(); ++row_idx) {
            const auto s = usable_ids[row_idx];
            rows.push_back({
                std::to_string(row_idx),
                std::to_string(s),
                format_csv_double(site_centers[s][0]),
                format_csv_double(site_centers[s][1]),
                format_csv_double(site_centers[s][2]),
                format_csv_double(epsilon[s]),
                format_csv_double(epsilon_se[s]),
                format_csv_double(delta_g_bind[s]),
                format_csv_double(epsilon_se[s]),
                std::to_string(n_mu_points[s]),
                format_csv_double(p_mu_eq[s]),
                format_csv_double(fit_rmse_p[s]),
                std::to_string(total_occupied[s]),
                std::to_string(total_frames[s]),
                format_csv_double(delta_g_bind[s]),
                format_csv_double(peak_scores[s]),
            });
        }
        write_csv_table(out_dir + "/sites.csv", headers, rows);
    }

    JsonWriter meta;
    meta.add("method", std::string("capacity-safe bounded independent-site binomial MLE"))
        .add("temperature_K", cfg.temperature)
        .add("bulk_mu_kcal_mol", cfg.mu_bulk)
        .add("assignment_cutoff_A", cfg.assignment_cutoff)
        .add("peak_percentile", cfg.peak_percentile)
        .add("sampler_volume_A3", sampler_volume)
        .add("n_insertion_points", static_cast<long long>(n_insertion_points))
        .add("bulk_water_density_A-3", cfg.bulk_water_density)
        .add("reconstructed_water_capacity", static_cast<long long>(water_capacity))
        .add("capacity_filter_enabled", cfg.capacity_filter)
        .add("max_capacity_hit_fraction", cfg.max_capacity_hit_fraction)
        .add("max_mean_capacity_fraction", cfg.max_mean_capacity_fraction)
        .add("n_sampled_mu_windows", static_cast<long long>(n_mu))
        .add("n_capacity_safe_mu_windows", static_cast<long long>(n_valid))
        .add("nearest_bulk_mu_kcal_mol", mu_values[mu_eq_idx])
        .add("n_detected_sites", static_cast<long long>(n_sites))
        .add("n_usable_sites", static_cast<long long>(usable_ids.size()));
    meta.write(out_dir + "/binomial_metadata.json");

    return out_dir;
}

// ===========================================================================
// ProtoMS-style GCI (whole-region + local logistic-sum) analysis
// ===========================================================================

std::string run_gci_analysis(
    const std::vector<MuWindowFrames>& frames_by_mu, const std::vector<double>& mu_values,
    const PostAnalysisConfig& cfg, std::size_t n_insertion_points, double sampler_volume,
    const std::string& save_path) {
    const std::string out_dir =
        save_path + "/gci/" + std::to_string(static_cast<int>(cfg.peak_percentile)) + ".0_protoms_gci";
    fs::create_directories(out_dir);

    const double kT = static_cast<double>(KT);
    const double standard_volume = static_cast<double>(STANDARD_VOLUME);
    const auto water_capacity = static_cast<std::size_t>(std::floor(sampler_volume * cfg.bulk_water_density));
    if (water_capacity == 0) throw std::runtime_error("Reconstructed water-buffer capacity is zero.");

    const std::size_t n_mu = mu_values.size();
    std::vector<double> B(n_mu);
    for (std::size_t j = 0; j < n_mu; ++j) B[j] = mu_values[j] / kT + std::log(sampler_volume / standard_volume);

    auto capacity = scan_capacity_and_collect_density_points(
        frames_by_mu, mu_values, water_capacity, cfg.capacity_filter, cfg.max_capacity_hit_fraction,
        cfg.max_mean_capacity_fraction);

    const std::size_t n_valid = std::count(capacity.valid_mask.begin(), capacity.valid_mask.end(), true);
    if (n_valid < 4) {
        throw std::runtime_error(
            "Only " + std::to_string(n_valid) + " chemical-potential windows remain after capacity "
            "filtering; need at least four for GCI fitting.");
    }

    const std::size_t mu_eq_idx = argmin_abs_diff(mu_values, cfg.mu_bulk);
    if (!capacity.valid_mask[mu_eq_idx]) {
        throw std::runtime_error(
            "The sampled window nearest bulk water chemical potential is capacity-limited. Increase the "
            "sampler water-buffer capacity and rerun.");
    }

    const auto peaks = detect_sites(capacity.density_points, cfg);
    std::vector<std::array<double, 3>> site_centers;
    std::vector<double> peak_scores;
    for (const auto& p : peaks) {
        site_centers.push_back(p.center);
        peak_scores.push_back(p.score);
    }
    const std::size_t n_sites = site_centers.size();

    write_density_map(capacity.density_points, cfg, out_dir + "/gO.dx");

    const auto counts = count_region_and_local_by_mu(frames_by_mu, site_centers, cfg.local_radius);

    // region_titration.csv
    {
        std::vector<std::string> headers = {
            "mu_index", "mu_kcal_mol", "B_sampler", "n_frames", "N_region_mean", "N_region_std",
            "N_region_max_observed", "water_capacity", "capacity_mean_fraction", "capacity_hit_fraction",
            "valid_for_gci", "capacity_filter_reason",
        };
        std::vector<std::vector<std::string>> rows;
        for (std::size_t j = 0; j < n_mu; ++j) {
            const auto& cap = capacity.diagnostics[j];
            rows.push_back({
                std::to_string(j),
                format_csv_double(mu_values[j]),
                format_csv_double(B[j]),
                std::to_string(counts.n_frames[j]),
                format_csv_double(counts.region_mean[j]),
                format_csv_double(counts.region_std[j]),
                std::to_string(cap.n_region_max_observed),
                std::to_string(water_capacity),
                format_csv_double(cap.capacity_mean_fraction),
                format_csv_double(cap.capacity_hit_fraction),
                capacity.valid_mask[j] ? "True" : "False",
                cap.capacity_filter_reason,
            });
        }
        write_csv_table(out_dir + "/region_titration.csv", headers, rows);
    }

    // local_titration.csv
    {
        std::vector<std::string> headers = {
            "site", "mu_index", "mu_kcal_mol", "B_sampler", "n_frames", "N_local_mean", "N_local_std",
            "valid_for_gci", "capacity_mean_fraction", "capacity_hit_fraction",
        };
        std::vector<std::vector<std::string>> rows;
        for (std::size_t s = 0; s < n_sites; ++s) {
            for (std::size_t j = 0; j < n_mu; ++j) {
                const auto& cap = capacity.diagnostics[j];
                rows.push_back({
                    std::to_string(s),
                    std::to_string(j),
                    format_csv_double(mu_values[j]),
                    format_csv_double(B[j]),
                    std::to_string(counts.n_frames[j]),
                    format_csv_double(counts.local_mean[s * n_mu + j]),
                    format_csv_double(counts.local_std[s * n_mu + j]),
                    capacity.valid_mask[j] ? "True" : "False",
                    format_csv_double(cap.capacity_mean_fraction),
                    format_csv_double(cap.capacity_hit_fraction),
                });
            }
        }
        write_csv_table(out_dir + "/local_titration.csv", headers, rows);
    }

    const auto B_fit = select_by_mask(B, capacity.valid_mask);
    const auto region_mean_fit = select_by_mask(counts.region_mean, capacity.valid_mask);
    const auto mu_fit_values = select_by_mask(mu_values, capacity.valid_mask);

    LogisticSumFitOptions region_opts;
    region_opts.max_terms = cfg.region_max_terms;
    region_opts.random_starts = cfg.random_starts;
    region_opts.seed = cfg.fit_seed;
    const auto region_model = fit_logistic_sum(B_fit, region_mean_fit, region_opts);
    const auto region_pmf = gci_pmf(region_model, B_fit, kT, sampler_volume, standard_volume, cfg.mu_bulk);
    write_csv_table(
        out_dir + "/region_gci_pmf.csv",
        {"N", "deltaN_from_min", "beta_dF_raw", "deltaG_insert_kcal_mol", "deltaG_bind_network_kcal_mol",
         "deltaG_bind_incremental_kcal_mol"},
        [&] {
            std::vector<std::vector<std::string>> rows;
            for (const auto& r : region_pmf) {
                rows.push_back({
                    std::to_string(r.n),
                    format_csv_double(r.delta_n_from_min),
                    format_csv_double(r.beta_dF_raw),
                    format_csv_double(r.delta_g_insert_kcal_mol),
                    format_csv_double(r.delta_g_bind_network_kcal_mol),
                    format_csv_double(r.delta_g_bind_incremental_kcal_mol),
                });
            }
            return rows;
        }());

    double local_analysis_volume = sampler_volume;
    if (cfg.local_volume_mode == "standard") {
        local_analysis_volume = standard_volume;
    } else if (cfg.local_volume_mode == "sphere") {
        constexpr double kPi = 3.14159265358979323846;
        local_analysis_volume = (4.0 / 3.0) * kPi * cfg.local_radius * cfg.local_radius * cfg.local_radius;
    } else if (cfg.local_volume_mode != "sampler") {
        throw std::invalid_argument("Unknown local_volume_mode: " + cfg.local_volume_mode);
    }

    std::vector<double> site_dg_insert(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> site_dg_bind(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<std::string> site_status(n_sites, "ok");
    std::vector<double> site_rmse(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<int> site_terms(n_sites, 0);
    std::vector<double> site_n_min(n_sites, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> site_n_max(n_sites, std::numeric_limits<double>::quiet_NaN());

    for (std::size_t s = 0; s < n_sites; ++s) {
        std::vector<double> y_fit(B_fit.size());
        std::size_t k = 0;
        for (std::size_t j = 0; j < n_mu; ++j) {
            if (capacity.valid_mask[j]) y_fit[k++] = counts.local_mean[s * n_mu + j];
        }

        try {
            const double nmin_round = std::round(*std::min_element(y_fit.begin(), y_fit.end()));
            const double nmax_round = std::round(*std::max_element(y_fit.begin(), y_fit.end()));
            if (nmax_round <= nmin_round) {
                site_status[s] = "no_integer_occupancy_transition_in_capacity_safe_range";
                continue;
            }

            int local_steps = std::max(1, static_cast<int>(std::lround(nmax_round - nmin_round)));
            local_steps = std::min(local_steps, cfg.local_max_terms);

            LogisticSumFitOptions local_opts;
            local_opts.n_steps = local_steps;
            local_opts.max_terms = cfg.local_max_terms;
            local_opts.random_starts = cfg.random_starts;
            local_opts.seed = cfg.fit_seed + s + 1;

            const auto model = fit_logistic_sum(B_fit, y_fit, local_opts);
            site_rmse[s] = model.rmse;
            site_terms[s] = static_cast<int>(model.amplitudes.size());
            site_n_min[s] = model.n_min;
            site_n_max[s] = model.n_max;

            const auto pmf = gci_pmf(model, B_fit, kT, local_analysis_volume, standard_volume, cfg.mu_bulk);
            const auto score = first_water_score_from_pmf(pmf);
            site_dg_insert[s] = score.delta_g_insert;
            site_dg_bind[s] = score.delta_g_bind;
            site_status[s] = score.status;
        } catch (const std::exception& exc) {
            site_status[s] = std::string(exc.what()).substr(0, 180);
        }
    }

    {
        std::vector<std::string> headers = {
            "site", "x", "y", "z", "peak_density_score", "local_radius_A", "local_volume_mode",
            "local_analysis_volume_A3", "N_local_min_observed", "N_local_max_observed",
            "N_local_min_observed_valid", "N_local_max_observed_valid", "N_local_min_observed_all",
            "N_local_max_observed_all", "N_local_at_bulk_mu", "fit_n_windows", "fit_mu_min_kcal_mol",
            "fit_mu_max_kcal_mol", "fit_B_min", "fit_B_max", "capacity_filter_applied", "N_min_model",
            "N_max_model", "fit_terms", "fit_rmse_N", "deltaG_insert_first_water", "deltaG_bind_first_water",
            "deltaG_bind", "epsilon_mean", "status",
        };
        std::vector<std::vector<std::string>> rows;
        for (std::size_t s = 0; s < n_sites; ++s) {
            std::vector<double> y_all(n_mu), y_fit;
            for (std::size_t j = 0; j < n_mu; ++j) {
                y_all[j] = counts.local_mean[s * n_mu + j];
                if (capacity.valid_mask[j]) y_fit.push_back(y_all[j]);
            }
            rows.push_back({
                std::to_string(s),
                format_csv_double(site_centers[s][0]),
                format_csv_double(site_centers[s][1]),
                format_csv_double(site_centers[s][2]),
                format_csv_double(s < peak_scores.size() ? peak_scores[s] : std::numeric_limits<double>::quiet_NaN()),
                format_csv_double(cfg.local_radius),
                cfg.local_volume_mode,
                format_csv_double(local_analysis_volume),
                format_csv_double(*std::min_element(y_fit.begin(), y_fit.end())),
                format_csv_double(*std::max_element(y_fit.begin(), y_fit.end())),
                format_csv_double(*std::min_element(y_fit.begin(), y_fit.end())),
                format_csv_double(*std::max_element(y_fit.begin(), y_fit.end())),
                format_csv_double(*std::min_element(y_all.begin(), y_all.end())),
                format_csv_double(*std::max_element(y_all.begin(), y_all.end())),
                format_csv_double(y_all[mu_eq_idx]),
                std::to_string(n_valid),
                format_csv_double(mu_fit_values.front()),
                format_csv_double(mu_fit_values.back()),
                format_csv_double(B_fit.front()),
                format_csv_double(B_fit.back()),
                cfg.capacity_filter ? "True" : "False",
                format_csv_double(site_n_min[s]),
                format_csv_double(site_n_max[s]),
                std::to_string(site_terms[s]),
                format_csv_double(site_rmse[s]),
                format_csv_double(site_dg_insert[s]),
                format_csv_double(site_dg_bind[s]),
                format_csv_double(site_dg_bind[s]),
                format_csv_double(site_dg_bind[s]),
                site_status[s],
            });
        }
        write_csv_table(out_dir + "/sites.csv", headers, rows);
        write_csv_table(out_dir + "/local_gci_sites.csv", headers, rows);
    }

    {
        std::vector<double> local_at_bulk(n_sites);
        for (std::size_t s = 0; s < n_sites; ++s) local_at_bulk[s] = counts.local_mean[s * n_mu + mu_eq_idx];
        write_sites_pdb(
            out_dir + "/sites.pdb", site_centers, local_at_bulk, site_dg_bind,
            {"MCSwell local ProtoMS-style GCI sites",
             "occupancy column = mean local water count at nearest bulk mu",
             "B-factor = first-water bulk-referenced local GCI score"});
    }

    std::size_t n_usable = 0;
    for (std::size_t s = 0; s < n_sites; ++s) {
        if (std::isfinite(site_dg_bind[s])) ++n_usable;
    }

    const int n_eq_int = static_cast<int>(std::clamp(
        std::lround(counts.region_mean[mu_eq_idx]), std::lround(region_model.n_min), std::lround(region_model.n_max)));
    const GciPmfRow* eq_row = nullptr;
    const GciPmfRow* min_row = nullptr;
    for (const auto& r : region_pmf) {
        if (r.n == n_eq_int) eq_row = &r;
        if (!min_row || r.delta_g_bind_network_kcal_mol < min_row->delta_g_bind_network_kcal_mol) min_row = &r;
    }

    JsonWriter meta;
    meta.add("method", std::string("ProtoMS-style GCI using monotone sum-of-logistics fit"))
        .add("kT_kcal_mol", kT)
        .add("standard_volume_A3", standard_volume)
        .add("bulk_mu_kcal_mol", cfg.mu_bulk)
        .add("sampler_volume_A3", sampler_volume)
        .add("n_insertion_points", static_cast<long long>(n_insertion_points))
        .add("bulk_water_density_waters_A3", cfg.bulk_water_density)
        .add("reconstructed_water_capacity", static_cast<long long>(water_capacity))
        .add("capacity_filter_enabled", cfg.capacity_filter)
        .add("n_gci_valid_mu_windows", static_cast<long long>(n_valid))
        .add("nearest_bulk_mu_kcal_mol", mu_values[mu_eq_idx])
        .add("peak_percentile", cfg.peak_percentile)
        .add("local_radius_A", cfg.local_radius)
        .add("local_volume_mode", cfg.local_volume_mode)
        .add("local_analysis_volume_A3", local_analysis_volume)
        .add("n_sites", static_cast<long long>(n_sites))
        .add("n_local_usable", static_cast<long long>(n_usable))
        .add("region_fit_N_min", region_model.n_min)
        .add("region_fit_N_max", region_model.n_max)
        .add("region_fit_rmse_N", region_model.rmse);
    if (eq_row) meta.add("region_binding_dG_at_rounded_bulk_N", eq_row->delta_g_bind_network_kcal_mol);
    else meta.add_null("region_binding_dG_at_rounded_bulk_N");
    if (min_row) {
        meta.add("region_binding_pmf_min_N", static_cast<long long>(min_row->n))
            .add("region_binding_pmf_min_dG_kcal_mol", min_row->delta_g_bind_network_kcal_mol);
    }
    meta.write(out_dir + "/gci_metadata.json");

    return out_dir;
}

} // namespace mcswell::analysis
