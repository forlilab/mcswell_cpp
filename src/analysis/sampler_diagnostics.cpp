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

#include "analysis/sampler_diagnostics.hpp"

#include "analysis/writers.hpp"

#include <algorithm>
#include <string>

namespace mcswell::analysis {

const char* move_type_name(int move_type) {
    switch (move_type) {
        case MOVE_INSERTION: return "insertion";
        case MOVE_DELETION: return "deletion";
        case MOVE_DISPLACEMENT: return "displacement";
        default: return "unknown";
    }
}

std::uint64_t SamplerDiagnostics::window_begin(std::size_t w) const {
    return static_cast<std::uint64_t>(w) * steps_per_window;
}

std::uint64_t SamplerDiagnostics::window_end(std::size_t w) const {
    return std::min(window_begin(w) + steps_per_window, total_steps);
}

std::uint64_t SamplerDiagnostics::window_steps(std::size_t w) const {
    const std::uint64_t begin = window_begin(w);
    const std::uint64_t end = window_end(w);
    return end > begin ? end - begin : 0;
}

std::size_t SamplerDiagnostics::first_production_window() const {
    if (steps_per_window == 0) return 0;
    // Ceiling division: the first window that starts at or after the
    // equilibration boundary, so a straddling window is left out of the
    // production slice entirely.
    const std::uint64_t w = (equilibration_steps + steps_per_window - 1) / steps_per_window;
    return static_cast<std::size_t>(std::min<std::uint64_t>(w, n_windows));
}

double AcceptanceSummary::rate(int move_type) const {
    const std::uint64_t p = proposed[move_type];
    return p == 0 ? 0.0 : static_cast<double>(accepted[move_type]) / static_cast<double>(p);
}

std::uint64_t AcceptanceSummary::total_proposed() const {
    return proposed[MOVE_INSERTION] + proposed[MOVE_DELETION] + proposed[MOVE_DISPLACEMENT];
}

std::uint64_t AcceptanceSummary::total_accepted() const {
    return accepted[MOVE_INSERTION] + accepted[MOVE_DELETION] + accepted[MOVE_DISPLACEMENT];
}

double AcceptanceSummary::overall_rate() const {
    const std::uint64_t p = total_proposed();
    return p == 0 ? 0.0 : static_cast<double>(total_accepted()) / static_cast<double>(p);
}

std::vector<AcceptanceSummary> summarize_acceptance(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values, std::size_t w_begin,
    std::size_t w_end) {
    std::vector<AcceptanceSummary> out;
    if (diag.empty()) return out;

    w_end = std::min(w_end, diag.n_windows);
    out.reserve(diag.n_mu);

    for (std::size_t mu = 0; mu < diag.n_mu; ++mu) {
        AcceptanceSummary s;
        s.mu_index = mu;
        s.mu_value = mu < mu_values.size() ? mu_values[mu] : 0.0;

        std::uint64_t n_sum = 0;
        for (std::size_t w = w_begin; w < w_end; ++w) {
            const std::size_t flat = mu * diag.n_windows + w;
            for (int m = 0; m < N_MOVE_TYPES; ++m) {
                s.proposed[m] += diag.proposed[flat * N_MOVE_TYPES + m];
                s.accepted[m] += diag.accepted[flat * N_MOVE_TYPES + m];
            }
            s.capacity_blocked += diag.capacity_blocked[flat];
            n_sum += diag.n_waters_sum[flat];
            s.steps += diag.window_steps(w);
        }

        s.mean_n_waters = s.steps == 0 ? 0.0 : static_cast<double>(n_sum) / static_cast<double>(s.steps);
        s.final_n_waters = w_end > w_begin ? diag.n_waters_end[mu * diag.n_windows + (w_end - 1)] : 0u;
        out.push_back(s);
    }
    return out;
}

std::vector<AcceptanceSummary> summarize_production(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values) {
    return summarize_acceptance(diag, mu_values, diag.first_production_window(), diag.n_windows);
}

std::vector<AcceptanceSummary> summarize_equilibration(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values) {
    return summarize_acceptance(diag, mu_values, 0, diag.first_production_window());
}

void write_acceptance_windows_csv(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values, const std::string& path) {
    const std::vector<std::string> headers = {
        "mu_index",
        "mu_kcal_mol",
        "window",
        "step_begin",
        "step_end",
        "n_steps",
        "phase",
        "insertion_proposed",
        "insertion_accepted",
        "insertion_rate",
        "deletion_proposed",
        "deletion_accepted",
        "deletion_rate",
        "displacement_proposed",
        "displacement_accepted",
        "displacement_rate",
        "overall_rate",
        "capacity_blocked",
        "mean_n_waters",
        "n_waters_end",
        "energy_end_kcal_mol",
    };

    std::vector<std::vector<std::string>> rows;
    if (!diag.empty()) rows.reserve(diag.n_mu * diag.n_windows);

    const std::size_t first_prod = diag.first_production_window();

    for (std::size_t mu = 0; mu < diag.n_mu; ++mu) {
        for (std::size_t w = 0; w < diag.n_windows; ++w) {
            const std::size_t flat = mu * diag.n_windows + w;
            const std::uint64_t steps = diag.window_steps(w);

            std::uint64_t prop[N_MOVE_TYPES];
            std::uint64_t acc[N_MOVE_TYPES];
            std::uint64_t prop_total = 0, acc_total = 0;
            for (int m = 0; m < N_MOVE_TYPES; ++m) {
                prop[m] = diag.proposed[flat * N_MOVE_TYPES + m];
                acc[m] = diag.accepted[flat * N_MOVE_TYPES + m];
                prop_total += prop[m];
                acc_total += acc[m];
            }

            const auto rate = [](std::uint64_t a, std::uint64_t p) {
                return p == 0 ? 0.0 : static_cast<double>(a) / static_cast<double>(p);
            };

            std::vector<std::string> row;
            row.reserve(headers.size());
            row.push_back(std::to_string(mu));
            row.push_back(format_csv_double(mu < mu_values.size() ? mu_values[mu] : 0.0));
            row.push_back(std::to_string(w));
            row.push_back(std::to_string(diag.window_begin(w)));
            row.push_back(std::to_string(diag.window_end(w)));
            row.push_back(std::to_string(steps));
            row.push_back(w < first_prod ? "equilibration" : "production");
            for (int m = 0; m < N_MOVE_TYPES; ++m) {
                row.push_back(std::to_string(prop[m]));
                row.push_back(std::to_string(acc[m]));
                row.push_back(format_csv_double(rate(acc[m], prop[m])));
            }
            row.push_back(format_csv_double(rate(acc_total, prop_total)));
            row.push_back(std::to_string(diag.capacity_blocked[flat]));
            row.push_back(format_csv_double(
                steps == 0 ? 0.0 : static_cast<double>(diag.n_waters_sum[flat]) / static_cast<double>(steps)));
            row.push_back(std::to_string(diag.n_waters_end[flat]));
            row.push_back(format_csv_double(static_cast<double>(diag.energy_end[flat])));
            rows.push_back(std::move(row));
        }
    }

    write_csv_table(path, headers, rows);
}

namespace {

void append_summary_rows(
    const std::vector<AcceptanceSummary>& summaries, const std::string& phase,
    std::vector<std::vector<std::string>>& rows) {
    for (const auto& s : summaries) {
        std::vector<std::string> row;
        row.push_back(std::to_string(s.mu_index));
        row.push_back(format_csv_double(s.mu_value));
        row.push_back(phase);
        row.push_back(std::to_string(s.steps));
        for (int m = 0; m < N_MOVE_TYPES; ++m) {
            row.push_back(std::to_string(s.proposed[m]));
            row.push_back(std::to_string(s.accepted[m]));
            row.push_back(format_csv_double(s.rate(m)));
        }
        row.push_back(std::to_string(s.total_proposed()));
        row.push_back(std::to_string(s.total_accepted()));
        row.push_back(format_csv_double(s.overall_rate()));
        row.push_back(std::to_string(s.capacity_blocked));
        row.push_back(format_csv_double(s.mean_n_waters));
        row.push_back(std::to_string(s.final_n_waters));
        rows.push_back(std::move(row));
    }
}

} // namespace

void write_acceptance_summary_csv(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values, const std::string& path) {
    const std::vector<std::string> headers = {
        "mu_index",
        "mu_kcal_mol",
        "phase",
        "n_steps",
        "insertion_proposed",
        "insertion_accepted",
        "insertion_rate",
        "deletion_proposed",
        "deletion_accepted",
        "deletion_rate",
        "displacement_proposed",
        "displacement_accepted",
        "displacement_rate",
        "total_proposed",
        "total_accepted",
        "overall_rate",
        "capacity_blocked",
        "mean_n_waters",
        "final_n_waters",
    };

    std::vector<std::vector<std::string>> rows;
    append_summary_rows(summarize_equilibration(diag, mu_values), "equilibration", rows);
    append_summary_rows(summarize_production(diag, mu_values), "production", rows);
    append_summary_rows(summarize_acceptance(diag, mu_values, 0, diag.n_windows), "total", rows);

    write_csv_table(path, headers, rows);
}

} // namespace mcswell::analysis
