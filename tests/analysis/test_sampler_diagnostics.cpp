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

#include "harness.hpp"

#include "analysis/sampler_diagnostics.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace mcswell::analysis;

namespace {

// Two mu, ten windows of 100 steps, equilibration ending exactly on the
// window-4/5 boundary. Window w of mu m gets a distinct, easily checked
// pattern so that mis-indexing shows up as a wrong total rather than as a
// plausible-looking number.
SamplerDiagnostics make_diag() {
    SamplerDiagnostics d;
    d.n_mu = 2;
    d.n_windows = 10;
    d.steps_per_window = 100;
    d.total_steps = 1000;
    d.equilibration_steps = 500;

    const std::size_t n = d.n_mu * d.n_windows;
    d.proposed.assign(n * N_MOVE_TYPES, 0);
    d.accepted.assign(n * N_MOVE_TYPES, 0);
    d.capacity_blocked.assign(n, 0);
    d.n_waters_sum.assign(n, 0);
    d.n_waters_end.assign(n, 0);
    d.energy_end.assign(n, 0.0f);

    for (std::size_t mu = 0; mu < d.n_mu; ++mu) {
        for (std::size_t w = 0; w < d.n_windows; ++w) {
            const std::size_t flat = mu * d.n_windows + w;
            // Proposals split 40/30/30 so each window sums to the 100 steps.
            d.proposed[flat * N_MOVE_TYPES + MOVE_INSERTION] = 40;
            d.proposed[flat * N_MOVE_TYPES + MOVE_DELETION] = 30;
            d.proposed[flat * N_MOVE_TYPES + MOVE_DISPLACEMENT] = 30;
            // One accepted insertion per window, two deletions, three
            // displacements -- all independent of mu and window, so the
            // phase sums are exact multiples of the window count.
            d.accepted[flat * N_MOVE_TYPES + MOVE_INSERTION] = 1;
            d.accepted[flat * N_MOVE_TYPES + MOVE_DELETION] = 2;
            d.accepted[flat * N_MOVE_TYPES + MOVE_DISPLACEMENT] = 3;
            d.capacity_blocked[flat] = mu;  // 0 for mu 0, 1 per window for mu 1
            d.n_waters_sum[flat] = 100 * (mu + 1);  // <N> = mu + 1
            d.n_waters_end[flat] = static_cast<std::uint32_t>(w);
            d.energy_end[flat] = -1.0f * static_cast<float>(w);
        }
    }
    return d;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::size_t count_lines(const std::string& text) {
    std::size_t n = 0;
    for (char c : text) {
        if (c == '\n') ++n;
    }
    return n;
}

void test_window_geometry() {
    const auto d = make_diag();

    MCSWELL_CHECK(!d.empty());
    MCSWELL_CHECK(d.window_begin(0) == 0);
    MCSWELL_CHECK(d.window_end(0) == 100);
    MCSWELL_CHECK(d.window_steps(0) == 100);
    MCSWELL_CHECK(d.window_begin(9) == 900);
    MCSWELL_CHECK(d.window_end(9) == 1000);

    // Equilibration ends at 500, i.e. exactly at the start of window 5.
    MCSWELL_CHECK(d.first_production_window() == 5);

    // Every window must be accounted for exactly once.
    std::uint64_t covered = 0;
    for (std::size_t w = 0; w < d.n_windows; ++w) covered += d.window_steps(w);
    MCSWELL_CHECK(covered == d.total_steps);
}

void test_window_geometry_clamps_to_total_steps() {
    // A run whose length is not a multiple of the window size: the last
    // window must be short rather than running past the end.
    SamplerDiagnostics d = make_diag();
    d.total_steps = 950;
    MCSWELL_CHECK(d.window_end(9) == 950);
    MCSWELL_CHECK(d.window_steps(9) == 50);

    std::uint64_t covered = 0;
    for (std::size_t w = 0; w < d.n_windows; ++w) covered += d.window_steps(w);
    MCSWELL_CHECK(covered == 950);
}

void test_first_production_window_rounds_up() {
    // Equilibration ending mid-window: the straddling window is excluded from
    // production rather than being counted as if it were all production.
    SamplerDiagnostics d = make_diag();
    d.equilibration_steps = 450;
    MCSWELL_CHECK(d.first_production_window() == 5);

    d.equilibration_steps = 0;
    MCSWELL_CHECK(d.first_production_window() == 0);

    // Equilibration covering the whole run leaves no production windows.
    d.equilibration_steps = 1000;
    MCSWELL_CHECK(d.first_production_window() == d.n_windows);
}

void test_summaries_split_by_phase() {
    const auto d = make_diag();
    const std::vector<double> mu_values = {-10.0, -5.0};

    const auto equil = summarize_equilibration(d, mu_values);
    const auto prod = summarize_production(d, mu_values);
    const auto total = summarize_acceptance(d, mu_values, 0, d.n_windows);

    MCSWELL_CHECK(equil.size() == 2);
    MCSWELL_CHECK(prod.size() == 2);

    // 5 equilibration windows, 5 production windows, 100 steps each.
    MCSWELL_CHECK(equil[0].steps == 500);
    MCSWELL_CHECK(prod[0].steps == 500);
    MCSWELL_CHECK(total[0].steps == 1000);

    // Counts add up across phases.
    for (int m = 0; m < N_MOVE_TYPES; ++m) {
        MCSWELL_CHECK(equil[0].proposed[m] + prod[0].proposed[m] == total[0].proposed[m]);
        MCSWELL_CHECK(equil[0].accepted[m] + prod[0].accepted[m] == total[0].accepted[m]);
    }

    // 5 windows x 40 insertion proposals, 1 accepted each.
    MCSWELL_CHECK(prod[0].proposed[MOVE_INSERTION] == 200);
    MCSWELL_CHECK(prod[0].accepted[MOVE_INSERTION] == 5);
    MCSWELL_CHECK_NEAR(prod[0].rate(MOVE_INSERTION), 5.0 / 200.0, 1e-12);
    MCSWELL_CHECK_NEAR(prod[0].rate(MOVE_DELETION), 10.0 / 150.0, 1e-12);
    MCSWELL_CHECK_NEAR(prod[0].rate(MOVE_DISPLACEMENT), 15.0 / 150.0, 1e-12);

    // Overall rate pools the move types rather than averaging their rates.
    MCSWELL_CHECK(prod[0].total_proposed() == 500);
    MCSWELL_CHECK(prod[0].total_accepted() == 30);
    MCSWELL_CHECK_NEAR(prod[0].overall_rate(), 30.0 / 500.0, 1e-12);

    // <N> is the step-weighted mean, not the mean of window means.
    MCSWELL_CHECK_NEAR(prod[0].mean_n_waters, 1.0, 1e-12);
    MCSWELL_CHECK_NEAR(prod[1].mean_n_waters, 2.0, 1e-12);

    // mu is carried through, and capacity blocks are per-mu.
    MCSWELL_CHECK_NEAR(prod[0].mu_value, -10.0, 1e-12);
    MCSWELL_CHECK_NEAR(prod[1].mu_value, -5.0, 1e-12);
    MCSWELL_CHECK(prod[0].capacity_blocked == 0);
    MCSWELL_CHECK(prod[1].capacity_blocked == 5);

    // final_n_waters is read off the last window of the slice, not the run.
    MCSWELL_CHECK(equil[0].final_n_waters == 4);
    MCSWELL_CHECK(prod[0].final_n_waters == 9);
}

void test_zero_proposal_rate_is_zero_not_nan() {
    SamplerDiagnostics d = make_diag();
    for (auto& v : d.proposed) v = 0;
    for (auto& v : d.accepted) v = 0;

    const auto prod = summarize_production(d, {0.0, 0.0});
    for (int m = 0; m < N_MOVE_TYPES; ++m) {
        MCSWELL_CHECK(prod[0].rate(m) == 0.0);
    }
    MCSWELL_CHECK(prod[0].overall_rate() == 0.0);
}

void test_empty_diagnostics_are_harmless() {
    SamplerDiagnostics d;
    MCSWELL_CHECK(d.empty());
    MCSWELL_CHECK(summarize_production(d, {}).empty());
    MCSWELL_CHECK(summarize_equilibration(d, {}).empty());
}

void test_windows_csv_shape_and_phase_labels() {
    const auto d = make_diag();
    const std::string path = "test_acceptance_windows.csv";
    write_acceptance_windows_csv(d, {-10.0, -5.0}, path);

    const std::string text = read_file(path);
    MCSWELL_CHECK(!text.empty());
    // Header plus one row per (mu, window).
    MCSWELL_CHECK(count_lines(text) == 1 + d.n_mu * d.n_windows);
    MCSWELL_CHECK(text.find("insertion_rate") != std::string::npos);
    MCSWELL_CHECK(text.find("capacity_blocked") != std::string::npos);
    MCSWELL_CHECK(text.find("equilibration") != std::string::npos);
    MCSWELL_CHECK(text.find("production") != std::string::npos);

    std::remove(path.c_str());
}

void test_summary_csv_has_three_phases_per_mu() {
    const auto d = make_diag();
    const std::string path = "test_acceptance_summary.csv";
    write_acceptance_summary_csv(d, {-10.0, -5.0}, path);

    const std::string text = read_file(path);
    // Header plus equilibration/production/total for each of the two mu.
    MCSWELL_CHECK(count_lines(text) == 1 + 3 * d.n_mu);
    MCSWELL_CHECK(text.find(",total,") != std::string::npos);

    std::remove(path.c_str());
}

void test_move_type_names() {
    MCSWELL_CHECK(std::string(move_type_name(MOVE_INSERTION)) == "insertion");
    MCSWELL_CHECK(std::string(move_type_name(MOVE_DELETION)) == "deletion");
    MCSWELL_CHECK(std::string(move_type_name(MOVE_DISPLACEMENT)) == "displacement");
    MCSWELL_CHECK(std::string(move_type_name(99)) == "unknown");
}

} // namespace

void run_sampler_diagnostics_tests() {
    test_window_geometry();
    test_window_geometry_clamps_to_total_steps();
    test_first_production_window_rounds_up();
    test_summaries_split_by_phase();
    test_zero_proposal_rate_is_zero_not_nan();
    test_empty_diagnostics_are_harmless();
    test_windows_csv_shape_and_phase_labels();
    test_summary_csv_has_three_phases_per_mu();
    test_move_type_names();
}
