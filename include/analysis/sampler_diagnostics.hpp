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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mcswell::analysis {

// Move types proposed by the GCMC kernel, in the order the kernel's
// rand_int(&st, n_moves) draw assigns them (see src/cuda/cuda_kernels.cu).
enum MoveType : int {
    MOVE_INSERTION = 0,
    MOVE_DELETION = 1,
    MOVE_DISPLACEMENT = 2,
    N_MOVE_TYPES = 3,
};

const char* move_type_name(int move_type);

// Per-(mu, window) sampling diagnostics collected by the GCMC kernel.
//
// The kernel runs one independent chain per mu, so every quantity here is
// resolved by mu as well as by window; windows tile the *whole* run,
// equilibration included, so the equilibration/production split is a host-side
// slice rather than something the kernel has to know about.
//
// Counters are per window, not cumulative: window w holds only what happened
// between window_begin(w) and window_end(w). Cumulative or run-total figures
// are built by summing, which is what summarize_acceptance() does.
//
// Collecting these costs the kernel a handful of register increments per step
// plus one total-energy evaluation per window, and -- importantly for
// reproducing published runs -- consumes no random numbers, so an instrumented
// run follows exactly the same Markov chain as an uninstrumented one with the
// same seed.
struct SamplerDiagnostics {
    std::size_t n_mu = 0;
    std::size_t n_windows = 0;
    std::uint64_t steps_per_window = 0;
    std::uint64_t total_steps = 0;
    std::uint64_t equilibration_steps = 0;

    // Indexed [(mu * n_windows + w) * N_MOVE_TYPES + move_type].
    std::vector<std::uint64_t> proposed;
    std::vector<std::uint64_t> accepted;

    // Indexed [mu * n_windows + w].
    //
    // capacity_blocked counts insertion proposals that passed the Metropolis
    // test but were refused anyway because the chain had already reached
    // target_n_waters (the fixed-size water buffer). These are *not* Metropolis
    // rejections: they are a hard wall, and a run in which they are not
    // negligible is one whose box is too small for the chemical potential being
    // sampled, independently of how well the chain is converged.
    std::vector<std::uint64_t> capacity_blocked;

    // Sum of the water count over the steps of the window; divide by the
    // window's step count for <N> over that window.
    std::vector<std::uint64_t> n_waters_sum;

    // Water count, and total water-receptor plus water-water interaction
    // energy (kcal/mol), of the configuration at the end of the window. The
    // energy omits the constant receptor-receptor term, so it is meaningful as
    // a relative trace rather than an absolute system energy.
    std::vector<std::uint32_t> n_waters_end;
    std::vector<float> energy_end;

    bool empty() const { return n_mu == 0 || n_windows == 0; }

    // Step range covered by window w, as [begin, end).
    std::uint64_t window_begin(std::size_t w) const;
    std::uint64_t window_end(std::size_t w) const;
    std::uint64_t window_steps(std::size_t w) const;

    // Index of the first window whose steps lie entirely in the production
    // phase. Windows straddling the equilibration boundary are excluded from
    // production sums rather than being split, which is exact as long as
    // steps_per_window divides equilibration_steps (the default windowing
    // makes it do so for the usual round step counts) and conservative
    // otherwise.
    std::size_t first_production_window() const;
};

// Acceptance statistics for one mu over one phase of the run.
struct AcceptanceSummary {
    std::size_t mu_index = 0;
    double mu_value = 0.0;

    std::uint64_t steps = 0;
    std::uint64_t proposed[N_MOVE_TYPES] = {0, 0, 0};
    std::uint64_t accepted[N_MOVE_TYPES] = {0, 0, 0};
    std::uint64_t capacity_blocked = 0;

    double mean_n_waters = 0.0;
    std::uint32_t final_n_waters = 0;

    double rate(int move_type) const;
    double overall_rate() const;
    std::uint64_t total_proposed() const;
    std::uint64_t total_accepted() const;
};

// Aggregate windows [w_begin, w_end) into one summary per mu.
std::vector<AcceptanceSummary> summarize_acceptance(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values, std::size_t w_begin,
    std::size_t w_end);

// Convenience wrappers for the two slices that get reported.
std::vector<AcceptanceSummary> summarize_production(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values);
std::vector<AcceptanceSummary> summarize_equilibration(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values);

// acceptance_windows.csv: one row per (mu, window). This is the trace the
// convergence analysis reads -- <N>, energy and per-move acceptance against
// step number -- so it is written whenever diagnostics are collected at all.
void write_acceptance_windows_csv(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values, const std::string& path);

// acceptance_summary.csv: one row per (mu, phase) with phase in
// {equilibration, production, total}. The table quoted in a paper.
void write_acceptance_summary_csv(
    const SamplerDiagnostics& diag, const std::vector<double>& mu_values, const std::string& path);

} // namespace mcswell::analysis
