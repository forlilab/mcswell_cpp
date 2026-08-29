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

// A small, dense, unconstrained Levenberg-Marquardt solver with a
// finite-difference Jacobian. Written specifically for the GCI
// logistic-sum fit (include/analysis/gci_fit.hpp) rather than vendoring a
// general-purpose linear-algebra dependency for one fit: parameter counts
// here are tiny (at most ~35, since fit_logistic_sum caps at 12 logistic
// terms x 3 params each), so a hand-rolled dense normal-equations solve is
// easy to get right.
//
// This is intentionally unconstrained: the caller keeps parameters
// well-behaved through reparameterization instead (e.g. optimizing
// log(slope) so the decoded slope is always positive, and softmax'd
// logits so decoded amplitudes are always non-negative and sum to a fixed
// total) -- the same trick the original scipy-based implementation used
// on top of scipy's own bounded trust-region solver. Here it's the only
// mechanism, which is a deliberate scope reduction versus scipy's
// bounds=(lower, upper); see gci_fit.cpp for where this could matter.

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mcswell::analysis {

struct LMResult {
    std::vector<double> params;
    double cost = 0.0; // sum of squared residuals at `params`
    int iterations = 0;
    bool converged = false;
};

namespace detail {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Solves A x = b via Gaussian elimination with partial pivoting. Returns
// false (leaving x unspecified) if A is numerically singular.
inline bool solve_linear_system(std::vector<std::vector<double>> A, std::vector<double> b, std::vector<double>& x) {
    const std::size_t n = b.size();
    x.assign(n, 0.0);

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot_row = col;
        double pivot_val = std::fabs(A[col][col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            if (std::fabs(A[row][col]) > pivot_val) {
                pivot_val = std::fabs(A[row][col]);
                pivot_row = row;
            }
        }
        if (pivot_val < 1e-14) return false;

        if (pivot_row != col) {
            std::swap(A[col], A[pivot_row]);
            std::swap(b[col], b[pivot_row]);
        }

        for (std::size_t row = col + 1; row < n; ++row) {
            const double factor = A[row][col] / A[col][col];
            if (factor == 0.0) continue;
            for (std::size_t k = col; k < n; ++k) A[row][k] -= factor * A[col][k];
            b[row] -= factor * b[col];
        }
    }

    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t k = i + 1; k < n; ++k) s -= A[i][k] * x[k];
        x[i] = s / A[i][i];
    }
    return true;
}

} // namespace detail

// `residual_fn(params, out_residuals)` fills out_residuals (already sized
// to n_residuals) given the current parameter vector.
template <class ResidualFn>
LMResult levenberg_marquardt(
    ResidualFn residual_fn, std::vector<double> params, std::size_t n_residuals,
    int max_iterations = 200, double lambda0 = 1e-3, double xtol = 1e-11, double ftol = 1e-11) {
    const std::size_t n = params.size();

    std::vector<double> r(n_residuals);
    residual_fn(params, r);
    double cost = detail::dot(r, r);

    LMResult result;
    result.params = params;
    result.cost = cost;

    double lambda = lambda0;
    const double h_eps = std::sqrt(std::numeric_limits<double>::epsilon());

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<std::vector<double>> J(n_residuals, std::vector<double>(n));
        std::vector<double> r_plus(n_residuals), r_minus(n_residuals);
        std::vector<double> perturbed = params;

        for (std::size_t j = 0; j < n; ++j) {
            const double h = h_eps * std::max(1.0, std::fabs(params[j]));
            perturbed[j] = params[j] + h;
            residual_fn(perturbed, r_plus);
            perturbed[j] = params[j] - h;
            residual_fn(perturbed, r_minus);
            perturbed[j] = params[j];
            for (std::size_t i = 0; i < n_residuals; ++i) {
                J[i][j] = (r_plus[i] - r_minus[i]) / (2.0 * h);
            }
        }

        std::vector<std::vector<double>> JTJ(n, std::vector<double>(n, 0.0));
        std::vector<double> JTr(n, 0.0);
        for (std::size_t i = 0; i < n_residuals; ++i) {
            for (std::size_t a = 0; a < n; ++a) {
                JTr[a] += J[i][a] * r[i];
                for (std::size_t b = 0; b < n; ++b) JTJ[a][b] += J[i][a] * J[i][b];
            }
        }

        bool step_accepted = false;
        for (int trial = 0; trial < 20 && !step_accepted; ++trial) {
            std::vector<std::vector<double>> A = JTJ;
            for (std::size_t a = 0; a < n; ++a) A[a][a] += lambda * std::max(JTJ[a][a], 1e-12);

            std::vector<double> rhs(n);
            for (std::size_t a = 0; a < n; ++a) rhs[a] = -JTr[a];

            std::vector<double> dx;
            if (!detail::solve_linear_system(A, rhs, dx)) {
                lambda *= 10.0;
                continue;
            }

            std::vector<double> new_params(n);
            for (std::size_t a = 0; a < n; ++a) new_params[a] = params[a] + dx[a];

            std::vector<double> new_r(n_residuals);
            residual_fn(new_params, new_r);
            const double new_cost = detail::dot(new_r, new_r);

            if (new_cost < cost) {
                double max_param_change = 0.0;
                for (double d : dx) max_param_change = std::max(max_param_change, std::fabs(d));
                const double cost_change = std::fabs(cost - new_cost);

                params = new_params;
                r = new_r;
                cost = new_cost;
                lambda = std::max(lambda / 10.0, 1e-12);
                step_accepted = true;

                if (max_param_change < xtol || cost_change < ftol * std::max(1.0, cost)) {
                    result.params = params;
                    result.cost = cost;
                    result.iterations = iter + 1;
                    result.converged = true;
                    return result;
                }
            } else {
                lambda *= 10.0;
            }
        }

        if (!step_accepted) {
            result.params = params;
            result.cost = cost;
            result.iterations = iter;
            result.converged = false;
            return result;
        }
    }

    result.params = params;
    result.cost = cost;
    result.iterations = max_iterations;
    result.converged = false;
    return result;
}

} // namespace mcswell::analysis
