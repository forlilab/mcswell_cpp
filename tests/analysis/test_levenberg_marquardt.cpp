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

#include "analysis/levenberg_marquardt.hpp"
#include "harness.hpp"

#include <cmath>
#include <vector>

using mcswell::analysis::levenberg_marquardt;

namespace {

void test_recovers_exponential_curve() {
    // Noiseless synthetic data from y = a*exp(b*x), a=2.5, b=-0.3.
    const double true_a = 2.5, true_b = -0.3;
    std::vector<double> xs, ys;
    for (double x = 0.0; x <= 5.0; x += 1.0) {
        xs.push_back(x);
        ys.push_back(true_a * std::exp(true_b * x));
    }

    auto residual = [&](const std::vector<double>& p, std::vector<double>& out) {
        out.resize(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) {
            out[i] = p[0] * std::exp(p[1] * xs[i]) - ys[i];
        }
    };

    // Deliberately off initial guess.
    auto result = levenberg_marquardt(residual, {1.0, 0.0}, xs.size());

    MCSWELL_CHECK(result.converged);
    MCSWELL_CHECK_NEAR(result.params[0], true_a, 1e-6);
    MCSWELL_CHECK_NEAR(result.params[1], true_b, 1e-6);
    MCSWELL_CHECK(result.cost < 1e-16);
}

void test_recovers_linear_curve() {
    // y = 3x + 1, trivially convex quadratic cost -> should converge fast.
    std::vector<double> xs = {0, 1, 2, 3, 4};
    std::vector<double> ys;
    for (double x : xs) ys.push_back(3.0 * x + 1.0);

    auto residual = [&](const std::vector<double>& p, std::vector<double>& out) {
        out.resize(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = p[0] * xs[i] + p[1] - ys[i];
    };

    auto result = levenberg_marquardt(residual, {0.0, 0.0}, xs.size());
    MCSWELL_CHECK(result.converged);
    MCSWELL_CHECK_NEAR(result.params[0], 3.0, 1e-8);
    MCSWELL_CHECK_NEAR(result.params[1], 1.0, 1e-8);
}

} // namespace

void run_levenberg_marquardt_tests() {
    test_recovers_exponential_curve();
    test_recovers_linear_curve();
}
