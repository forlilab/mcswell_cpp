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

#include "analysis/root_finding.hpp"
#include "harness.hpp"

#include <cmath>
#include <stdexcept>

using mcswell::analysis::brentq;

namespace {

void test_cubic_root_matches_scipy() {
    auto f = [](double x) { return x * x * x - x - 2.0; };
    // scipy.optimize.brentq(f, 0.0, 2.0)
    const double expected = 1.5213797068045676;
    const double root = brentq(f, 0.0, 2.0);
    MCSWELL_CHECK_NEAR(root, expected, 1e-10);
}

void test_exp_root_matches_scipy() {
    auto g = [](double x) { return std::exp(x) - 3.0; };
    // scipy.optimize.brentq(g, 0.0, 2.0)
    const double expected = 1.0986122886680403;
    const double root = brentq(g, 0.0, 2.0);
    MCSWELL_CHECK_NEAR(root, expected, 1e-10);
}

void test_unbracketed_throws() {
    auto f = [](double x) { return x * x + 1.0; }; // never crosses zero
    bool threw = false;
    try {
        brentq(f, -1.0, 1.0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    MCSWELL_CHECK(threw);
}

} // namespace

void run_root_finding_tests() {
    test_cubic_root_matches_scipy();
    test_exp_root_matches_scipy();
    test_unbracketed_throws();
}
