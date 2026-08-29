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

// Minimal, dependency-free test harness for the CUDA-free analysis library.
// No gtest/doctest: the project has no test infrastructure today and these
// suites are small enough that hand-rolled CHECK macros are less overhead
// than vendoring a framework.

#include <cmath>
#include <cstdlib>
#include <iostream>

inline int g_mcswell_test_failures = 0;

#define MCSWELL_CHECK(cond)                                                            \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << "  " << #cond     \
                      << "\n";                                                         \
            ++g_mcswell_test_failures;                                                 \
        }                                                                              \
    } while (0)

#define MCSWELL_CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                               \
        const double _mcswell_a = (a);                                                 \
        const double _mcswell_b = (b);                                                 \
        const double _mcswell_tol = (tol);                                             \
        if (!(std::fabs(_mcswell_a - _mcswell_b) <= _mcswell_tol)) {                   \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << "  " << #a        \
                      << " ~= " << #b << "  (" << _mcswell_a << " vs " << _mcswell_b   \
                      << ", tol=" << _mcswell_tol << ")\n";                            \
            ++g_mcswell_test_failures;                                                 \
        }                                                                              \
    } while (0)
