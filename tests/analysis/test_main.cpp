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

#include <iostream>

// One `run_*_tests()` per test_*.cpp file, registered here by hand -- see
// harness.hpp for why this project doesn't pull in a test framework.
void run_kdtree_tests();
void run_frame_extraction_tests();
void run_density_grid_tests();

int main() {
    run_kdtree_tests();
    run_frame_extraction_tests();
    run_density_grid_tests();

    if (g_mcswell_test_failures == 0) {
        std::cout << "[PASS] all analysis unit tests passed\n";
        return 0;
    }
    std::cerr << "[FAIL] " << g_mcswell_test_failures << " check(s) failed\n";
    return 1;
}
