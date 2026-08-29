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

#include "analysis/kdtree.hpp"
#include "harness.hpp"

using mcswell::analysis::KDTree3D;

namespace {

void test_nearest() {
    KDTree3D tree({{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}});
    MCSWELL_CHECK(tree.size() == 3);

    auto hit = tree.nearest({0.5, 0.0, 0.0}, 2.0);
    MCSWELL_CHECK(hit.found);
    MCSWELL_CHECK(hit.index == 0);
    MCSWELL_CHECK_NEAR(hit.distance, 0.5, 1e-9);

    auto miss = tree.nearest({5.0, 5.0, 5.0}, 1.0);
    MCSWELL_CHECK(!miss.found);

    auto batch = tree.nearest_batch({{0.1, 0.0, 0.0}, {10.1, 0.0, 0.0}}, 1.0);
    MCSWELL_CHECK(batch.size() == 2);
    MCSWELL_CHECK(batch[0].found && batch[0].index == 0);
    MCSWELL_CHECK(batch[1].found && batch[1].index == 1);
}

void test_count_within_radius() {
    KDTree3D tree({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}});

    // Inclusive-radius convention (matches scipy): an exact-boundary
    // distance of 1.5 must count, not just distances strictly below it.
    MCSWELL_CHECK(tree.count_within_radius({0.0, 0.0, 0.0}, 1.5) == 2);
    // Distances 0, 1, 2 are <= 2.5; distance 3 is not.
    MCSWELL_CHECK(tree.count_within_radius({0.0, 0.0, 0.0}, 2.5) == 3);
    // Exact-boundary case: radius exactly equal to a point's distance.
    MCSWELL_CHECK(tree.count_within_radius({0.0, 0.0, 0.0}, 1.0) == 2);

    auto counts = tree.count_within_radius_batch(
        {{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}}, 1.5);
    MCSWELL_CHECK(counts.size() == 2);
    MCSWELL_CHECK(counts[0] == 2);
    MCSWELL_CHECK(counts[1] == 2); // distances 0 and 1 from point index 3
}

void test_radius_pairs() {
    KDTree3D tree({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}, {5.5, 0.0, 0.0}});

    auto pairs = tree.radius_pairs(1.2);
    MCSWELL_CHECK(pairs.size() == 2);

    bool has_01 = false;
    bool has_23 = false;
    for (const auto& p : pairs) {
        if (p.first == 0 && p.second == 1) has_01 = true;
        if (p.first == 2 && p.second == 3) has_23 = true;
        MCSWELL_CHECK(p.first < p.second); // always i < j
    }
    MCSWELL_CHECK(has_01);
    MCSWELL_CHECK(has_23);

    // Exact-boundary case: radius exactly equal to the (0,1) distance must
    // still count that pair (inclusive, matching scipy's query_pairs).
    // (2,3) (distance 0.5) is also within this radius, so 2 pairs total.
    auto exact = tree.radius_pairs(1.0);
    MCSWELL_CHECK(exact.size() == 2);
    bool exact_has_01 = false;
    for (const auto& p : exact) {
        if (p.first == 0 && p.second == 1) exact_has_01 = true;
    }
    MCSWELL_CHECK(exact_has_01);
}

void test_empty_tree() {
    KDTree3D tree({});
    MCSWELL_CHECK(tree.size() == 0);

    auto hit = tree.nearest({0.0, 0.0, 0.0}, 100.0);
    MCSWELL_CHECK(!hit.found);
    MCSWELL_CHECK(tree.count_within_radius({0.0, 0.0, 0.0}, 100.0) == 0);
    MCSWELL_CHECK(tree.radius_pairs(100.0).empty());
}

} // namespace

void run_kdtree_tests() {
    test_nearest();
    test_count_within_radius();
    test_radius_pairs();
    test_empty_tree();
}
