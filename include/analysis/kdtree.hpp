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

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace mcswell::analysis {

struct NeighborResult {
    std::size_t index = static_cast<std::size_t>(-1);
    double distance = std::numeric_limits<double>::infinity();
    bool found = false;
};

// Fixed 3D point-cloud KD-tree for nearest-neighbor and radius queries.
// Pure host-side utility, independent of CUDA. Points are copied at
// construction, so the tree stays valid regardless of the caller's storage
// lifetime. nanoflann.hpp is only included in the .cpp (compile firewall).
class KDTree3D {
public:
    explicit KDTree3D(std::vector<std::array<double, 3>> points);
    ~KDTree3D();

    KDTree3D(const KDTree3D&) = delete;
    KDTree3D& operator=(const KDTree3D&) = delete;
    KDTree3D(KDTree3D&&) noexcept;
    KDTree3D& operator=(KDTree3D&&) noexcept;

    std::size_t size() const;

    // Nearest indexed point to `query`, or found=false if none lies within
    // `max_distance` (Euclidean). Mirrors scipy.spatial.cKDTree.query with
    // distance_upper_bound=max_distance.
    NeighborResult nearest(const std::array<double, 3>& query, double max_distance) const;
    std::vector<NeighborResult> nearest_batch(
        const std::vector<std::array<double, 3>>& queries, double max_distance) const;

    // Number of indexed points within `radius` (inclusive) of `query`.
    // Mirrors scipy.spatial.cKDTree.query_ball_point(..., return_length=True).
    std::size_t count_within_radius(const std::array<double, 3>& query, double radius) const;
    std::vector<std::size_t> count_within_radius_batch(
        const std::vector<std::array<double, 3>>& queries, double radius) const;

    // All unordered index pairs (i < j) among the indexed points whose
    // distance is at most `radius`. Mirrors scipy.spatial.cKDTree.query_pairs.
    std::vector<std::pair<std::size_t, std::size_t>> radius_pairs(double radius) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mcswell::analysis
