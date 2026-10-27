//  Copyright (c) 2026 Scripps Research, Forli Lab.
//  All rights reserved.
//
//  Author: Niccolo Bruciaferri
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
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/consts.cuh"

namespace mcswell {

// Every snapshot the titration took, stored compactly: frame f (= mu *
// n_snapshots + snap) holds waters [offsets[f], offsets[f+1]), each one
// WATER_SIZE floats in the kernel's compact layout (OX/OY/OZ/ORES, H1..., H2...
// from include/cuda/consts.cuh).
//
// Unlike the dense buffer it replaces, nothing is padded to a fixed water
// count, so a frame's size is exactly its water count and no coordinate value
// doubles as an "empty slot" marker.
struct WaterSnapshots {
    std::size_t n_mu = 0;
    std::size_t n_snapshots = 0;
    std::vector<std::uint64_t> offsets;  // n_mu * n_snapshots + 1 entries, in waters
    std::vector<float> waters;           // offsets.back() * WATER_SIZE floats

    std::size_t frame_index(std::size_t mu, std::size_t snap) const {
        if (mu >= n_mu || snap >= n_snapshots) {
            throw std::out_of_range("WaterSnapshots: (mu " + std::to_string(mu) + ", snapshot " +
                                    std::to_string(snap) + ") out of range");
        }
        return mu * n_snapshots + snap;
    }

    std::size_t n_waters(std::size_t mu, std::size_t snap) const {
        const std::size_t f = frame_index(mu, snap);
        return static_cast<std::size_t>(offsets[f + 1] - offsets[f]);
    }

    // First float of the frame's first water.
    const float* frame(std::size_t mu, std::size_t snap) const {
        return waters.data() + offsets[frame_index(mu, snap)] * static_cast<std::size_t>(WATER_SIZE);
    }

    // Checks that offsets and waters describe each other; the analysis code
    // indexes without bounds checks, so a malformed buffer fails here instead.
    void validate() const {
        if (offsets.size() != n_mu * n_snapshots + 1) {
            throw std::invalid_argument("WaterSnapshots: offsets must have n_mu * n_snapshots + 1 entries");
        }
        if (offsets.front() != 0) throw std::invalid_argument("WaterSnapshots: offsets[0] must be 0");
        for (std::size_t f = 0; f + 1 < offsets.size(); ++f) {
            if (offsets[f + 1] < offsets[f]) {
                throw std::invalid_argument("WaterSnapshots: offsets must be non-decreasing");
            }
        }
        if (waters.size() != offsets.back() * static_cast<std::size_t>(WATER_SIZE)) {
            throw std::invalid_argument("WaterSnapshots: waters must hold offsets.back() * WATER_SIZE floats");
        }
    }
};

} // namespace mcswell
