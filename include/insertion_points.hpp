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
#include <array>
#include <cstdint>
#include <vector>

struct Atom;

struct GridSpec {
    std::array<double,3> size;    // (x,y,z)
    std::array<double,3> center;  // (cx,cy,cz)
    double spacing = 0.0;
};

// The insertion lattice, as both the list of kept points and a dense
// occupancy mask over the full grid.
//
// The two describe the same set. The list is what insertion draws from; the
// mask answers "is this arbitrary position inside the sampled region?" in O(1),
// which the displacement move needs in order to stay inside it. A search over
// the point list would be the same question answered in O(n_points).
//
// The sampled volume used in the Adams parameter is n_points * spacing^3, i.e.
// the union of the voxels around the kept points -- so voxel membership, not
// distance-to-receptor, is the criterion that keeps the volume bookkeeping
// exactly consistent.
struct InsertionGrid {
    std::vector<double> points_xyz;    // 3 * n_points, {x0,y0,z0, x1,...}
    std::vector<std::uint8_t> mask;    // dims[0]*dims[1]*dims[2], 1 = kept
    std::array<std::int64_t, 3> dims{0, 0, 0};
    std::array<double, 3> origin{0.0, 0.0, 0.0};  // lattice point (0,0,0)
    double spacing = 0.0;

    std::size_t n_points() const { return points_xyz.size() / 3; }

    // Flat index of lattice cell (i,j,k). Matches the generation loop order
    // (i outermost over x, k innermost over z); the kernel indexes the same way.
    std::size_t index(std::int64_t i, std::int64_t j, std::int64_t k) const {
        return static_cast<std::size_t>((i * dims[1] + j) * dims[2] + k);
    }
};

InsertionGrid make_insertion_grid(
    const std::vector<Atom>& atoms,
    const GridSpec& grid,
    double max_distance,
    double min_distance = 1.5
);

// The InsertionGrid of an existing point list: the lattice is regular, so
// the points themselves fix it (origin = the per-axis minimum, which is
// always a lattice coordinate). Throws if a point is off that lattice.
// Lets the displacement constraint be built when only the points, not the
// box that generated them, are known.
InsertionGrid insertion_grid_from_points(const float* points_xyz, std::size_t n_points, double spacing);

// Returns (N,3) points flattened as {x0,y0,z0, x1,y1,z1, ...}
std::vector<double> make_insertion_points_flat(
    const std::vector<Atom>& atoms,
    const GridSpec& grid,
    double max_distance,
    double min_distance = 1.5
);