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

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "insertion_points.hpp"
#include "atom.hpp"

static inline double min_dist2_to_atoms(const std::array<double,3>& p,
                                        const std::vector<Atom>& atoms) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& a : atoms) {
        const auto& c = a.coords;
        const double dx = p[0] - c[0];
        const double dy = p[1] - c[1];
        const double dz = p[2] - c[2];
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best) best = d2;
    }
    return best;
}

InsertionGrid make_insertion_grid(
    const std::vector<Atom>& atoms,
    const GridSpec& grid,
    double max_distance,
    double min_distance
) {
    if (grid.spacing <= 0.0) throw std::invalid_argument("spacing must be > 0");
    if (max_distance <= 0.0) throw std::invalid_argument("max_distance must be > 0");
    if (min_distance < 0.0) throw std::invalid_argument("min_distance must be >= 0");
    if (min_distance > max_distance) throw std::invalid_argument("min_distance must be <= max_distance");

    const double cx = grid.center[0], cy = grid.center[1], cz = grid.center[2];
    const double sx = grid.size[0],   sy = grid.size[1],   sz = grid.size[2];
    const double spacing = grid.spacing;

    const double x_min = cx - sx / 2.0;
    const double x_max = cx + sx / 2.0;
    const double y_min = cy - sy / 2.0;
    const double y_max = cy + sy / 2.0;
    const double z_min = cz - sz / 2.0;
    const double z_max = cz + sz / 2.0;

    const std::int64_t x_points = static_cast<std::int64_t>(std::ceil((x_max - x_min) / spacing)) + 1;
    const std::int64_t y_points = static_cast<std::int64_t>(std::ceil((y_max - y_min) / spacing)) + 1;
    const std::int64_t z_points = static_cast<std::int64_t>(std::ceil((z_max - z_min) / spacing)) + 1;

    const double max2 = max_distance * max_distance;
    const double min2 = min_distance * min_distance;

    InsertionGrid out;
    out.dims = {x_points, y_points, z_points};
    out.origin = {x_min, y_min, z_min};
    out.spacing = spacing;
    out.points_xyz.reserve(static_cast<size_t>(x_points * y_points * z_points) * 3);
    out.mask.assign(static_cast<size_t>(x_points * y_points * z_points), 0u);

    for (std::int64_t i = 0; i < x_points; i++) {
        const double x = x_min + static_cast<double>(i) * spacing;
        for (std::int64_t j = 0; j < y_points; j++) {
            const double y = y_min + static_cast<double>(j) * spacing;
            for (std::int64_t k = 0; k < z_points; k++) {
                const double z = z_min + static_cast<double>(k) * spacing;

                if (!atoms.empty()) {
                    const std::array<double,3> p{x,y,z};
                    const double d2 = min_dist2_to_atoms(p, atoms);
                    if (d2 > max2 || d2 < min2) continue;
                }

                out.points_xyz.push_back(x);
                out.points_xyz.push_back(y);
                out.points_xyz.push_back(z);
                out.mask[out.index(i, j, k)] = 1u;
            }
        }
    }

    return out;
}

InsertionGrid insertion_grid_from_points(const float* points_xyz, std::size_t n_points, double spacing) {
    if (n_points == 0 || spacing <= 0.0) {
        throw std::invalid_argument("insertion_grid_from_points: need points and a positive spacing");
    }
    InsertionGrid out;
    out.spacing = spacing;
    std::array<double, 3> hi{};
    for (int a = 0; a < 3; ++a) {
        out.origin[a] = std::numeric_limits<double>::infinity();
        hi[a] = -std::numeric_limits<double>::infinity();
    }
    for (std::size_t p = 0; p < n_points; ++p) {
        for (int a = 0; a < 3; ++a) {
            const double v = points_xyz[3 * p + a];
            out.origin[a] = std::min(out.origin[a], v);
            hi[a] = std::max(hi[a], v);
        }
    }
    for (int a = 0; a < 3; ++a) {
        out.dims[a] = static_cast<std::int64_t>(std::llround((hi[a] - out.origin[a]) / spacing)) + 1;
    }
    out.mask.assign(static_cast<std::size_t>(out.dims[0] * out.dims[1] * out.dims[2]), 0u);
    out.points_xyz.reserve(3 * n_points);

    for (std::size_t p = 0; p < n_points; ++p) {
        std::int64_t idx[3];
        for (int a = 0; a < 3; ++a) {
            const double v = points_xyz[3 * p + a];
            const double f = (v - out.origin[a]) / spacing;
            idx[a] = static_cast<std::int64_t>(std::llround(f));
            // Points arrive as float32; allow for that rounding, nothing more.
            if (std::fabs(f - static_cast<double>(idx[a])) > 1e-3) {
                throw std::invalid_argument("insertion_grid_from_points: point " + std::to_string(p) +
                                            " is not on a lattice of spacing " + std::to_string(spacing));
            }
            out.points_xyz.push_back(v);
        }
        out.mask[out.index(idx[0], idx[1], idx[2])] = 1u;
    }
    return out;
}

std::vector<double> make_insertion_points_flat(
    const std::vector<Atom>& atoms,
    const GridSpec& grid,
    double max_distance,
    double min_distance
) {
    return make_insertion_grid(atoms, grid, max_distance, min_distance).points_xyz;
}