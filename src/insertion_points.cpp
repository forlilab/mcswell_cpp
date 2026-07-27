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

#include <cmath>
#include <limits>
#include <stdexcept>

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

std::vector<double> make_insertion_points_flat(
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

    std::vector<double> out;
    out.reserve(static_cast<size_t>(x_points * y_points * z_points) * 3);

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

                out.push_back(x);
                out.push_back(y);
                out.push_back(z);
            }
        }
    }

    return out;
}