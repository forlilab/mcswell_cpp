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

// Returns (N,3) points flattened as {x0,y0,z0, x1,y1,z1, ...}
std::vector<double> make_insertion_points_flat(
    const std::vector<Atom>& atoms,
    const GridSpec& grid,
    double max_distance,
    double min_distance = 1.5
);