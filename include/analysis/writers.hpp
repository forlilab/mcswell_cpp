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

#include "analysis/density_grid.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace mcswell::analysis {

// Shortest round-trip decimal representation of a double (via
// std::to_chars), or "nan"/"inf"/"-inf". Every CSV writer in this project
// (the many small per-analysis tables assembled in the pipeline layer,
// not exposed here as one function per Python script) goes through this,
// rather than each hand-rolling ostringstream precision.
std::string format_csv_double(double v);

// Bare-bones CSV table writer: fixed header row, then pre-formatted string
// cells (use format_csv_double for numeric columns). Deliberately generic
// rather than one named function per Python write_*_csv(): those were
// pure field-list/formatting differences with no algorithmic content, so
// call sites in the pipeline layer assemble headers/rows directly and
// hand them here instead of this header growing a dozen narrowly-used
// row structs.
void write_csv_table(
    const std::string& path, const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows);

// Pseudo-atom PDB for hydration sites (used by both the binomial and GCI
// analyses, which differ only in what they put in the occupancy/B-factor
// columns): one HETATM record per center, occupancy = occupancy_col[i],
// B-factor = bfactor_col[i]. `remark_lines` are emitted verbatim as
// leading "REMARK ..." lines.
void write_sites_pdb(
    const std::string& path, const std::vector<std::array<double, 3>>& centers,
    const std::vector<double>& occupancy_col, const std::vector<double>& bfactor_col,
    const std::vector<std::string>& remark_lines);

// OpenDX volumetric density map for visualization (e.g. in PyMOL/VMD).
void density_to_dx(const DensityGrid& grid, const std::string& path);

} // namespace mcswell::analysis
