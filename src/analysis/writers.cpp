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

#include "analysis/writers.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace mcswell::analysis {

std::string format_csv_double(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, res.ptr);
}

void write_csv_table(
    const std::string& path, const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("unable to create file: " + path);

    for (std::size_t i = 0; i < headers.size(); ++i) {
        if (i) f << ',';
        f << headers[i];
    }
    f << '\n';

    for (const auto& row : rows) {
        if (row.size() != headers.size()) {
            throw std::invalid_argument("write_csv_table: row width does not match header width");
        }
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i) f << ',';
            f << row[i];
        }
        f << '\n';
    }
}

void write_sites_pdb(
    const std::string& path, const std::vector<std::array<double, 3>>& centers,
    const std::vector<double>& occupancy_col, const std::vector<double>& bfactor_col,
    const std::vector<std::string>& remark_lines) {
    if (centers.size() != occupancy_col.size() || centers.size() != bfactor_col.size()) {
        throw std::invalid_argument("write_sites_pdb: centers/occupancy_col/bfactor_col size mismatch");
    }

    std::ofstream f(path);
    if (!f) throw std::runtime_error("unable to create file: " + path);

    for (const auto& line : remark_lines) f << "REMARK " << line << "\n";

    char buf[128];
    for (std::size_t idx = 0; idx < centers.size(); ++idx) {
        const int i = static_cast<int>(idx) + 1;
        const auto& c = centers[idx];
        std::snprintf(
            buf, sizeof(buf), "HETATM%5d  O   SIT A%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n", i, i, c[0], c[1], c[2],
            occupancy_col[idx], bfactor_col[idx]);
        f << buf;
    }
    f << "END\n";
}

void density_to_dx(const DensityGrid& grid, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("unable to create file: " + path);

    const std::size_t nx = grid.shape[0], ny = grid.shape[1], nz = grid.shape[2];
    const double x0 = grid.edges[0][0], y0 = grid.edges[1][0], z0 = grid.edges[2][0];
    const double dx = grid.edges[0][1] - grid.edges[0][0];
    const double dy = grid.edges[1][1] - grid.edges[1][0];
    const double dz = grid.edges[2][1] - grid.edges[2][0];

    f << "object 1 class gridpositions counts " << nx << " " << ny << " " << nz << "\n";
    f << "origin " << format_csv_double(x0) << " " << format_csv_double(y0) << " " << format_csv_double(z0) << "\n";
    f << "delta " << format_csv_double(dx) << " 0 0\n";
    f << "delta 0 " << format_csv_double(dy) << " 0\n";
    f << "delta 0 0 " << format_csv_double(dz) << "\n";
    f << "object 2 class gridconnections counts " << nx << " " << ny << " " << nz << "\n";
    f << "object 3 class array type double rank 0 items " << (nx * ny * nz) << " data follows\n";

    for (double v : grid.values) f << format_csv_double(v) << "\n";

    f << "object \"density\" class field\n";
    f << "component \"positions\" value 1\n";
    f << "component \"connections\" value 2\n";
    f << "component \"data\" value 3\n";
}

} // namespace mcswell::analysis
