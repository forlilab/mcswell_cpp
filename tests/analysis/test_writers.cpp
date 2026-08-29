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
#include "harness.hpp"

#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

using mcswell::analysis::density_to_dx;
using mcswell::analysis::DensityGrid;
using mcswell::analysis::format_csv_double;
using mcswell::analysis::write_csv_table;
using mcswell::analysis::write_sites_pdb;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void test_format_csv_double() {
    MCSWELL_CHECK(format_csv_double(3.0) == "3");
    MCSWELL_CHECK(format_csv_double(std::numeric_limits<double>::quiet_NaN()) == "nan");
    MCSWELL_CHECK(format_csv_double(std::numeric_limits<double>::infinity()) == "inf");
    MCSWELL_CHECK(format_csv_double(-std::numeric_limits<double>::infinity()) == "-inf");
    // Round-trips exactly back to the same double via strtod.
    const double v = 0.1 + 0.2;
    MCSWELL_CHECK(std::stod(format_csv_double(v)) == v);
}

void test_write_csv_table_roundtrip() {
    const std::string path = "/tmp/mcswell_test_table.csv";
    write_csv_table(
        path, {"a", "b", "status"},
        {
            {"1", format_csv_double(2.5), "ok"},
            {"2", format_csv_double(-1.0), "all_empty"},
        });

    const std::string content = read_file(path);
    MCSWELL_CHECK(content.find("a,b,status\n") == 0);
    MCSWELL_CHECK(content.find("1,2.5,ok\n") != std::string::npos);
    MCSWELL_CHECK(content.find("2,-1,all_empty\n") != std::string::npos);
    std::remove(path.c_str());
}

void test_write_sites_pdb_format() {
    const std::string path = "/tmp/mcswell_test_sites.pdb";
    write_sites_pdb(
        path, {{1.234, -5.6, 78.9}, {0.0, 0.0, 0.0}}, {0.5, 1.0}, {-3.21, 2.0},
        {"test remark line"});

    const std::string content = read_file(path);
    MCSWELL_CHECK(content.find("REMARK test remark line\n") == 0);
    // Fixed-width columns: serial=1, resSeq=1, x=1.234, y=-5.600, z=78.900, occ=0.50, bfac=-3.21
    MCSWELL_CHECK(
        content.find("HETATM    1  O   SIT A   1       1.234  -5.600  78.900  0.50 -3.21\n") != std::string::npos);
    MCSWELL_CHECK(content.rfind("END\n") == content.size() - 4);
}

void test_density_to_dx_header() {
    DensityGrid grid;
    grid.shape = {2, 2, 1};
    grid.edges[0] = {0.0, 1.0, 2.0};
    grid.edges[1] = {0.0, 0.5, 1.0};
    grid.edges[2] = {0.0, 1.0};
    grid.values = {1.0, 2.0, 3.0, 4.0};

    const std::string path = "/tmp/mcswell_test_density.dx";
    density_to_dx(grid, path);
    const std::string content = read_file(path);

    MCSWELL_CHECK(content.find("object 1 class gridpositions counts 2 2 1\n") != std::string::npos);
    MCSWELL_CHECK(content.find("origin 0 0 0\n") != std::string::npos);
    MCSWELL_CHECK(content.find("delta 1 0 0\n") != std::string::npos);
    MCSWELL_CHECK(content.find("delta 0 0.5 0\n") != std::string::npos);
    MCSWELL_CHECK(content.find("object 3 class array type double rank 0 items 4 data follows\n1\n2\n3\n4\n") !=
                  std::string::npos);
    std::remove(path.c_str());
}

} // namespace

void run_writers_tests() {
    test_format_csv_double();
    test_write_csv_table_roundtrip();
    test_write_sites_pdb_format();
    test_density_to_dx_header();
}
