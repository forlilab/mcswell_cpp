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