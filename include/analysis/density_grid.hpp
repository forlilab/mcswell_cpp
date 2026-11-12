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
#include <vector>

namespace mcswell::analysis {

// Bin edges along one axis, ascending. Built the same way as
// numpy.linspace(center-halfsize, center+halfsize, ceil(2*halfsize/spacing)+1),
// which is what the old Python analysis used -- kept exactly (not simplified
// to "spacing"-wide bins) so grid geometry matches historical runs bit for
// bit whenever 2*halfsize isn't an exact multiple of spacing.
std::array<std::vector<double>, 3> make_edges(
    const std::array<double, 3>& center, const std::array<double, 3>& halfsize, double spacing);

// A 3D histogram/density grid, stored row-major (C order): index(i,j,k) =
// i*(ny*nz) + j*nz + k, matching numpy's default flatten order (used later
// by the OpenDX writer).
struct DensityGrid {
    std::vector<double> values;
    std::array<std::size_t, 3> shape{};
    std::array<std::vector<double>, 3> edges;

    std::size_t index(std::size_t i, std::size_t j, std::size_t k) const {
        return i * shape[1] * shape[2] + j * shape[2] + k;
    }
};

// 3D histogram of `points` into `edges`. Mirrors numpy.histogramdd: bins are
// half-open [edges[i], edges[i+1]) except the last bin along each axis,
// which is closed on both ends; a point outside the edge range on any axis
// is dropped.
DensityGrid compute_density_grid(
    const std::vector<std::array<double, 3>>& points, const std::array<std::vector<double>, 3>& edges);

// Separable 3D Gaussian smoothing, matching
// scipy.ndimage.gaussian_filter(sigma=sigma) defaults: truncate=4.0 (kernel
// half-width = int(4*sigma + 0.5)) and mode='reflect' (half-sample
// symmetric boundary, e.g. for "abcd": "dcba|abcd|dcba").
DensityGrid gaussian_smooth(const DensityGrid& grid, double sigma);

struct Peak {
    std::array<double, 3> center{};
    double score = 0.0;
};

// Hydration-site finder: the greedy procedure of WaterKit's HydrationSites,
// which also built the MD/GIST reference sites MCSwell is benchmarked on.
// Voxels whose smoothed density is at or above the given percentile of the
// non-zero values (numpy.percentile, linear interpolation) are candidates.
// The densest remaining candidate becomes a site, every candidate within
// `min_separation` (inclusive) of it is discarded, and this repeats until
// none is left. Sites are therefore voxel centres more than `min_separation`
// apart, so they can be placed as explicit waters without clashes. A site's
// score is the smoothed density at its voxel. Equal densities go to the
// lower voxel index, so the result is deterministic. Returned in decreasing
// score.
std::vector<Peak> find_hydration_sites(const DensityGrid& smoothed, double percentile, double min_separation);

} // namespace mcswell::analysis
