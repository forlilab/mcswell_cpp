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

#include "analysis/density_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcswell::analysis {

namespace {

std::vector<double> linspace(double start, double stop, std::size_t num) {
    std::vector<double> out(num);
    if (num == 0) return out;
    if (num == 1) {
        out[0] = start;
        return out;
    }
    const double step = (stop - start) / static_cast<double>(num - 1);
    for (std::size_t i = 0; i < num; ++i) out[i] = start + static_cast<double>(i) * step;
    out[num - 1] = stop; // exact endpoint, matches numpy.linspace
    return out;
}

// Bin index in [0, edges.size()-2], or -1 if v lies outside [edges.front(),
// edges.back()]. Bins are half-open except the last, which is closed on
// both ends -- matches numpy.histogramdd.
long long bin_index(const std::vector<double>& edges, double v) {
    const std::size_t n_bins = edges.size() - 1;
    if (v < edges.front() || v > edges.back()) return -1;
    if (v == edges.back()) return static_cast<long long>(n_bins - 1);

    auto it = std::upper_bound(edges.begin(), edges.end(), v);
    std::size_t idx = static_cast<std::size_t>(it - edges.begin()) - 1;
    if (idx >= n_bins) idx = n_bins - 1;
    return static_cast<long long>(idx);
}

// scipy.ndimage's default mode='reflect' ("half-sample symmetric"): for
// "abcd" the extension is "dcba|abcd|dcba" -- the edge sample is not
// duplicated, unlike mode='nearest' or mode='mirror'.
long long reflect_index(long long i, long long n) {
    if (n == 1) return 0;
    const long long period = 2 * n;
    long long m = i % period;
    if (m < 0) m += period;
    return (m < n) ? m : (period - 1 - m);
}

std::vector<double> gaussian_kernel1d(double sigma) {
    const int radius = static_cast<int>(4.0 * sigma + 0.5); // scipy: int(truncate*sigma + 0.5), truncate=4.0
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    const double sigma2 = sigma * sigma;
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const double w = std::exp(-0.5 * static_cast<double>(k * k) / sigma2);
        kernel[static_cast<std::size_t>(k + radius)] = w;
        sum += w;
    }
    for (double& w : kernel) w /= sum;
    return kernel;
}

// numpy.percentile default ('linear' interpolation) on already-collected
// values. Copies and sorts `values`.
double percentile_linear(std::vector<double> values, double percentile) {
    std::sort(values.begin(), values.end());
    const double rank = (percentile / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(rank));
    const auto hi = static_cast<std::size_t>(std::ceil(rank));
    if (hi >= values.size()) return values.back();
    const double frac = rank - static_cast<double>(lo);
    return values[lo] + frac * (values[hi] - values[lo]);
}

} // namespace

std::array<std::vector<double>, 3> make_edges(
    const std::array<double, 3>& center, const std::array<double, 3>& halfsize, double spacing) {
    std::array<std::vector<double>, 3> edges;
    for (int d = 0; d < 3; ++d) {
        const double lo = center[d] - halfsize[d];
        const double hi = center[d] + halfsize[d];
        const auto n_points =
            static_cast<std::size_t>(std::ceil(2.0 * halfsize[d] / spacing)) + 1;
        edges[static_cast<std::size_t>(d)] = linspace(lo, hi, n_points);
    }
    return edges;
}

DensityGrid compute_density_grid(
    const std::vector<std::array<double, 3>>& points, const std::array<std::vector<double>, 3>& edges) {
    DensityGrid grid;
    grid.edges = edges;
    for (int d = 0; d < 3; ++d) grid.shape[static_cast<std::size_t>(d)] = edges[static_cast<std::size_t>(d)].size() - 1;
    grid.values.assign(grid.shape[0] * grid.shape[1] * grid.shape[2], 0.0);

    for (const auto& p : points) {
        const long long i = bin_index(edges[0], p[0]);
        if (i < 0) continue;
        const long long j = bin_index(edges[1], p[1]);
        if (j < 0) continue;
        const long long k = bin_index(edges[2], p[2]);
        if (k < 0) continue;
        grid.values[grid.index(
            static_cast<std::size_t>(i), static_cast<std::size_t>(j), static_cast<std::size_t>(k))] += 1.0;
    }
    return grid;
}

DensityGrid gaussian_smooth(const DensityGrid& grid, double sigma) {
    const auto kernel = gaussian_kernel1d(sigma);
    const auto radius = static_cast<long long>(kernel.size() / 2);

    const auto nx = static_cast<long long>(grid.shape[0]);
    const auto ny = static_cast<long long>(grid.shape[1]);
    const auto nz = static_cast<long long>(grid.shape[2]);

    std::vector<double> buf_a = grid.values;
    std::vector<double> buf_b(buf_a.size());

    auto convolve_axis = [&](const std::vector<double>& src, std::vector<double>& dst, int axis) {
        for (long long i = 0; i < nx; ++i) {
            for (long long j = 0; j < ny; ++j) {
                for (long long k = 0; k < nz; ++k) {
                    double acc = 0.0;
                    for (long long t = -radius; t <= radius; ++t) {
                        long long ii = i, jj = j, kk = k;
                        if (axis == 0) ii = reflect_index(i + t, nx);
                        else if (axis == 1) jj = reflect_index(j + t, ny);
                        else kk = reflect_index(k + t, nz);
                        acc += kernel[static_cast<std::size_t>(t + radius)] *
                               src[static_cast<std::size_t>(ii * ny * nz + jj * nz + kk)];
                    }
                    dst[static_cast<std::size_t>(i * ny * nz + j * nz + k)] = acc;
                }
            }
        }
    };

    convolve_axis(buf_a, buf_b, 0);
    convolve_axis(buf_b, buf_a, 1);
    convolve_axis(buf_a, buf_b, 2);

    DensityGrid out;
    out.edges = grid.edges;
    out.shape = grid.shape;
    out.values = std::move(buf_b);
    return out;
}

std::vector<Peak> find_hydration_sites(const DensityGrid& smoothed, double percentile, double min_separation) {
    if (!(min_separation > 0.0)) throw std::invalid_argument("min_separation must be > 0");

    std::vector<double> nonzero;
    nonzero.reserve(smoothed.values.size());
    for (double v : smoothed.values) {
        if (v > 0.0) nonzero.push_back(v);
    }
    if (nonzero.empty()) {
        throw std::runtime_error("No non-zero density in smoothed grid.");
    }
    const double threshold = percentile_linear(std::move(nonzero), percentile);

    // Candidates, densest first; equal densities in voxel-index order.
    std::vector<std::size_t> order;
    for (std::size_t idx = 0; idx < smoothed.values.size(); ++idx) {
        if (smoothed.values[idx] > 0.0 && smoothed.values[idx] >= threshold) order.push_back(idx);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return smoothed.values[a] > smoothed.values[b];
    });

    // Voxel centres per axis, and how many voxels the exclusion sphere can
    // span along each axis (bins may be slightly narrower than the nominal
    // spacing, see make_edges, so use the narrowest one).
    std::array<std::vector<double>, 3> centers;
    std::array<long long, 3> reach{};
    std::array<long long, 3> n{};
    for (std::size_t d = 0; d < 3; ++d) {
        const auto& e = smoothed.edges[d];
        n[d] = static_cast<long long>(smoothed.shape[d]);
        centers[d].resize(smoothed.shape[d]);
        double min_width = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < smoothed.shape[d]; ++i) {
            centers[d][i] = 0.5 * (e[i] + e[i + 1]);
            min_width = std::min(min_width, e[i + 1] - e[i]);
        }
        reach[d] = static_cast<long long>(std::ceil(min_separation / min_width));
    }

    const double sep2 = min_separation * min_separation;
    const auto nyz = n[1] * n[2];
    std::vector<char> excluded(smoothed.values.size(), 0);
    std::vector<Peak> sites;
    for (std::size_t idx : order) {
        if (excluded[idx]) continue;
        const long long i = static_cast<long long>(idx) / nyz;
        const long long j = (static_cast<long long>(idx) / n[2]) % n[1];
        const long long k = static_cast<long long>(idx) % n[2];

        Peak p;
        p.center = {centers[0][static_cast<std::size_t>(i)], centers[1][static_cast<std::size_t>(j)],
                    centers[2][static_cast<std::size_t>(k)]};
        p.score = smoothed.values[idx];
        sites.push_back(p);

        for (long long ii = std::max(0LL, i - reach[0]); ii <= std::min(n[0] - 1, i + reach[0]); ++ii) {
            const double dx = centers[0][static_cast<std::size_t>(ii)] - p.center[0];
            for (long long jj = std::max(0LL, j - reach[1]); jj <= std::min(n[1] - 1, j + reach[1]); ++jj) {
                const double dy = centers[1][static_cast<std::size_t>(jj)] - p.center[1];
                for (long long kk = std::max(0LL, k - reach[2]); kk <= std::min(n[2] - 1, k + reach[2]); ++kk) {
                    const double dz = centers[2][static_cast<std::size_t>(kk)] - p.center[2];
                    if (dx * dx + dy * dy + dz * dz <= sep2) {
                        excluded[static_cast<std::size_t>(ii * nyz + jj * n[2] + kk)] = 1;
                    }
                }
            }
        }
    }
    return sites;
}

} // namespace mcswell::analysis
