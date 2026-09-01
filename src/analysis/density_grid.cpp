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

#include "analysis/kdtree.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

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

// scipy.ndimage.maximum_filter's mode='nearest': out-of-range neighbors
// clamp to the nearest valid index (edge value repeats).
long long clamp_index(long long i, long long n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

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

std::vector<Peak> find_peaks(const DensityGrid& smoothed, double percentile, int neighborhood) {
    std::vector<double> nonzero;
    nonzero.reserve(smoothed.values.size());
    for (double v : smoothed.values) {
        if (v > 0.0) nonzero.push_back(v);
    }
    if (nonzero.empty()) {
        throw std::runtime_error("No non-zero density in smoothed grid.");
    }

    const double threshold = percentile_linear(nonzero, percentile);

    std::vector<double> masked(smoothed.values.size());
    for (std::size_t idx = 0; idx < smoothed.values.size(); ++idx) {
        masked[idx] = (smoothed.values[idx] >= threshold) ? smoothed.values[idx] : 0.0;
    }

    const auto nx = static_cast<long long>(smoothed.shape[0]);
    const auto ny = static_cast<long long>(smoothed.shape[1]);
    const auto nz = static_cast<long long>(smoothed.shape[2]);
    const auto nb = static_cast<long long>(neighborhood);

    std::vector<Peak> peaks;
    for (long long i = 0; i < nx; ++i) {
        for (long long j = 0; j < ny; ++j) {
            for (long long k = 0; k < nz; ++k) {
                const double center_val =
                    masked[smoothed.index(static_cast<std::size_t>(i), static_cast<std::size_t>(j), static_cast<std::size_t>(k))];
                if (center_val <= 0.0) continue;

                double max_val = -std::numeric_limits<double>::infinity();
                for (long long di = -nb; di <= nb; ++di) {
                    for (long long dj = -nb; dj <= nb; ++dj) {
                        for (long long dk = -nb; dk <= nb; ++dk) {
                            const long long ii = clamp_index(i + di, nx);
                            const long long jj = clamp_index(j + dj, ny);
                            const long long kk = clamp_index(k + dk, nz);
                            max_val = std::max(
                                max_val,
                                masked[smoothed.index(
                                    static_cast<std::size_t>(ii), static_cast<std::size_t>(jj),
                                    static_cast<std::size_t>(kk))]);
                        }
                    }
                }

                if (center_val == max_val) {
                    Peak p;
                    p.center = {
                        0.5 * (smoothed.edges[0][static_cast<std::size_t>(i)] +
                               smoothed.edges[0][static_cast<std::size_t>(i) + 1]),
                        0.5 * (smoothed.edges[1][static_cast<std::size_t>(j)] +
                               smoothed.edges[1][static_cast<std::size_t>(j) + 1]),
                        0.5 * (smoothed.edges[2][static_cast<std::size_t>(k)] +
                               smoothed.edges[2][static_cast<std::size_t>(k) + 1]),
                    };
                    // Score is the *unmasked* smoothed density, matching the
                    // Python implementation (H_smooth, not H_masked).
                    p.score = smoothed.values[smoothed.index(
                        static_cast<std::size_t>(i), static_cast<std::size_t>(j), static_cast<std::size_t>(k))];
                    peaks.push_back(p);
                }
            }
        }
    }
    return peaks;
}

std::vector<Peak> merge_close_sites(const std::vector<Peak>& peaks, double cutoff) {
    if (peaks.empty()) return {};

    std::vector<std::array<double, 3>> centers;
    centers.reserve(peaks.size());
    for (const auto& p : peaks) centers.push_back(p.center);

    KDTree3D tree(centers);
    const auto pairs = tree.radius_pairs(cutoff);

    std::vector<std::size_t> parent(peaks.size());
    std::iota(parent.begin(), parent.end(), 0);

    std::function<std::size_t(std::size_t)> find = [&](std::size_t i) -> std::size_t {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };

    for (const auto& pr : pairs) {
        const std::size_t ri = find(pr.first);
        const std::size_t rj = find(pr.second);
        if (ri != rj) parent[rj] = ri;
    }

    std::unordered_map<std::size_t, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < peaks.size(); ++i) groups[find(i)].push_back(i);

    std::vector<Peak> merged;
    merged.reserve(groups.size());
    for (const auto& [root, idxs] : groups) {
        (void)root;
        double total_weight = 0.0;
        std::array<double, 3> weighted_center{0.0, 0.0, 0.0};
        for (std::size_t idx : idxs) {
            total_weight += peaks[idx].score;
            for (int d = 0; d < 3; ++d) {
                weighted_center[static_cast<std::size_t>(d)] +=
                    peaks[idx].center[static_cast<std::size_t>(d)] * peaks[idx].score;
            }
        }

        Peak m;
        if (total_weight > 0.0) {
            for (int d = 0; d < 3; ++d) {
                m.center[static_cast<std::size_t>(d)] = weighted_center[static_cast<std::size_t>(d)] / total_weight;
            }
        } else {
            // Degenerate (all-zero-score cluster): fall back to an
            // unweighted mean rather than dividing by zero.
            std::array<double, 3> mean_center{0.0, 0.0, 0.0};
            for (std::size_t idx : idxs) {
                for (int d = 0; d < 3; ++d) {
                    mean_center[static_cast<std::size_t>(d)] += peaks[idx].center[static_cast<std::size_t>(d)];
                }
            }
            for (int d = 0; d < 3; ++d) {
                m.center[static_cast<std::size_t>(d)] =
                    mean_center[static_cast<std::size_t>(d)] / static_cast<double>(idxs.size());
            }
        }
        m.score = total_weight;
        merged.push_back(m);
    }
    return merged;
}

} // namespace mcswell::analysis
