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

#include "analysis/frame_extraction.hpp"

#include "cuda/consts.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcswell::analysis {

std::vector<MuWindowFrames> extract_oxygen_frames(
    const std::vector<float>& snapshots_flat,
    std::size_t n_mu,
    std::size_t n_snapshots,
    std::size_t max_n_waters
) {
    const auto water_size = static_cast<std::size_t>(WATER_SIZE);
    std::vector<MuWindowFrames> frames_by_mu(n_mu);

    for (std::size_t mu_idx = 0; mu_idx < n_mu; ++mu_idx) {
        auto& mu_frames = frames_by_mu[mu_idx].frames;
        mu_frames.resize(n_snapshots);

        for (std::size_t snap_idx = 0; snap_idx < n_snapshots; ++snap_idx) {
            const std::size_t snapshot_base =
                (mu_idx * n_snapshots * max_n_waters * water_size) +
                (snap_idx * max_n_waters * water_size);

            auto& oxygens = mu_frames[snap_idx].oxygens;
            oxygens.reserve(max_n_waters);

            for (std::size_t water_idx = 0; water_idx < max_n_waters; ++water_idx) {
                const std::size_t wat_index = snapshot_base + water_idx * water_size;
                if (wat_index + water_size - 1 >= snapshots_flat.size()) break;

                const double o_x = static_cast<double>(snapshots_flat[wat_index + OX]);
                const double o_y = static_cast<double>(snapshots_flat[wat_index + OY]);
                const double o_z = static_cast<double>(snapshots_flat[wat_index + OZ]);

                // Inactive slot: the GPU kernel zero-initializes the water
                // buffer and never writes unused slots, exactly the
                // convention the old PDB writer skipped on.
                if (o_x == 0.0 && o_y == 0.0 && o_z == 0.0) continue;

                oxygens.push_back({o_x, o_y, o_z});
            }
        }
    }

    return frames_by_mu;
}

CapacityScanResult scan_capacity_and_collect_density_points(
    const std::vector<MuWindowFrames>& frames_by_mu,
    const std::vector<double>& mu_values,
    std::size_t water_capacity,
    bool capacity_filter,
    double max_capacity_hit_fraction,
    double max_mean_capacity_fraction
) {
    if (water_capacity == 0) {
        throw std::invalid_argument("water_capacity must be positive");
    }
    if (frames_by_mu.size() != mu_values.size()) {
        throw std::invalid_argument("frames_by_mu and mu_values must have the same length");
    }

    CapacityScanResult result;
    result.diagnostics.reserve(frames_by_mu.size());
    result.valid_mask.assign(frames_by_mu.size(), false);

    bool prefix_open = true;

    for (std::size_t j = 0; j < frames_by_mu.size(); ++j) {
        const auto& frames = frames_by_mu[j].frames;

        CapacityDiagnostics diag;
        diag.mu_index = j;
        diag.mu_kcal_mol = mu_values[j];
        diag.n_frames = frames.size();
        diag.water_capacity = water_capacity;

        bool triggers_capacity = false;

        if (frames.empty()) {
            diag.n_region_mean = std::numeric_limits<double>::quiet_NaN();
            diag.n_region_std = std::numeric_limits<double>::quiet_NaN();
            diag.n_region_max_observed = 0;
            diag.capacity_mean_fraction = std::numeric_limits<double>::quiet_NaN();
            diag.capacity_hit_fraction = std::numeric_limits<double>::quiet_NaN();
        } else {
            double sum = 0.0;
            std::size_t max_n = 0;
            std::size_t hits = 0;
            for (const auto& f : frames) {
                const std::size_t c = f.oxygens.size();
                sum += static_cast<double>(c);
                max_n = std::max(max_n, c);
                if (c >= water_capacity) ++hits;
            }
            const double mean_n = sum / static_cast<double>(frames.size());

            double sq_sum = 0.0;
            for (const auto& f : frames) {
                const double d = static_cast<double>(f.oxygens.size()) - mean_n;
                sq_sum += d * d;
            }
            const double std_n = frames.size() >= 2
                ? std::sqrt(sq_sum / static_cast<double>(frames.size() - 1))
                : 0.0;

            diag.n_region_mean = mean_n;
            diag.n_region_std = std_n;
            diag.n_region_max_observed = max_n;
            diag.capacity_mean_fraction = mean_n / static_cast<double>(water_capacity);
            diag.capacity_hit_fraction = static_cast<double>(hits) / static_cast<double>(frames.size());

            triggers_capacity = capacity_filter &&
                (diag.capacity_hit_fraction > max_capacity_hit_fraction ||
                 diag.capacity_mean_fraction >= max_mean_capacity_fraction);
        }

        bool valid;
        std::string reason;
        if (!capacity_filter) {
            valid = true;
            reason = "capacity_filter_disabled";
        } else if (prefix_open && triggers_capacity) {
            prefix_open = false;
            valid = false;
            reason = "first_capacity_limited_window";
        } else if (prefix_open) {
            valid = true;
            reason = "capacity_safe";
        } else {
            valid = false;
            reason = "after_first_capacity_limited_window";
        }

        diag.valid_for_analysis = valid;
        diag.capacity_filter_reason = reason;
        result.valid_mask[j] = valid;

        if (valid) {
            for (const auto& f : frames) {
                result.density_points.insert(
                    result.density_points.end(), f.oxygens.begin(), f.oxygens.end());
            }
        }

        result.diagnostics.push_back(std::move(diag));
    }

    return result;
}

} // namespace mcswell::analysis
