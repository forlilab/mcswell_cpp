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

#include "analysis/kdtree.hpp"

#include <nanoflann.hpp>

#include <cmath>
#include <limits>

namespace mcswell::analysis {

namespace {

struct PointCloudAdaptor {
    const std::vector<std::array<double, 3>>& pts;

    inline std::size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(std::size_t idx, std::size_t dim) const { return pts[idx][dim]; }

    template <class BBox>
    bool kdtree_get_bbox(BBox&) const {
        return false;
    }
};

using KDTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>, PointCloudAdaptor, 3, std::size_t>;

// nanoflann's RadiusResultSet keeps points with squared-distance strictly
// less than the value passed to radiusSearch(). scipy's query_pairs /
// query_ball_point / query(distance_upper_bound=...) are all inclusive
// ("distance <= r") instead, and grid bin-center coordinates can land
// exactly on a cutoff distance (not just as floating-point noise), so the
// distinction matters here. Passing the smallest representable value above
// radius^2 makes an exact-boundary match satisfy nanoflann's strict "<"
// while not admitting any point actually farther away.
double inclusive_radius_sqr(double radius) {
    return std::nextafter(radius * radius, std::numeric_limits<double>::infinity());
}

} // namespace

struct KDTree3D::Impl {
    std::vector<std::array<double, 3>> points;
    PointCloudAdaptor adaptor{points};
    KDTreeIndex index;

    explicit Impl(std::vector<std::array<double, 3>> pts)
        : points(std::move(pts)), index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {}
};

KDTree3D::KDTree3D(std::vector<std::array<double, 3>> points)
    : impl_(std::make_unique<Impl>(std::move(points))) {}

KDTree3D::~KDTree3D() = default;
KDTree3D::KDTree3D(KDTree3D&&) noexcept = default;
KDTree3D& KDTree3D::operator=(KDTree3D&&) noexcept = default;

std::size_t KDTree3D::size() const { return impl_->points.size(); }

NeighborResult KDTree3D::nearest(const std::array<double, 3>& query, double max_distance) const {
    NeighborResult result;
    if (impl_->points.empty()) return result;

    std::size_t ret_index = 0;
    double out_dist_sqr = 0.0;
    const std::size_t n_found = impl_->index.knnSearch(query.data(), 1, &ret_index, &out_dist_sqr);
    if (n_found == 0) return result;

    const double dist = std::sqrt(out_dist_sqr);
    if (dist <= max_distance) {
        result.index = ret_index;
        result.distance = dist;
        result.found = true;
    }
    return result;
}

std::vector<NeighborResult> KDTree3D::nearest_batch(
    const std::vector<std::array<double, 3>>& queries, double max_distance) const {
    std::vector<NeighborResult> out;
    out.reserve(queries.size());
    for (const auto& q : queries) out.push_back(nearest(q, max_distance));
    return out;
}

std::size_t KDTree3D::count_within_radius(const std::array<double, 3>& query, double radius) const {
    if (impl_->points.empty()) return 0;
    std::vector<nanoflann::ResultItem<std::size_t, double>> matches;
    return impl_->index.radiusSearch(query.data(), inclusive_radius_sqr(radius), matches);
}

std::vector<std::size_t> KDTree3D::count_within_radius_batch(
    const std::vector<std::array<double, 3>>& queries, double radius) const {
    std::vector<std::size_t> out;
    out.reserve(queries.size());
    for (const auto& q : queries) out.push_back(count_within_radius(q, radius));
    return out;
}

std::vector<std::pair<std::size_t, std::size_t>> KDTree3D::radius_pairs(double radius) const {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    const auto& pts = impl_->points;
    const double radius_sqr = inclusive_radius_sqr(radius);

    std::vector<nanoflann::ResultItem<std::size_t, double>> matches;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        static_cast<void>(impl_->index.radiusSearch(pts[i].data(), radius_sqr, matches));
        for (const auto& m : matches) {
            if (m.first > i) pairs.emplace_back(i, m.first);
        }
    }
    return pairs;
}

} // namespace mcswell::analysis
