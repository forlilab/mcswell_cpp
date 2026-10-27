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
#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>

#include "consts.cuh"
#include "geometry.cuh"     // normalize + rodrigues_rotate_point
#include "water_model.cuh"  // create_water_std(...)
#include "cuda_rng.cuh"


__device__ __forceinline__
void load_water_from_array(const float* __restrict__ src, size_t base, float w[WATER_SIZE]) {
#pragma unroll
    for (int i=0;i<WATER_SIZE;i++) w[i] = src[base + i];
}

__device__ __forceinline__
void move_water_in_array(float* buf, size_t src, size_t dst) {
#pragma unroll
  for (int i = 0; i < 12; ++i) buf[dst + i] = buf[src + i];
}

__device__ __forceinline__
void copy_water_to_array(const float* src12, float* buf, size_t dst) {
#pragma unroll
  for (int i = 0; i < 12; ++i) buf[dst + i] = src12[i];
}

__device__ __forceinline__
void clear_water_in_array(float* buf, size_t base) {
#pragma unroll
  for (int i = 0; i < 12; ++i) buf[base + i] = 0.0f;
}

// Give the standard water a uniformly random orientation about its oxygen.
//
// The insertion acceptance rule, exp(B) / (N+1) * exp(-beta dU), takes the
// inserted molecule's orientation to be drawn uniformly over all rotations, so
// the orientation proposal has to be exactly that. u1..u3 are independent
// uniforms on (0,1]; Shoemake's construction maps them to a unit quaternion
// uniform over SO(3). (A random axis with a uniform rotation angle is *not*
// uniform: it over-weights small rotations of the template orientation.)
__device__ __forceinline__
void randomize_water_uniform(float u1, float u2, float u3, float w[WATER_SIZE]) {
    const float two_pi = 2.0f * CUDART_PI_F;
    const float a = sqrtf(1.0f - u1), b = sqrtf(u1);
    float s2, c2, s3, c3;
    sincosf(two_pi * u2, &s2, &c2);
    sincosf(two_pi * u3, &s3, &c3);
    const float3u q = make_float3u(a * s2, a * c2, b * s3);  // vector part
    const float qw = b * c3;                                  // scalar part

    const float3u pivot = make_float3u(w[OX], w[OY], w[OZ]);
    const auto rotate = [&](float3u p) {
        // v' = v + 2 qw (q x v) + 2 q x (q x v)
        const float3u v = sub(p, pivot);
        const float3u t = mul(cross(q, v), 2.0f);
        return add(pivot, add(add(v, mul(t, qw)), cross(q, t)));
    };
    const float3u h1 = rotate(make_float3u(w[H1X], w[H1Y], w[H1Z]));
    const float3u h2 = rotate(make_float3u(w[H2X], w[H2Y], w[H2Z]));

    w[H1X]=h1.x; w[H1Y]=h1.y; w[H1Z]=h1.z;
    w[H2X]=h2.x; w[H2Y]=h2.y; w[H2Z]=h2.z;
}

// Insertion: move the whole water rigidly so its oxygen lands on (x,y,z);
// resnum stored in each atom slot. The translation is (x,y,z) minus the
// current oxygen position, not (x,y,z) itself: the TIP4P template's oxygen is
// not at the origin, and shifting only the hydrogens by (x,y,z) would distort
// the molecule (O-H 0.89-1.02 A instead of 0.9572, varying with orientation).
__device__ __forceinline__
void propose_insertion_xyz(float x, float y, float z, float resnum_f, float w[WATER_SIZE]) {
    const float tx = x - w[OX], ty = y - w[OY], tz = z - w[OZ];
    w[OX]=x; w[OY]=y; w[OZ]=z; w[ORES]=resnum_f;

    w[H1X]+=tx; w[H1Y]+=ty; w[H1Z]+=tz; w[H1RES]=resnum_f;
    w[H2X]+=tx; w[H2Y]+=ty; w[H2Z]+=tz; w[H2RES]=resnum_f;
}

// Is (x,y,z) inside the sampled region?
//
// The region is the union of voxels around the kept lattice points, and its
// volume (n_points * spacing^3) is what enters the Adams parameter -- so voxel
// membership is the criterion that keeps the volume bookkeeping consistent.
// The lattice is regular, so the containing voxel is an index computation and
// the lookup is O(1); searching the point list would be O(n_points).
__device__ __forceinline__
bool inside_cavity(
    float x, float y, float z,
    const unsigned char* __restrict__ mask,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float spacing
){
    if (mask == nullptr) return true;          // mask absent: unconstrained
    // rintf rather than __float2int_rn: this header is pulled into host
    // translation units via snapshots_io.hpp -> water_model.cuh, where the
    // CUDA intrinsics are not declared.
    const int i = (int)rintf((x - ox) / spacing);
    const int j = (int)rintf((y - oy) / spacing);
    const int k = (int)rintf((z - oz) / spacing);
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
    return mask[((long long)i * ny + j) * nz + k] != 0u;
}

// Displacement proposal:
// - translate O by (dx,dy,dz)
// - rotate Hs around new O pivot by Rodrigues axis+angle
// - reject outright if the new O leaves the sampled region
//
// That last step is what makes insertion and deletion reciprocal. Insertion
// can only create a water inside the region, so without the constraint a water
// could be displaced somewhere it could never have been inserted and then be
// deleted from there -- a move whose reverse has zero proposal probability.
// The proposal stays symmetric within the region, so rejecting excursions out
// of it preserves detailed balance.
__device__ __forceinline__
bool propose_perturbation_cloud(
    const float oldw[WATER_SIZE],
    uint32_t boundaries_max,
    curandStatePhilox4_32_10_t* st,
    const unsigned char* __restrict__ cavity_mask,
    int mask_nx, int mask_ny, int mask_nz,
    float mask_ox, float mask_oy, float mask_oz,
    float mask_spacing,
    float dx, float dy, float dz,
    float ax, float ay, float az, float angle_rad,
    float neww[WATER_SIZE]
){
    // This draw is not used. It is kept deliberately: the original
    // implementation drew a random cloud point here (to test a constraint it
    // then discarded), and removing the draw would shift the Philox stream for
    // every subsequent step, so runs would no longer reproduce published
    // trajectories at the same seed. Keeping it costs one RNG call per
    // displacement proposal and keeps seeds comparable across versions.
    (void)rand_int(st, (int)boundaries_max);

#pragma unroll
    for (int i=0;i<WATER_SIZE;i++) neww[i]=oldw[i];

    const float new_ox = oldw[OX] + dx;
    const float new_oy = oldw[OY] + dy;
    const float new_oz = oldw[OZ] + dz;

    if (!inside_cavity(new_ox, new_oy, new_oz, cavity_mask,
                       mask_nx, mask_ny, mask_nz,
                       mask_ox, mask_oy, mask_oz, mask_spacing)) {
        return false;
    }

    float3u h1 = make_float3u(neww[H1X] + dx, neww[H1Y] + dy, neww[H1Z] + dz);
    float3u h2 = make_float3u(neww[H2X] + dx, neww[H2Y] + dy, neww[H2Z] + dz);

    const float3u pivot = make_float3u(new_ox, new_oy, new_oz);
    const float3u axis = normalize(make_float3u(ax, ay, az));

    const float3u h1r = rodrigues_rotate_point(h1, axis, angle_rad, pivot);
    const float3u h2r = rodrigues_rotate_point(h2, axis, angle_rad, pivot);

    neww[OX]=new_ox; neww[OY]=new_oy; neww[OZ]=new_oz;
    neww[H1X]=h1r.x; neww[H1Y]=h1r.y; neww[H1Z]=h1r.z;
    neww[H2X]=h2r.x; neww[H2Y]=h2r.y; neww[H2Z]=h2r.z;

    return true;
}
