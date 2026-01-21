#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>

#include "consts.cuh"
#include "geometry.cuh"     // normalize + rodrigues_rotate_point
#include "water_model.cuh"  // create_water_std(...)
#include "cuda_rng.cuh"


__device__ __forceinline__
void load_water_from_array(const float* __restrict__ src, uint32_t base, float w[WATER_SIZE]) {
#pragma unroll
    for (int i=0;i<WATER_SIZE;i++) w[i] = src[base + i];
}

__device__ __forceinline__
void move_water_in_array(float* buf, uint32_t src, uint32_t dst) {
#pragma unroll
  for (int i = 0; i < 12; ++i) buf[dst + i] = buf[src + i];
}

__device__ __forceinline__
void copy_water_to_array(const float* src12, float* buf, uint32_t dst) {
#pragma unroll
  for (int i = 0; i < 12; ++i) buf[dst + i] = src12[i];
}

__device__ __forceinline__
void clear_water_in_array(float* buf, uint32_t base) {
#pragma unroll
  for (int i = 0; i < 12; ++i) buf[base + i] = 0.0f;
}

// Randomize orientation of the standard water around its oxygen pivot using axis+angle
__device__ __forceinline__
void randomize_water_axis_angle(float ax, float ay, float az, float angle_rad, float w[WATER_SIZE]) {
    const float3u axis = normalize(make_float3u(ax, ay, az));
    const float3u pivot = make_float3u(w[OX], w[OY], w[OZ]);

    const float3u h1 = make_float3u(w[H1X], w[H1Y], w[H1Z]);
    const float3u h2 = make_float3u(w[H2X], w[H2Y], w[H2Z]);

    const float3u h1r = rodrigues_rotate_point(h1, axis, angle_rad, pivot);
    const float3u h2r = rodrigues_rotate_point(h2, axis, angle_rad, pivot);

    w[H1X]=h1r.x; w[H1Y]=h1r.y; w[H1Z]=h1r.z;
    w[H2X]=h2r.x; w[H2Y]=h2r.y; w[H2Z]=h2r.z;
}

// Insertion: set oxygen to (x,y,z) and translate Hs by same vector, resnum stored in each atom slot
__device__ __forceinline__
void propose_insertion_xyz(float x, float y, float z, float resnum_f, float w[WATER_SIZE]) {
    w[OX]=x; w[OY]=y; w[OZ]=z; w[ORES]=resnum_f;

    w[H1X]+=x; w[H1Y]+=y; w[H1Z]+=z; w[H1RES]=resnum_f;
    w[H2X]+=x; w[H2Y]+=y; w[H2Z]+=z; w[H2RES]=resnum_f;
}

// Displacement proposal:
// - translate O by (dx,dy,dz)
// - rotate Hs around new O pivot by Rodrigues axis+angle
// - additionally enforce that new O is close to allowed insertion cloud point (nearest within radius)
//   (this is how you can keep the “cloud” constraint for displacement too)
__device__ __forceinline__
bool propose_perturbation_cloud(
    const float oldw[WATER_SIZE],
    const float* __restrict__ boundary_xyz, // [boundaries_max*3]
    uint32_t boundaries_max,
    float max_dist_to_cloud,                // e.g. 0.5 Å
    curandStatePhilox4_32_10_t* st,
    float dx, float dy, float dz,
    float ax, float ay, float az, float angle_rad,
    float neww[WATER_SIZE]
){
#pragma unroll
    for (int i=0;i<WATER_SIZE;i++) neww[i]=oldw[i];

    const float new_ox = oldw[OX] + dx;
    const float new_oy = oldw[OY] + dy;
    const float new_oz = oldw[OZ] + dz;

    // Enforce “cloud” membership approximately:
    // pick a random cloud point and require O is within radius.
    // (If you instead want true nearest-neighbor, that needs a KD-tree / grid accel.)
    const uint32_t idx = (uint32_t)rand_int(st, (int)boundaries_max);
    const uint32_t bi = idx * 3;
    const float bx = boundary_xyz[bi+0];
    const float by = boundary_xyz[bi+1];
    const float bz = boundary_xyz[bi+2];

    const float ddx = new_ox - bx;
    const float ddy = new_oy - by;
    const float ddz = new_oz - bz;
    const float d2 = ddx*ddx + ddy*ddy + ddz*ddz;

    // Translate H positions first (as in Rust: h = old_h + delta)
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
