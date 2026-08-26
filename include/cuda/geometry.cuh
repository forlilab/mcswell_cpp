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

struct float3u { float x,y,z; };

__host__ __device__ __forceinline__ float3u make_float3u(float x,float y,float z){ return {x,y,z}; }

__host__ __device__ __forceinline__ float3u add(const float3u&a,const float3u&b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__host__ __device__ __forceinline__ float3u sub(const float3u&a,const float3u&b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__host__ __device__ __forceinline__ float3u mul(const float3u&a,float s){ return {a.x*s,a.y*s,a.z*s}; }

__host__ __device__ __forceinline__ float dot(const float3u&a,const float3u&b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ __forceinline__ float3u cross(const float3u&a,const float3u&b){
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

__host__ __device__ __forceinline__ float rsqrt_safe(float x){
#if defined(__CUDA_ARCH__)
    return rsqrtf(fmaxf(x, 1e-20f));
#else
    return 1.0f / sqrtf(fmaxf(x, 1e-20f));
#endif
}

__host__ __device__ __forceinline__ float3u normalize(const float3u& v){
    const float inv = rsqrt_safe(dot(v,v));
    return mul(v, inv);
}

// Rodrigues rotation: rotate point p around axis k by angle, about pivot c
__host__ __device__ __forceinline__
float3u rodrigues_rotate_point(const float3u& p, const float3u& k_unit, float angle, const float3u& pivot){
    // v = p - pivot
    const float3u v = sub(p, pivot);

#if defined(__CUDA_ARCH__)
    float s, c;
    sincosf(angle, &s, &c);
#else
    const float s = sinf(angle);
    const float c = cosf(angle);
#endif

    // v_rot = v*c + (k x v)*s + k*(k·v)*(1-c)
    const float3u kxv = cross(k_unit, v);
    const float kv = dot(k_unit, v);

    const float3u term1 = mul(v, c);
    const float3u term2 = mul(kxv, s);
    const float3u term3 = mul(k_unit, kv * (1.0f - c));

    const float3u vrot = add(add(term1, term2), term3);
    return add(pivot, vrot);
}

// Computes M-site position for TIP4P water model. 
// dOM is the distance from oxygen to M-site along the HOH bisector 
__host__ __device__ __forceinline__ 
float3u tip4p_m_site(const float ox, const float oy, const float oz, 
    const float h1_x, const float h1_y, const float h1_z,
    const float h2_x, const float h2_y, const float h2_z,
    float dOM) {
    // Unit vectors from oxygen to hydrogens
    float3u o = make_float3u(ox, oy, oz);
    float3u h1 = make_float3u(h1_x, h1_y, h1_z);
    float3u h2 = make_float3u(h2_x, h2_y, h2_z);

    float3u u1 = normalize(sub(h1, o));
    float3u u2 = normalize(sub(h2, o));

    // Bisector direction (points toward the hydrogens)
    float3u b = add(u1, u2);
    float3u bhat = normalize(b);

    // M-site is displaced from O along the bisector direction
    return add(o, mul(bhat, dOM));

}