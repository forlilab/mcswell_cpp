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
#include <cstdint>
#include "water_ops.cuh"
#include "consts.cuh"

// Define ONE of these via nvcc flags:
//   -DTIP3P
//   -DTIP3PFP

namespace water_model {

    #if defined(TIP3P) && defined(TIP3PFP)
    #error "Define only one water model: TIP3P or TIP3PFP or TIP4P"
    #endif
    #if defined(TIP3P) && defined(TIP4P)
    #error "Define only one water model: TIP3P or TIP3PFP or TIP4P"
    #endif
    #if defined(TIP3PFP) && defined(TIP4P)
    #error "Define only one water model: TIP3P or TIP3PFP or TIP4P"
    #endif

    #if !defined(TIP3P) && !defined(TIP3PFP) && !defined(TIP4P)
    #error "No water model selected. Compile with -DTIP3P or -DTIP3PFP or -DTIP4P"
    #endif

    #if defined(TIP3P)
    static constexpr float Q_O      = -0.8340f;
    static constexpr float Q_H      =  0.4170f;
    static constexpr float EPS_O    =  0.15210325f;
    static constexpr float RMINH_O  =  1.7682f;
    #elif defined(TIP3PFP)
    static constexpr float Q_O      = -0.8484f;
    static constexpr float Q_H      =  0.4242f;
    static constexpr float EPS_O    =  0.15586604f;
    static constexpr float RMINH_O  =  1.7835723f;
    #elif defined(TIP4P)
    static constexpr float Q_O      =  0.f;
    static constexpr float Q_H      =  0.520f;
    static constexpr float Q_M      =  -2.0f * Q_H;
    static constexpr float EPS_O    =  0.1550f;
    static constexpr float RMINH_O  =  1.7698915f;
    static constexpr float dOM      =  0.15f;
    #endif

    // Helpers (optional but nice)
    __host__ __device__ __forceinline__
    void params_for_atom(int atom_idx, float& q, float& eps, float& rminh) {
        if (atom_idx == 0) { // Oxygen
            q = Q_O; eps = EPS_O; rminh = RMINH_O;
        } else {             // Hydrogen
            q = Q_H; eps = 0.0f; rminh = 0.0f;
        }
    }
    __host__ __device__ __forceinline__
    void create_water_std(uint32_t resnum, float w[WATER_SIZE]) {
        // Zero first
    #pragma unroll
        for (int i = 0; i < WATER_SIZE; ++i) w[i] = 0.0f;

    #if defined(TIP3P)
        w[OX]=0.000f; w[OY]=0.000f; w[OZ]=0.000f; w[ORES]=(float)resnum;
        w[H1X]=0.000f; w[H1Y]=0.756f; w[H1Z]=0.586f; w[H1RES]=(float)resnum;
        w[H2X]=0.000f; w[H2Y]=-0.761f; w[H2Z]=0.594f; w[H2RES]=(float)resnum;
    #elif defined(TIP3PFP)
        w[OX]=0.000f; w[OY]=0.000f; w[OZ]=-0.018f; w[ORES]=(float)resnum;
        w[H1X]=0.000f; w[H1Y]=0.761f; w[H1Z]=0.595f; w[H1RES]=(float)resnum;
        w[H2X]=0.000f; w[H2Y]=-0.761f; w[H2Z]=0.594f; w[H2RES]=(float)resnum;
    #elif defined(TIP4P)
        w[OX]=0.000f; w[OY]=-0.06556f; w[OZ]=-0.018f; w[ORES]=(float)resnum;
        w[H1X]=0.75695f; w[H1Y]=0.52032f; w[H1Z]=0.000f; w[H1RES]=(float)resnum;
        w[H2X]=-0.75695f; w[H2Y]=0.52032f; w[H2Z]=0.000f; w[H2RES]=(float)resnum;
    #endif
    }
}