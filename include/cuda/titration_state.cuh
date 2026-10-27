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

// The titration kernel is resumable: a block can stop at the top of any step,
// before that step draws a random number, and a later launch picks the chain
// up exactly where it left off. The host uses this to grow the water buffer
// and to drain the snapshot buffer without either one having to be sized for
// the whole run up front. Plain device pointers only, so this header is safe
// to include from host translation units.

// Why a block stopped. Read by the host after every launch.
enum TitrationExit : std::uint32_t {
    TITRATION_DONE = 0,                  // every step has run
    TITRATION_WATER_BUFFER_FULL = 1,     // N reached the water capacity; grow and resume
    TITRATION_SNAPSHOT_BUFFER_FULL = 2,  // next snapshot may not fit; drain and resume
};

// Per-window diagnostic accumulators that have to survive a relaunch, in
// this order: proposed[3], accepted[3], capacity_blocked, n_waters_sum.
static constexpr int DIAG_CARRY_SLOTS = 8;

// Per-mu chain state carried from one launch to the next. The RNG state and
// the water count live in their own arrays (rng_states, d_wat_num), as before.
struct TitrationCarry {
    std::uint32_t* step;          // [n_mu] next step to run
    std::uint32_t* snap_id;       // [n_mu] snapshots taken so far
    std::uint32_t* exit_reason;   // [n_mu] TitrationExit of the last launch
    unsigned long long* diag;     // [n_mu * DIAG_CARRY_SLOTS]; null when diagnostics are off
};

// Snapshot staging buffer. Each mu owns floats_per_mu floats and appends
// whole frames to it (WATER_SIZE floats per water, no padding); the host
// drains it between launches. counts is indexed by the run-wide snapshot id,
// so it covers every snapshot of the run and is only read at the end.
struct SnapshotArena {
    float* data;                       // [n_mu * floats_per_mu]
    unsigned long long floats_per_mu;
    unsigned long long* used;          // [n_mu] floats written since the last drain
    std::uint32_t* counts;             // [n_mu * n_snapshots] waters in each snapshot
};
