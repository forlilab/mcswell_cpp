#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Atom;

// CUDA dummy: returns flat snapshots buffer on host
// Layout is whatever your kernel produces; keep it consistent with Python reshape.
std::vector<float> run_mcswell_gpu_titration(
    const std::vector<Atom>& receptor_points,
    const float* insertion_points_xyz,   // length = 3*n_points
    std::size_t n_points,
    float distance_cutoff,
    float spacing,
    std::size_t gcmc_steps,
    std::size_t equilibration_steps,
    const std::string& save_path,
    const std::vector<float>& mu_values,
    std::size_t n_snapshots,
    std::uint64_t seed = 12345ULL
);
