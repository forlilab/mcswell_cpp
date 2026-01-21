#pragma once
#include <cstdint>
#include <cstddef>

std::size_t dummy_rng_state_bytes();

void init_rng_launch(void* d_states, std::uint64_t seed, int n_mu);

// --------------------------------------------
// NEW: “Rust simulate_titration” launch wrapper
// --------------------------------------------
void run_gcmc_titration_launch(
    const float* d_boundaries_xyz,   // (3*n_points)
    int n_points,

    const float* d_receptor,         // (ATOM_FEATURES*n_receptor_atoms)
    int n_receptor_atoms,

    float* d_waters,                 // (n_mu*target_n_waters*WATER_SIZE)
    std::uint32_t* d_wat_num,         // (n_mu)

    const float* d_scalars,           // (5) [last_residue, volume, num_steps, target_n_waters, n_moves]
    float* d_snapshots,               // output

    const float* d_mu_values,         // (n_mu)
    int n_mu,

    float distance_cutoff,
    std::uint32_t gcmc_steps,
    std::uint32_t equil_steps,
    std::uint32_t n_snapshots,
    std::uint32_t target_n_waters,

    void* d_states                   // opaque RNG states (n_mu * dummy_rng_state_bytes())
);
