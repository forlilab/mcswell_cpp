#pragma once
#include <cstdint>

#ifdef __CUDACC__
// Only visible to NVCC compilation units (.cu)
#include <curand_kernel.h>

// Device kernels can use the real type here
__global__ void init_rng(curandStatePhilox4_32_10_t* states, std::uint64_t seed, int n_mu);

__global__ void run_gcmc_titration_kernel(
    const float* d_boundaries_xyz, int n_points,
    const float* d_receptor, int n_receptor_atoms,
    float* d_waters,
    std::uint32_t* d_wat_num,
    const float* d_scalars,
    float* d_snapshots,
    const float* d_mu_values, int n_mu,
    float distance_cutoff,
    std::uint32_t gcmc_steps,
    std::uint32_t equil_steps,
    std::uint32_t n_snapshots,
    std::uint32_t target_n_waters,
    curandStatePhilox4_32_10_t* states
);

#endif // __CUDACC__
