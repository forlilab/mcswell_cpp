#include <thread>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "snapshots_io.hpp"
#include "atom.hpp"
#include "run_mcswell.hpp"
#include "cuda/launch.cuh"
#include "cuda/consts.cuh"

namespace {
inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("[CUDA] ") + what + ": " + cudaGetErrorString(e));
    }
}
} // namespace

// - build receptor_atoms_buffer (flattened float)
// - upload boundaries (insertion_points_xyz)
// - allocate waters buffer and wat_num buffer
// - allocate snapshots buffer
// - allocate mu array + scalars array
// - create RNG states (1 per mu)
// - launch the CUDA kernel (1 block per mu, 512 threads)
// - read back snapshots and return
std::vector<float> run_mcswell_gpu_titration(
    const std::vector<Atom>& receptor_points,
    const float* insertion_points_xyz,   // length = 3*n_points
    std::size_t n_points,
    float distance_cutoff,               // kept for API; pass into kernel if/when needed
    float spacing,
    std::size_t gcmc_steps,
    std::size_t equilibration_steps,
    const std::string& save_path,        // unused in this in-memory version
    const std::vector<float>& mu_values,
    std::size_t n_snapshots,
    std::uint64_t seed
) {
    // -------------------------
    // Input checks
    // -------------------------
    if (spacing <= 0.0f) throw std::invalid_argument("spacing must be > 0");
    if (!insertion_points_xyz) throw std::invalid_argument("insertion_points_xyz is null");
    if (n_points == 0) throw std::invalid_argument("n_points must be > 0");
    if (mu_values.empty()) throw std::invalid_argument("mu_values is empty");
    if (n_snapshots == 0) throw std::invalid_argument("n_snapshots must be > 0");
    if (receptor_points.empty()) throw std::invalid_argument("receptor_points is empty");

    const int n_mu = static_cast<int>(mu_values.size());

    // -------------------------
    // Receptor_atoms_buffer
    // -------------------------
    // Rust pushes 7 floats per atom:
    //   x, y, z, charge, epsilon, rmin_half, residue_number
    std::vector<float> h_receptor;
    h_receptor.reserve(receptor_points.size() * ATOM_FEATURES);

    for (const auto& a : receptor_points) {
        h_receptor.push_back(static_cast<float>(a.coords[0]));
        h_receptor.push_back(static_cast<float>(a.coords[1]));
        h_receptor.push_back(static_cast<float>(a.coords[2]));
        h_receptor.push_back(static_cast<float>(a.charge));
        h_receptor.push_back(static_cast<float>(a.epsilon));
        h_receptor.push_back(static_cast<float>(a.rmin_half));
        h_receptor.push_back(static_cast<float>(a.residue_number));
    }

    const std::size_t n_receptor_atoms = receptor_points.size();
    const double voxel_volume = static_cast<double>(spacing) * spacing * spacing;
    const double total_volume = voxel_volume * static_cast<double>(n_points);
    const std::size_t target_n_waters =
        static_cast<std::size_t>(std::floor(total_volume * BULK_WATER_DENSITY));

    if (target_n_waters == 0) {
        throw std::runtime_error("target_n_waters computed as 0 (check spacing/n_points/units).");
    }

    const std::size_t waters_len =
        static_cast<std::size_t>(n_mu) * target_n_waters * WATER_SIZE;

    const std::size_t snapshots_len =
        static_cast<std::size_t>(n_mu) * n_snapshots * target_n_waters * WATER_SIZE;

    std::vector<std::uint32_t> h_wat_num(static_cast<std::size_t>(n_mu), 0u);

    float h_scalars[4] = {
        static_cast<float>(receptor_points.back().residue_number), // same “last element” debug
        static_cast<float>(total_volume),                          // volume
        static_cast<float>(gcmc_steps),                             // num_steps
        static_cast<float>(target_n_waters),                        // target_n_waters
    };

    // -------------------------
    // Debug prints (optional)
    // -------------------------
    

    std::cout << "[run_mcswell_gpu_titration]\n"
              << "  Water model: " << WATER_MODEL_NAME << "\n"
              << "  receptor_atoms: " << n_receptor_atoms << "\n"
              << "  n_points: " << n_points << "\n"
              << "  spacing: " << spacing << "\n"
              << "  total_volume: " << total_volume << "\n"
              << "  target_n_waters: " << target_n_waters << "\n"
              << "  n_mu: " << n_mu << "\n"
              << "  gcmc_steps: " << gcmc_steps << "\n"
              << "  equilibration_steps: " << equilibration_steps << "\n"
              << "  n_snapshots: " << n_snapshots << "\n"
              << "  snapshots_len: " << snapshots_len << "\n"
              << "  save_path: " << save_path << "\n";

    // -------------------------
    // Device allocations
    // -------------------------
    float* d_boundaries = nullptr;   // insertion points, flattened float[3*n_points]
    float* d_receptor   = nullptr;   // receptor_atoms_buffer
    float* d_waters     = nullptr;   // waters_buffer
    std::uint32_t* d_wat_num = nullptr; // wat_num buffer
    float* d_scalars    = nullptr;   // scalars[4]
    float* d_mu         = nullptr;   // mu_values[n_mu]
    float* d_snapshots  = nullptr;   // snapshots_buffer
    void*  d_states     = nullptr;   // RNG states (opaque bytes; 1 per mu)

    // boundaries
    cuda_check(cudaMalloc(&d_boundaries, n_points * 3 * sizeof(float)), "cudaMalloc d_boundaries");
    cuda_check(cudaMemcpy(d_boundaries, insertion_points_xyz,
                          n_points * 3 * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy boundaries");

    // receptor
    cuda_check(cudaMalloc(&d_receptor, h_receptor.size() * sizeof(float)), "cudaMalloc d_receptor");
    cuda_check(cudaMemcpy(d_receptor, h_receptor.data(),
                          h_receptor.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy receptor");

    // waters (zero)
    cuda_check(cudaMalloc(&d_waters, waters_len * sizeof(float)), "cudaMalloc d_waters");
    cuda_check(cudaMemset(d_waters, 0, waters_len * sizeof(float)), "cudaMemset d_waters");

    // wat_num (zero)
    cuda_check(cudaMalloc(&d_wat_num, static_cast<std::size_t>(n_mu) * sizeof(std::uint32_t)),
               "cudaMalloc d_wat_num");
    cuda_check(cudaMemcpy(d_wat_num, h_wat_num.data(),
                          static_cast<std::size_t>(n_mu) * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy wat_num");

    // scalars
    cuda_check(cudaMalloc(&d_scalars, 4 * sizeof(float)), "cudaMalloc d_scalars");
    cuda_check(cudaMemcpy(d_scalars, h_scalars, 4 * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy scalars");

    // mu array
    cuda_check(cudaMalloc(&d_mu, static_cast<std::size_t>(n_mu) * sizeof(float)), "cudaMalloc d_mu");
    cuda_check(cudaMemcpy(d_mu, mu_values.data(),
                          static_cast<std::size_t>(n_mu) * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy mu_values");

    // snapshots (zero)
    cuda_check(cudaMalloc(&d_snapshots, snapshots_len * sizeof(float)), "cudaMalloc d_snapshots");
    cuda_check(cudaMemset(d_snapshots, 0, snapshots_len * sizeof(float)), "cudaMemset d_snapshots");

    // rng states (opaque; launch layer tells you bytes per state)
    const std::size_t state_bytes = dummy_rng_state_bytes(); // provided by your launch layer
    cuda_check(cudaMalloc(&d_states, static_cast<std::size_t>(n_mu) * state_bytes), "cudaMalloc d_states");

    // -------------------------
    // Init random seeds (seed_tensor) + kernel launch
    // -------------------------
    // CUDA: initialize n_mu RNG states with a master seed.
    init_rng_launch(d_states, seed, n_mu);
    cuda_check(cudaGetLastError(), "init_rng_launch");
    cuda_check(cudaDeviceSynchronize(), "init_rng sync");

    run_gcmc_titration_launch(
        /*boundaries_xyz*/ d_boundaries,
        /*n_points*/       static_cast<int>(n_points),
        /*receptor*/       d_receptor,
        /*n_receptor*/     static_cast<int>(n_receptor_atoms),
        /*waters*/         d_waters,
        /*wat_num*/        d_wat_num,
        /*scalars*/        d_scalars,
        /*snapshots*/      d_snapshots,
        /*mu_values*/      d_mu,
        /*n_mu*/           n_mu,
        /*distance_cutoff*/distance_cutoff,
        /*gcmc_steps*/     static_cast<std::uint32_t>(gcmc_steps),
        /*equil_steps*/    static_cast<std::uint32_t>(equilibration_steps),
        /*n_snapshots*/    static_cast<std::uint32_t>(n_snapshots),
        /*target_n_waters*/static_cast<std::uint32_t>(target_n_waters),
        /*rng_states*/     d_states
    );
    cuda_check(cudaGetLastError(), "run_gcmc_titration_launch");
    cuda_check(cudaDeviceSynchronize(), "run_gcmc_titration sync");

    // -------------------------
    // Read back snapshots_handle
    // -------------------------
    std::vector<float> h_snapshots(snapshots_len);
    cuda_check(cudaMemcpy(h_snapshots.data(), d_snapshots,
                          snapshots_len * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy snapshots back");

    cuda_check(cudaDeviceSynchronize(), "sync before inspection");

    mcswell::save_pdbs_by_mu_snapshot(
        h_snapshots,
        (std::size_t)n_mu,
        (std::size_t)n_snapshots,
        (std::size_t)target_n_waters,
        save_path
    );

    // -------------------------
    // Cleanup
    // -------------------------
    cudaFree(d_states);
    cudaFree(d_snapshots);
    cudaFree(d_mu);
    cudaFree(d_scalars);
    cudaFree(d_wat_num);
    cudaFree(d_waters);
    cudaFree(d_receptor);
    cudaFree(d_boundaries);

    return h_snapshots;
}
