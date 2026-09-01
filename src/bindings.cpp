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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <vector>
#include <array>
#include <cstring>   // std::memcpy  (important on MSVC)

#include "atom.hpp"
#include "insertion_points.hpp"
#include "pipeline.hpp"
#include "run_mcswell.hpp"

namespace py = pybind11;

namespace {

std::size_t n_points_from_boundaries(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& boundaries_xyz) {
    auto buf = boundaries_xyz.request();
    if (buf.ndim == 2) {
        if (buf.shape[1] != 3) throw std::runtime_error("boundaries must be (N,3)");
        return static_cast<std::size_t>(buf.shape[0]);
    }
    if (buf.ndim == 1) {
        if (buf.shape[0] % 3 != 0) throw std::runtime_error("boundaries must be (3N,)");
        return static_cast<std::size_t>(buf.shape[0]) / 3;
    }
    throw std::runtime_error("boundaries must be shape (N,3) or (3N,)");
}

// Runs the GCMC titration only, with no post-simulation analysis -- kept
// for debugging/advanced use. Returns the full in-memory snapshot buffer
// (and the water-buffer sizing needed to index into it) rather than
// writing anything to disk, unless dump_debug_pdbs is requested.
py::dict run_mcswell_py(
    const std::vector<Atom>& receptor_points,
    py::array_t<float, py::array::c_style | py::array::forcecast> boundaries_xyz, // (N,3) or (3N,)
    float distance_cutoff,
    float spacing,
    std::size_t gcmc_steps,
    std::size_t equilibration_steps,
    const std::string& save_path,
    const std::vector<float>& mu_values,
    std::size_t n_snapshots,
    unsigned long long seed = 42ULL,
    bool dump_debug_pdbs = false
) {
    const std::size_t n_points = n_points_from_boundaries(boundaries_xyz);
    const float* ptr = static_cast<const float*>(boundaries_xyz.request().ptr);

    const auto result = run_mcswell_gpu_titration(
        receptor_points, ptr, n_points, distance_cutoff, spacing, gcmc_steps, equilibration_steps, save_path,
        mu_values, n_snapshots, seed, dump_debug_pdbs
    );

    py::array_t<float> snapshots(static_cast<py::ssize_t>(result.snapshots.size()));
    std::memcpy(snapshots.mutable_data(), result.snapshots.data(), result.snapshots.size() * sizeof(float));

    py::dict out;
    out["snapshots"] = snapshots;
    out["target_n_waters"] = result.target_n_waters;
    out["n_mu"] = mu_values.size();
    out["n_snapshots"] = n_snapshots;
    return out;
}

// Runs the GCMC titration and, without ever writing a per-snapshot PDB to
// disk, feeds its in-memory result directly into the requested
// post-simulation analyses. See include/pipeline.hpp.
py::dict run_mcswell_and_analyze_py(
    const std::vector<Atom>& receptor_points,
    py::array_t<float, py::array::c_style | py::array::forcecast> boundaries_xyz,
    float distance_cutoff,
    float spacing,
    std::size_t gcmc_steps,
    std::size_t equilibration_steps,
    const std::string& save_path,
    const std::vector<float>& mu_values,
    std::size_t n_snapshots,
    double temperature,
    double mu_bulk,
    std::array<double, 3> box_center,
    std::array<double, 3> box_halfsize,
    bool run_binomial = true,
    bool run_gci = true,
    double grid_spacing = 0.5,
    double grid_sigma = 1.4,
    double peak_percentile = 90.0,
    double peak_merge_cutoff = 1.4,
    bool capacity_filter = true,
    double bulk_water_density = 0.0334,
    double max_capacity_hit_fraction = 0.01,
    double max_mean_capacity_fraction = 0.90,
    double assignment_cutoff = 2.4,
    double local_radius = 1.4,
    const std::string& local_volume_mode = "sampler",
    int region_max_terms = 12,
    int local_max_terms = 4,
    int random_starts = 64,
    unsigned long long fit_seed = 20260812ULL,
    unsigned long long seed = 12345ULL,
    bool dump_debug_pdbs = false
) {
    const std::size_t n_points = n_points_from_boundaries(boundaries_xyz);
    const float* ptr = static_cast<const float*>(boundaries_xyz.request().ptr);

    mcswell::analysis::PostAnalysisConfig cfg;
    cfg.temperature = temperature;
    cfg.mu_bulk = mu_bulk;
    cfg.box_center = box_center;
    cfg.box_halfsize = box_halfsize;
    cfg.grid_spacing = grid_spacing;
    cfg.grid_sigma = grid_sigma;
    cfg.peak_percentile = peak_percentile;
    cfg.peak_merge_cutoff = peak_merge_cutoff;
    cfg.capacity_filter = capacity_filter;
    cfg.bulk_water_density = bulk_water_density;
    cfg.max_capacity_hit_fraction = max_capacity_hit_fraction;
    cfg.max_mean_capacity_fraction = max_mean_capacity_fraction;
    cfg.assignment_cutoff = assignment_cutoff;
    cfg.local_radius = local_radius;
    cfg.local_volume_mode = local_volume_mode;
    cfg.region_max_terms = region_max_terms;
    cfg.local_max_terms = local_max_terms;
    cfg.random_starts = random_starts;
    cfg.fit_seed = fit_seed;

    const auto result = mcswell::run_mcswell_full_pipeline(
        receptor_points, ptr, n_points, distance_cutoff, spacing, gcmc_steps, equilibration_steps, save_path,
        mu_values, n_snapshots, cfg, run_binomial, run_gci, seed, dump_debug_pdbs
    );

    py::dict out;
    out["n_insertion_points"] = result.n_insertion_points;
    out["sampler_volume"] = result.sampler_volume;
    out["binomial_output_dir"] = result.binomial_output_dir;
    out["gci_output_dir"] = result.gci_output_dir;
    return out;
}

} // namespace

static py::array_t<double> make_insertion_points_py(
    const std::vector<Atom>& atoms,
    std::array<double,3> size,
    double spacing,
    std::array<double,3> center,
    double max_distance,
    double min_distance
) {
    GridSpec g{size, center, spacing};
    std::vector<double> flat = make_insertion_points_flat(atoms, g, max_distance, min_distance);

    const py::ssize_t npts = static_cast<py::ssize_t>(flat.size() / 3);
    py::array_t<double> arr({npts, (py::ssize_t)3});
    std::memcpy(arr.mutable_data(), flat.data(), flat.size() * sizeof(double));
    return arr;
}

static void print_atoms(const std::vector<Atom>& atoms, std::size_t max_print = 10) {
    py::print("[mcswell_cpp] Received", atoms.size(), "Atom(s)");
    const std::size_t n = std::min(atoms.size(), max_print);
    for (std::size_t i = 0; i < n; i++) {
        py::print(" ", i, atoms[i].repr());
    }
    if (atoms.size() > n) {
        py::print(" ...", (atoms.size() - n), "more not shown");
    }
}

PYBIND11_MODULE(_mcswell_cpp, m) {
    m.doc() = "Minimal C++ bindings scaffold for MCSwell porting (CPU-only stub).";

    // Water model compiled into this extension (see CMakeLists.txt WATER_MODEL
    // and include/cuda/water_model.cuh). Lets Python post-processing pick a
    // matching bulk chemical potential instead of assuming TIP3P.
    m.attr("WATER_MODEL_NAME") = WATER_MODEL_NAME;

    py::class_<Atom>(m, "Atom")
        .def(py::init<std::string,
                      std::string,
                      std::array<double,3>,
                      double, double, double>(),
             py::arg("atom_type"),
             py::arg("atom_id"),
             py::arg("coords"),
             py::arg("rmin_half"),
             py::arg("epsilon"),
             py::arg("charge"))
        .def_readwrite("atom_type", &Atom::atom_type)
        .def_readwrite("atom_id", &Atom::atom_id)
        .def_readwrite("coords", &Atom::coords)
        .def_readwrite("rmin_half", &Atom::rmin_half)
        .def_readwrite("epsilon", &Atom::epsilon)
        .def_readwrite("charge", &Atom::charge)
        .def_readonly("residue_number", &Atom::residue_number)
        .def("__repr__", [](const Atom& a) { return a.repr(); });

    m.def("make_insertion_points",
      &make_insertion_points_py,
      py::arg("atoms"),
      py::arg("size"),
      py::arg("spacing"),
      py::arg("center"),
      py::arg("max_distance"),
      py::arg("min_distance") = 1.5,
      "Generate valid insertion points on a regular lattice.\n"
      "Returns numpy array of shape (N,3), dtype float64.");

    m.def("print_atoms", &print_atoms,
          py::arg("atoms"),
          py::arg("max_print") = 10,
          "Print Atom values (debug helper).");

    m.def("run_mcswell",
      &run_mcswell_py,
      py::arg("receptor_points"),
      py::arg("boundaries"),
      py::arg("distance_cutoff"),
      py::arg("spacing"),
      py::arg("gcmc_steps"),
      py::arg("equilibration_steps"),
      py::arg("save_path"),
      py::arg("mu_values"),
      py::arg("n_snapshots"),
      py::arg("seed") = 42ULL,
      py::arg("dump_debug_pdbs") = false,
      "Run the GCMC titration only (no post-simulation analysis). Returns "
      "a dict with the full in-memory snapshot buffer ('snapshots'), plus "
      "the sizing needed to index into it ('target_n_waters', 'n_mu', "
      "'n_snapshots'). Nothing is written to disk unless dump_debug_pdbs "
      "is set. Prefer run_mcswell_and_analyze for normal use.");

    m.def("run_mcswell_and_analyze",
      &run_mcswell_and_analyze_py,
      py::arg("receptor_points"),
      py::arg("boundaries"),
      py::arg("distance_cutoff"),
      py::arg("spacing"),
      py::arg("gcmc_steps"),
      py::arg("equilibration_steps"),
      py::arg("save_path"),
      py::arg("mu_values"),
      py::arg("n_snapshots"),
      py::arg("temperature"),
      py::arg("mu_bulk"),
      py::arg("box_center"),
      py::arg("box_halfsize"),
      py::arg("run_binomial") = true,
      py::arg("run_gci") = true,
      py::arg("grid_spacing") = 0.5,
      py::arg("grid_sigma") = 1.4,
      py::arg("peak_percentile") = 90.0,
      py::arg("peak_merge_cutoff") = 1.4,
      py::arg("capacity_filter") = true,
      py::arg("bulk_water_density") = 0.0334,
      py::arg("max_capacity_hit_fraction") = 0.01,
      py::arg("max_mean_capacity_fraction") = 0.90,
      py::arg("assignment_cutoff") = 2.4,
      py::arg("local_radius") = 1.4,
      py::arg("local_volume_mode") = "sampler",
      py::arg("region_max_terms") = 12,
      py::arg("local_max_terms") = 4,
      py::arg("random_starts") = 64,
      py::arg("fit_seed") = 20260812ULL,
      py::arg("seed") = 12345ULL,
      py::arg("dump_debug_pdbs") = false,
      "Run the GCMC titration and the requested post-simulation "
      "analyses (binomial site-occupancy fit and/or ProtoMS-style GCI) "
      "entirely in memory: no per-snapshot PDB is written to disk. "
      "Writes only the final result artifacts (sites.csv, sites.pdb, "
      "titration CSVs, gO.dx, metadata JSON) under save_path/gci/. "
      "Returns a dict with n_insertion_points, sampler_volume, "
      "binomial_output_dir, gci_output_dir.");
}
