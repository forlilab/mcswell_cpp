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
#include "run_mcswell.hpp"

namespace py = pybind11;

static void run_mcswell_py(
    const std::vector<Atom>& receptor_points,
    py::array_t<float, py::array::c_style | py::array::forcecast> boundaries_xyz, // (N,3) or (3N,)
    float distance_cutoff,
    float spacing,
    std::size_t gcmc_steps,
    std::size_t equilibration_steps,
    const std::string& save_path,
    const std::vector<float>& mu_values,
    std::size_t n_snapshots,
    unsigned long long seed = 42ULL
) {
    auto buf = boundaries_xyz.request();
    const float* ptr = static_cast<const float*>(buf.ptr);

    std::size_t n_points = 0;
    if (buf.ndim == 2) {
        if (buf.shape[1] != 3) throw std::runtime_error("boundaries must be (N,3)");
        n_points = (std::size_t)buf.shape[0];
    } else if (buf.ndim == 1) {
        if (buf.shape[0] % 3 != 0) throw std::runtime_error("boundaries must be (3N,)");
        n_points = (std::size_t)buf.shape[0] / 3;
    } else {
        throw std::runtime_error("boundaries must be shape (N,3) or (3N,)");
    }

    std::vector<float> out = run_mcswell_gpu_titration(
        receptor_points,
        ptr,
        n_points,
        distance_cutoff,
        spacing,
        gcmc_steps, 
        equilibration_steps,
        save_path,
        mu_values,
        n_snapshots,
        seed
    );
}

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
      py::arg("seed") = 12345ULL,
      "");
}
