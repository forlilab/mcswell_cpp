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
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cstddef>

#include "atom.hpp"          
#include "cuda/consts.cuh"
#include "cuda/water_model.cuh"

namespace mcswell {

static constexpr int WATER_SIZE_LOCAL = 12; // keep local to avoid include issues

inline Atom make_water_atom_O(double x, double y, double z, std::size_t resnum) {
    Atom a(
        /*atom_type*/ "OW",
        /*atom_id*/   "A:HOH:" + std::to_string(resnum) + ":O",
        /*coords*/    {x, y, z},
        /*rmin_half*/ (double)water_model::RMINH_O,
        /*epsilon*/   (double)water_model::EPS_O,
        /*charge*/    (double)water_model::Q_O
    );
    a.set_residue_number(resnum);
    return a;
}

inline Atom make_water_atom_H(double x, double y, double z, std::size_t resnum, int which) {
    // which: 1 -> H1, 2 -> H2
    const std::string name = (which == 1) ? "H1" : "H2";
    Atom a(
        /*atom_type*/ "HW",
        /*atom_id*/   "A:HOH:" + std::to_string(resnum) + ":" + name,
        /*coords*/    {x, y, z},
        /*rmin_half*/ 0.0,
        /*epsilon*/   0.0,
        /*charge*/    (double)water_model::Q_H
    );
    a.set_residue_number(resnum);
    return a;
}

inline std::string atom_name_from_id(const std::string& atom_id) {
    // atom_id: chain:resname:resnum:atom
    auto pos = atom_id.rfind(':');
    if (pos == std::string::npos) return "X";
    return atom_id.substr(pos + 1);
}

inline std::vector<std::vector<std::vector<Atom>>>
reconstruct_snapshots_by_mu(
    const std::vector<float>& waters,
    std::size_t n_mu,
    std::size_t n_snapshots,
    std::size_t max_n_waters
) {
    const std::size_t WATER_SIZE = (std::size_t)WATER_SIZE_LOCAL;

    std::vector<std::vector<std::vector<Atom>>> frames_by_mu(n_mu);

    for (std::size_t mu_idx = 0; mu_idx < n_mu; ++mu_idx) {
        frames_by_mu[mu_idx].reserve(n_snapshots);

        for (std::size_t snap_idx = 0; snap_idx < n_snapshots; ++snap_idx) {
            const std::size_t snapshot_base =
                (mu_idx * n_snapshots * max_n_waters * WATER_SIZE) +
                (snap_idx * max_n_waters * WATER_SIZE);

            std::vector<Atom> atoms_snapshot;
            atoms_snapshot.reserve(max_n_waters * 3);

            for (std::size_t water_idx = 0; water_idx < max_n_waters; ++water_idx) {
                const std::size_t wat_index = snapshot_base + water_idx * WATER_SIZE;
                if (wat_index + (WATER_SIZE - 1) >= waters.size()) break;

                const double o_x = (double)waters[wat_index + 0];
                const double o_y = (double)waters[wat_index + 1];
                const double o_z = (double)waters[wat_index + 2];

                // inactive slot
                if (o_x == 0.0 && o_y == 0.0 && o_z == 0.0) continue;

                const std::size_t o_res  = (std::size_t)waters[wat_index + 3];
                const std::size_t h1_res = (std::size_t)waters[wat_index + 7];
                const std::size_t h2_res = (std::size_t)waters[wat_index + 11];

                // Choose best residue number (and sanity-check)
                std::size_t resnum = o_res;
                if (resnum == 0) resnum = h1_res;
                if (resnum == 0) resnum = h2_res;
                if (resnum == 0) {
                    resnum = water_idx + 1;
                }

                const double h1_x = (double)waters[wat_index + 4];
                const double h1_y = (double)waters[wat_index + 5];
                const double h1_z = (double)waters[wat_index + 6];

                const double h2_x = (double)waters[wat_index + 8];
                const double h2_y = (double)waters[wat_index + 9];
                const double h2_z = (double)waters[wat_index + 10];

                atoms_snapshot.push_back(make_water_atom_O(o_x,  o_y,  o_z,  resnum));
                atoms_snapshot.push_back(make_water_atom_H(h1_x, h1_y, h1_z, resnum, 1));
                atoms_snapshot.push_back(make_water_atom_H(h2_x, h2_y, h2_z, resnum, 2));
            }

            frames_by_mu[mu_idx].push_back(std::move(atoms_snapshot));
        }
    }

    return frames_by_mu;
}

inline void to_pdb(
    const std::vector<Atom>& atoms,
    const std::string& fname,
    const std::vector<double>* energies_per_water = nullptr
) {
    std::ofstream f(fname);
    if (!f) throw std::runtime_error("unable to create file: " + fname);

    const bool include_energies = (energies_per_water != nullptr);

    auto atom_name_from_id = [](const std::string& atom_id) -> std::string {
        auto pos = atom_id.rfind(':');
        if (pos == std::string::npos) return "X";
        return atom_id.substr(pos + 1); // "O","H1","H2"
    };

    std::size_t water_idx = 0;
    int resnumber = 0;

    for (std::size_t i = 0; i < atoms.size(); ++i) {
        if (i % 3 == 0) water_idx++;

        const Atom& a = atoms[i];
        const auto c = a.coords_arr();

        double bfac = 0.0;
        if (include_energies) {
            if (water_idx == 0 || (water_idx - 1) >= energies_per_water->size())
                throw std::runtime_error("energies vector too small for to_pdb()");
            bfac = (*energies_per_water)[water_idx - 1];
        }

        std::string aname = atom_name_from_id(a.atom_id_str()); // "O","H1","H2"
        std::string element = (aname.size() && aname[0] == 'H') ? "H" : "O";

        // PDB atom name is 4 chars; common for waters is " O  ", " H1 ", " H2 "
        char atom_name_4[5] = {' ', ' ', ' ', ' ', '\0'};
        if (aname.size() == 1) {
            atom_name_4[1] = aname[0];            // " O  "
        } else if (aname.size() == 2) {
            atom_name_4[1] = aname[0]; atom_name_4[2] = aname[1];  // " H1 "
        } else if (aname.size() == 3) {
            atom_name_4[0] = aname[0]; atom_name_4[1] = aname[1]; atom_name_4[2] = aname[2];
        } else {
            // truncate
            atom_name_4[0] = aname[0]; atom_name_4[1] = aname[1]; atom_name_4[2] = aname[2]; atom_name_4[3] = aname[3];
        }

        const int serial = (int)(i + 1);                 // 1-based
        const int resseq = (int)a.residue_number;        // use stored residue number

        char line[200];
        // Columns per PDB spec (enough for most readers):
        // ATOM  serial name resName chain resSeq   x       y       z   occ  temp element
        std::snprintf(
            line, sizeof(line),
            "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n",
            serial,
            atom_name_4,
            "HOH",
            "A",
            water_idx,
            c[0], c[1], c[2],
            0.0, bfac,
            element.c_str()
        );
        f << line;
    }

    f << "TER\n";
}

inline void save_pdbs_by_mu_snapshot(
    const std::vector<float>& snapshots_flat,
    std::size_t n_mu,
    std::size_t n_snapshots,
    std::size_t max_n_waters,
    const std::string& out_dir
) {
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    auto frames = reconstruct_snapshots_by_mu(snapshots_flat, n_mu, n_snapshots, max_n_waters);

    for (std::size_t mu = 0; mu < n_mu; ++mu) {
        // optional: per-mu folder
        std::ostringstream mu_dir;
        mu_dir << out_dir << "/mu_" << std::setw(3) << std::setfill('0') << mu;
        fs::create_directories(mu_dir.str());

        for (std::size_t s = 0; s < n_snapshots; ++s) {
            std::ostringstream fname;
            fname << mu_dir.str()
                  << "/snap_" << std::setw(5) << std::setfill('0') << s
                  << ".pdb";
            to_pdb(frames[mu][s], fname.str(), nullptr);
        }
    }
}

} // namespace mcswell
