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
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

struct Atom {
    // Atom type for the forcefield
    std::string atom_type;

    // Atom id with chain:resname:residue:atom
    std::string atom_id;

    // Atom number (residue number)
    std::size_t residue_number = 0;

    // 3D coordinates of the atom
    std::array<double, 3> coords{};

    // sigma value of the VdW
    double rmin_half = 0.0;

    // epsilon value of the VdW
    double epsilon = 0.0;

    // partial charge for coulomb
    double charge = 0.0;

    Atom() = default;

    Atom(std::string atom_type_,
         std::string atom_id_,
         std::array<double, 3> coords_point,
         double rmin_half_,
         double epsilon_,
         double charge_)
        : atom_type(std::move(atom_type_)),
          atom_id(std::move(atom_id_)),
          residue_number(0),
          coords(coords_point),
          rmin_half(rmin_half_),
          epsilon(epsilon_),
          charge(charge_) {
        set_atom_number();
    }

    // ---------- getters ----------
    const std::string& atom_type_str() const { return atom_type; }
    const std::string& atom_id_str()   const { return atom_id; }

    std::array<double, 3> coords_arr() const { return coords; }

    double rmin_half_val() const { return rmin_half; }
    double epsilon_val()   const { return epsilon; }
    double charge_val()    const { return charge; }

    // ---------- setters ----------
    void set_coords(std::array<double, 3> new_coords) { coords = new_coords; }

    void set_atom_number() {
        // split atom_id by ':'
        std::vector<std::string> parts;
        parts.reserve(4);
        std::stringstream ss(atom_id);
        std::string item;
        while (std::getline(ss, item, ':')) {
            parts.push_back(item);
        }

        if (parts.size() < 3) {
            throw std::runtime_error(
                "Failed to parse atom number: atom_id has fewer than 3 ':'-separated fields: " + atom_id
            );
        }

        try {
            residue_number = static_cast<std::size_t>(std::stoull(parts[2]));
        } catch (...) {
            throw std::runtime_error(
                "Failed to parse atom number from atom_id field[2]: " + atom_id
            );
        }
    }

    void set_residue_number(std::size_t res_number) {
        // split atom_id by ':'
        std::vector<std::string> parts;
        parts.reserve(4);
        std::stringstream ss(atom_id);
        std::string item;
        while (std::getline(ss, item, ':')) {
            parts.push_back(item);
        }

        if (parts.size() < 4) {
            throw std::runtime_error(
                "set_residue_number: atom_id must have at least 4 ':'-separated fields (chain:resname:resnum:atom_type): "
                + atom_id
            );
        }

        const std::string& chain     = parts[0];
        const std::string& resname   = parts[1];
        const std::string& atom_type = parts[3];

        atom_id = chain + ":" + resname + ":" + std::to_string(res_number) + ":" + atom_type;
        residue_number = res_number;
    }

    bool is_heavy_atom() const {
        return !(atom_type.size() >= 1 && atom_type[0] == 'H');
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Atom(atom_type=" << atom_type
            << ", atom_id=" << atom_id
            << ", residue_number=" << residue_number
            << ", coords=[" << coords[0] << ", " << coords[1] << ", " << coords[2] << "]"
            << ", rmin_half=" << rmin_half
            << ", epsilon=" << epsilon
            << ", charge=" << charge
            << ")";
        return oss.str();
    }
};
