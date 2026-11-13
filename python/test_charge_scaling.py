#   Copyright (c) 2026 Scripps Research, Forli Lab.
#   All rights reserved.
#
#   Author: Niccolo Bruciaferri
#
#   This library is free software; you can redistribute it and/or
#   modify it under the terms of the GNU Lesser General Public
#   License as published by the Free Software Foundation; either
#   version 2.1 of the License, or (at your option) any later version.
#
#   This library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public
#   License along with this library; if not, write to the Free Software
#   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

"""Self-checks for charge_scaling.py. Run directly: `python test_charge_scaling.py`.

Small hand-built residues, far apart, whose charged groups are known: a
carboxylate, an ammonium, a guanidinium, a calcium ion, a neutral amide and a
water. No pytest, as for test_convergence.py.
"""

import numpy as np

import charge_scaling as cs

# (atom id, atom name, xyz, charge); residues 20 A apart so no bond crosses them
ATOMS = [
    # Asp-like side chain: CB-CG(OD1)(OD2), net -1
    ("A:ASP:1:CB", "CB", (0.00, 0.00, 0.00), -0.20),
    ("A:ASP:1:CG", "CG", (1.52, 0.00, 0.00), 0.80),
    ("A:ASP:1:OD1", "OD1", (2.15, 1.08, 0.00), -0.80),
    ("A:ASP:1:OD2", "OD2", (2.15, -1.08, 0.00), -0.80),
    # Lys-like end: CE-NZ(H)3, net +1
    ("A:LYS:2:CE", "CE", (20.00, 0.00, 0.00), 0.31),
    ("A:LYS:2:NZ", "NZ", (21.49, 0.00, 0.00), -0.30),
    ("A:LYS:2:HZ1", "HZ1", (21.83, 0.96, 0.00), 0.33),
    ("A:LYS:2:HZ2", "HZ2", (21.83, -0.48, 0.83), 0.33),
    ("A:LYS:2:HZ3", "HZ3", (21.83, -0.48, -0.83), 0.33),
    # Arg-like guanidinium: CZ(NH1 H2)(NH2 H2), net +1
    ("A:ARG:3:CZ", "CZ", (40.00, 0.00, 0.00), 0.80),
    ("A:ARG:3:NH1", "NH1", (40.66, 1.15, 0.00), -0.86),
    ("A:ARG:3:HH11", "HH11", (41.67, 1.15, 0.00), 0.45),
    ("A:ARG:3:HH12", "HH12", (40.16, 2.02, 0.00), 0.45),
    ("A:ARG:3:NH2", "NH2", (40.66, -1.15, 0.00), -0.86),
    ("A:ARG:3:HH21", "HH21", (41.67, -1.15, 0.00), 0.45),
    ("A:ARG:3:HH22", "HH22", (40.16, -2.02, 0.00), 0.45),
    ("A:ARG:3:CD", "CD", (38.50, 0.00, 0.00), 0.12),
    # calcium ion, +2
    ("B:CA:4:CA", "CA", (60.00, 0.00, 0.00), 2.00),
    # Asn-like amide: CG(=OD1)ND2(H)2, neutral
    ("A:ASN:5:CB", "CB", (80.00, 0.00, 0.00), -0.20),
    ("A:ASN:5:CG", "CG", (81.52, 0.00, 0.00), 0.71),
    ("A:ASN:5:OD1", "OD1", (82.15, 1.08, 0.00), -0.59),
    ("A:ASN:5:ND2", "ND2", (82.20, -1.15, 0.00), -0.92),
    ("A:ASN:5:HD21", "HD21", (83.21, -1.15, 0.00), 0.50),
    ("A:ASN:5:HD22", "HD22", (81.70, -2.02, 0.00), 0.50),
    # water
    ("W:HOH:6:O", "O", (100.00, 0.00, 0.00), -0.834),
    ("W:HOH:6:H1", "H1", (100.96, 0.00, 0.00), 0.417),
    ("W:HOH:6:H2", "H2", (99.76, 0.93, 0.00), 0.417),
]
IDS = [a[0] for a in ATOMS]
NAMES = [a[1] for a in ATOMS]
XYZ = np.array([a[2] for a in ATOMS])
Q = np.array([a[3] for a in ATOMS])


def residue_charge(q, res):
    return sum(qi for i, qi in zip(IDS, q) if i.rsplit(":", 1)[0] == res)


def test_groups_found():
    resnames = [i.split(":")[1] for i in IDS]
    groups = cs.find_charged_groups(NAMES, resnames, XYZ, Q)
    found = sorted((IDS[g[0]].rsplit(":", 1)[0], z) for g, z in groups)
    assert found == [("A:ARG:3", 1), ("A:ASP:1", -1), ("A:LYS:2", 1)], found


def test_residue_net_charges():
    new, report = cs.scale_charges(IDS, NAMES, XYZ, Q, 0.75)
    expected = {"A:ASP:1": -0.75, "A:LYS:2": 0.75, "A:ARG:3": 0.75, "B:CA:4": 1.50,
                "A:ASN:5": 0.0, "W:HOH:6": 0.0}
    for res, target in expected.items():
        assert abs(residue_charge(new, res) - target) < 1e-9, (res, residue_charge(new, res))
    assert sorted(r["residue"] for r in report) == ["A:ARG:3", "A:ASP:1", "A:LYS:2", "B:CA:4"]


def test_group_atoms_scaled_and_neutral_untouched():
    new, _ = cs.scale_charges(IDS, NAMES, XYZ, Q, 0.75)
    for i, atom in enumerate(IDS):
        if atom in ("A:ASP:1:CG", "A:ASP:1:OD1", "A:ASP:1:OD2", "A:LYS:2:NZ", "A:ARG:3:NH1"):
            assert abs(new[i] - 0.75 * Q[i]) < 1e-12, atom
        if atom.startswith(("A:ASN:5", "W:HOH:6")):
            assert new[i] == Q[i], atom


def test_factor_one_is_identity():
    new, report = cs.scale_charges(IDS, NAMES, XYZ, Q, 1.0)
    assert np.array_equal(new, Q) and report == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all charge-scaling checks passed")
