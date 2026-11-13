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

"""Electronic continuum correction (ECC): scale the charges of formally charged groups.

Fixed-charge force fields lack electronic polarization, which screens the
interactions between charged groups in a condensed phase. The ECC includes it in
a mean-field way by scaling the charges of ions and charged groups by
1/sqrt(eps_el) ~ 0.75, where eps_el ~ 1.78 is the electronic (high-frequency)
dielectric constant of water (Leontyev & Stuchebrukhov, PCCP 2011, 13, 2613;
Duboue-Dijon et al., J. Chem. Phys. 2020, 153, 050901; the prosECCo75 force
field, Nencini et al., JCTC 2024, 20, 7546).

Charged groups are found from the chemistry of the receptor and ligand atoms
(bonds inferred from distances):

  carboxylate                        C bonded to exactly two terminal O        -1
  sulfonate / phosphate              S or P with >= 3 terminal O               -1
  amidinium / guanidinium /          C with no O neighbour, bonded to >= 2 N   +1
  protonated imidazole               that all carry H
  ammonium                           N with 4 neighbours, at least one H       +1
  quaternary ammonium                N with 4 heavy neighbours                 +1

A candidate is kept only if its summed charge has the expected sign and is
larger than 0.5 in magnitude (0.2 for ammonium, whose charge an N-terminal
proline spreads over the ring), so neutral look-alikes are left alone. The
charges of each group are multiplied by the factor, and the other atoms of the
same residue absorb the remainder, so that every residue ends with a net charge
of factor x its formal charge. A charged residue in which no group is found (a
metal ion, for example) is scaled as a whole. Water is never scaled: the water
models already include electronic polarization effectively.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_CHARGE_SCALING = 0.75

_COVALENT_RADIUS = dict(H=0.31, C=0.76, N=0.71, O=0.66, S=1.05, P=1.07, F=0.57, CL=1.02, BR=1.20, I=1.39)
_WATER = {"HOH", "WAT", "TIP3", "TIP", "SOL", "FIX"}


def _element(name, resname):
    n = name.upper().lstrip("0123456789")
    if resname == "UNL" and n[:2] in ("CL", "BR"):
        return n[:2]
    return n[0] if n else "X"


def find_charged_groups(names, resnames, xyz, charges):
    """Charged groups as (atom indices, formal charge) pairs.

    names, resnames: per-atom atom and residue names; xyz: (n, 3) coordinates
    in angstrom; charges: per-atom partial charges.
    """
    xyz = np.asarray(xyz, float)
    q = np.asarray(charges, float)
    el = [_element(n, r) for n, r in zip(names, resnames)]
    rad = np.array([_COVALENT_RADIUS.get(e, 0.8) for e in el])
    neighbours = defaultdict(list)
    for i, j in cKDTree(xyz).query_pairs(2.2):
        if np.linalg.norm(xyz[i] - xyz[j]) < 1.2 * (rad[i] + rad[j]):
            neighbours[i].append(j)
            neighbours[j].append(i)

    def terminal_o(o, centre):  # an O bonded to nothing but centre (and other O)
        return [k for k in neighbours[o] if el[k] != "O"] == [centre]

    candidates = []   # (indices, formal charge, minimum |summed charge|)
    for i, e in enumerate(el):
        nb = neighbours[i]
        if e == "C":
            o = [j for j in nb if el[j] == "O"]
            n = [j for j in nb if el[j] == "N"]
            if len(o) == 2 and all(terminal_o(j, i) for j in o):
                candidates.append(([i] + o, -1, 0.5))
            elif not o and len(n) >= 2:
                h_on = {j: [h for h in neighbours[j] if el[h] == "H"] for j in n}
                if all(h_on.values()):
                    group = [i] + n + [h for j in n for h in h_on[j]] + [h for h in nb if el[h] == "H"]
                    candidates.append((group, +1, 0.5))
        elif e in ("S", "P"):
            o = [j for j in nb if el[j] == "O" and terminal_o(j, i)]
            if len(o) >= 3:
                candidates.append(([i] + o, -1, 0.5))
        elif e == "N" and len(nb) == 4:
            if any(el[h] == "H" for h in nb):
                candidates.append(([i] + [h for h in nb if el[h] == "H"], +1, 0.2))
            else:
                group = [i] + nb + [h for c in nb for h in neighbours[c] if el[h] == "H"]
                candidates.append((group, +1, 0.5))

    groups = []
    for group, z, threshold in candidates:
        group = sorted(set(group))
        total = q[group].sum()
        if np.sign(total) == np.sign(z) and abs(total) > threshold:
            groups.append((group, z))
    return groups


def scale_charges(atom_ids, names, xyz, charges, factor=DEFAULT_CHARGE_SCALING):
    """Scaled charges and a per-residue report.

    atom_ids are MCSwell atom ids, "<chain>:<resname>:<resnum>:<atom>". Returns
    (new_charges, report), where report has one entry per scaled residue:
    dict(residue, formal_charge, charge_before, charge_after, n_groups).
    """
    q = np.asarray(charges, float)
    if factor == 1.0:
        return q.copy(), []
    residue = np.array([a.rsplit(":", 1)[0] for a in atom_ids])
    resname = [r.split(":")[1] if r.count(":") >= 1 else "" for r in residue]
    is_water = np.array([r in _WATER for r in resname])
    keep = np.where(~is_water)[0]

    groups = find_charged_groups([names[i] for i in keep], [resname[i] for i in keep], np.asarray(xyz)[keep],
                                 q[keep])
    new = q.copy()
    in_group = np.zeros(len(q), bool)
    n_groups = defaultdict(int)
    for group, z in groups:
        idx = keep[group]
        new[idx] = factor * q[idx]
        in_group[idx] = True
        n_groups[residue[idx[0]]] += 1

    report = []
    for res in sorted(set(residue[keep])):
        idx = np.where(residue == res)[0]
        formal = int(round(q[idx].sum()))
        if res in n_groups:
            # every residue ends at factor x its formal charge (zwitterions stay neutral)
            rest = idx[~in_group[idx]]
            pool = rest if len(rest) else idx
            new[pool] += (factor * formal - new[idx].sum()) / len(pool)
        elif formal != 0:
            new[idx] = factor * q[idx]     # no group found: an ion, scaled as a whole
        else:
            continue
        report.append(dict(residue=res, formal_charge=formal, charge_before=float(q[idx].sum()),
                           charge_after=float(new[idx].sum()), n_groups=n_groups.get(res, 0)))
    return new, report
