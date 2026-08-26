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

import numpy as np
def closest_index(values, target):
    return min(range(len(values)), key=lambda i: abs(values[i] - target))

def expand_ranges(ranges):
    return np.unique(np.concatenate([
        np.arange(ranges["start"], ranges["stop"] + 1e-8, ranges["step"])
    ]))

def save_box_corners_pdb(center, size, outfile="box_corners.pdb"):
    """
    Save the 8 corner points of an axis-aligned box to a PDB file.
    """

    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)

    # Half-sizes
    hx, hy, hz = size / 2.0

    # Corners: all 8 combinations of ± half-size
    offsets = np.array([
        [-hx, -hy, -hz],
        [-hx, -hy, +hz],
        [-hx, +hy, -hz],
        [-hx, +hy, +hz],
        [+hx, -hy, -hz],
        [+hx, -hy, +hz],
        [+hx, +hy, -hz],
        [+hx, +hy, +hz],
    ])

    corners = center + offsets

    with open(outfile, "w") as f:
        for i, (x, y, z) in enumerate(corners, start=1):
            atom_name = f"C{i}"        # <- fixed
            f.write(
                f"ATOM  {i:5d}  {atom_name:>3s} BOX A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")

    print(f"Saved 8 box corner atoms to {outfile}")
    return

ION_SMARTS = {
        # Alkali metals
        "LI":  "[#3]",
        "NA":  "[#11]",
        "K":   "[#19]",
        "RB":  "[#37]",
        "CS":  "[#55]",

        # Alkaline earth metals
        "BE":  "[#4]",
        "MG":  "[#12]",
        "CA":  "[#20]",
        "SR":  "[#38]",
        "BA":  "[#56]",

        # Transition metals
        "SC":  "[#21]",
        "TI":  "[#22]",
        "V":   "[#23]",
        "CR":  "[#24]",
        "MN":  "[#25]",
        "FE":  "[#26]",
        "CO":  "[#27]",
        "NI":  "[#28]",
        "CU":  "[#29]",
        "ZN":  "[#30]",

        "Y":   "[#39]",
        "ZR":  "[#40]",
        "NB":  "[#41]",
        "MO":  "[#42]",
        "RU":  "[#44]",
        "RH":  "[#45]",
        "PD":  "[#46]",
        "AG":  "[#47]",
        "CD":  "[#48]",

        "HF":  "[#72]",
        "TA":  "[#73]",
        "W":   "[#74]",
        "RE":  "[#75]",
        "PT":  "[#78]",
        "AU":  "[#79]",
        "HG":  "[#80]",

        # Nonmetals
        "F":   "[#9]",
        "CL":  "[#17]",
        "BR":  "[#35]",
        "I":   "[#53]",
        "O":   "[#8]",
        "S":   "[#16]",
        "SE":  "[#34]",

        # p-block metals
        "AL": "[#13]",
        "GA": "[#31]",
        "IN": "[#49]",
        "SN": "[#50]",
        "SB": "[#51]",
        "PB": "[#82]",
        "BI": "[#83]",

        # Lanthanides
        "LA": "[#57]",
        "CE": "[#58]",
        "PR": "[#59]",
        "ND": "[#60]",
        "SM": "[#62]",
        "EU": "[#63]",
        "GD": "[#64]",
        "TB": "[#65]",
        "DY": "[#66]",
        "HO": "[#67]",
        "ER": "[#68]",
        "TM": "[#69]",
        "YB": "[#70]",
        "LU": "[#71]",

        # Actinides
        "TH": "[#90]",
        "U":  "[#92]",
    }

def _is_ion(mol):
    for ion_name, smarts in ION_SMARTS.items():
        ion_query = Chem.MolFromSmarts(smarts)
        # print(f"Checking {ion_name} with query: {ion_query}")
        if mol.HasSubstructMatch(ion_query):
            print("Has substruct match")
            print(mol.GetNumAtoms())
            if mol.GetNumAtoms() == 1:
                return True
    return False

def rmin_half(sigma):
        """
        The VdW rmin_half term from sigma
        """
        rmin = 2.0 ** (1.0 / 6) * sigma
        rmin_half = rmin / 2
        return rmin_half


# -----------------------------------------------------------------------------
# Water-model-dependent bulk chemical potential
# -----------------------------------------------------------------------------
# Excess chemical potential of bulk water (kcal/mol, ~300 K), keyed by the
# WATER_MODEL compiled into the mcswell_cpp CUDA extension (see
# CMakeLists.txt and include/cuda/water_model.cuh).
#
# These are published particle-insertion/FEP literature values (e.g. Mahoney &
# Jorgensen, J. Chem. Phys. 112, 8910 (2000), for TIP3P/TIP4P), computed with
# a DIFFERENT energy function, LJ cutoff, and long-range electrostatics
# treatment than the MCSwell GCMC sampler (include/cuda/energy.cuh). Treat
# them as starting points only -- recalibrate mu_bulk with MCSwell's own
# sampler (e.g. a bulk-box titration) before trusting absolute DeltaG_bind
# values for production work.
MU_BULK_BY_WATER_MODEL = {
    "TIP3P": -6.09,
    "TIP4P": -6.87,
}


def detect_compiled_water_model():
    """
    Return the WATER_MODEL_NAME the mcswell_cpp CUDA extension was compiled
    with (e.g. "TIP3P", "TIP4P"), or None if the extension is not importable
    in the current environment (e.g. pure post-processing without the built
    C++ module available).
    """
    try:
        import mcswell_cpp as mc
    except Exception:
        return None
    return getattr(mc, "WATER_MODEL_NAME", None)


def resolve_mu_bulk(explicit_mu_bulk):
    """
    Resolve the bulk-water excess chemical potential (kcal/mol) used to
    reference hydration free energies (DeltaG_bind = epsilon - mu_bulk).

    Priority:
      1. ``explicit_mu_bulk`` if not None (CLI/config override) -- trusted
         as-is, no warning.
      2. A literature default keyed by the water model the mcswell_cpp
         extension was compiled with (see MU_BULK_BY_WATER_MODEL above), with
         a loud warning that it is an unverified literature value for this
         specific sampler.
      3. If the compiled model can't be detected, fall back to the legacy
         TIP3P default with a loud warning. If the model IS detected but has
         no table entry, raise rather than guess.

    Returns
    -------
    (mu_bulk, water_model, source) : (float, Optional[str], str)
    """
    if explicit_mu_bulk is not None:
        return float(explicit_mu_bulk), detect_compiled_water_model(), "explicit_override"

    water_model = detect_compiled_water_model()

    if water_model is None:
        mu_bulk = MU_BULK_BY_WATER_MODEL["TIP3P"]
        print(
            "[mu_bulk] WARNING: mu_bulk was not supplied and the compiled "
            "mcswell_cpp water model could not be detected (extension not "
            "importable in this environment). Assuming the legacy default "
            f"(TIP3P, {mu_bulk} kcal/mol). If this run used a different "
            "water model, pass --mu-bulk / gci.mu_bulk explicitly."
        )
        return float(mu_bulk), None, "fallback_tip3p_model_undetected"

    if water_model not in MU_BULK_BY_WATER_MODEL:
        raise RuntimeError(
            f"No literature bulk-water excess chemical potential is known for "
            f"compiled water model '{water_model}'. Pass --mu-bulk / "
            f"gci.mu_bulk explicitly (known models: "
            f"{sorted(MU_BULK_BY_WATER_MODEL)})."
        )

    mu_bulk = MU_BULK_BY_WATER_MODEL[water_model]
    print(
        f"[mu_bulk] WARNING: no explicit mu_bulk given. Auto-selected the "
        f"literature bulk excess chemical potential for the compiled water "
        f"model '{water_model}': {mu_bulk} kcal/mol. This is a published "
        "value computed with a different energy function/cutoff than the "
        "MCSwell GCMC sampler -- recalibrate with this sampler before "
        "trusting absolute DeltaG_bind values. Override with --mu-bulk / "
        "gci.mu_bulk to silence this warning."
    )
    return float(mu_bulk), water_model, f"literature_default_{water_model.lower()}"
