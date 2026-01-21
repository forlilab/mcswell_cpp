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