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

import mcswell_cpp as mc
import argparse
import io
import numpy as np
import itertools
import time
import os
from utils import *
import tomllib
import estimate_free_energies as fe
import estimate_free_energies_GCI as fe_gci
from rdkit import Chem 
from rdkit.Chem import SDMolSupplier
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit
import pdbfixer
import openff
from openff.toolkit import Molecule
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from openff.units.openmm import to_openmm

def _build_box(positions, padding=2*openmmunit.angstrom):
        """Builds the simulation box
        """
        padding = padding.value_in_unit(openmmunit.nanometer)
        positions = positions.value_in_unit(openmmunit.nanometer)
        minRange = Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
        maxRange = Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
        center = 0.5*(minRange+maxRange)
        radius = max(unit.norm(center-pos) for pos in positions)
        width = max(2*radius+padding, 2*padding)
        vectors = (Vec3(width, 0, 0), Vec3(0, width, 0), Vec3(0, 0, width))
        box = Vec3(vectors[0][0], vectors[1][1], vectors[2][2])
        origin = center - (np.ceil(np.array((width, width, width))))
        _box_origin = origin
        _box_size = np.ceil(np.array([maxRange[0]-minRange[0],
                                           maxRange[1]-minRange[1],
                                           maxRange[2]-minRange[2]])).astype(int)
        lowerBound = center-box/2
        upperBound = center+box/2
        return vectors


def fix_pdb(pdbfile: str, pdbxfile: str, keep_heterogens: bool=False):
    """Fixes common problems in PDB such as:
            - missing atoms
            - missing residues
            - missing hydrogens
            - remove nonstandard residues

    :param pdbfile: pdb string old format
    :type pdbfile: str
    :param pdbxfile: pdb string new format
    :type pdbxfile: str
    :param keep_heterogens: if False all heterogen atoms but waters are deleted, defaults to False
    :type keep_heterogens: bool, optional
    :return: new topology and positions
    :rtype: tuple[Topology, list]
    """
    fixer = pdbfixer.PDBFixer(pdbfile=pdbfile, pdbxfile=pdbxfile)
    fixer.findMissingResidues()
    
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=False)

    fixer.findMissingAtoms() 
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7)
    return fixer.topology, fixer.positions

def characterize_receptor(receptors, project_path, protein_ff=["amber14-all.xml", 
        "amber14/tip3p.xml", "implicit/gbn2.xml"]):
    waters_residue_names = ["HOH", "WAT"]
    mcswell_atoms = list()

    for pdb_file in receptors:
        with open(pdb_file) as f:
            pdb_string = f.read()
        pdb_string = io.StringIO(pdb_string)
        
        pdbfile = None
        pdbxfile = None
        if pdb_file.endswith(".pdb"):
            pdbfile = pdb_string
        else:
            pdbxfile = pdb_string
        protein_topology, protein_positions = fix_pdb(pdbfile, pdbxfile, keep_heterogens=False)
        protein_topology.createStandardBonds()
        protein_modeller = Modeller(protein_topology, protein_positions)
        # Clean the receptor (to add an option to delete or not the water)
        # protein_modeller.deleteWater()
        ions_to_delete = []
        for residue in protein_modeller.topology.residues():
            if residue.name in ['NA', 'CL', 'K']:  # Adjust residue names based on your PDB
                ions_to_delete.append(residue)

        # Delete the identified ions
        protein_modeller.delete(ions_to_delete)
        
        vectors = _build_box(protein_positions)
        protein_modeller.topology.setPeriodicBoxVectors(vectors)
        protein_atoms = [atom for atom in protein_topology.atoms() if atom.residue.name not in ["NA", "CL", "K"]]
        forcefield = ForceField(*protein_ff)
        system = forcefield.createSystem(protein_modeller.topology,
                                        # nonbondedMethod=PME,
                                        # nonbondedCutoff=10*openmmunit.angstrom,
                                        switchDistance=9*openmmunit.angstrom,
                                        removeCMMotion=True,
                                        # constraints=HBonds,
                                        hydrogenMass=1.0*openmmunit.amu,
                                        soluteDielectric=1.0,
                                        solventDielectric=78.5)
        
        print("Minimizing hydrogen geometry...")
        _integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds
        )
        _simulation = Simulation(protein_modeller.topology, system, _integrator)
        _simulation.context.setPositions(protein_modeller.positions)
        _simulation.minimizeEnergy(maxIterations=500)
        positions = _simulation.context.getState(getPositions=True).getPositions()
        protein_modeller.positions = positions
        print("[Minimization done")

        # Save the cleaned receptor
        rec_name = pdb_file.split("/")[-1].split(".")[0]
        PDBFile.writeFile(protein_modeller.topology, protein_modeller.positions, open(f'{project_path}/system_cleaned.pdb', 'w'))

        # Retrieve all Force objects from the System
        forces = system.getForces()
        # Find the NonbondedForce object
        nonbonded_force = [f for f in forces if isinstance(f, NonbondedForce)][0]
        for idx, atom in enumerate(protein_atoms):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom.index)
            residue = atom.residue
            
            if residue.name in waters_residue_names:
                if atom.name == "O":
                    atom_type = "OW"
                else:
                    atom_type = "HW"
            else:
                atom_type = atom.name    
            chain = residue.chain
            chain_id = chr(ord('@')+ chain.index + 1)
            unique_id = f"{chain_id}:{residue.name}:{residue.index}"
            # print(unique_id)
            atom_id = f"{unique_id}:{atom_type}"
            coords = protein_positions[idx].in_units_of(openmmunit.angstrom)._value
            if atom_type == "HW":
                rmin_half_value = 0.0
                epsilon_value = 0.0
            else:
                rmin_half_value = rmin_half(sigma.in_units_of(openmmunit.angstrom))._value
                epsilon_value = epsilon.in_units_of(openmmunit.kilocalories_per_mole)._value
            charge_value = charge.in_units_of(openmmunit.elementary_charge)._value
            new_atom = mc.Atom(atom_type, atom_id, [coords.x, coords.y, coords.z], rmin_half_value, epsilon_value, charge_value)
            mcswell_atoms.append(new_atom)
    return mcswell_atoms

def characterize_small_molecule(small_molecules, small_molecule_forcefield, protein_ff=["amber14-all.xml", 
    "amber14/tip3p.xml"]):
    waters_residue_names = ["HOH", "WAT"]
    mcswell_atoms = list()
    molecules = list()
    forcefield = ForceField(*protein_ff)
    modeller = Modeller(Topology(), None)
    for molecule in small_molecules:
        rdkit_mol = SDMolSupplier(molecule)[0]
        small_mol = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
        # is_ion = _is_ion(rdkit_mol)
        # print(f"Is ION: {is_ion}")
        # if not is_ion:
        if small_molecule_forcefield == "espaloma":
            template_generator = EspalomaTemplateGenerator(
                molecules=small_mol, forcefield="espaloma-0.3.2"
            )
        elif small_molecule_forcefield.upper() == "SMIRNOFF":
            template_generator = SMIRNOFFTemplateGenerator(
                molecules=small_mol, forcefield="openff-1.2.0"
            )
        elif small_molecule_forcefield.upper() == "GAFF":
            template_generator = GAFFTemplateGenerator(
                molecules=small_mol, forcefield="gaff-2.11"
            )
        forcefield.registerTemplateGenerator(template_generator.generator)

        # make an OpenFF Topology of the ligand
        small_mol_off_topology = openff.toolkit.Topology.from_molecules(molecules=[small_mol])

        # convert it to an OpenMM Topology
        small_mol_topology = small_mol_off_topology.to_openmm()

        # get the positions of the ligand
        small_mol_positions = to_openmm(small_mol.conformers[0])
        
        for res in small_mol_topology.residues():
            res.name = "UNL"
        
        vectors = _build_box(small_mol_positions)
        modeller.topology.setPeriodicBoxVectors(vectors)
        modeller.add(small_mol_topology, small_mol_positions)

    print("Creating system...")
    system = forcefield.createSystem(modeller.topology,
                                    nonbondedMethod=PME,
                                    nonbondedCutoff=10*openmmunit.angstrom,
                                    switchDistance=9*openmmunit.angstrom,
                                    removeCMMotion=True,
                                    constraints=HBonds,
                                    hydrogenMass=1.0*openmmunit.amu)
    print("System created!")
    forces = system.getForces()
    small_mols_atoms = [atom for atom in modeller.topology.atoms() if atom.residue.name not in ["NA", "K", "CA"]]
    nonbonded_force = [f for f in forces if isinstance(f, NonbondedForce)][0]
    for idx, atom in enumerate(small_mols_atoms):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom.index)
        residue = atom.residue
        
        if residue.name in waters_residue_names:
            if atom.name == "O":
                atom_type = "OW"
            else:
                atom_type = "HW"
        else:
            atom_type = atom.name    
        chain = residue.chain
        chain_id = chr(ord('@')+ chain.index + 1)
        unique_id = f"{chain_id}:{residue.name}:{residue.index}"
        atom_id = f"{unique_id}:{atom_type}"
        coords = modeller.positions[idx].in_units_of(openmmunit.angstrom)._value
        if atom_type == "HW":
            rmin_half_value = 0.0
            epsilon_value = 0.0
        else:
            rmin_half_value = rmin_half(sigma.in_units_of(openmmunit.angstrom))._value
            epsilon_value = epsilon.in_units_of(openmmunit.kilocalories_per_mole)._value
        charge_value = charge.in_units_of(openmmunit.elementary_charge)._value
        new_atom = mc.Atom(atom_type, atom_id, [coords[0], coords[1], coords[2]], rmin_half_value, epsilon_value, charge_value)
        mcswell_atoms.append(new_atom)
    return mcswell_atoms

def get_data_from_openmm(receptors, project_path, small_molecules=None, small_molecule_forcefield=None):
    mcswell_atoms = list()
    if small_molecules is not None:
        print("Parametrizing small molecule..")
        atoms_small_mol = characterize_small_molecule(small_molecules, small_molecule_forcefield)
        for atom in atoms_small_mol:
            mcswell_atoms.append(atom)
    if receptors is not None:
        print("Parametrizing receptor..")
        atoms_receptor = characterize_receptor(receptors, project_path)
        for atom in atoms_receptor:
            mcswell_atoms.append(atom)
    return mcswell_atoms


def run_mcswell(config, project_path, n_gcmc_steps=None, energy_estimation_method="both"):
    #io
    # project_path = config["io"]["save_path"]
    os.makedirs(project_path, exist_ok=True)

    #receptor
    receptor_path = None
    if "receptor" in config:
        receptor_path = config["receptor"]["path"]
        print(receptor_path)

    #ligand
    small_molecule_path = None
    small_molecule_ff = None
    if "ligand" in config:
        small_molecule_path = config["ligand"]["small_molecule_path"]
        small_molecule_ff = config["ligand"]["small_molecule_forcefield"]

    #simulation_parameters
    n_snapshots = config["simulation_parameters"]["n_snapshots"]
    if n_gcmc_steps is None:
        n_gcmc_steps = config["simulation_parameters"]["n_gcmc_steps"]
    n_equilibration_steps = config["simulation_parameters"]["n_equilibration_steps"]
    distance_cutoff = config["simulation_parameters"]["distance_cutoff"]

    #mu_range
    mu_values = expand_ranges(config["mu_range"])
    
    #simulation_box
    spacing = config["simulation_box"]["spacing"]
    center_x = config["simulation_box"]["center_x"]
    center_y = config["simulation_box"]["center_y"]
    center_z = config["simulation_box"]["center_z"]
    x_size = config["simulation_box"]["x_size"]
    y_size = config["simulation_box"]["y_size"]
    z_size = config["simulation_box"]["z_size"]

    save_box_corners_pdb([center_x, center_y, center_z], [x_size, y_size, z_size], outfile=f"{project_path}/box.pdb")
    
    print("Parametrizing system with OpenMM...")
    parametrized_atoms = get_data_from_openmm(receptors=receptor_path, 
        project_path=project_path, 
        small_molecules=small_molecule_path, 
        small_molecule_forcefield=small_molecule_ff,)

    print("Starting MCSwell!")
    center = [center_x, center_y, center_z]
    pts = mc.make_insertion_points(
         parametrized_atoms,
         size=(x_size, y_size, z_size),
         spacing=spacing,
         center=center,
         max_distance=distance_cutoff,
         min_distance=1.5,
     )

    #gci
    gci_cfg = config.get("gci", {})

    def gci_pick(key, default):
        return gci_cfg.get(key, default)

    mu_bulk_override = gci_pick("mu_bulk", None)
    mu_bulk, mu_bulk_water_model, mu_bulk_source = resolve_mu_bulk(
        None if mu_bulk_override is None else float(mu_bulk_override)
    )

    start = time.time()
    print("Performing TITRATION and GRAND CANONICAL INTEGRATION")
    # Runs the GCMC titration and the requested free-energy analyses in one
    # C++ call, entirely in memory: no per-snapshot PDB is written to disk
    # (pass dump_debug_pdbs=True below for that, e.g. for visual inspection
    # in PyMOL/VMD -- it writes under project_path/mu_###/snap_#####.pdb).
    result = mc.run_mcswell_and_analyze(
        receptor_points=parametrized_atoms,
        boundaries=pts,
        distance_cutoff=distance_cutoff,
        spacing=spacing,
        gcmc_steps=n_gcmc_steps,
        equilibration_steps=n_equilibration_steps,
        save_path=project_path,
        mu_values=mu_values,
        n_snapshots=n_snapshots,
        temperature=float(gci_pick("temperature", 300.0)),
        mu_bulk=mu_bulk,
        box_center=[center_x, center_y, center_z],
        box_halfsize=[x_size / 2.0, y_size / 2.0, z_size / 2.0],
        run_binomial=energy_estimation_method in ("binomial", "both"),
        run_gci=energy_estimation_method in ("gci", "both"),
        peak_percentile=float(gci_pick("peak_percentile", 90.0)),
        capacity_filter=bool(gci_pick("capacity_filter", True)),
        bulk_water_density=float(gci_pick("bulk_water_density", 0.0334)),
        max_capacity_hit_fraction=float(gci_pick("max_capacity_hit_fraction", 0.01)),
        max_mean_capacity_fraction=float(gci_pick("max_mean_capacity_fraction", 0.90)),
        local_radius=float(gci_pick("local_radius", 1.4)),
        local_volume_mode=str(gci_pick("local_volume_mode", "sampler")),
        region_max_terms=int(gci_pick("region_max_terms", 12)),
        local_max_terms=int(gci_pick("local_max_terms", 4)),
        random_starts=int(gci_pick("random_starts", 64)),
        fit_seed=int(gci_pick("seed", 20260812)),
    )

    exec_time = time.time() - start
    print(f"Time necessary for the C++ part: {exec_time/60} minutes - {exec_time} seconds")
    print(
        f"[mu_bulk] resolved to {mu_bulk:.6f} kcal/mol "
        f"(water model: {mu_bulk_water_model or 'undetected'}, source: {mu_bulk_source})"
    )
    result["mu_bulk"] = mu_bulk
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run an MCSwell GCMC titration and its free-energy post-processing."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the MCSwell TOML configuration file.",
    )
    parser.add_argument(
        "--energy-estimation-method",
        dest="energy_estimation_method",
        choices=("gci", "binomial", "both"),
        default="both",
        help=(
            "Free-energy post-processing method to run: 'gci' (ProtoMS-style "
            "Grand Canonical Integration), 'binomial' (independent-site "
            "binomial MLE occupancy fit), or 'both' (default: runs both)."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(args.config)
    with open(args.config, "rb") as fi:
        config = tomllib.load(fi)
    project_path_base = config["io"]["save_path"]
    print("Starting")
    run_info = run_mcswell(
        config, project_path_base, energy_estimation_method=args.energy_estimation_method
    )

    # The titration + free-energy fits already ran (in C++, in memory) as
    # part of run_mcswell() above; this just draws the diagnostic plots
    # from the small CSVs it wrote.
    if run_info["binomial_output_dir"]:
        fe.main(run_info["binomial_output_dir"], mu_bulk=run_info["mu_bulk"])
    if run_info["gci_output_dir"]:
        fe_gci.main(run_info["gci_output_dir"])
