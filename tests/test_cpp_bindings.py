import mcswell_cpp as mc
import io
import numpy as np
import time
import os
import tomllib
from utils import *
from rdkit import Chem 
from rdkit.Chem import SDMolSupplier
from openmm.app import *
from openmm import *
import openmm.unit as openmmunit
import pdbfixer

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
        "amber14/tip3p.xml"]):
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
                                        nonbondedMethod=PME,
                                        nonbondedCutoff=10*openmmunit.angstrom,
                                        switchDistance=9*openmmunit.angstrom,
                                        removeCMMotion=True,
                                        constraints=HBonds,
                                        hydrogenMass=1.0*openmmunit.amu)
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

def get_data_from_openmm(receptors, project_path, small_molecules=None, small_molecule_forcefield=None):
    mcswell_atoms = list()
    if receptors is not None:
        print("Parametrizing receptor..")
        atoms_receptor = characterize_receptor(receptors, project_path)
        for atom in atoms_receptor:
            mcswell_atoms.append(atom)
    return mcswell_atoms


def run_mcswell(config, project_path, n_gcmc_steps=None):
    titration = True
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
    # flat = pts.astype(np.float32, copy=False).ravel()
    start = time.time()
    save_path = f"{project_path}/frames/"
    os.makedirs(save_path, exist_ok=True)
    print("Performing TITRATION and GRAND CANONICAL INTEGRATION")    
    # mc.run_mcswell_gpu_titration(
    #     parametrized_atoms, 
    #     pts,
    #     distance_cutoff,
    #     spacing,
    #     n_gcmc_steps,
    #     n_equilibration_steps, 
    #     save_path, 
    #     mu_values,
    #     n_snapshots)
    mc.run_mcswell_dummy(
        parametrized_atoms, 
        pts,
        distance_cutoff,
        spacing,
        n_gcmc_steps,
        n_equilibration_steps, 
        save_path, 
        mu_values,
        n_snapshots)

    exec_time = time.time() - start
    print(f"Time necessary for the rust part: {exec_time/60} minutes - {exec_time} seconds")
    return


if __name__ == "__main__":
    # print("HELLO!")
    config_path = "/data/MCSwell_C++/tests/config.toml"
    print(config_path)
    with open(config_path, "rb") as fi:
        config = tomllib.load(fi)
    project_path_base = config["io"]["save_path"]
    print("Starting")
    run_mcswell(config, project_path_base)
