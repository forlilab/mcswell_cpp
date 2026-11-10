#!/usr/bin/env python
"""Equilibrated water boxes for the mu_bulk calibration: a fixed shell around an empty cube.

MCSwell has no periodic boundaries, so bulk water is represented by a frozen
snapshot of equilibrated water. For each water model, a 52 A box of water is equilibrated
with OpenMM (NPT, 300 K, 1 bar, PME, rigid water; TIP3P from OpenMM's tip3p.xml, TIP4P
from tip4p_mcswell.xml, the original TIP4P compiled into mcswell_cpp). Three snapshots,
200 ps apart, are each cut, for a 26, 20 and 14 A central cube, into:

  shells/<model>_box<k>_r<size>_shell.pdb   every water whose O lies outside the central cube
                                            (13 / 16 / 19 A of water around it, beyond the 12 A LJ
                                            cutoff), held fixed in the titration (receptor.fixed_waters)
  the central cube                          emptied; the titration samples it

Three cube sizes, because the frozen shell is not bulk water: waters at its surface may bind
better or worse than in bulk, so mu_bulk is checked against the cube size (in production the
20 and 26 A cubes agree; see README.md).

boxes.json records for each snapshot and cube size the cube's centre and size and n_target,
the number of waters whose O was inside the cube. mu_bulk is the mu at which the titration holds
n_target waters in the cube, i.e. the chemical potential at which MCSwell's water is in
equilibrium with bulk water at the model's own density.

    python make_water_boxes.py [--platform CUDA]          # about 5 minutes on a GPU
"""

import argparse
import json
import os

import numpy as np
import openmm
import openmm.unit as u
from openmm.app import PME, ForceField, Modeller, PDBFile, Simulation, Topology

HERE = os.path.dirname(os.path.abspath(__file__))
BOX_NM = 5.2
REGIONS_A = (26.0, 20.0, 14.0)
N_SNAPSHOTS = 3
EQUIL_PS = 500
SPACING_PS = 200
STEP_FS = 2.0
MODELS = {
    "tip3p": (["tip3p.xml"], "tip3p"),
    "tip4p": ([os.path.join(HERE, "tip4p_mcswell.xml")], "tip4pew"),   # tip4pew: 4-site template for addSolvent
}


def write_shell(path, topology, pos_A, keep):
    with open(path, "w") as fo:
        n = 1
        for k, res in enumerate(r for r in topology.residues() if r.name == "HOH"):
            if not keep[k]:
                continue
            atoms = {a.name: a.index for a in res.atoms()}
            resnum = n // 3 + 1
            if resnum > 9999:
                raise ValueError("more than 9999 shell waters: PDB numbering would repeat")
            for name in ("O", "H1", "H2"):
                x, y, z = pos_A[atoms[name]]
                fo.write(f"HETATM{n % 100000:5d} {name:<4} HOH W{resnum:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}\n")
                n += 1
        fo.write("END\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--platform", default=None, help="OpenMM platform (default: fastest available)")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "shells"), exist_ok=True)
    out = {}

    for k_model, (model, (ff_files, solvent_model)) in enumerate(MODELS.items()):
        print(f"== {model}")
        ff = ForceField(*ff_files)
        modeller = Modeller(Topology(), [])
        modeller.addSolvent(ff, model=solvent_model, boxSize=openmm.Vec3(BOX_NM, BOX_NM, BOX_NM) * u.nanometer)
        n_water = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
        system = ff.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * u.nanometer,
                                 rigidWater=True, constraints=None)
        for f in system.getForces():
            if isinstance(f, openmm.NonbondedForce):
                f.setUseDispersionCorrection(True)
        system.addForce(openmm.MonteCarloBarostat(1 * u.bar, 300 * u.kelvin, 25))
        integrator = openmm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, STEP_FS * u.femtosecond)
        integrator.setRandomNumberSeed(1234 + k_model)
        platform = openmm.Platform.getPlatformByName(args.platform) if args.platform else None
        sim = Simulation(modeller.topology, system, integrator, platform)
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(300 * u.kelvin, 4321 + k_model)
        steps = lambda ps: int(round(ps * 1000 / STEP_FS))
        print(f"  {n_water} waters, platform {sim.context.getPlatform().getName()}; equilibrating {EQUIL_PS} ps")
        sim.step(steps(EQUIL_PS))

        mass_g = sum(system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())) \
            / 6.02214076e23
        snaps = []
        for k in range(N_SNAPSHOTS):
            densities = []
            for _ in range(20):
                sim.step(steps(SPACING_PS) // 20)
                v = sim.context.getState().getPeriodicBoxVolume().value_in_unit(u.nanometer ** 3) * 1e-21
                densities.append(mass_g / v)
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            box_A = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.angstrom).diagonal()
            pos_A = state.getPositions(asNumpy=True).value_in_unit(u.angstrom)
            centre = box_A / 2
            o_xyz = np.array([pos_A[next(a.index for a in r.atoms() if a.name == "O")]
                              for r in modeller.topology.residues() if r.name == "HOH"])
            print(f"  snapshot {k}: box {box_A[0]:.2f} A, density {np.mean(densities):.4f} g/cm3")
            for region in REGIONS_A:
                inside = np.all(np.abs(o_xyz - centre) <= region / 2, axis=1)
                path = os.path.join(HERE, "shells", f"{model}_box{k}_r{region:g}_shell.pdb")
                write_shell(path, modeller.topology, pos_A, ~inside)
                snaps.append({"snapshot": k, "shell": os.path.relpath(path, HERE),
                              "box_A": [float(x) for x in box_A], "center_A": [float(x) for x in centre],
                              "region_A": region, "n_target": int(inside.sum()),
                              "n_shell": int((~inside).sum()),
                              "density_g_cm3_last_200ps": float(np.mean(densities))})
                print(f"    {region:g} A cube: {inside.sum()} waters in the cube, {(~inside).sum()} in the shell")
        out[model] = snaps

    with open(os.path.join(HERE, "boxes.json"), "w") as fo:
        json.dump(out, fo, indent=1)
    print(f"Wrote {os.path.join(HERE, 'boxes.json')}")


if __name__ == "__main__":
    main()
