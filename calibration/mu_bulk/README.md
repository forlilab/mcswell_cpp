# μ_bulk calibration for MCSwell's energy function

Every MCSwell free energy is referenced to bulk water: ΔG_bind = μ½ − μ_bulk. The published
excess chemical potentials of TIP3P and TIP4P (−6.09 and −6.87 kcal/mol, Mahoney & Jorgensen
2000) were computed with periodic boundaries, Ewald sums and their own cutoffs. MCSwell uses
Coulomb without cutoff, O–O LJ to 12 Å, no tail correction and no periodic boundaries, so its
water is in equilibrium with bulk at a different μ. This folder calibrates μ_bulk for MCSwell's
Hamiltonian; the results are the defaults in `python/utils.py` (`MU_BULK_BY_WATER_MODEL`).

## Result

| model | 14 Å cube | 20 Å cube | 26 Å cube | **default (20 + 26 Å)** | literature |
|---|---|---|---|---|---|
| TIP3P | −6.07 ± 0.31 | −6.29 ± 0.05 | −6.31 ± 0.06 | **−6.30 ± 0.05** | −6.09 |
| TIP4P | −6.23 ± 0.10 | −6.12 ± 0.09 | −6.07 ± 0.04 | **−6.09 ± 0.07** | −6.87 |

kcal/mol, 300 K; ± is the standard deviation over snapshots, cube sizes and starts.
`results/` holds the per-run and summary tables and the plots (`n_vs_mu.png`: ⟨N⟩(μ) of every
run against its target; `mu_bulk_vs_size.png`: μ_bulk against the cube size).

- All 36 runs pass the convergence check (the two halves of production agree within
  0.1 kcal/mol), and empty and full starts agree within 0.03 kcal/mol.
- The 20 and 26 Å cubes agree, so the frozen shell's surface effect is small. The 14 Å cubes
  scatter by snapshot: their water counts differ from the mean-density target by up to 5 %, and
  1 % in density moves μ_bulk by about 0.08 kcal/mol. The 1/L extrapolation (the `inf` rows of
  `mu_bulk_summary.csv`) is driven by those points and goes in opposite directions for the two
  models, so it isn't used.
- Systematic error: the target density is OpenMM's NPT density (PME, dispersion correction), not
  the density of MCSwell's Hamiltonian. A 1 % density difference would move μ_bulk by about
  0.1 kcal/mol.
- Below about −6.7 kcal/mol the cube holds a vapour-like phase; it condenses abruptly between
  −7 and −6.5 kcal/mol, just below μ_bulk.

Changing μ_bulk shifts every ΔG by the same amount, so correlations and slopes are unchanged
and results from earlier runs are re-referenced without a rerun:
ΔG_new = ΔG_old + μ_bulk,old − μ_bulk,new. In the case study against RETI (45 matched waters,
GCI), the calibrated values move every TIP3P ΔG by +0.21 kcal/mol (mean signed error
−2.36 → −2.15) and every TIP4P ΔG by −0.78 (−1.86 → −2.64): most of TIP4P's apparent edge over
TIP3P came from the literature μ_bulk.

Recalibrate if the energy function changes (cutoffs, electrostatics, water parameters).

## Method

μ_bulk is the μ at which MCSwell's water is in equilibrium with bulk water at the model's own
density. MCSwell has no periodic boundaries, so bulk water is represented by a frozen snapshot:

1. `make_water_boxes.py` equilibrates a 52 Å box of water per model with OpenMM (NPT, 300 K,
   1 bar, PME; TIP3P from OpenMM's `tip3p.xml`, TIP4P from `tip4p_mcswell.xml`, the original
   TIP4P compiled into mcswell_cpp, which OpenMM doesn't ship). From 3 snapshots 200 ps apart it
   removes the waters in a central cube of 26, 20 or 14 Å and writes the rest, the shell, to
   `shells/` (13–19 Å thick, beyond the 12 Å LJ cutoff). `boxes.json` records each cube and its
   target count. Densities: TIP3P 0.983–0.985, TIP4P 0.992–0.993 g/cm³.
2. Each run titrates the empty cube (μ = −10 … −2 kcal/mol, step 0.25) with the shell held fixed
   (`receptor.fixed_waters`) and nothing else in the system, from an empty and from a pre-filled
   start: 2 models × 3 snapshots × 3 cubes × 2 starts = 36 runs, each 40M equilibration +
   200M production steps.
3. `analyze_mub.py`: μ_bulk is where ⟨N⟩(μ) in the cube reaches the bulk count (mean MD density ×
   cube volume), by linear interpolation. It is also computed from each half of production, and
   a run is flagged when the halves differ by more than max(0.1, 2 × error).

Sampling is the hard part: at 1/100 of these steps a 20 Å cube was still filling during
production and μ_bulk came out 1–2 kcal/mol too positive. The `full` start does not start from
bulk density (the prefill puts 100–160 waters in a 20 Å cube that holds ~260), so it is not an
upper bracket; its agreement with `empty` is a check on equilibration, not a bound.

## Reproducing

The shells (18 MB) are not in the repository; `make_water_boxes.py` makes new ones (new
snapshots, so expect results within the spread above). The water model is fixed at compile
time, so the TIP3P and TIP4P runs each need a build of the extension (`-DWATER_MODEL=TIP3P` /
`TIP4P`); `run_task.sh` checks the build before running.

```console
cd calibration/mu_bulk
python make_water_boxes.py --platform CUDA        # shells/ and boxes.json, ~5 min on a GPU
python make_configs_mub.py                        # configs/ and tasks.txt (36 runs)
PYTHON_TIP3P=<python with a TIP3P build> PYTHON_TIP4P=<python with a TIP4P build> \
    bash run_task.sh --line 1                     # one run; or every line of tasks.txt
python analyze_mub.py                             # results/
```

On a SLURM cluster, `submit_array_mps.slurm` runs three titrations per GPU under MPS
(`mkdir -p logs/slurm && sbatch submit_array_mps.slurm`; adapt the partition, modules and
environment activation). If the jobs see this folder under another path, pass it to
`make_configs_mub.py --root`. The 26 Å TIP4P runs are the slowest, about 2.5 h on one GPU
under MPS.

The published results came from mcswell_cpp `f645829` with the receptor options of this branch
(cluster runs, 2026-10-28).
