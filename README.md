# MCSwell

MCSwell is a new tool to predict hydration sites positions and thermodynamics
using Monte Carlo (MC) sampling. MCSwell is part of the Waterkit suite.

At the moment MCSwell is not supported on Windows.

- [Requirements](#requirements)
- [Installation](#installation)
- [Compilation](#compilation)
- [Configuration](#configuration)
- [Running MCSwell](#running-mcswell)

## Requirements

- NVIDIA GPU with a compute capability supported by your CUDA Toolkit (CUDA
  12.x supports Maxwell 5.0 and newer; CUDA 13.x dropped everything below
  Turing 7.5). The build queries the toolkit itself, so you do not need to
  list architectures manually — see
  [GPU architecture / CUDA compatibility](#gpu-architecture--cuda-compatibility).
- CUDA Toolkit >= 12.0 (enforced at configure time)
- GCC >= 7 (or any C++17-capable compiler)
- CMake >= 3.18
- Ninja build system
- Python >= 3.9
- Network access the first time you configure the build: CMake fetches
  [nanoflann](https://github.com/jlblancoc/nanoflann) (a small header-only
  KD-tree library used by the post-simulation hydration-site analysis) via
  `FetchContent`. It's cached under the build directory afterward, so
  subsequent configures/rebuilds don't need network access again.

## Installation

### 1. Install system packages

```console
# Update system
foo@bar:~$ sudo apt update && sudo apt upgrade -y

# Install GCC, CMake, Ninja
foo@bar:~$ sudo apt install -y gcc g++ cmake ninja-build

# Install NVIDIA CUDA Toolkit (>= 12.0)
foo@bar:~$ sudo apt install -y nvidia-cuda-toolkit
```

> **Note:** The `nvidia-cuda-toolkit` package version depends on your Ubuntu release.
> CUDA 12.0+ is required due to glibc compatibility with the `_FloatN` types used in
> newer system headers. Verify with `nvcc --version` after installation.
> If your distribution provides an older version, install CUDA from the
> [NVIDIA CUDA downloads page](https://developer.nvidia.com/cuda-downloads) instead.

### 2. Install Conda (miniconda)

```console
foo@bar:~$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
foo@bar:~$ bash Miniconda3-latest-Linux-x86_64.sh
# restart shell or source ~/.bashrc
```

### 3. Create environment & install Python dependencies

```console
foo@bar:~$ conda create -n mcswell python=3.11 -y && conda activate mcswell
(mcswell) foo@bar:~$ conda install -c conda-forge scikit-learn pandas scipy numpy matplotlib \
  openmm openmmforcefields openff-toolkit pdbfixer parmed \
  mdanalysis griddataformats mdtraj -y
(mcswell) foo@bar:~$ pip install scikit-build-core pybind11
```

### 4. Resolve the conda/system CUDA conflict

OpenMM pulls in conda CUDA packages (`cuda-nvcc`, etc.) that, if left as-is,
override the system `nvcc` and can cause the CUDA build to fail with obscure
glibc header errors (`_Float64`/`_Float128`/`issignaling`, seen on e.g. Ubuntu
24.04) — this was tracked down in
[issue #4](https://github.com/forlilab/mcswell_cpp/issues/4). Pick **one** of
the two options below.

#### Option A: remove the conda CUDA packages, build against the system toolkit

Simplest if you already have a working system-wide CUDA Toolkit
(see [Requirements](#requirements)).

```console
(mcswell) foo@bar:~$ conda remove --force cuda-nvcc cuda-cudart cuda-cudart-dev \
  cuda-driver-dev cuda-nvrtc cuda-nvrtc-dev cuda-profiler-api 2>/dev/null; true
(mcswell) foo@bar:~$ unset NVCC_PREPEND_FLAGS CXX CC
(mcswell) foo@bar:~$ ln -sf /usr/bin/strip "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-strip"
```

> **Note:** This does not affect OpenMM at runtime — it only uses the CUDA shared
> libraries already installed system-wide by the GPU driver.

Then confirm that `nvcc` now resolves to the **system** installation, not the
conda environment:

```console
(mcswell) foo@bar:~$ which nvcc
/usr/local/cuda/bin/nvcc
(mcswell) foo@bar:~$ nvcc --version
```

If `which nvcc` still points to `$CONDA_PREFIX/bin/nvcc`, explicitly set the
system CUDA toolkit before compiling:

```console
(mcswell) foo@bar:~$ export CUDA_HOME=/usr/local/cuda
(mcswell) foo@bar:~$ export CUDACXX=/usr/local/cuda/bin/nvcc
(mcswell) foo@bar:~$ export PATH=/usr/local/cuda/bin:$PATH
```

#### Option B: build entirely inside the conda/micromamba environment

No sudo and no system CUDA needed. Instead of removing conda's CUDA packages,
install a *complete, matched* CUDA toolchain directly into the same
environment, so `nvcc` and its headers stay fully self-contained and never
touch (or conflict with) the system's glibc. This never requires root and
works even on machines with no system-wide CUDA Toolkit at all.

**This only works if you get one thing right: `cuda-version` must not exceed
what your GPU driver actually supports**, or the build will succeed but
OpenMM will fail at runtime with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Check
the ceiling first:

```console
(mcswell) foo@bar:~$ nvidia-smi | head -3
# ... | Driver Version: 570.211.01   CUDA Version: 12.8 |
```

Use that reported version (`12.8` in this example, not higher) for
`cuda-version` below:

```console
(mcswell) foo@bar:~$ conda install -c conda-forge "cuda-version=12.8" \
  gcc_linux-64 gxx_linux-64 cuda-nvcc cuda-crt cuda-cudart-dev libcurand-dev -y
```

`gcc_linux-64`/`gxx_linux-64` matter here: they bring conda's own bundled,
version-matched sysroot, so `nvcc` compiles against *that* instead of falling
through to the system's (possibly newer, incompatible) glibc headers — that
fallback is exactly what causes the errors from issue #4. `cuda-crt` and
`cuda-cudart-dev`/`libcurand-dev` provide the headers the plain `cuda-nvcc`
metapackage doesn't include on its own.

The `mcswell_cpp` build will print a CMake warning that `nvcc` resolves into a
conda/mamba environment — that warning exists to flag the issue-#4 failure
mode; with this exact package set it's a known, harmless false positive.

## Compilation

To compile the C++ application with default settings (TIP3P ff):

```console
(mcswell) foo@bar:~$ cd /path/to/mcswell_cpp
(mcswell) foo@bar/mcswell_cpp:~$ pip install -e . -v --config-settings=cmake.args="--preset defaults"
```

To change the water model used, modify the field `WATER MODEL` in
`CMakePresets.json`. As of now only TIP3P and TIP4P water models are available.

### GPU architecture / CUDA compatibility

By default, MCSwell asks your CUDA Toolkit which architectures it supports
(`nvcc --list-gpu-arch`) and builds native (SASS) code for all of them, plus a
PTX ("virtual") fallback for the newest one. Thanks to PTX's forward
compatibility, the resulting build also runs on GPUs newer than anything the
toolkit knows about — the driver JIT-compiles the fallback PTX the first time
a kernel is launched (a one-time delay on first run, no rebuild needed).

Because the list is derived from the toolkit at configure time, upgrading CUDA
never leaves a stale, unsupported architecture behind. This matters because the
supported set changes between major versions: CUDA 13 removed Maxwell, Pascal
and Volta, so a hardcoded `50`/`60`/`70` (or CMake's `all` keyword, which
expands from CMake's own built-in table) makes even compiler detection fail
with:

```
nvcc fatal   : Unsupported gpu architecture 'compute_50'
```

If a value is supplied by your environment or command line, MCSwell validates
it against the toolkit and drops entries `nvcc` cannot compile, warning about
each one. The architectures actually used are printed at configure time:

```
-- MCSwell: CUDA architectures: 75;80;86;87;88;89;90;100;103;110;120;121;121-virtual
```

To shorten build times you can restrict the list explicitly (it is still
validated):

```console
(mcswell) foo@bar:~$ pip install -e . -v --config-settings=cmake.args="--preset defaults -DCMAKE_CUDA_ARCHITECTURES=90;100;120"
```

Or build only for the GPU(s) actually present on the machine:

```console
(mcswell) foo@bar:~$ pip install -e . -v --config-settings=cmake.args="--preset defaults -DCMAKE_CUDA_ARCHITECTURES=native"
```

The build fails fast with a clear error if your CUDA Toolkit is below the
required 12.0 floor, instead of failing later with a confusing compiler error.

> **Known issue:** some CUDA >= 12.0 installations (reported with CUDA 12.6
> on Ubuntu 24.04/GCC 13) fail during CMake's own compiler-detection step with
> `_Float64`/`_Float128`/`issignaling` errors from glibc headers. This has
> been verified to **not** reproduce with CUDA 12.0 on the same Ubuntu
> 24.04/glibc 2.39/GCC 13.3 combination, so it looks specific to certain CUDA
> point releases (or to a conda/pip-provided `nvcc` shadowing the system one —
> see [Option A](#option-a-remove-the-conda-cuda-packages-build-against-the-system-toolkit)).
> If you hit this, please open an issue with `which nvcc`, `nvcc --version`,
> and your full error log.

## Configuration

MCSwell is driven by a TOML config file; a working example lives in
`example/config.toml`.

### Top-level

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `title` | ✓ | string | non-empty | `"TOML configuration file for MCSwell"` |

### `[io]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `io.save_path` | ✓ | string | valid directory path | `"/data/mcswell_runs/3std_monomer_rep_0"` |
| `io.prepared_system` |  | string | file path | `"/data/mcswell_runs/prepared/3std.json"` |

`prepared_system` lets several runs of the same system share one OpenMM prep.
Write the file once with `--prepare-only` (see below). Every run whose config
names the same file then loads the parametrized atoms from it instead of
preparing the receptor and ligand again. This skips minutes of ligand charge
fitting per run, and every replicate and water model scores against exactly
the same atoms. A run refuses the file if its receptor or ligand files
(compared by content), ligand force field or minimization setting differ from
the ones the file was made with.

### `[receptor]` (optional)

Used when hydrating a **protein receptor**.
Omit this section for ligand-only hydration.

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `receptor.path` | ✓* | array[string] | ≥1 file; `.pdb`, `.cif`, `.mmcif` | `["3std_monomer_clean.pdb"]` |

\* Required **only if** `[receptor]` section is present.

### `[ligand]` (optional)

Used when hydrating a **small molecule ligand**.
Omit this section for receptor-only hydration.

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `ligand.small_molecule_path` | ✓* | array[string] | ≥1 file; `.sdf`, `.mol2` | `["3std_monomer_clean_ligand.sdf"]` |
| `ligand.small_molecule_forcefield` | ✓* | string | supported forcefield name | `"gaff"` |

\* Required **only if** `[ligand]` section is present.

### `[simulation_parameters]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `simulation_parameters.n_snapshots` | ✓ | int | ≥ 1 | `1000` |
| `simulation_parameters.n_equilibration_steps` | ✓ | int | 0 ≤ value ≤ `n_gcmc_steps` | `5000000` |
| `simulation_parameters.n_gcmc_steps` | ✓ | int | ≥ 1 | `50000000` |
| `simulation_parameters.distance_cutoff` | ✓ | float | > 0 Å | `9.0` |
| `simulation_parameters.seed` |  | int | GCMC master RNG seed, default `12345` | `20260901` |
| `simulation_parameters.n_diagnostic_windows` |  | int | ≥ 0, default `1000`; `0` disables | `1000` |
| `simulation_parameters.diagnostic_energy_stride` |  | int | ≥ 1, default `1` | `20` |
| `simulation_parameters.max_waters` |  | int | ≥ 0, default `0` (no limit) | `0` |
| `simulation_parameters.constrain_displacement` |  | bool | default `true` | `true` |
| `simulation_parameters.initial_fill` |  | string | `"auto"` (default), `"empty"`, `"full"` | `"auto"` |
| `simulation_parameters.max_rotation_deg` |  | float | (0, 180], default `30` | `30` |

`constrain_displacement` rejects displacement moves that would carry a water
out of the sampled region (the insertion lattice's voxels). Without it waters
drift out of the region and out of the box, where insertion can no longer
reach them: detailed balance breaks and the titration's N is inflated (in the
example, 17-38% of the waters ended up outside the box). Set it to `false`
only to reproduce runs made before it became the default.

`initial_fill` sets each µ window's starting state. `"auto"` starts the
windows above bulk µ pre-filled (bulk density, waters at least 2.8 Å apart on
random insertion points, random orientations) and the rest empty: an empty
start at high µ spends most of the run inserting waters one at a time, since
insertion is rarely accepted in dense water. `"empty"` and `"full"` apply one
start to every window; the stationary distribution is the same either way, so
running both and comparing ⟨N⟩ is a direct equilibration check.

`max_rotation_deg` caps the rotation a displacement move applies. Any value is
valid; full ±180° reorientations are almost always rejected in a
hydrogen-bonded network, so the 30° default keeps displacement acceptance
useful.

There is no limit on the number of waters per µ window. The water buffer
starts at the bulk-water estimate (insertion volume × 0.0334 Å⁻³) and doubles
whenever a window fills it, and snapshots are streamed from a fixed-size GPU
staging buffer to host memory, so GPU memory no longer grows with
`n_snapshots`. Growing the buffer does not perturb the chain: a run follows
exactly the chain the old fixed-buffer code did, for as long as that code would
not have refused an insertion. `max_waters` restores a hard limit: insertions
beyond it are refused and counted as capacity-blocked, and the `[gci]`
capacity filter then excludes the windows that ran into it.

Set `seed` explicitly when running replicates of the same system, so that the
Markov chains differ by design rather than by accident.

**`seed` alone does not make a run reproducible end to end.** It fixes the
GCMC chain, but the receptor prep that feeds it does not: PDBFixer's hydrogen
placement is unseeded, so two preps of the same input file give bit-identical
heavy atoms and *different* hydrogen positions (up to ~2.3 Å apart on freely
rotating groups). Since every receptor atom carries a partial charge the
sampler scores against, the insertion lattice and the energy landscape differ
between preps. To reproduce a run exactly, cache the prepped receptor and
replay it rather than re-prepping.

`n_diagnostic_windows` splits the run into equal windows and records, for each
(µ, window), the number of insertion/deletion/displacement moves proposed and
accepted, insertions blocked by `max_waters` (always 0 without one), the mean and final
water count, and the total water-subsystem energy. These go to
`acceptance_windows.csv` and `acceptance_summary.csv` under `save_path` (see
[Output](#output)). The counters draw no random numbers, so an instrumented
run follows exactly the same chain as an uninstrumented one with the same
seed.

The per-window energy is the only part that costs real work
(O(`n_waters` × `n_receptor_atoms`) per evaluation).
`diagnostic_energy_stride` evaluates it every *N*th window instead, which is
what makes a high-resolution occupancy trace affordable: e.g. 20000 windows
with `diagnostic_energy_stride = 20` resolves correlation times in steps while
keeping the energy cost at the 1000-window level. Skipped windows record `nan`
for the energy.

### `[mu_range]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `mu_range.start` | ✓ | float | `< stop` | `-38.0` |
| `mu_range.stop` | ✓ | float | `> start` | `-4.0` |
| `mu_range.step` | ✓ | float | > 0 | `1.0` |

Going much above bulk µ (−6.87 kcal/mol for TIP4P, −6.09 for TIP3P) buys
little: past roughly bulk + 3 the region keeps compressing water beyond bulk
density and does not equilibrate in a feasible run, and those densest windows
also set the run's wall time.

### `[openmm]` (optional)

Controls the OpenMM system-prep step only. The GCMC titration itself is a
separate CUDA kernel and is unaffected by these keys.

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `openmm.minimize` |  | bool | default `false` | `false` |
| `openmm.minimize_max_iterations` |  | int | default `500` | `500` |
| `openmm.platform` |  | string | `"auto"` (default), `CUDA`, `OpenCL`, `CPU`, `Reference` | `"auto"` |

Minimization is off by default: input structures are assumed to be already
equilibrated. Both keys can be overridden per-run from the command line with
`--minimize`/`--no-minimize` and `--openmm-platform`.

### `[gci]` (optional)

Every key has a default, so the whole section may be omitted.

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `gci.peak_percentile` |  | float | 0 < value ≤ 100, default `90.0` | `90.0` |
| `gci.mu_bulk` |  | float | kcal/mol | `-6.09` |
| `gci.temperature` |  | float | K, default `300.0` | `300.0` |
| `gci.capacity_filter` |  | bool | default `true`; only acts with `max_waters` set | `true` |
| `gci.max_capacity_hit_fraction` |  | float | 0–1, default `0.01` | `0.01` |
| `gci.max_mean_capacity_fraction` |  | float | 0–1, default `0.90` | `0.90` |
| `gci.local_radius` |  | float | Å, default `1.4` | `1.4` |
| `gci.local_volume_mode` |  | string | `"sampler"` (default), `"standard"`, or `"sphere"` | `"sampler"` |
| `gci.region_max_terms` |  | int | default `12` | `12` |
| `gci.local_max_terms` |  | int | default `4` | `4` |
| `gci.random_starts` |  | int | default `64` | `64` |
| `gci.seed` |  | int | fit RNG seed, default `20260812` | `20260812` |

`gci.mu_bulk` is optional: if omitted, it's auto-selected from the
compiled water model's published literature value (with a loud warning
-- see `utils.MU_BULK_BY_WATER_MODEL`), since that value was fit with a
different energy function/cutoff than MCSwell's own sampler.

### `[simulation_box]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `simulation_box.spacing` | ✓ | float | > 0 Å | `0.375` |
| `simulation_box.center_x` | ✓ | float | finite | `27.61085` |
| `simulation_box.center_y` | ✓ | float | finite | `10.1136` |
| `simulation_box.center_z` | ✓ | float | finite | `33.31735` |
| `simulation_box.x_size` | ✓ | float | > 0 Å | `29.1773` |
| `simulation_box.y_size` | ✓ | float | > 0 Å | `27.1828` |
| `simulation_box.z_size` | ✓ | float | > 0 Å | `30.4509` |

### Cross-field validation rules

| Rule |
|------|
| At least one of `[receptor]` or `[ligand]` must be present |
| `[receptor]` and `[ligand]` may both be present |
| `n_gcmc_steps >= n_equilibration_steps` |
| `mu_range.start < mu_range.stop` |
| `mu_range.step > 0` |
| `n_gcmc_steps - n_equilibration_steps >= n_snapshots` |
| `spacing > 0` |
| `x_size > 0`, `y_size > 0`, `z_size > 0` |

## Running MCSwell

```console
(mcswell) foo@bar:~$ python /path_to_mcswell_cpp/python/run_mcswell.py --config <path_to_the_config_file>
```

Or run the provided example:

```console
(mcswell) foo@bar:~$ python python/run_mcswell.py --config example/config.toml
```

### Command-line options

| Flag | Default | Effect |
|------|---------|--------|
| `--config` | — | Path to the TOML config file (required) |
| `--energy-estimation-method` | `both` | `gci`, `binomial`, or `both` |
| `--minimize` / `--no-minimize` | from `[openmm]` | Energy-minimize the prepped receptor, or skip it |
| `--openmm-platform` | from `[openmm]` | Platform for the prep step (`auto`, `CUDA`, `OpenCL`, `CPU`, `Reference`) |
| `--prepare-only` | off | Only run the OpenMM prep, write it to `io.prepared_system`, and stop |

By default a run performs the GCMC titration and both free-energy
post-processing methods (ProtoMS-style GCI and the independent-site binomial
fit). Restrict to a single method with `--energy-estimation-method`:

```console
(mcswell) foo@bar:~$ python /path_to_mcswell_cpp/python/run_mcswell.py \
    --config <path_to_the_config_file> \
    --energy-estimation-method gci
```

### Output

The GCMC titration and both free-energy analyses run entirely in memory:
no per-snapshot PDB is written to disk. Under `[io].save_path`, a run
writes only the final result artifacts:

```
<save_path>/
  box.pdb, system_cleaned.pdb        # pre-simulation setup, from OpenMM
  acceptance_summary.csv             # sampling diagnostics, per (mu, phase)
  acceptance_windows.csv             # sampling diagnostics, per (mu, window)
  acceptance_by_mu.png               # production acceptance rate per move vs mu
  n_waters_trace.png                 # water count over the run, equilibration check
  gci/<peak_percentile>_binomial/    # independent-site binomial fit
    sites.csv, sites.pdb, sites_all.csv, titration.csv,
    titration_diagnostics.csv, capacity_diagnostics.csv, gO.dx,
    binomial_metadata.json, titration_curves.png
  gci/<peak_percentile>_protoms_gci/ # ProtoMS-style GCI
    sites.csv, sites.pdb, region_titration.csv, local_titration.csv,
    region_gci_pmf.csv, region_gci_model.csv, local_gci_models.csv,
    gci_metadata.json, region_titration.png, local_titration_curves.png
```

#### Sampling diagnostics

`acceptance_summary.csv` has one row per (µ, phase) with phase in
`equilibration` / `production` / `total`: moves proposed and accepted for each
of the three move types, the resulting acceptance rates, insertions refused by
`max_waters` (always 0 without one), and the mean and final water count.
`acceptance_windows.csv` is the same breakdown resolved per window, plus the
water-subsystem energy, and is what a convergence analysis reads.

`run_mcswell.py` draws both tables (`python/plot_acceptance.py`, also usable
standalone as `python python/plot_acceptance.py <save_path> --mu-bulk <mu>`):

- `acceptance_by_mu.png` -- production-phase acceptance rate of each move type
  against µ on a log scale, with ⟨N⟩(µ) underneath for context. Moves proposed
  but never accepted sit in a separate "0%" row; moves never proposed (no water
  to delete or displace) are gaps, not zeros. Above the very dilute regime the
  insertion and deletion curves should coincide: at equilibrium both are
  proposed equally often and accepted equally often.
- `n_waters_trace.png` -- the water count over the whole run at six µ values
  spread across the titration (the one nearest bulk µ always among them), with
  the equilibration phase shaded and the production mean dashed. Each panel
  carries a verdict from `python/convergence.py`: equilibrated, stationary only
  after the configured equilibration (the detected point is drawn dotted), or
  still changing at the end of the run. Anything but "equilibrated" means that
  µ needs a longer equilibration phase or a longer run before its ⟨N⟩ can be
  trusted.

`python/convergence.py` turns that trace into the usual Markov-chain
diagnostics — statistical inefficiency, integrated autocorrelation time in MC
steps, effective sample size, a block-averaged standard error on ⟨N⟩, and an
automatically detected equilibration point (Chodera's reverse-cumulative
scan). It depends on nothing but numpy:

```python
from convergence import load_trace, analyse_run

trace = load_trace("<save_path>/acceptance_windows.csv")
for r in analyse_run(trace):
    print(r.mu_value, r.mean_n_waters, r.sem_n_waters,
          r.autocorrelation_steps, r.n_effective_samples)
```

Because the trace holds window *means*, correlations shorter than one window
are averaged away before they can be measured; `autocorrelation_resolved`
flags the µ values where that happened, and the fix is to re-record with more
windows (and a larger `diagnostic_energy_stride` to keep it cheap).
`autocorrelation_reliable` flags the opposite problem — a production phase
spanning too few correlation times for `g` to be estimated at all, in which
case `second_half_drift_fraction` (which assumes nothing about the correlation
structure) is the statistic to trust.

Self-checks for these estimators, against series whose statistical
inefficiency is known analytically:

```bash
python python/test_convergence.py
```

The C++ side, including the diagnostics tables, is covered by the analysis
unit tests:

```bash
cmake -S . -B build -DMCSWELL_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
```

If you want per-snapshot PDBs for visual inspection in PyMOL/VMD (e.g.
while debugging), pass `dump_debug_pdbs=True` to `mc.run_mcswell_and_analyze`
(or the lower-level `mc.run_mcswell`) if calling the Python module directly;
they're written under `<save_path>/mu_###/snap_#####.pdb`. This isn't
exposed as a `run_mcswell.py` CLI flag, since it's meant for one-off
debugging, not routine runs.
