# MCSwell
MCSwell is a new tool to predict hydration sites positions and thermodynamics using Monte Carlo (MC) sampling. MCSwell is part of the Waterkit suite.

At the moment MCSwell is not supported on Windows.

# Requirements
- NVIDIA GPU with CUDA Compute Capability >= 3.5
- CUDA Toolkit >= 11.0 (required for C++17 support)
- GCC >= 7 (or any C++17-capable compiler)
- CMake >= 3.18
- Ninja build system
- Python >= 3.9

# Installation
1. Install system packages

```console
# Update system
foo@bar:~$ sudo apt update && sudo apt upgrade -y
```

```console
# Install GCC, CMake, Ninja
foo@bar:~$ sudo apt install -y gcc g++ cmake ninja-build
```

```console
# Install NVIDIA CUDA Toolkit (>= 11.0)
foo@bar:~$ sudo apt install -y nvidia-cuda-toolkit
```

> **Note:** The `nvidia-cuda-toolkit` package version depends on your Ubuntu release.
> On Ubuntu 20.04+ this provides CUDA 11+. Verify with `nvcc --version` after installation.
> If your distribution provides an older version, install CUDA from the
> [NVIDIA CUDA downloads page](https://developer.nvidia.com/cuda-downloads) instead.

2. Install Conda (miniconda)
```console
foo@bar:~$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
foo@bar:~$ bash Miniconda3-latest-Linux-x86_64.sh
# restart shell or source ~/.bashrc
```
3. Create environment & install Python dependencies

```console
foo@bar:~$ conda create -n mcswell python=3.11 -y && conda activate mcswell
(mcswell) foo@bar:~$ conda install -c conda-forge scikit-learn pandas scipy numpy matplotlib \
  openmm openmmforcefields openff-toolkit pdbfixer parmed \
  mdanalysis griddataformats mdtraj -y
(mcswell) foo@bar:~$ pip install scikit-build-core pybind11
```

# Compilation
To compile the C++ application with default settings (TIP3P ff):

```console
(mcswell) foo@bar:~$ cd /path/to/mcswell_cpp
(mcswell) foo@bar/mcswell_cpp:~$ pip install -e . -v --config-settings=cmake.args="--preset default"
```

To change the water model used:
 modify the field "WATER MODEL" in CMakePresets.json

As of now only TIP3P, TIP3FB and TIP4P water models are available.

# Config file
You can find an example of the configuration file in tests (config.toml)

## Configuration schema

### Top-level

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `title` | ✓ | string | non-empty | `"TOML configuration file for MCSwell"` |

---

### `[io]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `io.save_path` | ✓ | string | valid directory path | `"/data/phd/mcswell_case_study/mcswell_gci/scytalone/3std_monomer_clean_rep_0"` |

---

### `[receptor]` (optional)

Used when hydrating a **protein receptor**.  
Omit this section for ligand-only hydration.

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `receptor.path` | ✓* | array[string] | ≥1 file; `.pdb`, `.cif`, `.mmcif` | `["3std_monomer_clean.pdb"]` |

\* Required **only if** `[receptor]` section is present.

---

### `[ligand]` (optional)

Used when hydrating a **small molecule ligand**.  
Omit this section for receptor-only hydration.

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `ligand.small_molecule_path` | ✓* | array[string] | ≥1 file; `.sdf`, `.mol2` | `["3std_monomer_clean_ligand.sdf"]` |
| `ligand.small_molecule_forcefield` | ✓* | string | supported forcefield name | `"gaff"` |

\* Required **only if** `[ligand]` section is present.

---

### `[simulation_parameters]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `simulation_parameters.n_snapshots` | ✓ | int | ≥ 1 | `1000` |
| `simulation_parameters.n_equilibration_steps` | ✓ | int | 0 ≤ value ≤ `n_gcmc_steps` | `5000000` |
| `simulation_parameters.n_gcmc_steps` | ✓ | int | ≥ 1 | `50000000` |
| `simulation_parameters.distance_cutoff` | ✓ | float | > 0 Å | `9.0` |

---

### `[mu_range]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `mu_range.start` | ✓ | float | `< stop` | `-38.0` |
| `mu_range.stop` | ✓ | float | `> start` | `2.0` |
| `mu_range.step` | ✓ | float | > 0 | `1.0` |

---

### `[gci]`

| Key | Req | Type | Constraints | Example |
|-----|:---:|------|-------------|---------|
| `gci.peak_percentile` | ✓ | float | 0 < value ≤ 100 | `90.0` |

---

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

---

### Cross-field validation rules

| Rule |
|------|
| At least one of `[receptor]` or `[ligand]` must be present |
| `[receptor]` and `[ligand]` may both be present |
| `n_gcmc_steps >= n_equilibration_steps` |
| `mu_range.start < mu_range.stop` |
| `mu_range.step > 0` |
| `spacing > 0` |
| `x_size > 0`, `y_size > 0`, `z_size > 0` |



# Execution
To run MCSwell:

```console
(mcswell) foo@bar:~$ python /path_to_mcswell_cpp/python/run_mcswell.py <path_to_the_config_file>
```

You can run the example provided in the `example` folder:

```console
(mcswell) foo@bar:~$ python python/run_mcswell.py example/config.toml
```
