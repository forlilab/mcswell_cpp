#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""
MCSwell ProtoMS-style Grand Canonical Integration (GCI) analysis
================================================================

This module can be called directly from ``run_mcswell.py`` via
``main(config)`` and can also be used as a standalone post-processing script
for existing MCSwell titration snapshots laid out as:

    <save_path>/frames/
        mu_000/snap_00000.pdb
        mu_000/snap_00001.pdb
        ...
        mu_001/snap_00000.pdb
        ...

It implements two related analyses:

1) Whole-region GCI
   - Uses the total number of GCMC waters in each snapshot, <N>(B).
   - Reconstructs the Adams parameter used by the current MCSwell sampler:

         B = mu / kT + ln(V_sampler / V0)

   - Fits a monotone sum-of-logistics titration model.
   - Evaluates a ProtoMS-style insertion PMF and bulk-referenced binding PMF.

2) Local/site GCI
   - Identifies MCSwell hydration-site centers from the same density-peak
     procedure used by the original analysis.
   - Counts the *mean number of waters* in a sphere around each site for every
     chemical potential (default radius = 1.4 A). Counts are NOT capped at 1.
   - Fits the same monotone titration model and calculates a first-water GCI
     free-energy score for each site.

3) Capacity-safety filtering
   - Reconstructs the current MCSwell water-buffer capacity from the sampler
     volume and BULK_WATER_DENSITY.
   - Detects the first chemical-potential window that is too close to the hard
     water-buffer ceiling.
   - Uses only the contiguous low-mu prefix before that point for density-site
     detection and GCI fitting.
   - Still writes every sampled window to the titration CSVs, with a
     ``valid_for_gci`` flag, so excluded high-mu data remain visible.

IMPORTANT SCIENTIFIC NOTE
-------------------------
Whole-region GCI is the direct analogue of standard ProtoMS GCI because the
observable is the total particle count in the sampled grand-canonical region.
Post-hoc local/site GCI is a decomposition/score rather than a rigorous
independent grand-canonical free energy unless insertion/deletion moves are
actually restricted to the local site volume.

The default local standard-state correction uses the *sampler volume*. This
makes the local score invariant to the arbitrary additive ln(V_sampler/V0)
term in B and recovers the expected single-site free energy in the independent
site limit. For comparison with the legacy ProtoMS local-cluster convention,
use:

    --local-volume-mode standard

which sets the local analysis volume equal to V0 so that the volume correction
is zero.

Dependencies
------------
numpy, scipy, matplotlib, tomli/tomllib

Optional for automatic reconstruction of the sampler volume:
MCSwell Python environment (mcswell_cpp + run_mcswell.py dependencies).
You may instead supply the exact volume printed by the original run:

    --sampler-volume <A^3>

or the insertion-point count:

    --n-insertion-points <N>

Example
-------
python estimate_free_energies_gci.py /path/to/mcswell_config.toml

Recommended if you know the original sampler volume from the MCSwell log:
python estimate_free_energies_gci.py config.toml --sampler-volume 12345.678

Outputs are written to:
    <save_path>/gci/<peak_percentile>_protoms_gci/

Author of this analysis implementation: generated for MCSwell post-processing.
It is intentionally written from scratch around the published GCI equations
and MCSwell's current sampler conventions.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares, brentq
from scipy.spatial import cKDTree
from scipy.special import expit

import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tomllib  # Python >= 3.11
except ImportError:  # pragma: no cover
    import tomli as tomllib

from utils import resolve_mu_bulk


# -----------------------------------------------------------------------------
# Constants matching the current MCSwell CUDA sampler
# -----------------------------------------------------------------------------
# include/cuda/consts.cuh in the current MCSwell repository:
#   KT              = 0.59616123 kcal/mol
#   STANDARD_VOLUME = 29.914 A^3
MCSWELL_KT = 0.59616123
MCSWELL_STANDARD_VOLUME = 29.914
DEFAULT_TEMPERATURE = 300.0
# Bulk-water excess chemical potential is water-model dependent and is
# resolved at runtime via utils.resolve_mu_bulk() from the compiled
# mcswell_cpp water model (utils.MU_BULK_BY_WATER_MODEL), unless overridden
# via --bulk-mu / gci.bulk_mu.

# Density/site detection defaults from the original MCSwell analysis.
GRID_SPACING = 0.5       # A
GRID_SIGMA = 1.4         # Gaussian sigma in voxels
PEAK_PERCENTILE = 90.0
PEAK_MERGE_CUTOFF = 1.4  # A
DEFAULT_LOCAL_RADIUS = 1.4  # A, ProtoMS local-GCI convention
MCSWELL_BULK_WATER_DENSITY = 0.0334  # waters / A^3; current MCSwell buffer sizing
DEFAULT_MAX_CAPACITY_HIT_FRACTION = 0.01
DEFAULT_MAX_MEAN_CAPACITY_FRACTION = 0.90

WATER_RESNAMES = ("HOH", "WAT", "H2O", "TIP3", "SOL", "WA1")
WATER_O_NAMES = ("O", "OW", "O1", "OH2", "O00")


# =============================================================================
# Configuration and file helpers
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def expand_mu_range(cfg: dict) -> np.ndarray:
    """Match MCSwell utils.expand_ranges for config['mu_range']."""
    r = cfg["mu_range"]
    return np.unique(
        np.arange(float(r["start"]), float(r["stop"]) + 1e-8, float(r["step"]))
    )


def get_gc_box_from_config(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    sb = cfg["simulation_box"]
    center = np.array(
        [sb["center_x"], sb["center_y"], sb["center_z"]], dtype=float
    )
    halfsize = np.array(
        [sb["x_size"] / 2.0, sb["y_size"] / 2.0, sb["z_size"] / 2.0],
        dtype=float,
    )
    return center, halfsize


def detect_mu_and_frames(frames_dir: str) -> Tuple[List[int], Dict[int, List[int]]]:
    mu_pat = re.compile(r"^mu_(\d+)$")
    snap_pat = re.compile(r"^snap_(\d+)\.pdb$")

    if not os.path.isdir(frames_dir):
        raise RuntimeError(f"Frames directory does not exist: {frames_dir}")

    frames_per_mu: Dict[int, List[int]] = {}
    for entry in os.listdir(frames_dir):
        mm = mu_pat.match(entry)
        if not mm:
            continue
        mu_idx = int(mm.group(1))
        mu_dir = os.path.join(frames_dir, entry)
        if not os.path.isdir(mu_dir):
            continue
        ids = []
        for fn in os.listdir(mu_dir):
            sm = snap_pat.match(fn)
            if sm:
                ids.append(int(sm.group(1)))
        ids.sort()
        if ids:
            frames_per_mu[mu_idx] = ids

    if not frames_per_mu:
        raise RuntimeError(
            f"No mu_### folders containing snap_#####.pdb were found in {frames_dir}"
        )

    return sorted(frames_per_mu), frames_per_mu


def list_frames_for_mu(frames_dir: str, mu_idx: int) -> List[str]:
    mu_dir = os.path.join(frames_dir, f"mu_{mu_idx:03d}")
    paths = glob.glob(os.path.join(mu_dir, "snap_*.pdb"))

    def key(path: str) -> int:
        m = re.search(r"snap_(\d+)\.pdb$", os.path.basename(path))
        return int(m.group(1)) if m else 10**12

    paths.sort(key=key)
    return paths


def read_water_oxygen_coords_from_pdb(path: str) -> np.ndarray:
    """Read GCMC water oxygen coordinates from one snapshot PDB."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    coords = []
    with open(path, "r") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom = line[12:16].strip()
            resn = line[17:20].strip()
            if atom not in WATER_O_NAMES or resn not in WATER_RESNAMES:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))

    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


# =============================================================================
# Reconstruct the exact GCMC sampler volume used by MCSwell
# =============================================================================

def _count_insertion_points(points) -> int:
    arr = np.asarray(points)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return int(arr.shape[0])
    if arr.size % 3 != 0:
        raise RuntimeError(
            f"Could not infer insertion-point count from shape {arr.shape}."
        )
    return int(arr.size // 3)


def reconstruct_sampler_volume(cfg: dict, work_dir: str) -> Tuple[float, int]:
    """
    Rebuild insertion points using the same Python/C++ functions as run_mcswell.py.

    MCSwell C++ uses:
        V_sampler = n_insertion_points * spacing^3

    This path may run OpenMM receptor/ligand parameterization. If the original
    log's total_volume is available, --sampler-volume is preferred because it
    is the exact value used in that run.
    """
    try:
        import mcswell_cpp as mc
        import run_mcswell as runner
    except Exception as exc:
        raise RuntimeError(
            "Automatic sampler-volume reconstruction requires running this script "
            "inside the MCSwell Python environment. Alternatively supply "
            "--sampler-volume or --n-insertion-points."
        ) from exc

    os.makedirs(work_dir, exist_ok=True)

    receptor_path = cfg.get("receptor", {}).get("path")
    ligand_cfg = cfg.get("ligand", {})
    ligand_path = ligand_cfg.get("small_molecule_path")
    ligand_ff = ligand_cfg.get("small_molecule_forcefield")

    sb = cfg["simulation_box"]
    sim = cfg["simulation_parameters"]
    spacing = float(sb["spacing"])
    center = [float(sb["center_x"]), float(sb["center_y"]), float(sb["center_z"])]
    size = (float(sb["x_size"]), float(sb["y_size"]), float(sb["z_size"]))
    distance_cutoff = float(sim["distance_cutoff"])

    print("[volume] Reconstructing MCSwell insertion cloud...")
    parametrized_atoms = runner.get_data_from_openmm(
        receptors=receptor_path,
        project_path=work_dir,
        small_molecules=ligand_path,
        small_molecule_forcefield=ligand_ff,
    )

    points = mc.make_insertion_points(
        parametrized_atoms,
        size=size,
        spacing=spacing,
        center=center,
        max_distance=distance_cutoff,
        min_distance=1.5,
    )
    n_points = _count_insertion_points(points)
    volume = n_points * spacing**3
    return float(volume), int(n_points)


def resolve_sampler_volume(
    cfg: dict,
    out_dir: str,
    sampler_volume: Optional[float],
    n_insertion_points: Optional[int],
) -> Tuple[float, Optional[int], str]:
    spacing = float(cfg["simulation_box"]["spacing"])

    if sampler_volume is not None:
        if sampler_volume <= 0:
            raise ValueError("--sampler-volume must be > 0")
        return float(sampler_volume), None, "cli_sampler_volume"

    if n_insertion_points is not None:
        if n_insertion_points <= 0:
            raise ValueError("--n-insertion-points must be > 0")
        volume = int(n_insertion_points) * spacing**3
        return float(volume), int(n_insertion_points), "cli_n_insertion_points"

    work = os.path.join(out_dir, "_volume_reconstruction")
    volume, npts = reconstruct_sampler_volume(cfg, work)
    return volume, npts, "reconstructed_from_mcswell"


# =============================================================================
# Density grid and hydration-site detection (same logic as original analysis)
# =============================================================================

def make_edges(center: np.ndarray, halfsize: np.ndarray, spacing: float):
    lo = center - halfsize
    hi = center + halfsize
    return [
        np.linspace(lo[d], hi[d], int(np.ceil(2 * halfsize[d] / spacing)) + 1)
        for d in range(3)
    ]


def collect_all_points(
    frames_dir: str,
    mu_indices: Sequence[int],
    frames_per_mu: Dict[int, List[int]],
) -> np.ndarray:
    pts = []
    total_files = sum(len(frames_per_mu[i]) for i in mu_indices)
    done = 0
    for mu_idx in mu_indices:
        mu_dir = os.path.join(frames_dir, f"mu_{mu_idx:03d}")
        for fr in frames_per_mu[mu_idx]:
            path = os.path.join(mu_dir, f"snap_{fr:05d}.pdb")
            xyz = read_water_oxygen_coords_from_pdb(path)
            if xyz.size:
                pts.append(xyz)
            done += 1
        print(f"[density] mu_{mu_idx:03d}: read {len(frames_per_mu[mu_idx])} frames "
              f"({done}/{total_files})")

    if not pts:
        raise RuntimeError("No water oxygen coordinates were found in the snapshots.")
    return np.vstack(pts)



def scan_capacity_and_collect_density_points(
    frames_dir: str,
    mu_values: np.ndarray,
    frames_per_mu: Dict[int, List[int]],
    water_capacity: int,
    *,
    capacity_filter: bool = True,
    max_capacity_hit_fraction: float = DEFAULT_MAX_CAPACITY_HIT_FRACTION,
    max_mean_capacity_fraction: float = DEFAULT_MAX_MEAN_CAPACITY_FRACTION,
):
    """
    Scan each mu window once to diagnose the MCSwell water-buffer ceiling and
    collect water oxygens for density-site detection.

    The current sampler allocates a finite water buffer. Once a window begins
    to interact appreciably with that ceiling, later/higher-mu windows are not
    used for GCI fitting. We deliberately use a *contiguous low-mu prefix*:
    after the first capacity-limited window, all subsequent windows are
    excluded even if a later window happens to fall below the threshold.

    A window triggers truncation if either:
      1) more than ``max_capacity_hit_fraction`` of saved frames are exactly at
         the reconstructed water capacity, OR
      2) the mean region population is at least
         ``max_mean_capacity_fraction * water_capacity``.

    Returns
    -------
    density_points : ndarray, shape (n, 3)
        Water oxygen coordinates from GCI-valid windows only.
    diagnostics : list[dict]
        Per-window capacity diagnostics.
    valid_mask : ndarray[bool]
        Contiguous prefix used for site detection and all GCI fits.
    """
    if water_capacity <= 0:
        raise ValueError("water_capacity must be positive")
    if not (0.0 <= max_capacity_hit_fraction <= 1.0):
        raise ValueError("max_capacity_hit_fraction must be in [0, 1]")
    if not (0.0 < max_mean_capacity_fraction <= 1.0):
        raise ValueError("max_mean_capacity_fraction must be in (0, 1]")

    n_mu = len(mu_values)
    valid_mask = np.ones(n_mu, dtype=bool)
    diagnostics = []
    density_pts = []
    prefix_open = True

    for mi in range(n_mu):
        mu_dir = os.path.join(frames_dir, f"mu_{mi:03d}")
        frame_ids = frames_per_mu.get(mi, [])
        if not frame_ids:
            raise RuntimeError(f"No saved frames found for mu_{mi:03d}")

        counts = np.zeros(len(frame_ids), dtype=int)
        this_mu_pts = []

        for fi, fr in enumerate(frame_ids):
            path = os.path.join(mu_dir, f"snap_{fr:05d}.pdb")
            xyz = read_water_oxygen_coords_from_pdb(path)
            counts[fi] = int(xyz.shape[0])
            if xyz.size:
                this_mu_pts.append(xyz)

        mean_n = float(np.mean(counts))
        std_n = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
        max_n = int(np.max(counts))
        mean_fraction = mean_n / float(water_capacity)
        hit_fraction = float(np.mean(counts >= water_capacity))

        triggers_capacity = bool(
            capacity_filter
            and (
                hit_fraction > float(max_capacity_hit_fraction)
                or mean_fraction >= float(max_mean_capacity_fraction)
            )
        )

        if prefix_open and triggers_capacity:
            prefix_open = False
            valid = False
            reason = "first_capacity_limited_window"
        elif not prefix_open and capacity_filter:
            valid = False
            reason = "after_first_capacity_limited_window"
        else:
            valid = True
            reason = "valid"

        valid_mask[mi] = valid

        if valid and this_mu_pts:
            density_pts.extend(this_mu_pts)

        diagnostics.append({
            "mu_index": mi,
            "mu_kcal_mol": float(mu_values[mi]),
            "n_frames": int(len(counts)),
            "N_region_mean_scan": mean_n,
            "N_region_std_scan": std_n,
            "N_region_max_observed": max_n,
            "water_capacity": int(water_capacity),
            "capacity_mean_fraction": mean_fraction,
            "capacity_hit_fraction": hit_fraction,
            "valid_for_gci": bool(valid),
            "capacity_filter_reason": reason,
        })

        flag = "KEEP" if valid else "EXCLUDE"
        print(
            f"[capacity] mu_{mi:03d} ({mu_values[mi]:8.3f}): "
            f"<N>={mean_n:8.3f}, max={max_n:4d}/{water_capacity}, "
            f"mean/cap={mean_fraction:6.3f}, hit-cap={hit_fraction:6.3f} -> {flag}"
        )

    if not np.any(valid_mask):
        raise RuntimeError(
            "Capacity filter rejected every chemical-potential window. "
            "Check sampler_volume / bulk_water_density / thresholds."
        )

    if not density_pts:
        raise RuntimeError(
            "No water oxygen coordinates were found in the GCI-valid windows."
        )

    return np.vstack(density_pts), diagnostics, valid_mask


def compute_density_grid(points: np.ndarray, edges):
    H, edges = np.histogramdd(points, bins=edges)
    return H.astype(float), edges


def find_peaks(H_smooth: np.ndarray, edges, percentile: float, neighborhood: int = 1):
    nonzero = H_smooth[H_smooth > 0]
    if nonzero.size == 0:
        raise RuntimeError("No non-zero density in smoothed grid.")

    threshold = np.percentile(nonzero, percentile)
    footprint = np.ones((2 * neighborhood + 1,) * 3, dtype=bool)
    masked = np.where(H_smooth >= threshold, H_smooth, 0.0)
    local_max = maximum_filter(masked, footprint=footprint, mode="nearest")
    peak_mask = (masked == local_max) & (masked > 0)
    idxs = np.argwhere(peak_mask)

    if idxs.size == 0:
        return np.empty((0, 3)), np.empty(0)

    xs = 0.5 * (edges[0][:-1] + edges[0][1:])
    ys = 0.5 * (edges[1][:-1] + edges[1][1:])
    zs = 0.5 * (edges[2][:-1] + edges[2][1:])

    centers = np.column_stack(
        [xs[idxs[:, 0]], ys[idxs[:, 1]], zs[idxs[:, 2]]]
    )
    scores = H_smooth[tuple(idxs.T)]
    return centers, scores


def merge_close_sites(centers: np.ndarray, scores: np.ndarray, cutoff: float):
    if len(centers) == 0:
        return centers, scores

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=cutoff)
    parent = np.arange(len(centers))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    groups: Dict[int, List[int]] = {}
    for i in range(len(centers)):
        groups.setdefault(find(i), []).append(i)

    merged_centers = []
    merged_scores = []
    for idxs in groups.values():
        w = scores[idxs]
        c = centers[idxs]
        total = float(w.sum())
        if total <= 0:
            merged_centers.append(c.mean(axis=0))
        else:
            merged_centers.append((c * w[:, None]).sum(axis=0) / total)
        merged_scores.append(total)

    return np.asarray(merged_centers), np.asarray(merged_scores)


def density_to_dx(H: np.ndarray, edges, outpath: str):
    nx, ny, nz = H.shape
    x0, y0, z0 = edges[0][0], edges[1][0], edges[2][0]
    dx = edges[0][1] - edges[0][0]
    dy = edges[1][1] - edges[1][0]
    dz = edges[2][1] - edges[2][0]

    with open(outpath, "w") as fh:
        fh.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        fh.write(f"origin {x0} {y0} {z0}\n")
        fh.write(f"delta {dx} 0 0\n")
        fh.write(f"delta 0 {dy} 0\n")
        fh.write(f"delta 0 0 {dz}\n")
        fh.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        fh.write(
            f"object 3 class array type double rank 0 items {nx*ny*nz} data follows\n"
        )
        for value in H.flatten(order="C"):
            fh.write(f"{value}\n")
        fh.write('object "density" class field\n')
        fh.write('component "positions" value 1\n')
        fh.write('component "connections" value 2\n')
        fh.write('component "data" value 3\n')


# =============================================================================
# Count N(B): total-region and local spherical counts
# =============================================================================

def count_region_and_local_by_mu(
    frames_dir: str,
    mu_values: np.ndarray,
    centers: np.ndarray,
    local_radius: float,
):
    n_mu = len(mu_values)
    n_sites = len(centers)

    n_frames = np.zeros(n_mu, dtype=int)
    region_mean = np.full(n_mu, np.nan, dtype=float)
    region_std = np.full(n_mu, np.nan, dtype=float)
    local_mean = np.full((n_sites, n_mu), np.nan, dtype=float)
    local_std = np.full((n_sites, n_mu), np.nan, dtype=float)

    for mi in range(n_mu):
        frames = list_frames_for_mu(frames_dir, mi)
        if not frames:
            print(f"[count] WARNING: no frames for mu_{mi:03d}")
            continue

        region_counts = np.zeros(len(frames), dtype=float)
        local_counts = np.zeros((n_sites, len(frames)), dtype=float)

        for fi, path in enumerate(frames):
            xyz = read_water_oxygen_coords_from_pdb(path)
            region_counts[fi] = xyz.shape[0]

            if n_sites and xyz.shape[0]:
                water_tree = cKDTree(xyz)
                try:
                    counts = water_tree.query_ball_point(
                        centers, r=local_radius, return_length=True
                    )
                    local_counts[:, fi] = np.asarray(counts, dtype=float)
                except TypeError:
                    hits = water_tree.query_ball_point(centers, r=local_radius)
                    local_counts[:, fi] = np.fromiter(
                        (len(h) for h in hits), dtype=float, count=n_sites
                    )

        n_frames[mi] = len(frames)
        region_mean[mi] = region_counts.mean()
        region_std[mi] = region_counts.std(ddof=1) if len(frames) > 1 else 0.0
        if n_sites:
            local_mean[:, mi] = local_counts.mean(axis=1)
            local_std[:, mi] = (
                local_counts.std(axis=1, ddof=1) if len(frames) > 1 else 0.0
            )

        print(
            f"[count] mu_{mi:03d} ({mu_values[mi]:8.3f} kcal/mol): "
            f"frames={len(frames):4d}, <N_region>={region_mean[mi]:8.3f}"
        )

    return region_mean, region_std, local_mean, local_std, n_frames


# =============================================================================
# Monotone sum-of-logistics titration model
# =============================================================================

def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


@dataclass
class LogisticSumModel:
    n_min: float
    n_max: float
    amplitudes: np.ndarray
    slopes: np.ndarray
    centers: np.ndarray
    rmse: float
    sse: float

    @property
    def delta_n(self) -> float:
        return float(self.n_max - self.n_min)

    def predict(self, B):
        x = np.asarray(B, dtype=float)
        z = self.slopes[:, None] * (x.ravel()[None, :] - self.centers[:, None])
        vals = self.n_min + np.sum(self.amplitudes[:, None] * expit(z), axis=0)
        vals = vals.reshape(x.shape)
        if np.ndim(B) == 0:
            return float(vals)
        return vals

    def excess_integral_from_minus_inf(self, B: float) -> float:
        """
        Integral_{-inf}^{B} [N(b) - n_min] db.
        Integral of sigmoid(s*(b-c)) is log(1+exp(s*(b-c)))/s.
        """
        z = self.slopes * (float(B) - self.centers)
        return float(np.sum((self.amplitudes / self.slopes) * np.logaddexp(0.0, z)))

    def inverse(self, target_n: float, B_lo: float, B_hi: float) -> float:
        """Finite inverse for n_min < target < n_max."""
        if not (self.n_min < target_n < self.n_max):
            raise ValueError("inverse target must be strictly inside model asymptotes")

        def f(b):
            return self.predict(b) - target_n

        lo = float(B_lo)
        hi = float(B_hi)
        # Expand until the target is bracketed.
        for _ in range(20):
            flo, fhi = f(lo), f(hi)
            if flo <= 0 <= fhi:
                return float(brentq(f, lo, hi, maxiter=300))
            span = max(10.0, hi - lo)
            if flo > 0:
                lo -= span
            if fhi < 0:
                hi += span
        raise RuntimeError(f"Could not bracket inverse for N={target_n}")

    def dimensionless_transfer_from_nmin(self, target_n: float, B_lo: float, B_hi: float) -> float:
        """
        ProtoMS-style GCI transfer term relative to n_min:

          Delta(beta F)_raw = DeltaN * B_N
                              - integral_{-inf}^{B_N} [N(B)-n_min] dB

        This is algebraically the B_lower -> -inf limit of
          N_f B_f - N_i B_i - integral N(B)dB
        and avoids numerical problems at the asymptotic lower endpoint.

        For the upper asymptote N=n_max, the finite analytic limit is
          sum_j amplitude_j * center_j.
        """
        target_n = float(target_n)
        if target_n < self.n_min - 1e-8 or target_n > self.n_max + 1e-8:
            raise ValueError("target N outside model range")

        dn = target_n - self.n_min
        if abs(dn) < 1e-12:
            return 0.0

        if abs(target_n - self.n_max) < 1e-10:
            return float(np.sum(self.amplitudes * self.centers))

        Bn = self.inverse(target_n, B_lo, B_hi)
        return float(dn * Bn - self.excess_integral_from_minus_inf(Bn))


def fit_logistic_sum(
    B: Sequence[float],
    N: Sequence[float],
    n_steps: Optional[int] = None,
    max_terms: int = 12,
    random_starts: int = 8,
    seed: int = 20260812,
) -> LogisticSumModel:
    """
    Fit a monotone sum of logistic transitions with integer asymptotes.

    The asymptotes are set to round(min observed N) and round(max observed N),
    following the common GCI practice of evaluating integer occupancy states.
    The total fitted amplitude is constrained to n_max - n_min.
    """
    B = np.asarray(B, dtype=float)
    N = np.asarray(N, dtype=float)
    valid = np.isfinite(B) & np.isfinite(N)
    B, N = B[valid], N[valid]
    if B.size < 4:
        raise RuntimeError("Need at least four finite titration points for GCI fitting.")

    order = np.argsort(B)
    B, N = B[order], N[order]

    n_min = float(np.round(np.min(N)))
    n_max = float(np.round(np.max(N)))
    if n_max <= n_min:
        raise RuntimeError(
            f"Titration does not span distinct integer occupancies: "
            f"round(min)={n_min}, round(max)={n_max}."
        )

    delta = n_max - n_min
    if n_steps is None:
        n_steps = max(1, int(round(delta)))
    m = int(max(1, min(n_steps, max_terms)))

    bmin, bmax = float(B.min()), float(B.max())
    bspan = max(bmax - bmin, 1.0)

    # Initial centers based on occupancy quantiles/crossings.
    centers0 = []
    for j in range(m):
        target = n_min + delta * (j + 0.5) / m
        centers0.append(float(B[np.argmin(np.abs(N - target))]))
    centers0 = np.asarray(centers0)

    rng = np.random.default_rng(seed)

    # Parameterization:
    #   if m>1: m-1 free logits with the first fixed at 0 -> softmax fractions
    #   m log-slopes
    #   m centers
    def unpack(theta):
        pos = 0
        if m == 1:
            amps = np.array([delta], dtype=float)
        else:
            logits = np.concatenate(([0.0], theta[pos:pos + m - 1]))
            pos += m - 1
            amps = delta * _softmax(logits)
        slopes = np.exp(theta[pos:pos + m])
        pos += m
        centers = theta[pos:pos + m]
        return amps, slopes, centers

    def residual(theta):
        amps, slopes, centers = unpack(theta)
        z = slopes[:, None] * (B[None, :] - centers[:, None])
        pred = n_min + np.sum(amps[:, None] * expit(z), axis=0)
        return pred - N

    if m == 1:
        theta0 = np.concatenate((np.log(np.ones(m)), centers0))
        lower = np.concatenate((np.full(m, math.log(0.02)), np.full(m, bmin - bspan)))
        upper = np.concatenate((np.full(m, math.log(20.0)), np.full(m, bmax + bspan)))
    else:
        theta0 = np.concatenate((np.zeros(m - 1), np.log(np.ones(m)), centers0))
        lower = np.concatenate((
            np.full(m - 1, -6.0),
            np.full(m, math.log(0.02)),
            np.full(m, bmin - bspan),
        ))
        upper = np.concatenate((
            np.full(m - 1, 6.0),
            np.full(m, math.log(20.0)),
            np.full(m, bmax + bspan),
        ))

    best = None
    starts = max(1, int(random_starts))
    for s in range(starts):
        t0 = theta0.copy()
        if s > 0:
            # Jitter only inside the supplied bounds.
            if m > 1:
                t0[:m - 1] += rng.normal(0.0, 0.6, size=m - 1)
                slope_start = m - 1
            else:
                slope_start = 0
            t0[slope_start:slope_start + m] += rng.normal(0.0, 0.35, size=m)
            t0[-m:] += rng.normal(0.0, 0.12 * bspan, size=m)
            t0 = np.minimum(np.maximum(t0, lower + 1e-8), upper - 1e-8)

        fit = least_squares(
            residual,
            t0,
            bounds=(lower, upper),
            max_nfev=25000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        sse = float(np.sum(fit.fun**2))
        if best is None or sse < best[0]:
            best = (sse, fit.x)

    assert best is not None
    sse, theta = best
    amps, slopes, centers = unpack(theta)
    idx = np.argsort(centers)
    amps, slopes, centers = amps[idx], slopes[idx], centers[idx]
    rmse = math.sqrt(sse / B.size)

    return LogisticSumModel(
        n_min=n_min,
        n_max=n_max,
        amplitudes=np.asarray(amps),
        slopes=np.asarray(slopes),
        centers=np.asarray(centers),
        rmse=float(rmse),
        sse=float(sse),
    )


# =============================================================================
# GCI PMF calculations
# =============================================================================

def gci_pmf(
    model: LogisticSumModel,
    B_obs: np.ndarray,
    kT: float,
    analysis_volume: float,
    standard_volume: float,
    mu_bulk: float,
) -> List[dict]:
    """
    Calculate integer-state insertion and bulk-referenced binding PMFs.

    Following the ProtoMS-style volume correction:

      beta DeltaG_insert(N) = beta DeltaF_raw(N)
                              - DeltaN * ln(V_analysis / V0)

    and then reference each inserted water to bulk solvent:

      DeltaG_bind(N) = DeltaG_insert(N) - DeltaN * mu_bulk

    where mu_bulk is negative for favorable hydration of bulk water.
    """
    n0 = int(round(model.n_min))
    n1 = int(round(model.n_max))
    rows = []
    prev_bind = None

    blo = float(np.min(B_obs))
    bhi = float(np.max(B_obs))
    volume_log = math.log(float(analysis_volume) / float(standard_volume))

    for n in range(n0, n1 + 1):
        raw_dimless = model.dimensionless_transfer_from_nmin(n, blo, bhi)
        dn = n - n0
        corrected_dimless = raw_dimless - dn * volume_log
        dg_insert = corrected_dimless * kT
        dg_bind = dg_insert - dn * mu_bulk
        incremental = np.nan if prev_bind is None else dg_bind - prev_bind
        rows.append({
            "N": n,
            "deltaN_from_min": dn,
            "beta_dF_raw": raw_dimless,
            "deltaG_insert_kcal_mol": dg_insert,
            "deltaG_bind_network_kcal_mol": dg_bind,
            "deltaG_bind_incremental_kcal_mol": incremental,
        })
        prev_bind = dg_bind

    return rows


def first_water_score_from_pmf(rows: List[dict]) -> Tuple[float, float, str]:
    """Return insertion and binding dG for adding the first water from N=0 to 1."""
    by_n = {int(r["N"]): r for r in rows}
    if 0 not in by_n or 1 not in by_n:
        return np.nan, np.nan, "does_not_span_0_to_1"
    r0, r1 = by_n[0], by_n[1]
    dg_ins = r1["deltaG_insert_kcal_mol"] - r0["deltaG_insert_kcal_mol"]
    dg_bind = r1["deltaG_bind_network_kcal_mol"] - r0["deltaG_bind_network_kcal_mol"]
    return float(dg_ins), float(dg_bind), "ok"


# =============================================================================
# Output helpers
# =============================================================================

def write_csv(path: str, rows: List[dict], fieldnames: Optional[List[str]] = None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_sites_pdb(path: str, centers: np.ndarray, local_at_bulk: np.ndarray, dg_bind: np.ndarray):
    with open(path, "w") as fh:
        fh.write("REMARK MCSwell local ProtoMS-style GCI sites\n")
        fh.write("REMARK occupancy column = mean local water count at nearest bulk mu\n")
        fh.write("REMARK B-factor = first-water bulk-referenced local GCI score\n")
        for i, center in enumerate(centers, start=1):
            occ = local_at_bulk[i - 1] if i - 1 < len(local_at_bulk) else np.nan
            dg = dg_bind[i - 1] if i - 1 < len(dg_bind) else np.nan
            occ_out = 0.0 if not np.isfinite(occ) else float(occ)
            dg_out = 0.0 if not np.isfinite(dg) else float(dg)
            fh.write(
                f"HETATM{i:5d}  O   SIT A{i:4d}    "
                f"{center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}"
                f"{occ_out:6.2f}{dg_out:6.2f}\n"
            )
        fh.write("END\n")


def plot_region_titration(
    path: str,
    B: np.ndarray,
    N: np.ndarray,
    model: LogisticSumModel,
    B_eq: float,
    valid_mask: Optional[np.ndarray] = None,
):
    if valid_mask is None:
        valid_mask = np.ones(len(B), dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    B_valid = B[valid_mask]
    x = np.linspace(B_valid.min() - 2.0, B_valid.max() + 2.0, 600)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(B[valid_mask], N[valid_mask], s=28, label="Used for GCI fit")
    if np.any(~valid_mask):
        plt.scatter(
            B[~valid_mask], N[~valid_mask], s=36, marker="x",
            label="Excluded: capacity-limited"
        )
    plt.plot(x, model.predict(x), lw=2, label="Monotone logistic-sum fit")
    plt.axvline(B_eq, ls="--", lw=1.2, label="Bulk-equilibrium B")
    plt.xlabel("Adams parameter B")
    plt.ylabel("Mean number of GCMC waters <N>")
    plt.title("MCSwell whole-region GCMC titration")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_local_titrations(
    path: str,
    B: np.ndarray,
    local_mean: np.ndarray,
    models: List[Optional[LogisticSumModel]],
    site_order: np.ndarray,
    max_sites: int = 12,
    valid_mask: Optional[np.ndarray] = None,
):
    chosen = [int(i) for i in site_order if models[int(i)] is not None][:max_sites]
    if not chosen:
        return

    if valid_mask is None:
        valid_mask = np.ones(len(B), dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    B_valid = B[valid_mask]

    # One figure, individual curves overlaid; no subplots.
    x = np.linspace(B_valid.min() - 2.0, B_valid.max() + 2.0, 600)
    fig = plt.figure(figsize=(8, 5.5))
    for i in chosen:
        plt.scatter(B[valid_mask], local_mean[i, valid_mask], s=14, alpha=0.55)
        if np.any(~valid_mask):
            plt.scatter(
                B[~valid_mask], local_mean[i, ~valid_mask],
                s=20, marker="x", alpha=0.35
            )
        plt.plot(x, models[i].predict(x), lw=1.4, label=f"Site {i}")
    plt.xlabel("Adams parameter B")
    plt.ylabel("Mean waters in local sphere <N_site>")
    plt.ylim(bottom=-0.05)
    plt.title("Local GCI titration curves (x = excluded capacity-limited windows)")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================

def run_analysis(args, cfg: Optional[dict] = None) -> str:
    """Run the GCI analysis.

    Parameters
    ----------
    args
        Namespace-like object containing analysis options.
    cfg
        Optional already-loaded MCSwell configuration dictionary.  This is the
        path used when the module is called from ``run_mcswell.py``.  If omitted,
        ``args.config`` is loaded from disk for command-line use.
    """
    if cfg is None:
        if not getattr(args, "config", None):
            raise ValueError("run_analysis requires either cfg or args.config")
        cfg = load_config(args.config)
    save_path = cfg["io"]["save_path"]
    frames_dir = os.path.join(save_path, "frames")

    peak_percentile = float(
        args.peak_percentile
        if args.peak_percentile is not None
        else cfg.get("gci", {}).get("peak_percentile", PEAK_PERCENTILE)
    )

    out_dir = args.output_dir or os.path.join(
        save_path, "gci", f"{peak_percentile:g}_protoms_gci"
    )
    os.makedirs(out_dir, exist_ok=True)

    mu_all = expand_mu_range(cfg)
    mu_indices, frames_per_mu = detect_mu_and_frames(frames_dir)

    if min(mu_indices) < 0 or max(mu_indices) >= len(mu_all):
        raise RuntimeError(
            f"Snapshot mu indices {min(mu_indices)}..{max(mu_indices)} are incompatible "
            f"with config mu_range containing {len(mu_all)} values."
        )
    if len(mu_indices) != len(mu_all):
        print(
            f"[warning] Found {len(mu_indices)} mu directories but config defines "
            f"{len(mu_all)} chemical potentials. Missing windows will be ignored."
        )

    # For the current snapshot layout, downstream counting loops use 0..len(mu)-1,
    # so require a contiguous complete set. This prevents silent mu misalignment.
    expected = list(range(len(mu_all)))
    if mu_indices != expected:
        raise RuntimeError(
            "This ready-to-run implementation requires a complete contiguous set of "
            f"mu_000..mu_{len(mu_all)-1:03d}. Found: {mu_indices}"
        )

    kT = float(args.kt)
    standard_volume = float(args.standard_volume)
    mu_bulk, mu_bulk_water_model, mu_bulk_source = resolve_mu_bulk(args.bulk_mu)

    sampler_volume, npts, volume_source = resolve_sampler_volume(
        cfg, out_dir, args.sampler_volume, args.n_insertion_points
    )

    B = mu_all / kT + math.log(sampler_volume / standard_volume)
    B_eq = mu_bulk / kT + math.log(sampler_volume / standard_volume)
    mu_eq_idx = int(np.argmin(np.abs(mu_all - mu_bulk)))

    print("\n=== GCI thermodynamic constants ===")
    print(f"kT                       : {kT:.8f} kcal/mol")
    print(f"standard volume V0       : {standard_volume:.6f} A^3")
    print(
        f"bulk mu                  : {mu_bulk:.6f} kcal/mol "
        f"(water model: {mu_bulk_water_model or 'undetected'}, source: {mu_bulk_source})"
    )
    print(f"sampler volume           : {sampler_volume:.6f} A^3 ({volume_source})")
    if npts is not None:
        print(f"insertion points         : {npts}")
    print(f"ln(V/V0)                 : {math.log(sampler_volume / standard_volume):.6f}")
    print(f"bulk-equilibrium B        : {B_eq:.6f}")
    print(f"nearest sampled bulk mu   : {mu_all[mu_eq_idx]:.6f} (mu_{mu_eq_idx:03d})")

    # ------------------------------------------------------------------
    # Capacity-safety scan and site detection
    # ------------------------------------------------------------------
    water_capacity = int(
        math.floor(sampler_volume * float(args.bulk_water_density))
    )
    if water_capacity <= 0:
        raise RuntimeError(
            "Reconstructed water-buffer capacity is zero. "
            "Check sampler_volume and bulk_water_density."
        )

    print("\n=== Capacity-safety filter ===")
    print(f"bulk-water density used   : {args.bulk_water_density:.6f} waters/A^3")
    print(f"reconstructed capacity    : {water_capacity} waters")
    print(f"capacity filter enabled   : {bool(args.capacity_filter)}")
    print(f"max hit-cap fraction      : {args.max_capacity_hit_fraction:.4f}")
    print(f"max mean/capacity fraction: {args.max_mean_capacity_fraction:.4f}")

    print("\n[capacity] Scanning snapshots and collecting density points...")
    all_points, capacity_diagnostics, gci_valid_mask = (
        scan_capacity_and_collect_density_points(
            frames_dir,
            mu_all,
            frames_per_mu,
            water_capacity,
            capacity_filter=bool(args.capacity_filter),
            max_capacity_hit_fraction=float(args.max_capacity_hit_fraction),
            max_mean_capacity_fraction=float(args.max_mean_capacity_fraction),
        )
    )

    n_valid = int(np.count_nonzero(gci_valid_mask))
    if n_valid < 4:
        raise RuntimeError(
            f"Only {n_valid} chemical-potential windows remain after capacity "
            "filtering; need at least four for GCI fitting."
        )

    if not gci_valid_mask[mu_eq_idx]:
        raise RuntimeError(
            "The sampled window nearest bulk water chemical potential "
            f"(mu={mu_all[mu_eq_idx]:.3f} kcal/mol) is capacity-limited. "
            "The current simulation does not provide a trustworthy bulk-referenced "
            "GCI analysis. Increase the sampler water-buffer capacity and rerun."
        )

    B_fit = B[gci_valid_mask]
    mu_fit = mu_all[gci_valid_mask]
    valid_indices = np.flatnonzero(gci_valid_mask)
    first_excluded = np.flatnonzero(~gci_valid_mask)

    print(
        f"[capacity] GCI will use {n_valid}/{len(mu_all)} contiguous windows: "
        f"mu={mu_fit[0]:.3f} .. {mu_fit[-1]:.3f} kcal/mol"
    )
    if len(first_excluded):
        ii = int(first_excluded[0])
        print(
            f"[capacity] First excluded window: mu_{ii:03d}, "
            f"mu={mu_all[ii]:.3f} kcal/mol"
        )

    # Hydration-site density is also built only from the capacity-safe windows.
    gc_center, gc_halfsize = get_gc_box_from_config(cfg)
    print(
        f"\n[density] Building hydration-site density from "
        f"{n_valid} capacity-safe mu windows..."
    )
    print(f"[density] Total water oxygen observations used: {len(all_points):,}")

    edges = make_edges(gc_center, gc_halfsize, GRID_SPACING)
    H, edges = compute_density_grid(all_points, edges)
    H_smooth = gaussian_filter(H, sigma=GRID_SIGMA)
    raw_centers, raw_scores = find_peaks(
        H_smooth, edges, percentile=peak_percentile, neighborhood=1
    )
    centers, peak_scores = merge_close_sites(
        raw_centers, raw_scores, cutoff=PEAK_MERGE_CUTOFF
    )
    print(f"[density] Raw peaks: {len(raw_centers)}; merged sites: {len(centers)}")
    density_to_dx(H_smooth, edges, os.path.join(out_dir, "gO.dx"))

    # ------------------------------------------------------------------
    # Count total and local water populations at each mu.
    # ------------------------------------------------------------------
    print("\n[count] Computing <N>(mu) for whole region and local spheres...")
    region_mean, region_std, local_mean, local_std, n_frames = count_region_and_local_by_mu(
        frames_dir,
        mu_all,
        centers,
        local_radius=float(args.local_radius),
    )

    # Save raw titration data first. All sampled windows are retained in the
    # CSVs; valid_for_gci identifies the contiguous capacity-safe prefix used
    # for site detection and thermodynamic fitting.
    titration_rows = []
    for i, (mu, b) in enumerate(zip(mu_all, B)):
        cap = capacity_diagnostics[i]
        titration_rows.append({
            "mu_index": i,
            "mu_kcal_mol": mu,
            "B_sampler": b,
            "n_frames": int(n_frames[i]),
            "N_region_mean": region_mean[i],
            "N_region_std": region_std[i],
            "N_region_max_observed": cap["N_region_max_observed"],
            "water_capacity": int(water_capacity),
            "capacity_mean_fraction": cap["capacity_mean_fraction"],
            "capacity_hit_fraction": cap["capacity_hit_fraction"],
            "valid_for_gci": bool(gci_valid_mask[i]),
            "capacity_filter_reason": cap["capacity_filter_reason"],
        })
    write_csv(os.path.join(out_dir, "region_titration.csv"), titration_rows)

    local_long = []
    for s in range(len(centers)):
        for i, (mu, b) in enumerate(zip(mu_all, B)):
            cap = capacity_diagnostics[i]
            local_long.append({
                "site": s,
                "mu_index": i,
                "mu_kcal_mol": mu,
                "B_sampler": b,
                "n_frames": int(n_frames[i]),
                "N_local_mean": local_mean[s, i],
                "N_local_std": local_std[s, i],
                "valid_for_gci": bool(gci_valid_mask[i]),
                "capacity_mean_fraction": cap["capacity_mean_fraction"],
                "capacity_hit_fraction": cap["capacity_hit_fraction"],
            })
    write_csv(os.path.join(out_dir, "local_titration.csv"), local_long)

    # ------------------------------------------------------------------
    # Whole-region ProtoMS-style GCI
    # ------------------------------------------------------------------
    print("\n[gci-region] Fitting whole-region N(B) using capacity-safe windows...")
    region_model = fit_logistic_sum(
        B_fit,
        region_mean[gci_valid_mask],
        n_steps=None,
        max_terms=int(args.region_max_terms),
        random_starts=int(args.random_starts),
        seed=int(args.seed),
    )
    print(
        f"[gci-region] asymptotes N={region_model.n_min:g} -> {region_model.n_max:g}; "
        f"terms={len(region_model.amplitudes)}; RMSE_N={region_model.rmse:.4f}"
    )

    region_pmf = gci_pmf(
        region_model,
        B_fit,
        kT=kT,
        analysis_volume=sampler_volume,
        standard_volume=standard_volume,
        mu_bulk=mu_bulk,
    )
    write_csv(os.path.join(out_dir, "region_gci_pmf.csv"), region_pmf)
    plot_region_titration(
        os.path.join(out_dir, "region_titration.png"),
        B,
        region_mean,
        region_model,
        B_eq,
        valid_mask=gci_valid_mask,
    )

    # Which region PMF state corresponds most closely to equilibrium occupancy?
    N_eq_obs = float(region_mean[mu_eq_idx])
    N_eq_int = int(np.clip(np.round(N_eq_obs), region_model.n_min, region_model.n_max))
    region_by_n = {int(r["N"]): r for r in region_pmf}
    eq_row = region_by_n.get(N_eq_int)
    min_row = min(region_pmf, key=lambda r: r["deltaG_bind_network_kcal_mol"])

    # ------------------------------------------------------------------
    # Local GCI fits
    # ------------------------------------------------------------------
    if args.local_volume_mode == "sampler":
        local_analysis_volume = sampler_volume
    elif args.local_volume_mode == "standard":
        local_analysis_volume = standard_volume
    elif args.local_volume_mode == "sphere":
        local_analysis_volume = 4.0 / 3.0 * math.pi * float(args.local_radius) ** 3
    else:  # pragma: no cover
        raise ValueError(args.local_volume_mode)

    print("\n[gci-local] Fitting local N_site(B) curves...")
    print(
        f"[gci-local] radius={args.local_radius:.3f} A; "
        f"volume correction mode={args.local_volume_mode}; "
        f"analysis volume={local_analysis_volume:.6f} A^3"
    )

    site_rows = []
    site_models: List[Optional[LogisticSumModel]] = [None] * len(centers)
    site_dg_bind = np.full(len(centers), np.nan, dtype=float)
    site_dg_insert = np.full(len(centers), np.nan, dtype=float)

    for s in range(len(centers)):
        y = local_mean[s]
        y_fit = y[gci_valid_mask]
        status = "ok"
        model = None
        dg_ins = np.nan
        dg_bind = np.nan
        nmin_round = np.nan
        nmax_round = np.nan
        rmse = np.nan
        terms = 0

        try:
            nmin_round = float(np.round(np.nanmin(y_fit)))
            nmax_round = float(np.round(np.nanmax(y_fit)))
            if nmax_round <= nmin_round:
                raise RuntimeError("no_integer_occupancy_transition_in_capacity_safe_range")

            local_steps = max(1, int(nmax_round - nmin_round))
            local_steps = min(local_steps, int(args.local_max_terms))
            model = fit_logistic_sum(
                B_fit,
                y_fit,
                n_steps=local_steps,
                max_terms=int(args.local_max_terms),
                random_starts=int(args.random_starts),
                seed=int(args.seed) + s + 1,
            )
            site_models[s] = model
            rmse = model.rmse
            terms = len(model.amplitudes)

            pmf = gci_pmf(
                model,
                B_fit,
                kT=kT,
                analysis_volume=local_analysis_volume,
                standard_volume=standard_volume,
                mu_bulk=mu_bulk,
            )
            dg_ins, dg_bind, status = first_water_score_from_pmf(pmf)

        except Exception as exc:
            status = str(exc).replace(",", ";")[:180]

        site_dg_insert[s] = dg_ins
        site_dg_bind[s] = dg_bind

        c = centers[s]
        site_rows.append({
            "site": s,
            "x": c[0],
            "y": c[1],
            "z": c[2],
            "peak_density_score": peak_scores[s] if s < len(peak_scores) else np.nan,
            "local_radius_A": float(args.local_radius),
            "local_volume_mode": args.local_volume_mode,
            "local_analysis_volume_A3": local_analysis_volume,

            # Values used by the fit (capacity-safe windows only).
            "N_local_min_observed": float(np.nanmin(y_fit)),
            "N_local_max_observed": float(np.nanmax(y_fit)),
            "N_local_min_observed_valid": float(np.nanmin(y_fit)),
            "N_local_max_observed_valid": float(np.nanmax(y_fit)),

            # Diagnostics retaining the complete sampled titration.
            "N_local_min_observed_all": float(np.nanmin(y)),
            "N_local_max_observed_all": float(np.nanmax(y)),
            "N_local_at_bulk_mu": float(y[mu_eq_idx]),
            "fit_n_windows": int(np.count_nonzero(gci_valid_mask)),
            "fit_mu_min_kcal_mol": float(mu_fit[0]),
            "fit_mu_max_kcal_mol": float(mu_fit[-1]),
            "fit_B_min": float(B_fit[0]),
            "fit_B_max": float(B_fit[-1]),
            "capacity_filter_applied": bool(args.capacity_filter),

            "N_min_model": model.n_min if model is not None else nmin_round,
            "N_max_model": model.n_max if model is not None else nmax_round,
            "fit_terms": terms,
            "fit_rmse_N": rmse,
            "deltaG_insert_first_water": dg_ins,
            "deltaG_bind_first_water": dg_bind,

            # Backward-compatible aliases for existing plotting pipelines.
            "deltaG_bind": dg_bind,
            "epsilon_mean": dg_bind,
            "status": status,
        })

        if (s + 1) % 25 == 0 or s + 1 == len(centers):
            usable = np.count_nonzero(np.isfinite(site_dg_bind[:s + 1]))
            print(
                f"[gci-local] processed {s+1}/{len(centers)} sites; "
                f"usable first-water scores={usable}"
            )

    write_csv(os.path.join(out_dir, "sites.csv"), site_rows)
    # Same file under an explicit name to prevent ambiguity in downstream work.
    write_csv(os.path.join(out_dir, "local_gci_sites.csv"), site_rows)

    local_at_bulk = local_mean[:, mu_eq_idx] if len(centers) else np.array([])
    write_sites_pdb(
        os.path.join(out_dir, "sites.pdb"), centers, local_at_bulk, site_dg_bind
    )

    if len(centers):
        order = np.argsort(-np.nan_to_num(local_at_bulk, nan=-np.inf))
        plot_local_titrations(
            os.path.join(out_dir, "local_titration_curves.png"),
            B,
            local_mean,
            site_models,
            order,
            max_sites=int(args.plot_max_sites),
            valid_mask=gci_valid_mask,
        )

    # ------------------------------------------------------------------
    # Save metadata and concise summary.
    # ------------------------------------------------------------------
    metadata = {
        "config": (
            os.path.abspath(args.config)
            if getattr(args, "config", None)
            else "<in-memory config passed by run_mcswell.py>"
        ),
        "frames_dir": os.path.abspath(frames_dir),
        "output_dir": os.path.abspath(out_dir),
        "method": "ProtoMS-style GCI using monotone sum-of-logistics fit",
        "kT_kcal_mol": kT,
        "temperature_K_label": float(args.temperature),
        "standard_volume_A3": standard_volume,
        "bulk_mu_kcal_mol": mu_bulk,
        "mu_bulk_water_model": mu_bulk_water_model,
        "mu_bulk_source": mu_bulk_source,
        "sampler_volume_A3": sampler_volume,
        "sampler_volume_source": volume_source,
        "n_insertion_points": npts,
        "bulk_water_density_waters_A3": float(args.bulk_water_density),
        "reconstructed_water_capacity": int(water_capacity),
        "capacity_filter_enabled": bool(args.capacity_filter),
        "max_capacity_hit_fraction": float(args.max_capacity_hit_fraction),
        "max_mean_capacity_fraction": float(args.max_mean_capacity_fraction),
        "n_gci_valid_mu_windows": int(np.count_nonzero(gci_valid_mask)),
        "gci_fit_mu_min_kcal_mol": float(mu_fit[0]),
        "gci_fit_mu_max_kcal_mol": float(mu_fit[-1]),
        "first_excluded_mu_index": (
            int(first_excluded[0]) if len(first_excluded) else None
        ),
        "first_excluded_mu_kcal_mol": (
            float(mu_all[int(first_excluded[0])]) if len(first_excluded) else None
        ),
        "B_definition": "B = mu/kT + ln(V_sampler/V0)",
        "B_equilibrium": B_eq,
        "nearest_bulk_mu_index": mu_eq_idx,
        "nearest_bulk_mu_kcal_mol": float(mu_all[mu_eq_idx]),
        "peak_percentile": peak_percentile,
        "grid_spacing_A": GRID_SPACING,
        "grid_sigma_voxels": GRID_SIGMA,
        "peak_merge_cutoff_A": PEAK_MERGE_CUTOFF,
        "local_radius_A": float(args.local_radius),
        "local_volume_mode": args.local_volume_mode,
        "local_analysis_volume_A3": local_analysis_volume,
        "n_sites": int(len(centers)),
        "n_local_usable": int(np.count_nonzero(np.isfinite(site_dg_bind))),
        "region_fit_N_min": region_model.n_min,
        "region_fit_N_max": region_model.n_max,
        "region_fit_terms": int(len(region_model.amplitudes)),
        "region_fit_rmse_N": region_model.rmse,
        "region_observed_N_at_bulk": N_eq_obs,
        "region_rounded_N_at_bulk": N_eq_int,
        "region_binding_dG_at_rounded_bulk_N": (
            eq_row["deltaG_bind_network_kcal_mol"] if eq_row is not None else None
        ),
        "region_binding_pmf_min_N": int(min_row["N"]),
        "region_binding_pmf_min_dG_kcal_mol": min_row["deltaG_bind_network_kcal_mol"],
    }
    with open(os.path.join(out_dir, "gci_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n=== GCI analysis complete ===")
    print(f"Output directory: {out_dir}")
    print(
        f"Capacity-safe GCI range: mu={mu_fit[0]:.3f} .. "
        f"{mu_fit[-1]:.3f} kcal/mol "
        f"({np.count_nonzero(gci_valid_mask)}/{len(mu_all)} windows)"
    )
    print(f"Whole-region fit RMSE in N: {region_model.rmse:.4f}")
    print(f"Observed <N_region> near bulk mu: {N_eq_obs:.3f}")
    if eq_row is not None:
        print(
            f"Whole-region network binding dG at rounded N={N_eq_int}: "
            f"{eq_row['deltaG_bind_network_kcal_mol']:.3f} kcal/mol"
        )
    print(
        f"Whole-region binding-PMF minimum: N={int(min_row['N'])}, "
        f"dG={min_row['deltaG_bind_network_kcal_mol']:.3f} kcal/mol"
    )
    print(
        f"Local sites with finite first-water GCI scores: "
        f"{np.count_nonzero(np.isfinite(site_dg_bind))}/{len(site_dg_bind)}"
    )
    print("For RETI comparison, use sites.csv column: deltaG_bind_first_water")
    if args.local_volume_mode == "standard":
        print(
            "NOTE: local-volume-mode=standard reproduces the legacy local-cluster "
            "choice of zeroing the volume correction; treat these as local GCI scores."
        )
    else:
        print(
            "NOTE: local site GCI remains a post-hoc local score; whole-region GCI is "
            "the direct grand-canonical integration of the sampled particle number."
        )

    return out_dir


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="ProtoMS-style GCI thermodynamic analysis for existing MCSwell snapshots."
    )
    p.add_argument("config", help="MCSwell TOML configuration used for the simulation")
    p.add_argument("--output-dir", default=None, help="Override output directory")

    # Exact MCSwell sampler constants / volume.
    p.add_argument(
        "--sampler-volume",
        type=float,
        default=None,
        help="Exact sampler total_volume in A^3 from the original MCSwell run log. Preferred if known.",
    )
    p.add_argument(
        "--n-insertion-points",
        type=int,
        default=None,
        help="Insertion-point count; sampler volume is N * spacing^3.",
    )
    p.add_argument("--kt", type=float, default=MCSWELL_KT,
                   help=f"kT in kcal/mol (default current MCSwell sampler: {MCSWELL_KT})")
    p.add_argument("--standard-volume", type=float, default=MCSWELL_STANDARD_VOLUME,
                   help=f"V0 in A^3 (default current MCSwell sampler: {MCSWELL_STANDARD_VOLUME})")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="Temperature label for metadata (K)")
    p.add_argument("--bulk-mu", type=float, default=None,
                   help=(
                       "Bulk-water excess chemical potential in kcal/mol. If "
                       "omitted, it is auto-selected from the compiled "
                       "mcswell_cpp water model (see utils.MU_BULK_BY_WATER_MODEL) "
                       "with a warning; some models require this to be set "
                       "explicitly."
                   ))

    # Site detection / local GCI.
    p.add_argument("--peak-percentile", type=float, default=None,
                   help="Density peak percentile; otherwise config[gci] or 90")
    p.add_argument("--local-radius", type=float, default=DEFAULT_LOCAL_RADIUS,
                   help=f"Local hydration-site sphere radius in A (default {DEFAULT_LOCAL_RADIUS})")
    p.add_argument(
        "--local-volume-mode",
        choices=("sampler", "standard", "sphere"),
        default="sampler",
        help=(
            "Volume used in the GCI standard-state correction for local sites. "
            "'sampler' (default) removes the same B-volume offset used by MCSwell; "
            "'standard' sets V=V0 (legacy ProtoMS local-cluster score convention); "
            "'sphere' uses 4/3*pi*r^3."
        ),
    )

    # Capacity-safety filtering.
    p.add_argument(
        "--no-capacity-filter",
        dest="capacity_filter",
        action="store_false",
        help=(
            "Disable automatic exclusion of high-mu windows that approach the "
            "finite MCSwell water-buffer capacity."
        ),
    )
    p.set_defaults(capacity_filter=True)
    p.add_argument(
        "--bulk-water-density",
        type=float,
        default=MCSWELL_BULK_WATER_DENSITY,
        help=(
            "Water number density used to reconstruct the current MCSwell buffer "
            f"capacity (default {MCSWELL_BULK_WATER_DENSITY} waters/A^3)."
        ),
    )
    p.add_argument(
        "--max-capacity-hit-fraction",
        type=float,
        default=DEFAULT_MAX_CAPACITY_HIT_FRACTION,
        help=(
            "First window with a larger fraction of saved frames exactly at the "
            "water-buffer capacity is excluded, together with all later windows."
        ),
    )
    p.add_argument(
        "--max-mean-capacity-fraction",
        type=float,
        default=DEFAULT_MAX_MEAN_CAPACITY_FRACTION,
        help=(
            "First window whose mean region population reaches this fraction of "
            "the water-buffer capacity is excluded, together with all later windows."
        ),
    )

    # Fit options.
    p.add_argument("--region-max-terms", type=int, default=12,
                   help="Maximum logistic terms for whole-region N(B) fit")
    p.add_argument("--local-max-terms", type=int, default=4,
                   help="Maximum logistic terms per local-site N(B) fit")
    p.add_argument("--random-starts", type=int, default=8,
                   help="Random starts for each nonlinear titration fit")
    p.add_argument("--seed", type=int, default=20260812,
                   help="Random seed for fit initialization")
    p.add_argument("--plot-max-sites", type=int, default=12,
                   help="Maximum local titration curves overlaid in diagnostic plot")
    return p


def _module_args_from_config(
    config: dict,
    *,
    sampler_volume: Optional[float] = None,
    n_insertion_points: Optional[int] = None,
    output_dir: Optional[str] = None,
    peak_percentile: Optional[float] = None,
    local_radius: Optional[float] = None,
    local_volume_mode: Optional[str] = None,
    capacity_filter: Optional[bool] = None,
    bulk_water_density: Optional[float] = None,
    max_capacity_hit_fraction: Optional[float] = None,
    max_mean_capacity_fraction: Optional[float] = None,
    region_max_terms: Optional[int] = None,
    local_max_terms: Optional[int] = None,
    random_starts: Optional[int] = None,
    seed: Optional[int] = None,
    plot_max_sites: Optional[int] = None,
    kt: Optional[float] = None,
    standard_volume: Optional[float] = None,
    temperature: Optional[float] = None,
    bulk_mu: Optional[float] = None,
):
    """Build an argparse-like namespace for module execution.

    Values can be provided explicitly to ``main(...)`` or, for convenience,
    under ``config["gci"]``. Explicit arguments take precedence.
    """
    gci_cfg = config.get("gci", {})

    def pick(explicit, key, default):
        return explicit if explicit is not None else gci_cfg.get(key, default)

    # Allow sampler metadata to live in [gci] in the TOML if desired.
    sampler_volume = pick(sampler_volume, "sampler_volume", None)
    n_insertion_points = pick(n_insertion_points, "n_insertion_points", None)

    if sampler_volume is not None and n_insertion_points is not None:
        raise ValueError(
            "Provide only one of sampler_volume or n_insertion_points, not both."
        )

    # None is intentional here too: resolve_mu_bulk() auto-selects a
    # water-model-dependent default (with a warning) when nothing is given.
    bulk_mu = pick(bulk_mu, "bulk_mu", None)
    bulk_mu = float(bulk_mu) if bulk_mu is not None else None

    return SimpleNamespace(
        # None is intentional: the config is already loaded in memory.
        config=None,
        output_dir=pick(output_dir, "output_dir", None),
        sampler_volume=sampler_volume,
        n_insertion_points=n_insertion_points,
        kt=float(pick(kt, "kt", MCSWELL_KT)),
        standard_volume=float(
            pick(standard_volume, "standard_volume", MCSWELL_STANDARD_VOLUME)
        ),
        temperature=float(pick(temperature, "temperature", DEFAULT_TEMPERATURE)),
        bulk_mu=bulk_mu,
        peak_percentile=pick(peak_percentile, "peak_percentile", None),
        local_radius=float(
            pick(local_radius, "local_radius", DEFAULT_LOCAL_RADIUS)
        ),
        local_volume_mode=str(
            pick(local_volume_mode, "local_volume_mode", "sampler")
        ),
        capacity_filter=bool(
            pick(capacity_filter, "capacity_filter", True)
        ),
        bulk_water_density=float(
            pick(
                bulk_water_density,
                "bulk_water_density",
                MCSWELL_BULK_WATER_DENSITY,
            )
        ),
        max_capacity_hit_fraction=float(
            pick(
                max_capacity_hit_fraction,
                "max_capacity_hit_fraction",
                DEFAULT_MAX_CAPACITY_HIT_FRACTION,
            )
        ),
        max_mean_capacity_fraction=float(
            pick(
                max_mean_capacity_fraction,
                "max_mean_capacity_fraction",
                DEFAULT_MAX_MEAN_CAPACITY_FRACTION,
            )
        ),
        region_max_terms=int(pick(region_max_terms, "region_max_terms", 12)),
        local_max_terms=int(pick(local_max_terms, "local_max_terms", 4)),
        random_starts=int(pick(random_starts, "random_starts", 8)),
        seed=int(pick(seed, "seed", 20260812)),
        plot_max_sites=int(pick(plot_max_sites, "plot_max_sites", 12)),
    )


def main(
    config,
    *,
    sampler_volume: Optional[float] = None,
    n_insertion_points: Optional[int] = None,
    output_dir: Optional[str] = None,
    **overrides,
):
    """Module entry point, matching ``estimate_free_energies.main(config)``.

    This is intended to be called directly from ``run_mcswell.py``::

        import estimate_free_energies_gci as fe_gci
        fe_gci.main(config)

    For the most exact/efficient integration, pass the sampler volume already
    known by ``run_mcswell.py``::

        fe_gci.main(config, sampler_volume=sampler_volume)

    Parameters
    ----------
    config : dict or path-like
        Loaded MCSwell config dictionary (preferred for module use), or a TOML
        path for convenience.
    sampler_volume : float, optional
        Exact GCMC insertion-cloud volume in A^3. If omitted, the script first
        checks ``config["gci"]["sampler_volume"]`` and otherwise reconstructs
        it using the MCSwell machinery.
    n_insertion_points : int, optional
        Alternative to sampler_volume. Volume = n_points * spacing^3.
    output_dir : str, optional
        Override the default GCI output directory.
    **overrides
        Optional analysis settings such as local_radius, local_volume_mode,
        region_max_terms, local_max_terms, random_starts, seed, kt,
        standard_volume, temperature, bulk_mu, and peak_percentile.

    Returns
    -------
    str
        Output directory containing the GCI results.
    """
    if isinstance(config, (str, os.PathLike)):
        cfg_path = os.fspath(config)
        cfg = load_config(cfg_path)
    elif isinstance(config, dict):
        cfg_path = None
        cfg = config
    else:
        raise TypeError("config must be a dict or a TOML path")

    args = _module_args_from_config(
        cfg,
        sampler_volume=sampler_volume,
        n_insertion_points=n_insertion_points,
        output_dir=output_dir,
        **overrides,
    )
    args.config = cfg_path
    return run_analysis(args, cfg=cfg)


def cli(argv=None):
    """Command-line entry point retained for standalone post-processing."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sampler_volume is not None and args.n_insertion_points is not None:
        parser.error("Use only one of --sampler-volume or --n-insertion-points")
    return run_analysis(args)


if __name__ == "__main__":
    cli()
