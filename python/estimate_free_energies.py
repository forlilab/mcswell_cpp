#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#   Copyright (c) 2026 Scripps Research, Forli Lab.
#   All rights reserved.
#
#   Original author: Niccolo Bruciaferri
#   Thermodynamic-analysis correction: bounded independent-site grand-canonical
#   occupancy model with binomial maximum-likelihood estimation.
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

import os
import re
import sys
import csv
import glob
import json
import math
import argparse
from types import SimpleNamespace
from typing import Optional

try:
    import tomllib as tomli  # Python >= 3.11
except ImportError:  # pragma: no cover
    import tomli
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import brentq
from scipy.special import expit
import matplotlib.pyplot as plt

from utils import *


# Boltzmann constant in kcal mol^-1 K^-1
K_B = 0.0019872041

# -----------------------------------------------------------------------------
# Physical parameters
# -----------------------------------------------------------------------------
TEMPERATURE = 300.0  # K

# Bulk-water excess chemical potential used by the GCMC simulations.
# The fitted logistic midpoint epsilon is on the same chemical-potential scale.
# The bulk-referenced site free energy is therefore:
#       DeltaG_bind = epsilon - mu_bulk
#
# mu_bulk is water-model dependent (see utils.MU_BULK_BY_WATER_MODEL) and is
# resolved automatically from the compiled mcswell_cpp water model unless
# overridden via gci.mu_bulk in the config or --mu-bulk on the CLI.

# Discrete site assignment cutoff (Angstrom)
ASSIGNMENT_CUTOFF = 2.4

# Grid parameters -- intentionally left as in the original analysis code.
GRID_SPACING = 0.5  # Angstrom voxel spacing
GRID_SIGMA = 1.4    # Gaussian smoothing sigma (voxels)

# Peak / zone thresholds (percentiles on non-zero density)
PEAK_PERCENTILE = 90

# Peak merging cutoff (Angstrom)
PEAK_MERGE_CUTOFF = 1.4

# Numerical bracket used when solving for the logistic midpoint epsilon.
# +/- 100 kT is ~60 kcal/mol at 300 K and is intentionally generous.
FIT_BRACKET_KT = 100.0

# Current MCSwell C++ water-buffer sizing:
#     capacity = floor(V_sampler * BULK_WATER_DENSITY)
#
# The capacity is a memory/sampler ceiling, not a physical upper bound on the
# grand-canonical particle number.  High-mu windows that approach this ceiling
# are therefore excluded from density construction and thermodynamic fitting.
MCSWELL_BULK_WATER_DENSITY = 0.0334  # waters / Angstrom^3
DEFAULT_MAX_CAPACITY_HIT_FRACTION = 0.01
DEFAULT_MAX_MEAN_CAPACITY_FRACTION = 0.90


# =============================================================================
# IO / PDB parsing
# =============================================================================

def read_water_oxygen_coords_from_pdb(
    pdb_path,
    accept_resnames=("HOH", "WAT", "H2O", "TIP3", "SOL", "WA1"),
    accept_atoms=("O", "OW", "O1", "OH2", "O00"),
):
    """Return an (N, 3) float32 array of water oxygen coordinates."""
    if (not os.path.exists(pdb_path)) or os.path.getsize(pdb_path) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom = line[12:16].strip()
            resn = line[17:20].strip()

            if atom not in accept_atoms:
                continue
            if resn not in accept_resnames:
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


def sort_clusters_by_occupancy(clust_ids):
    """Retained from the original implementation for API compatibility."""
    n_clusts = int(max(clust_ids))
    occ = [[i, list(clust_ids).count(i)] for i in range(1, n_clusts + 1)]
    occ = sorted(occ, key=lambda x: -x[1])
    old_order = [x[0] for x in occ]
    clust_occs = [x[1] for x in occ]
    clust_ids_sorted = np.asarray(
        [old_order.index(x) + 1 for x in clust_ids], dtype=int
    )
    return clust_ids_sorted, clust_occs


# =============================================================================
# Thermodynamic model
# =============================================================================

def independent_site_occupancy(mu_values, epsilon, T=TEMPERATURE):
    r"""
    Bounded independent-site grand-canonical occupancy isotherm:

        p(mu) = 1 / [1 + exp(beta * (epsilon - mu))]

    where epsilon is the chemical potential at which p = 0.5 and
    beta = 1 / (k_B T).
    """
    mu = np.asarray(mu_values, dtype=float)
    beta = 1.0 / (K_B * T)
    return expit(beta * (mu - epsilon))


def fit_epsilon_binomial(mu_values, occ_counts, n_frames, T=TEMPERATURE):
    r"""
    Fit the bounded independent-site isotherm by binomial maximum likelihood.

    At each chemical potential mu_i:

        k_i ~ Binomial(N_i, p(mu_i; epsilon))

    where k_i is the number of occupied frames and N_i is the total number
    of analyzed frames.

    Returns a dictionary containing:
      epsilon          fitted midpoint chemical potential (kcal/mol)
      epsilon_se       nominal binomial Fisher-information SE (kcal/mol)
      n_mu_points      number of chemical potentials containing frames
      total_occupied   total occupied-frame observations over all mu
      total_frames     total frame observations over all mu
      fit_rmse_p       RMSE between observed occupancy fractions and fit
      log_likelihood   binomial log-likelihood, excluding combinatorial constants
      status           'ok', 'all_empty', 'all_occupied', or failure message

    Important: epsilon_se treats saved frames as independent. For manuscript
    uncertainty, replicate-to-replicate variation is preferable when available.
    """
    mu = np.asarray(mu_values, dtype=float)
    k = np.asarray(occ_counts, dtype=float)
    n = np.asarray(n_frames, dtype=float)

    valid = (
        np.isfinite(mu)
        & np.isfinite(k)
        & np.isfinite(n)
        & (n > 0)
    )

    mu = mu[valid]
    k = k[valid]
    n = n[valid]

    if mu.size == 0:
        return {
            "epsilon": np.nan,
            "epsilon_se": np.nan,
            "n_mu_points": 0,
            "total_occupied": 0,
            "total_frames": 0,
            "fit_rmse_p": np.nan,
            "log_likelihood": np.nan,
            "status": "no_frames",
        }

    if np.any(k < 0) or np.any(k > n):
        raise ValueError("Occupancy counts must satisfy 0 <= k_i <= N_i.")

    total_occupied = int(np.sum(k))
    total_frames = int(np.sum(n))

    # A completely empty or completely occupied titration has no finite
    # maximum-likelihood midpoint within this one-parameter model.
    if total_occupied == 0:
        return {
            "epsilon": np.nan,
            "epsilon_se": np.nan,
            "n_mu_points": int(mu.size),
            "total_occupied": total_occupied,
            "total_frames": total_frames,
            "fit_rmse_p": np.nan,
            "log_likelihood": np.nan,
            "status": "all_empty",
        }

    if total_occupied == total_frames:
        return {
            "epsilon": np.nan,
            "epsilon_se": np.nan,
            "n_mu_points": int(mu.size),
            "total_occupied": total_occupied,
            "total_frames": total_frames,
            "fit_rmse_p": np.nan,
            "log_likelihood": np.nan,
            "status": "all_occupied",
        }

    beta = 1.0 / (K_B * T)

    # d(log L)/d(epsilon) = beta * sum_i [k_i - N_i p_i]
    # so the MLE satisfies sum_i N_i p_i = sum_i k_i.
    def score(epsilon):
        p_model = independent_site_occupancy(mu, epsilon, T=T)
        return np.sum(n * p_model - k)

    pad = FIT_BRACKET_KT * K_B * T
    lo = float(np.min(mu) - pad)
    hi = float(np.max(mu) + pad)

    f_lo = score(lo)
    f_hi = score(hi)

    if not (f_lo > 0 and f_hi < 0):
        # This should only occur for pathological numerical cases after the
        # all-empty/all-occupied checks above.
        return {
            "epsilon": np.nan,
            "epsilon_se": np.nan,
            "n_mu_points": int(mu.size),
            "total_occupied": total_occupied,
            "total_frames": total_frames,
            "fit_rmse_p": np.nan,
            "log_likelihood": np.nan,
            "status": "root_not_bracketed",
        }

    epsilon_hat = float(brentq(score, lo, hi))
    p_hat = independent_site_occupancy(mu, epsilon_hat, T=T)

    # Nominal binomial Fisher information for epsilon.
    fisher = (beta ** 2) * np.sum(n * p_hat * (1.0 - p_hat))
    epsilon_se = np.sqrt(1.0 / fisher) if fisher > 0 else np.nan

    p_obs = k / n
    fit_rmse_p = float(np.sqrt(np.mean((p_obs - p_hat) ** 2)))

    # Numerically stable log-likelihood (without log binomial coefficients).
    tiny = np.finfo(float).tiny
    p_clip = np.clip(p_hat, tiny, 1.0 - np.finfo(float).eps)
    log_likelihood = float(
        np.sum(k * np.log(p_clip) + (n - k) * np.log1p(-p_clip))
    )

    return {
        "epsilon": epsilon_hat,
        "epsilon_se": float(epsilon_se),
        "n_mu_points": int(mu.size),
        "total_occupied": total_occupied,
        "total_frames": total_frames,
        "fit_rmse_p": fit_rmse_p,
        "log_likelihood": log_likelihood,
        "status": "ok",
    }


# =============================================================================
# Outputs
# =============================================================================

def write_sites_pdb(out_pdb, centers, delta_g_bind, p_mu_eq):
    """
    Write usable hydration sites.

    PDB occupancy column: p(mu closest to mu_bulk)
    PDB B-factor column: bulk-referenced DeltaG_bind (kcal/mol)
    """
    with open(out_pdb, "w") as f:
        f.write(
            "REMARK Independent-site grand-canonical occupancy fit; "
            "B-factor = DeltaG_bind = epsilon - mu_bulk (kcal/mol)\n"
        )
        for i, (c, dg, occ) in enumerate(
            zip(centers, delta_g_bind, p_mu_eq), 1
        ):
            f.write(
                f"HETATM{i:5d}  O   SIT A{i:4d}    "
                f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}"
                f"{occ:6.2f}{dg:6.2f}\n"
            )
        f.write("END\n")


def write_sites_csv(
    out_csv,
    source_site_ids,
    centers,
    epsilon,
    epsilon_se,
    delta_g_bind,
    n_mu_points,
    p_mu_eq,
    fit_rmse_p,
    total_occupied,
    total_frames,
    clust_occs=None,
):
    """Write only sites with finite fitted free energies."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "site",
            "source_site_id",
            "x",
            "y",
            "z",
            "epsilon_midpoint",
            "epsilon_fit_se_binomial",
            "deltaG_bind",
            "deltaG_bind_se_binomial",
            "n_mu_points",
            "p_mu_eq",
            "fit_rmse_p",
            "total_occupied_frames",
            "total_frame_observations",
            # Backward-compatible alias so old plotting scripts that read
            # epsilon_mean do not silently plot the unreferenced midpoint.
            "epsilon_mean",
        ]
        if clust_occs is not None:
            header += ["density_peak_score"]

        w.writerow(header)

        for row_idx in range(len(centers)):
            row = [
                row_idx,
                int(source_site_ids[row_idx]),
                *centers[row_idx],
                epsilon[row_idx],
                epsilon_se[row_idx],
                delta_g_bind[row_idx],
                epsilon_se[row_idx],
                int(n_mu_points[row_idx]),
                p_mu_eq[row_idx],
                fit_rmse_p[row_idx],
                int(total_occupied[row_idx]),
                int(total_frames[row_idx]),
                # Intentionally alias epsilon_mean to DeltaG_bind for legacy
                # downstream scripts. New code should use deltaG_bind explicitly.
                delta_g_bind[row_idx],
            ]
            if clust_occs is not None:
                row += [clust_occs[row_idx]]
            w.writerow(row)


def write_sites_all_csv(
    out_csv,
    centers,
    epsilon,
    epsilon_se,
    delta_g_bind,
    n_mu_points,
    p_mu_eq,
    fit_rmse_p,
    total_occupied,
    total_frames,
    status,
    clust_occs=None,
):
    """Write every detected site, including thermodynamically unresolved ones."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "source_site_id",
            "x",
            "y",
            "z",
            "epsilon_midpoint",
            "epsilon_fit_se_binomial",
            "deltaG_bind",
            "deltaG_bind_se_binomial",
            "n_mu_points",
            "p_mu_eq",
            "fit_rmse_p",
            "total_occupied_frames",
            "total_frame_observations",
            "fit_status",
        ]
        if clust_occs is not None:
            header += ["density_peak_score"]

        w.writerow(header)

        for i in range(len(centers)):
            row = [
                i,
                *centers[i],
                epsilon[i],
                epsilon_se[i],
                delta_g_bind[i],
                epsilon_se[i],
                int(n_mu_points[i]),
                p_mu_eq[i],
                fit_rmse_p[i],
                int(total_occupied[i]),
                int(total_frames[i]),
                status[i],
            ]
            if clust_occs is not None:
                row += [clust_occs[i]]
            w.writerow(row)


def write_titration_csv(out_csv, mu_indices, mu_values, p):
    """Backward-compatible wide observed-occupancy table."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        labels = [
            f"p_muidx_{int(mi):03d}_mu_{float(mu):.6g}"
            for mi, mu in zip(mu_indices, mu_values)
        ]
        w.writerow(["source_site_id"] + labels)
        for i in range(p.shape[0]):
            w.writerow([i] + list(p[i]))


def write_titration_diagnostics_csv(
    out_csv,
    mu_indices,
    mu_values,
    p,
    occ_counts,
    n_frames,
    epsilon,
):
    """
    Long-format diagnostic table containing observed counts and fitted p(mu).
    This is useful for checking whether individual sites follow a clean sigmoid.
    """
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "source_site_id",
            "mu_index",
            "mu_kcal_mol",
            "n_frames",
            "n_occupied",
            "p_observed",
            "p_fitted",
        ])

        for site_i in range(p.shape[0]):
            if np.isfinite(epsilon[site_i]):
                p_fit = independent_site_occupancy(
                    mu_values, epsilon[site_i], T=TEMPERATURE
                )
            else:
                p_fit = np.full(len(mu_values), np.nan, dtype=float)

            for j, (mu_idx, mu) in enumerate(zip(mu_indices, mu_values)):
                w.writerow([
                    site_i,
                    int(mu_idx),
                    float(mu),
                    int(n_frames[j]),
                    int(occ_counts[site_i, j]),
                    float(p[site_i, j]) if n_frames[j] > 0 else np.nan,
                    float(p_fit[j]) if np.isfinite(p_fit[j]) else np.nan,
                ])


def plot_titration_curves(
    out_png,
    mu_values,
    p,
    n_frames,
    epsilon,
    mu_eq_index,
    max_sites=12,
):
    """Plot measured occupancies together with fitted bounded isotherms."""
    valid_mu = n_frames > 0
    if not np.any(valid_mu):
        print("[plot] No chemical potentials with frames — skipping.")
        return

    chosen = []
    for i in range(p.shape[0]):
        if not np.isfinite(epsilon[i]):
            continue

        p_valid = p[i, valid_mu]

        # Prefer sites that actually show a transition over the sampled range.
        if p_valid.size >= 3 and (np.max(p_valid) - np.min(p_valid)) >= 0.10:
            chosen.append(i)

        if len(chosen) >= max_sites:
            break

    # Fallback: plot any finite fits if few curves pass the transition criterion.
    if not chosen:
        chosen = [
            i for i in range(p.shape[0]) if np.isfinite(epsilon[i])
        ][:max_sites]

    if not chosen:
        print("[plot] No sites with finite thermodynamic fits — skipping.")
        return

    mu_min = float(np.min(mu_values[valid_mu]))
    mu_max = float(np.max(mu_values[valid_mu]))
    mu_dense = np.linspace(mu_min, mu_max, 500)

    plt.figure(figsize=(7, 5))

    for i in chosen:
        line, = plt.plot(
            mu_values[valid_mu],
            p[i, valid_mu],
            marker="o",
            linestyle="none",
            markersize=4,
            label=f"Site {i}",
        )

        p_dense = independent_site_occupancy(
            mu_dense, epsilon[i], T=TEMPERATURE
        )
        plt.plot(
            mu_dense,
            p_dense,
            linestyle="-",
            linewidth=1.2,
            color=line.get_color(),
        )

    plt.axvline(
        mu_values[mu_eq_index],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"bulk mu ~ {mu_values[mu_eq_index]:.2f}",
    )

    plt.xlabel("Chemical potential mu (kcal/mol)")
    plt.ylabel("Occupancy probability p(mu)")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[plot] wrote {out_png}")


# =============================================================================
# Config / geometry helpers
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)


def get_gc_box_from_config(cfg: dict):
    sb = cfg["simulation_box"]
    center = np.array(
        [sb["center_x"], sb["center_y"], sb["center_z"]],
        dtype=float,
    )
    halfsize = np.array(
        [sb["x_size"] / 2.0, sb["y_size"] / 2.0, sb["z_size"] / 2.0],
        dtype=float,
    )
    return center, halfsize


def in_gc_box(coords: np.ndarray, center: np.ndarray, halfsize: np.ndarray):
    d = np.abs(coords - center)
    return np.all(d <= halfsize, axis=-1)


# =============================================================================
# FRAME / PDB LOADING
# =============================================================================
# Expected layout:
# frames/
#   mu_000/
#     snap_00000.pdb
#   mu_001/
#     snap_00000.pdb
# ...
# =============================================================================

def detect_mu_and_frames(frames_dir):
    """
    Detect mu indices and frame indices from:
        frames/mu_###/snap_#####.pdb

    Returns:
        mu_indices: sorted list[int]
        frames_per_mu: dict[int, list[int]]
    """
    mu_pat = re.compile(r"^mu_(\d+)$")
    snap_pat = re.compile(r"^snap_(\d+)\.pdb$")

    frames_per_mu = {}

    if not os.path.isdir(frames_dir):
        raise RuntimeError(
            f"Frames dir does not exist or is not a directory: {frames_dir}"
        )

    for entry in os.listdir(frames_dir):
        mu_m = mu_pat.match(entry)
        if not mu_m:
            continue

        mu_idx = int(mu_m.group(1))
        mu_path = os.path.join(frames_dir, entry)

        if not os.path.isdir(mu_path):
            continue

        frame_idxs = []
        for fn in os.listdir(mu_path):
            m = snap_pat.match(fn)
            if not m:
                continue
            frame_idxs.append(int(m.group(1)))

        frame_idxs.sort()
        if frame_idxs:
            frames_per_mu[mu_idx] = frame_idxs

    if not frames_per_mu:
        raise RuntimeError(
            f"No mu_### folders with snap_#####.pdb found in {frames_dir}"
        )

    mu_indices = sorted(frames_per_mu.keys())
    return mu_indices, frames_per_mu


def list_frames_for_mu(frames_dir, mu_idx, pattern="snap_*.pdb"):
    """Return sorted PDB snapshot paths for a specific mu folder."""
    mu_dir = os.path.join(frames_dir, f"mu_{mu_idx:03d}")
    if not os.path.isdir(mu_dir):
        return []

    paths = glob.glob(os.path.join(mu_dir, pattern))

    def sort_key(path):
        m = re.search(r"snap_(\d+)\.pdb$", os.path.basename(path))
        return int(m.group(1)) if m else 10**12

    paths.sort(key=sort_key)
    return paths


def collect_all_points(
    frames_dir,
    mu_indices,
    frames_per_mu,
    gc_center,
    gc_halfsize,
):
    """
    Collect all water oxygens across all mu/snapshot files.

    gc_center/gc_halfsize are retained for compatibility with the original
    signature; the histogram itself discards coordinates outside its edges.
    """
    pts = []

    for mu_idx in mu_indices:
        mu_dir = os.path.join(frames_dir, f"mu_{mu_idx:03d}")
        for frame_idx in frames_per_mu[mu_idx]:
            pdb_path = os.path.join(mu_dir, f"snap_{frame_idx:05d}.pdb")
            coords = read_water_oxygen_coords_from_pdb(pdb_path)
            if coords.size:
                pts.append(coords)

    if not pts:
        raise RuntimeError("No water oxygens found in frames.")

    return np.vstack(pts)


def occupancy_probabilities_by_mu(
    frames_dir,
    mu_indices,
    mu_values,
    centers,
    cutoff=ASSIGNMENT_CUTOFF,
    frame_pattern="snap_*.pdb",
    verbose=True,
):
    """
    Compute binary site occupancy for every saved frame at every mu.

    A hydration site is counted at most once per frame. If several water
    oxygens are assigned to the same site, only the closest one is retained.

    Returns:
        p               shape (n_sites, n_mu), observed occupancy fractions
        occ_counts      shape (n_sites, n_mu), occupied-frame counts k_i
        n_frames        shape (n_mu,), number of frames N_i
    """
    mu_indices = list(mu_indices)
    mu_values = np.asarray(mu_values, dtype=float)

    if len(mu_indices) != len(mu_values):
        raise ValueError("mu_indices and mu_values must have the same length.")

    n_sites = len(centers)
    n_mu = len(mu_values)

    if n_sites == 0:
        return (
            np.zeros((0, n_mu), dtype=float),
            np.zeros((0, n_mu), dtype=int),
            np.zeros(n_mu, dtype=int),
        )

    tree = cKDTree(centers)
    p = np.zeros((n_sites, n_mu), dtype=float)
    occ_counts_by_mu = np.zeros((n_sites, n_mu), dtype=int)
    n_frames = np.zeros(n_mu, dtype=int)

    for j, (mu_idx, mu_value) in enumerate(zip(mu_indices, mu_values)):
        frames = list_frames_for_mu(
            frames_dir,
            mu_idx,
            pattern=frame_pattern,
        )

        if verbose:
            print(
                f"[mu folder {mu_idx:03d} | mu={mu_value:.6g}] "
                f"frames: {len(frames)}"
            )

        if not frames:
            continue

        occ_counts = np.zeros(n_sites, dtype=int)

        for fp in frames:
            xyz = read_water_oxygen_coords_from_pdb(fp)
            n_frames[j] += 1

            if xyz.shape[0] == 0:
                continue

            d, idx = tree.query(
                xyz,
                k=1,
                distance_upper_bound=cutoff,
            )

            valid = idx < n_sites
            if not np.any(valid):
                continue

            # Enforce at most one water per site per frame: choose closest.
            best = {}
            for dist, ci in zip(d[valid], idx[valid]):
                ci = int(ci)
                if (ci not in best) or (dist < best[ci]):
                    best[ci] = float(dist)

            for ci in best:
                occ_counts[ci] += 1

        occ_counts_by_mu[:, j] = occ_counts

        if n_frames[j] > 0:
            p[:, j] = occ_counts.astype(float) / float(n_frames[j])

    return p, occ_counts_by_mu, n_frames


# =============================================================================
# Density grid / site detection
# =============================================================================

def make_edges(center, halfsize, spacing):
    min_corner = center - halfsize
    max_corner = center + halfsize
    return [
        np.linspace(
            min_corner[d],
            max_corner[d],
            int(np.ceil(2 * halfsize[d] / spacing)) + 1,
        )
        for d in range(3)
    ]


def compute_density_grid(points, gc_center, gc_halfsize, edges):
    """3D histogram of all water O positions in the GC box."""
    H, edges = np.histogramdd(points, bins=edges)
    return H.astype(float), edges


def density_to_dx(H, edges, outpath):
    """Save volumetric density as OpenDX for visualization."""
    nx, ny, nz = H.shape

    x0 = edges[0][0]
    y0 = edges[1][0]
    z0 = edges[2][0]

    dx = edges[0][1] - edges[0][0]
    dy = edges[1][1] - edges[1][0]
    dz = edges[2][1] - edges[2][0]

    with open(outpath, "w") as f:
        f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        f.write(f"origin {x0} {y0} {z0}\n")
        f.write(f"delta {dx} 0 0\n")
        f.write(f"delta 0 {dy} 0\n")
        f.write(f"delta 0 0 {dz}\n")
        f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        f.write(
            f"object 3 class array type double rank 0 "
            f"items {nx * ny * nz} data follows\n"
        )

        flat = H.flatten(order="C")
        for value in flat:
            f.write(f"{value}\n")

        f.write('object "density" class field\n')
        f.write('component "positions" value 1\n')
        f.write('component "connections" value 2\n')
        f.write('component "data" value 3\n')

    print(f"DX map written: {outpath}")


def find_peaks(H_smooth, edges, percentile, neighborhood=1):
    """Find local maxima in the smoothed density grid."""
    nonzero = H_smooth[H_smooth > 0]
    if nonzero.size == 0:
        raise RuntimeError("No non-zero density in smoothed grid.")

    thresh = np.percentile(nonzero, percentile)
    footprint = np.ones((2 * neighborhood + 1,) * 3, dtype=bool)

    H_masked = np.where(H_smooth >= thresh, H_smooth, 0.0)
    H_max = maximum_filter(H_masked, footprint=footprint, mode="nearest")

    peak_mask = (H_masked == H_max) & (H_masked > 0)
    idxs = np.argwhere(peak_mask)

    if idxs.size == 0:
        return np.empty((0, 3)), np.empty(0)

    xs = 0.5 * (edges[0][:-1] + edges[0][1:])
    ys = 0.5 * (edges[1][:-1] + edges[1][1:])
    zs = 0.5 * (edges[2][:-1] + edges[2][1:])

    centers = np.column_stack([
        xs[idxs[:, 0]],
        ys[idxs[:, 1]],
        zs[idxs[:, 2]],
    ])
    scores = H_smooth[tuple(idxs.T)]

    return centers, scores


def merge_close_sites(centers, scores, cutoff):
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

    groups = {}
    for i in range(len(centers)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged_centers = []
    merged_scores = []

    for idxs in groups.values():
        w = scores[idxs]
        c = centers[idxs]
        total_weight = w.sum()

        merged_centers.append(
            (c * w[:, None]).sum(axis=0) / total_weight
        )
        merged_scores.append(total_weight)

    return np.array(merged_centers), np.array(merged_scores)


# =============================================================================
# Main analysis
# =============================================================================

# =============================================================================
# Capacity-safety helpers
# =============================================================================

def _count_insertion_points(points) -> int:
    """Infer the number of xyz insertion points returned by mcswell_cpp."""
    arr = np.asarray(points)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return int(arr.shape[0])
    if arr.size % 3 != 0:
        raise RuntimeError(
            f"Could not infer insertion-point count from shape {arr.shape}."
        )
    return int(arr.size // 3)


def reconstruct_sampler_volume(cfg: dict, work_dir: str):
    """
    Rebuild the MCSwell insertion cloud and return:
        sampler_volume_A3, n_insertion_points

    MCSwell uses:
        V_sampler = n_insertion_points * spacing^3

    Passing sampler_volume directly from run_mcswell.py is preferable because
    it guarantees use of the exact volume from the sampling run.
    """
    try:
        import mcswell_cpp as mc
        import run_mcswell as runner
    except Exception as exc:
        raise RuntimeError(
            "Automatic sampler-volume reconstruction requires the MCSwell "
            "Python environment. Prefer calling main(config, sampler_volume=...) "
            "from run_mcswell.py, or set gci.sampler_volume in the TOML."
        ) from exc

    os.makedirs(work_dir, exist_ok=True)

    receptor_path = cfg.get("receptor", {}).get("path")
    ligand_cfg = cfg.get("ligand", {})
    ligand_path = ligand_cfg.get("small_molecule_path")
    ligand_ff = ligand_cfg.get("small_molecule_forcefield")

    sb = cfg["simulation_box"]
    sim = cfg["simulation_parameters"]

    spacing = float(sb["spacing"])
    center = [
        float(sb["center_x"]),
        float(sb["center_y"]),
        float(sb["center_z"]),
    ]
    size = (
        float(sb["x_size"]),
        float(sb["y_size"]),
        float(sb["z_size"]),
    )
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
    sampler_volume = n_points * spacing**3

    return float(sampler_volume), int(n_points)


def resolve_sampler_volume(
    cfg: dict,
    out_dir: str,
    sampler_volume: Optional[float],
    n_insertion_points: Optional[int],
):
    """Resolve the exact volume used in MCSwell's Adams-parameter sampler."""
    spacing = float(cfg["simulation_box"]["spacing"])

    if sampler_volume is not None:
        if float(sampler_volume) <= 0:
            raise ValueError("sampler_volume must be > 0")
        return float(sampler_volume), None, "explicit_sampler_volume"

    if n_insertion_points is not None:
        if int(n_insertion_points) <= 0:
            raise ValueError("n_insertion_points must be > 0")
        volume = int(n_insertion_points) * spacing**3
        return float(volume), int(n_insertion_points), "explicit_n_insertion_points"

    work_dir = os.path.join(out_dir, "_volume_reconstruction")
    volume, npts = reconstruct_sampler_volume(cfg, work_dir)
    return volume, npts, "reconstructed_from_mcswell"


def scan_capacity_and_collect_density_points(
    frames_dir,
    mu_indices,
    mu_values,
    frames_per_mu,
    water_capacity: int,
    *,
    capacity_filter: bool = True,
    max_capacity_hit_fraction: float = DEFAULT_MAX_CAPACITY_HIT_FRACTION,
    max_mean_capacity_fraction: float = DEFAULT_MAX_MEAN_CAPACITY_FRACTION,
):
    """
    Scan total GCMC-water counts and define one contiguous capacity-safe prefix.

    The first window is excluded if either:
      * fraction(frames with N >= capacity) > max_capacity_hit_fraction, OR
      * mean(N) / capacity >= max_mean_capacity_fraction.

    Once a window is excluded, every later/higher-mu window is excluded too.
    Density points are collected ONLY from the retained windows.

    Returns
    -------
    all_points_valid : ndarray, shape (N, 3)
        Water oxygen coordinates from capacity-safe windows only.
    diagnostics : list[dict]
        One entry per chemical-potential window.
    valid_mask : ndarray[bool]
        Capacity-safe window mask in the same order as mu_values.
    """
    if water_capacity <= 0:
        raise ValueError("water_capacity must be positive")
    if not (0.0 <= max_capacity_hit_fraction <= 1.0):
        raise ValueError("max_capacity_hit_fraction must be in [0, 1]")
    if not (0.0 < max_mean_capacity_fraction <= 1.0):
        raise ValueError("max_mean_capacity_fraction must be in (0, 1]")

    mu_indices = list(mu_indices)
    mu_values = np.asarray(mu_values, dtype=float)

    if len(mu_indices) != len(mu_values):
        raise ValueError("mu_indices and mu_values must have the same length")

    diagnostics = []
    valid_mask = np.zeros(len(mu_values), dtype=bool)
    retained_points = []
    prefix_open = True

    for j, (mu_idx, mu) in enumerate(zip(mu_indices, mu_values)):
        counts = []
        window_points = []

        for frame_idx in frames_per_mu[mu_idx]:
            pdb_path = os.path.join(
                frames_dir,
                f"mu_{mu_idx:03d}",
                f"snap_{frame_idx:05d}.pdb",
            )
            xyz = read_water_oxygen_coords_from_pdb(pdb_path)
            n_waters = int(xyz.shape[0])
            counts.append(n_waters)

            if xyz.size:
                window_points.append(xyz)

        counts = np.asarray(counts, dtype=int)
        if counts.size == 0:
            mean_n = np.nan
            std_n = np.nan
            max_n = 0
            mean_fraction = np.nan
            hit_fraction = np.nan
            triggers_capacity = False
        else:
            mean_n = float(np.mean(counts))
            std_n = float(np.std(counts, ddof=1)) if counts.size >= 2 else 0.0
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

        if not capacity_filter:
            valid = True
            reason = "capacity_filter_disabled"
        elif prefix_open and triggers_capacity:
            prefix_open = False
            valid = False
            reason = "first_capacity_limited_window"
        elif prefix_open:
            valid = True
            reason = "capacity_safe"
        else:
            valid = False
            reason = "after_first_capacity_limited_window"

        valid_mask[j] = valid

        if valid and window_points:
            retained_points.extend(window_points)

        diagnostics.append({
            "mu_index": int(mu_idx),
            "mu_kcal_mol": float(mu),
            "n_frames": int(counts.size),
            "N_region_mean": mean_n,
            "N_region_std": std_n,
            "N_region_max_observed": max_n,
            "water_capacity": int(water_capacity),
            "capacity_mean_fraction": mean_fraction,
            "capacity_hit_fraction": hit_fraction,
            "valid_for_binomial": bool(valid),
            "capacity_filter_reason": reason,
        })

        print(
            f"[capacity] mu_{mu_idx:03d} ({mu:8.3f}): "
            f"<N>={mean_n:8.3f}, max={max_n:4d}/{water_capacity}, "
            f"mean/cap={mean_fraction:6.3f}, hit={hit_fraction:6.3f}, "
            f"valid={valid}"
        )

    if not retained_points:
        raise RuntimeError(
            "No water oxygen coordinates remain after capacity filtering."
        )

    return np.vstack(retained_points), diagnostics, valid_mask


def write_capacity_diagnostics_csv(out_csv, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_titration_diagnostics_capacity_safe_csv(
    out_csv,
    mu_indices,
    mu_values,
    p,
    occ_counts,
    n_frames,
    epsilon,
    valid_mask,
    capacity_diagnostics,
    T=TEMPERATURE,
):
    """
    Long-format site titration table.

    All sampled windows are retained in the CSV, but valid_for_binomial shows
    which contiguous capacity-safe prefix was actually used in the fit.
    """
    cap_by_position = list(capacity_diagnostics)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "source_site_id",
            "mu_index",
            "mu_kcal_mol",
            "n_frames",
            "n_occupied",
            "p_observed",
            "p_fitted",
            "valid_for_binomial",
            "N_region_mean",
            "water_capacity",
            "capacity_mean_fraction",
            "capacity_hit_fraction",
            "capacity_filter_reason",
        ])

        for site_i in range(p.shape[0]):
            if np.isfinite(epsilon[site_i]):
                p_fit = independent_site_occupancy(
                    mu_values,
                    epsilon[site_i],
                    T=T,
                )
            else:
                p_fit = np.full(len(mu_values), np.nan, dtype=float)

            for j, (mu_idx, mu) in enumerate(zip(mu_indices, mu_values)):
                cap = cap_by_position[j]
                w.writerow([
                    site_i,
                    int(mu_idx),
                    float(mu),
                    int(n_frames[j]),
                    int(occ_counts[site_i, j]),
                    float(p[site_i, j]) if n_frames[j] > 0 else np.nan,
                    float(p_fit[j]) if np.isfinite(p_fit[j]) else np.nan,
                    bool(valid_mask[j]),
                    cap["N_region_mean"],
                    cap["water_capacity"],
                    cap["capacity_mean_fraction"],
                    cap["capacity_hit_fraction"],
                    cap["capacity_filter_reason"],
                ])


def plot_titration_curves_capacity_safe(
    out_png,
    mu_values,
    p,
    n_frames,
    epsilon,
    mu_eq_index,
    valid_mask,
    *,
    T=TEMPERATURE,
    max_sites=12,
):
    """
    Plot measured occupancy over all sampled windows.

    Capacity-safe points used in the fit are circles; excluded high-mu windows
    are x markers.  The fitted sigmoid is based only on the valid prefix.
    """
    mu_values = np.asarray(mu_values, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    has_frames = np.asarray(n_frames) > 0
    fit_window_mask = has_frames & valid_mask
    excluded_mask = has_frames & (~valid_mask)

    if not np.any(fit_window_mask):
        print("[plot] No capacity-safe windows with frames — skipping.")
        return

    chosen = []
    for i in range(p.shape[0]):
        if not np.isfinite(epsilon[i]):
            continue
        pv = p[i, fit_window_mask]
        if pv.size >= 3 and (np.max(pv) - np.min(pv)) >= 0.10:
            chosen.append(i)
        if len(chosen) >= max_sites:
            break

    if not chosen:
        chosen = [
            i for i in range(p.shape[0]) if np.isfinite(epsilon[i])
        ][:max_sites]

    if not chosen:
        print("[plot] No finite binomial fits — skipping.")
        return

    mu_dense = np.linspace(
        float(np.min(mu_values[fit_window_mask])),
        float(np.max(mu_values[fit_window_mask])),
        500,
    )

    plt.figure(figsize=(7, 5))

    for i in chosen:
        line, = plt.plot(
            mu_values[fit_window_mask],
            p[i, fit_window_mask],
            marker="o",
            linestyle="none",
            markersize=4,
            label=f"Site {i}",
        )

        if np.any(excluded_mask):
            plt.plot(
                mu_values[excluded_mask],
                p[i, excluded_mask],
                marker="x",
                linestyle="none",
                markersize=4,
                color=line.get_color(),
                alpha=0.55,
            )

        plt.plot(
            mu_dense,
            independent_site_occupancy(mu_dense, epsilon[i], T=T),
            linestyle="-",
            linewidth=1.2,
            color=line.get_color(),
        )

    plt.axvline(
        mu_values[mu_eq_index],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"bulk mu ~ {mu_values[mu_eq_index]:.2f}",
    )

    if np.any(~valid_mask):
        first_bad = int(np.flatnonzero(~valid_mask)[0])
        plt.axvline(
            mu_values[first_bad],
            linestyle=":",
            linewidth=1.2,
            color="black",
            alpha=0.65,
            label="first excluded window",
        )

    plt.xlabel("Chemical potential mu (kcal/mol)")
    plt.ylabel("Occupancy probability p(mu)")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[plot] wrote {out_png}")


# =============================================================================
# Main capacity-safe binomial analysis
# =============================================================================

def run_analysis(
    cfg: dict,
    *,
    sampler_volume: Optional[float] = None,
    n_insertion_points: Optional[int] = None,
    output_dir: Optional[str] = None,
    capacity_filter: Optional[bool] = None,
    bulk_water_density: Optional[float] = None,
    max_capacity_hit_fraction: Optional[float] = None,
    max_mean_capacity_fraction: Optional[float] = None,
    peak_percentile: Optional[float] = None,
    temperature: Optional[float] = None,
    mu_bulk: Optional[float] = None,
    analysis_tag: Optional[str] = None,
):
    gci_cfg = cfg.get("gci", {})

    def pick(explicit, key, default):
        return explicit if explicit is not None else gci_cfg.get(key, default)

    peak_percentile = float(
        pick(peak_percentile, "peak_percentile", PEAK_PERCENTILE)
    )
    T = float(pick(temperature, "temperature", TEMPERATURE))
    mu_bulk_override = pick(mu_bulk, "mu_bulk", None)
    mu_bulk, mu_bulk_water_model, mu_bulk_source = resolve_mu_bulk(
        None if mu_bulk_override is None else float(mu_bulk_override)
    )
    analysis_tag = str(pick(analysis_tag, "analysis_tag", "binomial"))

    capacity_filter = bool(
        pick(capacity_filter, "capacity_filter", True)
    )
    bulk_water_density = float(
        pick(
            bulk_water_density,
            "bulk_water_density",
            MCSWELL_BULK_WATER_DENSITY,
        )
    )
    max_capacity_hit_fraction = float(
        pick(
            max_capacity_hit_fraction,
            "max_capacity_hit_fraction",
            DEFAULT_MAX_CAPACITY_HIT_FRACTION,
        )
    )
    max_mean_capacity_fraction = float(
        pick(
            max_mean_capacity_fraction,
            "max_mean_capacity_fraction",
            DEFAULT_MAX_MEAN_CAPACITY_FRACTION,
        )
    )

    sampler_volume = pick(sampler_volume, "sampler_volume", None)
    n_insertion_points = pick(
        n_insertion_points, "n_insertion_points", None
    )

    save_path = cfg["io"]["save_path"]
    frames_dir = os.path.join(save_path, "frames")

    # Preserve the existing matcher-compatible naming, e.g. 90.0_binomial.
    if output_dir is None:
        output_dir = os.path.join(
            save_path,
            "gci",
            f"{peak_percentile:.1f}_{analysis_tag}",
        )
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    gc_center, gc_halfsize = get_gc_box_from_config(cfg)

    print("\n=== Capacity-safe independent-site binomial analysis ===")
    print(f"Frames dir:               {frames_dir}")
    print(f"Output dir:               {out_dir}")
    print(f"Temperature:              {T:.2f} K")
    print(
        f"Bulk mu:                  {mu_bulk:.6f} kcal/mol "
        f"(water model: {mu_bulk_water_model or 'undetected'}, source: {mu_bulk_source})"
    )
    print(f"Peak percentile:          {peak_percentile:.1f}")
    print(f"Assignment cutoff:        {ASSIGNMENT_CUTOFF:.2f} A")

    mu_indices, frames_per_mu = detect_mu_and_frames(frames_dir)
    MU_VALUES = np.asarray(expand_ranges(cfg["mu_range"]), dtype=float)

    if min(mu_indices) < 0 or max(mu_indices) >= len(MU_VALUES):
        raise ValueError(
            f"mu_idx range ({min(mu_indices)}..{max(mu_indices)}) is "
            f"incompatible with configured mu_range length {len(MU_VALUES)}."
        )

    mu_indices_sorted = sorted(mu_indices)
    mu_values = MU_VALUES[mu_indices_sorted]

    if len(mu_indices_sorted) != len(MU_VALUES):
        print(
            f"WARNING: detected {len(mu_indices_sorted)} mu folders but "
            f"configuration contains {len(MU_VALUES)} chemical potentials. "
            "Only detected windows will be analyzed."
        )

    # Exact sampler volume -> current MCSwell buffer capacity.
    sampler_volume, npts, volume_source = resolve_sampler_volume(
        cfg,
        out_dir,
        sampler_volume,
        n_insertion_points,
    )
    water_capacity = int(
        math.floor(sampler_volume * bulk_water_density)
    )

    if water_capacity <= 0:
        raise RuntimeError(
            "Reconstructed MCSwell water-buffer capacity is zero."
        )

    print("\n=== Capacity-safety filter ===")
    print(f"Sampler volume:           {sampler_volume:.6f} A^3 ({volume_source})")
    if npts is not None:
        print(f"Insertion points:         {npts}")
    print(f"Bulk-water density:       {bulk_water_density:.6f} waters/A^3")
    print(f"Water-buffer capacity:    {water_capacity}")
    print(f"Capacity filter enabled:  {capacity_filter}")
    print(f"Max hit-cap fraction:     {max_capacity_hit_fraction:.4f}")
    print(f"Max mean/cap fraction:    {max_mean_capacity_fraction:.4f}")

    # Determine the valid contiguous prefix AND build density only from it.
    all_points, capacity_diagnostics, valid_mask = (
        scan_capacity_and_collect_density_points(
            frames_dir,
            mu_indices_sorted,
            mu_values,
            frames_per_mu,
            water_capacity,
            capacity_filter=capacity_filter,
            max_capacity_hit_fraction=max_capacity_hit_fraction,
            max_mean_capacity_fraction=max_mean_capacity_fraction,
        )
    )

    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid < 4:
        raise RuntimeError(
            f"Only {n_valid} chemical-potential windows remain after capacity "
            "filtering; too few for a meaningful titration fit."
        )

    mu_eq_index = int(np.argmin(np.abs(mu_values - mu_bulk)))
    if not valid_mask[mu_eq_index]:
        raise RuntimeError(
            f"The sampled window nearest bulk mu ({mu_values[mu_eq_index]:.3f}) "
            "is already capacity-limited. Post-processing cannot repair this. "
            "Increase the MCSwell water-buffer capacity and rerun."
        )

    valid_positions = np.flatnonzero(valid_mask)
    print(
        f"\n[capacity] Binomial fit will use {n_valid}/{len(mu_values)} "
        f"contiguous windows: mu={mu_values[valid_positions[0]]:.3f} .. "
        f"{mu_values[valid_positions[-1]]:.3f} kcal/mol"
    )

    if np.any(~valid_mask):
        first_bad = int(np.flatnonzero(~valid_mask)[0])
        print(
            f"[capacity] First excluded window: mu_{mu_indices_sorted[first_bad]:03d} "
            f"(mu={mu_values[first_bad]:.3f} kcal/mol)"
        )

    write_capacity_diagnostics_csv(
        os.path.join(out_dir, "capacity_diagnostics.csv"),
        capacity_diagnostics,
    )

    # -----------------------------------------------------------------
    # Capacity-safe density map and site identification
    # -----------------------------------------------------------------
    print(
        f"\n[density] Building hydration-site density from "
        f"{n_valid} capacity-safe windows..."
    )
    print(f"[density] Water oxygen points retained: {all_points.shape[0]}")

    edges = make_edges(
        gc_center,
        gc_halfsize,
        spacing=GRID_SPACING,
    )
    H, edges = compute_density_grid(
        all_points,
        gc_center,
        gc_halfsize,
        edges,
    )
    H_smooth = gaussian_filter(H, sigma=GRID_SIGMA)

    site_centers_raw, clust_occs_raw = find_peaks(
        H_smooth,
        edges,
        percentile=peak_percentile,
        neighborhood=1,
    )
    print(f"[density] Raw peaks found: {site_centers_raw.shape[0]}")

    site_centers, clust_occs = merge_close_sites(
        site_centers_raw,
        clust_occs_raw,
        cutoff=PEAK_MERGE_CUTOFF,
    )
    n_sites = site_centers.shape[0]
    print(f"[density] Merged hydration sites: {n_sites}")

    # -----------------------------------------------------------------
    # Compute observed binary occupancy for ALL windows.
    # Only valid_mask windows are passed to the MLE.
    # -----------------------------------------------------------------
    p, occ_counts, n_frames = occupancy_probabilities_by_mu(
        frames_dir,
        mu_indices_sorted,
        mu_values,
        site_centers,
        cutoff=ASSIGNMENT_CUTOFF,
        verbose=True,
    )

    mu_fit = mu_values[valid_mask]
    n_frames_fit = n_frames[valid_mask]

    # -----------------------------------------------------------------
    # Capacity-safe binomial MLE
    # -----------------------------------------------------------------
    epsilon = np.full(n_sites, np.nan, dtype=float)
    epsilon_se = np.full(n_sites, np.nan, dtype=float)
    delta_g_bind = np.full(n_sites, np.nan, dtype=float)
    n_mu_points = np.zeros(n_sites, dtype=int)
    fit_rmse_p = np.full(n_sites, np.nan, dtype=float)
    total_occupied = np.zeros(n_sites, dtype=int)
    total_frames = np.zeros(n_sites, dtype=int)
    status = np.empty(n_sites, dtype=object)

    print("\n[binomial] Fitting capacity-safe occupancy titrations...")

    for i in range(n_sites):
        fit = fit_epsilon_binomial(
            mu_fit,
            occ_counts[i, valid_mask],
            n_frames_fit,
            T=T,
        )

        epsilon[i] = fit["epsilon"]
        epsilon_se[i] = fit["epsilon_se"]
        n_mu_points[i] = fit["n_mu_points"]
        fit_rmse_p[i] = fit["fit_rmse_p"]
        total_occupied[i] = fit["total_occupied"]
        total_frames[i] = fit["total_frames"]
        status[i] = fit["status"]

        if np.isfinite(epsilon[i]):
            delta_g_bind[i] = epsilon[i] - mu_bulk

    p_mu_eq_all = p[:, mu_eq_index]

    print(
        f"[binomial] Nearest sampled bulk mu: "
        f"{mu_values[mu_eq_index]:.3f} kcal/mol "
        f"(mu_{mu_indices_sorted[mu_eq_index]:03d})"
    )

    usable = np.isfinite(delta_g_bind) & (status == "ok")
    print(
        f"[binomial] Usable thermodynamic sites: "
        f"{int(np.count_nonzero(usable))}/{n_sites}"
    )

    unresolved_ids = np.where(~usable)[0]
    if unresolved_ids.size:
        print("[binomial] Unresolved site fits:")
        for site_id in unresolved_ids:
            print(
                f"    site {site_id}: status={status[site_id]}, "
                f"occupied={total_occupied[site_id]}/{total_frames[site_id]}"
            )

    # -----------------------------------------------------------------
    # Diagnostics: preserve all sampled windows, but flag excluded ones.
    # -----------------------------------------------------------------
    write_sites_all_csv(
        os.path.join(out_dir, "sites_all.csv"),
        site_centers,
        epsilon,
        epsilon_se,
        delta_g_bind,
        n_mu_points,
        p_mu_eq_all,
        fit_rmse_p,
        total_occupied,
        total_frames,
        status,
        clust_occs=clust_occs,
    )

    write_titration_csv(
        os.path.join(out_dir, "titration.csv"),
        mu_indices_sorted,
        mu_values,
        p,
    )

    write_titration_diagnostics_capacity_safe_csv(
        os.path.join(out_dir, "titration_diagnostics.csv"),
        mu_indices_sorted,
        mu_values,
        p,
        occ_counts,
        n_frames,
        epsilon,
        valid_mask,
        capacity_diagnostics,
        T=T,
    )

    plot_titration_curves_capacity_safe(
        os.path.join(out_dir, "titration_curves.png"),
        mu_values,
        p,
        n_frames,
        epsilon,
        mu_eq_index,
        valid_mask,
        T=T,
    )

    density_to_dx(
        H_smooth,
        edges,
        os.path.join(out_dir, "gO.dx"),
    )

    # -----------------------------------------------------------------
    # Matcher-compatible usable-site outputs.
    # -----------------------------------------------------------------
    source_site_ids = np.where(usable)[0]
    centers_usable = site_centers[usable]
    epsilon_usable = epsilon[usable]
    epsilon_se_usable = epsilon_se[usable]
    delta_g_usable = delta_g_bind[usable]
    n_mu_points_usable = n_mu_points[usable]
    p_mu_eq_usable = p_mu_eq_all[usable]
    fit_rmse_p_usable = fit_rmse_p[usable]
    total_occupied_usable = total_occupied[usable]
    total_frames_usable = total_frames[usable]
    clust_occs_usable = np.asarray(clust_occs)[usable]

    write_sites_pdb(
        os.path.join(out_dir, "sites.pdb"),
        centers_usable,
        delta_g_usable,
        p_mu_eq_usable,
    )

    write_sites_csv(
        os.path.join(out_dir, "sites.csv"),
        source_site_ids,
        centers_usable,
        epsilon_usable,
        epsilon_se_usable,
        delta_g_usable,
        n_mu_points_usable,
        p_mu_eq_usable,
        fit_rmse_p_usable,
        total_occupied_usable,
        total_frames_usable,
        clust_occs=clust_occs_usable,
    )

    metadata = {
        "method": "capacity-safe bounded independent-site binomial MLE",
        "temperature_K": T,
        "bulk_mu_kcal_mol": mu_bulk,
        "mu_bulk_water_model": mu_bulk_water_model,
        "mu_bulk_source": mu_bulk_source,
        "assignment_cutoff_A": ASSIGNMENT_CUTOFF,
        "peak_percentile": peak_percentile,
        "sampler_volume_A3": sampler_volume,
        "sampler_volume_source": volume_source,
        "n_insertion_points": npts,
        "bulk_water_density_A-3": bulk_water_density,
        "reconstructed_water_capacity": water_capacity,
        "capacity_filter_enabled": capacity_filter,
        "max_capacity_hit_fraction": max_capacity_hit_fraction,
        "max_mean_capacity_fraction": max_mean_capacity_fraction,
        "n_sampled_mu_windows": int(len(mu_values)),
        "n_capacity_safe_mu_windows": n_valid,
        "fit_mu_min_kcal_mol": float(mu_fit[0]),
        "fit_mu_max_kcal_mol": float(mu_fit[-1]),
        "nearest_bulk_mu_kcal_mol": float(mu_values[mu_eq_index]),
        "n_detected_sites": int(n_sites),
        "n_usable_sites": int(np.count_nonzero(usable)),
    }

    with open(
        os.path.join(out_dir, "binomial_metadata.json"),
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Capacity-safe binomial analysis complete ===")
    print(f"Output directory: {out_dir}")
    print(
        f"Fit range: mu={mu_fit[0]:.3f} .. {mu_fit[-1]:.3f} kcal/mol "
        f"({n_valid}/{len(mu_values)} windows)"
    )
    print(
        "For RETI comparison use sites.csv column 'deltaG_bind' "
        "(or the sites.pdb B-factor)."
    )

    return out_dir


def main(
    config,
    *,
    sampler_volume: Optional[float] = None,
    n_insertion_points: Optional[int] = None,
    output_dir: Optional[str] = None,
    **overrides,
):
    """
    Module entry point.

    Examples
    --------
    From run_mcswell.py:

        fe.main(
            config,
            sampler_volume=run_info["sampler_volume"],
        )

    Standalone code may also pass a TOML path as ``config``.
    """
    if isinstance(config, (str, os.PathLike)):
        cfg = load_config(str(config))
    elif isinstance(config, dict):
        cfg = config
    else:
        raise TypeError("config must be a dict or TOML path")

    return run_analysis(
        cfg,
        sampler_volume=sampler_volume,
        n_insertion_points=n_insertion_points,
        output_dir=output_dir,
        **overrides,
    )


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Capacity-safe bounded independent-site/binomial thermodynamic "
            "analysis for MCSwell snapshots."
        )
    )
    p.add_argument("config", help="MCSwell TOML configuration")
    p.add_argument("--sampler-volume", type=float, default=None)
    p.add_argument("--n-insertion-points", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--peak-percentile", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--mu-bulk", type=float, default=None)
    p.add_argument("--analysis-tag", default=None)

    p.add_argument(
        "--no-capacity-filter",
        dest="capacity_filter",
        action="store_false",
        help="Disable exclusion of capacity-limited high-mu windows.",
    )
    p.set_defaults(capacity_filter=None)

    p.add_argument("--bulk-water-density", type=float, default=None)
    p.add_argument("--max-capacity-hit-fraction", type=float, default=None)
    p.add_argument("--max-mean-capacity-fraction", type=float, default=None)

    return p


def cli(argv=None):
    args = build_parser().parse_args(argv)

    kwargs = {
        "sampler_volume": args.sampler_volume,
        "n_insertion_points": args.n_insertion_points,
        "output_dir": args.output_dir,
        "peak_percentile": args.peak_percentile,
        "temperature": args.temperature,
        "mu_bulk": args.mu_bulk,
        "analysis_tag": args.analysis_tag,
        "capacity_filter": args.capacity_filter,
        "bulk_water_density": args.bulk_water_density,
        "max_capacity_hit_fraction": args.max_capacity_hit_fraction,
        "max_mean_capacity_fraction": args.max_mean_capacity_fraction,
    }

    # Remove unset optional CLI values so config defaults can take effect.
    kwargs = {
        k: v for k, v in kwargs.items()
        if v is not None
    }

    return main(args.config, **kwargs)


if __name__ == "__main__":
    cli()
