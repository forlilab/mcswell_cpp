#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, csv, glob, random
from dataclasses import dataclass
from collections import defaultdict

import tomli
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt
import MDAnalysis as mda
import hdbscan
from utils import *


K_B = 0.0019872041

# Physical parameters
TEMPERATURE = 300.0  # K
MU_BULK     = -6.09  # kcal/mol (bulk water chemical potential)
P_MIN       = 0.01   # min occupancy for ε fit for discrete sites

# Discrete site assignment cutoff (Å)
ASSIGNMENT_CUTOFF = 2.4

# Grid parameters
GRID_SPACING = 0.5   # Å voxel spacing
GRID_SIGMA   = 1.4   # Gaussian smoothing sigma (voxels)

# Peak / zone thresholds (percentiles on non-zero density)
PEAK_PERCENTILE = 90 # high-density peaks for discrete sites

# Peak merging cutoff (Å)
PEAK_MERGE_CUTOFF = 1.4

# -----------------------------
# IO / PDB parsing
# -----------------------------
def read_water_oxygen_coords_from_pdb(
    pdb_path,
    accept_resnames=("HOH", "WAT", "H2O", "TIP3", "SOL", "WA1"),
    accept_atoms=("O", "OW", "O1", "OH2", "O00"),
):
    """Return (N,3) float32 array of water oxygen coords from a frame PDB."""
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
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def sort_clusters_by_occupancy(clust_ids):
    n_clusts = int(max(clust_ids))
    occ = [[i, list(clust_ids).count(i)] for i in range(1, n_clusts + 1)]
    occ = sorted(occ, key=lambda x: -x[1])
    old_order = [x[0] for x in occ]
    clust_occs = [x[1] for x in occ]
    clust_ids_sorted = np.asarray([old_order.index(x) + 1 for x in clust_ids], dtype=int)
    return clust_ids_sorted, clust_occs


# -----------------------------
# Assign frames to clusters (per mu) and compute p(mu)
# -----------------------------

def epsilon_from_p_logit(mu_values, nbar_profile, T, p_low=0.02):
    mu = np.asarray(mu_values, float)
    nbar = np.asarray(nbar_profile, float)

    kT = K_B * T
    mask = nbar > p_low

    if not np.any(mask):
        return np.nan, np.nan, 0

    eps = mu[mask] - kT * np.log(nbar[mask])
    return eps.mean(), eps.std(), eps.size

# -----------------------------
# Outputs
# -----------------------------
def write_sites_pdb(out_pdb, centers, eps, p_mu_eq, occ_counts_mu_eq=None):
    """
    PDB columns mimic ProtoMS style:
    - occupancy column: p(mu_eq)
    - B-factor: epsilon
    (Optionally: write raw obs-counts as B-factor alt, but default is epsilon.)
    """
    with open(out_pdb, "w") as f:
        f.write("REMARK ProtoMS-style clusters + GCI epsilon\n")
        for i, (c, e, o) in enumerate(zip(centers, eps, p_mu_eq), 1):
            f.write(
                f"HETATM{i:5d}  O   SIT A{i:4d}    "
                f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}"
                f"{o:6.2f}{e:6.2f}\n"
            )
        f.write("END\n")

def write_sites_csv(out_csv, centers, eps, eps_std, npts, p_mu_eq, clust_occs=None):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["site","x","y","z","epsilon_mean","epsilon_std","n_mu_points","p_mu_eq"]
        if clust_occs is not None:
            header += ["obs_count_mu_eq_clustering"]
        w.writerow(header)
        for i in range(len(centers)):
            row = [i, *centers[i], eps[i], eps_std[i], npts[i], p_mu_eq[i]]
            if clust_occs is not None:
                row += [clust_occs[i]]
            w.writerow(row)

def write_titration_csv(out_csv, mu_values, p):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site"] + [f"p_mu_{i:02d}" for i in range(len(mu_values))])
        for i in range(p.shape[0]):
            w.writerow([i] + list(p[i]))

def plot_titration_curves(out_png, mu_values, p, n_frames, mu_eq_index, max_sites=12):
    valid_mu = n_frames > 0
    chosen = []
    for i in range(p.shape[0]):
        if np.count_nonzero((p[i] > 0.05) & valid_mu) >= 3:
            chosen.append(i)
        if len(chosen) >= max_sites:
            break
    if not chosen:
        print("[plot] No sites with enough data — skipping.")
        return

    plt.figure(figsize=(6,4))
    for i in chosen:
        plt.plot(mu_values[valid_mu], p[i, valid_mu], marker="o", label=f"Site {i}")
    plt.axvline(mu_values[mu_eq_index], ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Chemical potential μ (kcal/mol)")
    plt.ylabel("Occupancy probability p(μ)")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[plot] wrote {out_png}")
    
# prova
def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)


def get_gc_box_from_config(cfg: dict):
    sb = cfg["simulation_box"]
    center = np.array(
        [sb["center_x"], sb["center_y"], sb["center_z"]],
        dtype=float
    )
    halfsize = np.array(
        [sb["x_size"]/2.0, sb["y_size"]/2.0, sb["z_size"]/2.0],
        dtype=float
    )
    return center, halfsize


def in_gc_box(coords: np.ndarray, center: np.ndarray, halfsize: np.ndarray):
    d = np.abs(coords - center)
    return np.all(d <= halfsize, axis=-1)


# ======================================================
# FRAME / PDB LOADING
# ======================================================
# -----------------------------
# New folder-based layout helpers
# frames/
#   mu_000/
#     snap_00000.pdb
#   mu_001/
#     snap_00000.pdb
# -----------------------------

def detect_mu_and_frames(frames_dir):
    """
    Detect μ indices and frame indices from folder structure:
      frames/mu_000/snap_00000.pdb
    Returns:
      mu_indices: sorted list[int]
      frames_per_mu: dict[int, list[int]]  (frame indices per mu)
    """
    mu_pat = re.compile(r"^mu_(\d+)$")
    snap_pat = re.compile(r"^snap_(\d+)\.pdb$")

    frames_per_mu = {}

    if not os.path.isdir(frames_dir):
        raise RuntimeError(f"Frames dir does not exist or is not a dir: {frames_dir}")

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
    """
    List frame pdbs for a given mu_idx inside frames/mu_###/.
    Returns sorted list of absolute paths.
    """
    mu_dir = os.path.join(frames_dir, f"mu_{mu_idx:03d}")
    if not os.path.isdir(mu_dir):
        return []

    paths = glob.glob(os.path.join(mu_dir, pattern))

    def sort_key(p):
        m = re.search(r"snap_(\d+)\.pdb$", os.path.basename(p))
        return int(m.group(1)) if m else 10**12

    paths.sort(key=sort_key)
    return paths


def collect_all_points(frames_dir, mu_indices, frames_per_mu, gc_center, gc_halfsize):
    """
    Collect all water oxygens across all mu/snap files.
    (gc_center/gc_halfsize kept for signature compatibility; filtering not applied here,
     same as your current behavior.)
    """
    pts = []
    for mu_idx in mu_indices:
        mu_dir = os.path.join(frames_dir, f"mu_{mu_idx:03d}")
        for fr in frames_per_mu[mu_idx]:
            pdb_path = os.path.join(mu_dir, f"snap_{fr:05d}.pdb")
            coords = read_water_oxygen_coords_from_pdb(pdb_path)
            if coords.size:
                pts.append(coords)

    if not pts:
        raise RuntimeError("No water oxygens found in frames.")
    return np.vstack(pts)


def occupancy_probabilities_by_mu(
    frames_dir,
    mu_values,
    centers,
    cutoff=2.4,
    frame_pattern="snap_*.pdb",
    verbose=True
):
    """
    Folder layout version:
      frames/mu_###/snap_#####.pdb
    """
    n_sites = len(centers)
    n_mu = len(mu_values)
    if n_sites == 0:
        return np.zeros((0, n_mu), float), np.zeros(n_mu, int)

    tree = cKDTree(centers)
    p = np.zeros((n_sites, n_mu), float)
    n_frames = np.zeros(n_mu, int)

    for mi in range(n_mu):
        frames = list_frames_for_mu(frames_dir, mi, pattern=frame_pattern)
        if verbose:
            print(f"[mu {mi:03d} | {mu_values[mi]:.6g}] frames: {len(frames)}")
        if not frames:
            continue

        occ_counts = np.zeros(n_sites, int)

        for fp in frames:
            xyz = read_water_oxygen_coords_from_pdb(fp)
            n_frames[mi] += 1
            if xyz.shape[0] == 0:
                continue

            d, idx = tree.query(xyz, k=1, distance_upper_bound=cutoff)
            valid = idx < n_sites
            if not np.any(valid):
                continue

            # enforce at most one water per cluster per frame: choose closest
            best = {}
            for dist, ci in zip(d[valid], idx[valid]):
                ci = int(ci)
                if (ci not in best) or (dist < best[ci]):
                    best[ci] = float(dist)
            for ci in best.keys():
                occ_counts[ci] += 1

        if n_frames[mi] > 0:
            p[:, mi] = occ_counts / float(n_frames[mi])

    return p, n_frames


def make_edges(center, halfsize, spacing):
    min_corner = center - halfsize
    max_corner = center + halfsize
    return [np.linspace(min_corner[d], max_corner[d],
                         int(np.ceil(2 * halfsize[d] / spacing)) + 1)
            for d in range(3)]

def compute_density_grid(points, gc_center, gc_halfsize, edges):
    """
    3D histogram of all water O positions in GC box.
    Axes order: x, y, z  -> H[ix, iy, iz].
    """
    min_corner = gc_center - gc_halfsize
    max_corner = gc_center + gc_halfsize

    H, edges = np.histogramdd(points, bins=edges)
    return H.astype(float), edges


def density_to_dx(H, edges, outpath):
    """
    Save volumetric density as OpenDX (for PyMOL isosurface).
    """
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
        f.write(f"object 3 class array type double rank 0 items {nx*ny*nz} data follows\n")

        flat = H.flatten(order="C")
        for v in flat:
            f.write(f"{v}\n")

        f.write("object \"density\" class field\n")
        f.write("component \"positions\" value 1\n")
        f.write("component \"connections\" value 2\n")
        f.write("component \"data\" value 3\n")

    print(f"DX map written: {outpath}")


def find_peaks(H_smooth, edges, percentile, neighborhood=1):
    """
    Find local maxima in smoothed density grid.
    Returns peak centers and peak scores.
    """

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
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged_centers = []
    merged_scores = []

    for idxs in groups.values():
        w = scores[idxs]
        c = centers[idxs]
        tot = w.sum()
        merged_centers.append((c * w[:, None]).sum(axis=0) / tot)
        merged_scores.append(tot)

    return np.array(merged_centers), np.array(merged_scores)

def main(cfg):
    save_path  = cfg["io"]["save_path"]
    peak_percentile = cfg["gci"]["peak_percentile"]
    # peak_percentiles = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
    # for peak_percentile in peak_percentiles:
    frames_dir = os.path.join(save_path, "frames")
    out_dir    = os.path.join(save_path, "gci", f"{peak_percentile}")
    os.makedirs(out_dir, exist_ok=True)

    gc_center, gc_halfsize = get_gc_box_from_config(cfg)

    print(f"Frames dir: {frames_dir}")
    print(f"Output dir: {out_dir}")
    print(f"GC center:   {gc_center}")
    print(f"GC halfsize: {gc_halfsize}")

    print("Detecting μ indices and frames...")
    mu_indices, frames_per_mu = detect_mu_and_frames(frames_dir)
    print(f"Found {len(mu_indices)} μ indices and "
        f"{sum(len(v) for v in frames_per_mu.values())} frames total.")

    MU_VALUES = expand_ranges(cfg["mu_range"])
    
    # Sanity check vs MU_VALUES
    if len(mu_indices) != len(MU_VALUES):
        print(
            f"WARNING: detected {len(mu_indices)} mu indices but MU_VALUES has "
            f"length {len(MU_VALUES)}."
        )
    if min(mu_indices) != 0 or max(mu_indices) >= len(MU_VALUES):
        raise ValueError(
            f"mu_idx range ({min(mu_indices)}..{max(mu_indices)}) "
            f"is incompatible with MU_VALUES length {len(MU_VALUES)}."
        )

    mu_indices_sorted = sorted(mu_indices)
    mu_values = MU_VALUES[mu_indices_sorted]

    print("Collecting all water O positions for density grid...")
    all_points = collect_all_points(frames_dir, mu_indices_sorted, frames_per_mu,
                                    gc_center, gc_halfsize)
    print(f"Total points collected: {all_points.shape[0]}")
    edges = make_edges(gc_center, gc_halfsize, spacing=GRID_SPACING)
    print("Computing density grid...")
    H, edges = compute_density_grid(all_points, gc_center, gc_halfsize, edges)
    H_smooth = gaussian_filter(H, sigma=GRID_SIGMA)

    print("Detecting discrete sites as density peaks...")
    site_centers_raw, clust_occs_raw = find_peaks(H_smooth, edges,
                                                percentile=peak_percentile,
                                                neighborhood=1)
    print(f"Raw peaks found: {site_centers_raw.shape[0]}")

    print(f"Merging peaks closer than {PEAK_MERGE_CUTOFF:.1f} Å...")
    site_centers, clust_occs = merge_close_sites(site_centers_raw,
                                                clust_occs_raw,
                                                cutoff=PEAK_MERGE_CUTOFF)
    n_sites = site_centers.shape[0]
    print(f"[build] clustered sites: {n_sites}")
    cutoff = 2.4

    # --- Occupancy probabilities across mu ---
    p, n_frames = occupancy_probabilities_by_mu(
        frames_dir, mu_values, site_centers, cutoff=cutoff, verbose=True
    )    
    centers = site_centers
    # --- Fit epsilon per site ---
    eps = np.full(len(centers), np.nan, float)
    eps_std = np.full(len(centers), np.nan, float)
    npts = np.zeros(len(centers), int)

    for i in range(len(centers)):
        eps[i], eps_std[i], npts[i] = epsilon_from_p_logit(mu_values, p[i], T=300, p_low=0.02)

    usable = (npts >= 1) & np.isfinite(eps)
    print(f"[filter] usable sites: {usable.sum()}/{len(usable)}")

    centers = centers[usable]
    p = p[usable]
    eps = eps[usable]
    eps_std = eps_std[usable]
    npts = npts[usable]
    clust_occs = [clust_occs[i] for i, u in enumerate(usable) if u]
    mu_eq_index = closest_index(mu_values, MU_BULK)
    print(f"Closest chemical potential to the bulk ({MU_BULK}): {mu_values[mu_eq_index]}")
    p_mu_eq = p[:, mu_eq_index]

    write_sites_pdb(os.path.join(out_dir, "sites.pdb"), centers, eps, p_mu_eq)
    write_sites_csv(os.path.join(out_dir, "sites.csv"), centers, eps, eps_std, npts, p_mu_eq, clust_occs=clust_occs)
    write_titration_csv(os.path.join(out_dir, "titration.csv"), mu_values, p)
    plot_titration_curves(os.path.join(out_dir, "titration_curves.png"), mu_values, p, n_frames, mu_eq_index)
    density_to_dx(H_smooth, edges, os.path.join(out_dir, "gO.dx"))
    print(f"[done] outputs in: {out_dir}")
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python protoms_free_energies.py mcswell_config.toml")
        sys.exit(1)

    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    main(cfg)