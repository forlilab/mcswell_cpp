#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#   Copyright (c) 2026 Scripps Research, Forli Lab.
#   All rights reserved.
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
Plotting for MCSwell's ProtoMS-style GCI (whole-region + local
logistic-sum) analysis.

All of the actual numeric work formerly done here (reading per-snapshot
PDB frames, density-grid site detection, whole-region and per-site
logistic-sum fitting, and the integer-state PMF) now happens in C++ -- see
include/analysis/pipeline_analysis.hpp -- operating on the GCMC titration's
in-memory snapshot buffer instead of files on disk. This module only reads
the small CSV outputs C++ already wrote (region_titration.csv,
local_titration.csv, region_gci_model.csv, local_gci_models.csv) and draws
the diagnostic plots; it never touches a per-snapshot PDB.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_float(value):
    return float(value) if value not in ("", "nan") else float("nan")


def predict_logistic_sum(B, n_min, terms):
    """Re-evaluates predict(B) = n_min + sum_j amp_j*sigmoid(slope_j*(B-center_j))
    from the fitted (amplitude, slope, center) terms written by C++
    (region_gci_model.csv / local_gci_models.csv). This is the same
    one-line sigmoid sum as the C++ fit, re-derived here only so the plot
    can draw a smooth curve; the fit itself happens in C++."""
    B = np.asarray(B, dtype=float)
    out = np.full(B.shape, float(n_min))
    for amp, slope, center in terms:
        out = out + amp * (1.0 / (1.0 + np.exp(-slope * (B - center))))
    return out


def read_region_titration(out_dir):
    path = os.path.join(out_dir, "region_titration.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    B = np.zeros(n)
    N = np.zeros(n)
    valid_mask = np.zeros(n, dtype=bool)
    for r in rows:
        i = int(r["mu_index"])
        B[i] = float(r["B_sampler"])
        N[i] = float(r["N_region_mean"])
        valid_mask[i] = r["valid_for_gci"] == "True"
    return B, N, valid_mask


def read_region_model(out_dir):
    path = os.path.join(out_dir, "region_gci_model.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, []
    n_min = float(rows[0]["n_min"])
    terms = [(float(r["amplitude"]), float(r["slope"]), float(r["center"])) for r in rows]
    return n_min, terms


def read_local_titration(out_dir):
    path = os.path.join(out_dir, "local_titration.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    sites = sorted({int(r["site"]) for r in rows})
    mu_indices = sorted({int(r["mu_index"]) for r in rows})
    site_pos = {s: i for i, s in enumerate(sites)}
    mu_pos = {m: i for i, m in enumerate(mu_indices)}

    n_sites, n_mu = len(sites), len(mu_indices)
    B = np.zeros(n_mu)
    local_mean = np.zeros((n_sites, n_mu))
    valid_mask = np.zeros(n_mu, dtype=bool)
    for r in rows:
        si, mi = site_pos[int(r["site"])], mu_pos[int(r["mu_index"])]
        B[mi] = float(r["B_sampler"])
        local_mean[si, mi] = float(r["N_local_mean"])
        valid_mask[mi] = r["valid_for_gci"] == "True"
    return sites, B, local_mean, valid_mask


def read_local_models(out_dir):
    path = os.path.join(out_dir, "local_gci_models.csv")
    models = {}  # site_id -> (n_min, [(amp, slope, center), ...])
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            site_id = int(r["site"])
            n_min, terms = models.setdefault(site_id, (float(r["n_min"]), []))
            terms.append((float(r["amplitude"]), float(r["slope"]), float(r["center"])))
    return models


def plot_region_titration(out_dir, out_png=None):
    B, N, valid_mask = read_region_titration(out_dir)
    n_min, terms = read_region_model(out_dir)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(B[valid_mask], N[valid_mask], s=28, label="Used for GCI fit")
    if np.any(~valid_mask):
        plt.scatter(B[~valid_mask], N[~valid_mask], s=36, marker="x", label="Excluded: capacity-limited")

    if terms:
        x = np.linspace(B[valid_mask].min() - 2.0, B[valid_mask].max() + 2.0, 600)
        plt.plot(x, predict_logistic_sum(x, n_min, terms), lw=2, label="Monotone logistic-sum fit")

    plt.xlabel("Adams parameter B")
    plt.ylabel("Mean number of GCMC waters <N>")
    plt.title("MCSwell whole-region GCMC titration")
    plt.legend(frameon=False)
    plt.tight_layout()

    out_png = out_png or os.path.join(out_dir, "region_titration.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


def plot_local_titrations(out_dir, max_sites=12, out_png=None):
    sites, B, local_mean, valid_mask = read_local_titration(out_dir)
    models = read_local_models(out_dir)

    # Prefer sites with a usable fit, ranked by occupancy at the last valid window.
    fit_sites = [s for s in sites if s in models]
    if not fit_sites:
        print("[plot] No sites with a finite local GCI fit -- skipping.")
        return
    order = sorted(fit_sites, key=lambda s: -local_mean[sites.index(s), valid_mask][-1] if np.any(valid_mask) else 0)
    chosen = order[:max_sites]

    x = np.linspace(B[valid_mask].min() - 2.0, B[valid_mask].max() + 2.0, 600)
    fig = plt.figure(figsize=(8, 5.5))
    for site_id in chosen:
        row = sites.index(site_id)
        plt.scatter(B[valid_mask], local_mean[row, valid_mask], s=14, alpha=0.55)
        if np.any(~valid_mask):
            plt.scatter(B[~valid_mask], local_mean[row, ~valid_mask], s=20, marker="x", alpha=0.35)
        n_min, terms = models[site_id]
        plt.plot(x, predict_logistic_sum(x, n_min, terms), lw=1.4, label=f"Site {site_id}")

    plt.xlabel("Adams parameter B")
    plt.ylabel("Mean waters in local sphere <N_site>")
    plt.ylim(bottom=-0.05)
    plt.title("Local GCI titration curves (x = excluded capacity-limited windows)")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()

    out_png = out_png or os.path.join(out_dir, "local_titration_curves.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


def main(out_dir, max_sites=12):
    plot_region_titration(out_dir)
    plot_local_titrations(out_dir, max_sites=max_sites)
    print(f"For RETI comparison use {out_dir}/sites.csv column 'deltaG_bind_first_water'.")


def build_parser():
    p = argparse.ArgumentParser(
        description="Plot MCSwell ProtoMS-style GCI titration curves from the C++ analysis pipeline's CSV output."
    )
    p.add_argument("out_dir", help="GCI analysis output directory (save_path/gci/<pct>_protoms_gci)")
    p.add_argument("--max-sites", type=int, default=12)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args.out_dir, max_sites=args.max_sites)
