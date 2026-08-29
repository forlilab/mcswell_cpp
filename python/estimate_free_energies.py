#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#   Copyright (c) 2026 Scripps Research, Forli Lab.
#   All rights reserved.
#
#   Original author: Niccolo Bruciaferri
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
Plotting for MCSwell's capacity-safe independent-site binomial occupancy
analysis.

All of the actual numeric work formerly done here (reading per-snapshot
PDB frames, building the density grid, detecting hydration sites, and
fitting the binomial occupancy isotherm) now happens in C++ -- see
include/analysis/pipeline_analysis.hpp -- operating on the GCMC titration's
in-memory snapshot buffer instead of files on disk. This module only reads
the small CSV outputs that C++ already wrote (titration_diagnostics.csv,
sites_all.csv) and draws the diagnostic plot scipy/matplotlib is good at;
it never touches a per-snapshot PDB.
"""

import argparse
import csv
import math
import os

import numpy as np
import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_B = 0.0019872041


def _parse_float(value):
    return float(value) if value not in ("", "nan") else float("nan")


def independent_site_occupancy(mu_values, epsilon, temperature):
    """Same one-line sigmoid the C++ fit evaluates; re-derived here only so
    the plot can draw a smooth curve between the sampled mu points instead
    of connecting them with straight segments. The fit itself (epsilon) is
    read from sites_all.csv, not recomputed."""
    beta = 1.0 / (K_B * temperature)
    return 1.0 / (1.0 + np.exp(-beta * (np.asarray(mu_values, dtype=float) - epsilon)))


def read_titration_diagnostics(out_dir):
    """Long-format titration_diagnostics.csv -> per-site/per-mu arrays."""
    path = os.path.join(out_dir, "titration_diagnostics.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    sites = sorted({int(r["source_site_id"]) for r in rows})
    mu_indices = sorted({int(r["mu_index"]) for r in rows})
    site_pos = {s: i for i, s in enumerate(sites)}
    mu_pos = {m: i for i, m in enumerate(mu_indices)}

    n_sites, n_mu = len(sites), len(mu_indices)
    mu_values = np.zeros(n_mu)
    n_frames = np.zeros(n_mu, dtype=int)
    valid_mask = np.zeros(n_mu, dtype=bool)
    p_observed = np.full((n_sites, n_mu), np.nan)

    for r in rows:
        si, mi = site_pos[int(r["source_site_id"])], mu_pos[int(r["mu_index"])]
        mu_values[mi] = float(r["mu_kcal_mol"])
        n_frames[mi] = int(r["n_frames"])
        valid_mask[mi] = r["valid_for_binomial"] == "True"
        p_observed[si, mi] = _parse_float(r["p_observed"])

    return sites, mu_values, n_frames, valid_mask, p_observed


def read_sites_all(out_dir):
    path = os.path.join(out_dir, "sites_all.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    epsilon = {int(r["source_site_id"]): _parse_float(r["epsilon_midpoint"]) for r in rows}
    status = {int(r["source_site_id"]): r["fit_status"] for r in rows}
    return epsilon, status


def plot_titration_curves(
    out_dir,
    mu_bulk,
    temperature=300.0,
    max_sites=12,
    out_png=None,
):
    """Observed occupancy (circles; x for capacity-excluded windows) with
    the fitted isotherm overlaid (smooth curve, re-evaluated from the
    fitted epsilon -- see independent_site_occupancy above)."""
    sites, mu_values, n_frames, valid_mask, p_observed = read_titration_diagnostics(out_dir)
    epsilon, status = read_sites_all(out_dir)

    has_frames = n_frames > 0
    if not np.any(has_frames):
        print("[plot] No chemical potentials with frames -- skipping.")
        return

    fit_window_mask = has_frames & valid_mask
    excluded_mask = has_frames & (~valid_mask)

    chosen = []
    for site_row, site_id in enumerate(sites):
        eps = epsilon.get(site_id, float("nan"))
        if status.get(site_id) != "ok" or not math.isfinite(eps):
            continue
        pv = p_observed[site_row, fit_window_mask]
        if pv.size >= 3 and (np.nanmax(pv) - np.nanmin(pv)) >= 0.10:
            chosen.append(site_row)
        if len(chosen) >= max_sites:
            break

    if not chosen:
        chosen = [
            i for i, site_id in enumerate(sites)
            if status.get(site_id) == "ok" and math.isfinite(epsilon.get(site_id, float("nan")))
        ][:max_sites]

    if not chosen:
        print("[plot] No sites with a finite thermodynamic fit -- skipping.")
        return

    mu_fit_vals = mu_values[fit_window_mask]
    mu_dense = np.linspace(mu_fit_vals.min(), mu_fit_vals.max(), 500)

    plt.figure(figsize=(7, 5))
    for row in chosen:
        site_id = sites[row]
        line, = plt.plot(
            mu_values[fit_window_mask], p_observed[row, fit_window_mask],
            marker="o", linestyle="none", markersize=4, label=f"Site {site_id}",
        )
        if np.any(excluded_mask):
            plt.plot(
                mu_values[excluded_mask], p_observed[row, excluded_mask],
                marker="x", linestyle="none", markersize=4, color=line.get_color(), alpha=0.55,
            )
        plt.plot(
            mu_dense, independent_site_occupancy(mu_dense, epsilon[site_id], temperature),
            linestyle="-", linewidth=1.2, color=line.get_color(),
        )

    mu_eq_idx = int(np.argmin(np.abs(mu_values - mu_bulk)))
    plt.axvline(
        mu_values[mu_eq_idx], linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"bulk mu ~ {mu_values[mu_eq_idx]:.2f}",
    )
    if np.any(~valid_mask):
        first_bad = int(np.flatnonzero(~valid_mask)[0])
        plt.axvline(
            mu_values[first_bad], linestyle=":", linewidth=1.2, color="black", alpha=0.65,
            label="first excluded window",
        )

    plt.xlabel("Chemical potential mu (kcal/mol)")
    plt.ylabel("Occupancy probability p(mu)")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()

    out_png = out_png or os.path.join(out_dir, "titration_curves.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[plot] wrote {out_png}")


def main(out_dir, mu_bulk, temperature=300.0, max_sites=12):
    plot_titration_curves(out_dir, mu_bulk=mu_bulk, temperature=temperature, max_sites=max_sites)
    print(f"For RETI comparison use {out_dir}/sites.csv column 'deltaG_bind' (or the sites.pdb B-factor).")


def build_parser():
    p = argparse.ArgumentParser(
        description="Plot MCSwell binomial titration curves from the C++ analysis pipeline's CSV output."
    )
    p.add_argument("out_dir", help="Binomial analysis output directory (save_path/gci/<pct>_binomial)")
    p.add_argument("--mu-bulk", type=float, required=True)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--max-sites", type=int, default=12)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args.out_dir, mu_bulk=args.mu_bulk, temperature=args.temperature, max_sites=args.max_sites)
