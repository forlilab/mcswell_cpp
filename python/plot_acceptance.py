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

"""Sampling-diagnostics plots from MCSwell's acceptance tables.

Reads <save_path>/acceptance_summary.csv and acceptance_windows.csv and writes
two figures next to them:

  acceptance_by_mu.png  - production acceptance per move type against mu, with
                          the titration <N>(mu) underneath for context.
  n_waters_trace.png    - the water count over the whole run at a handful of
                          mu values across the titration, with the
                          equilibration phase marked and a per-mu verdict
                          from convergence.py (detected equilibration point,
                          drift between the halves of production).

Rates are recomputed from the counts so that a move that was never proposed
(e.g. deletion while the region is empty) shows as missing rather than as 0%.

usage: plot_acceptance.py <save_path> [--mu-bulk X]
"""
import argparse
import csv
import os

import matplotlib
import numpy as np

import convergence

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

MOVES = ("insertion", "deletion", "displacement")
# Production halves differing by more than this fraction of <N>, and by more
# than DRIFT_SEMS standard errors of the production mean, get flagged. Each
# half's mean has about sqrt(2) times that error, so their difference about
# twice it: 6 is a 3-sigma test on the difference.
MAX_DRIFT = 0.02
DRIFT_SEMS = 6.0
# A detected equilibration point later than the configured one is flagged
# when it falls more than this fraction of the production phase past it.
LATE_EQUILIBRATION_FRACTION = 0.10

# Reference palette, light mode: categorical slots 1-3. Text never takes a
# series color.
SERIES = {"insertion": "#2a78d6", "deletion": "#eb6834", "displacement": "#1baf7a"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"


# Applied with plt.rc_context so the look stays local to these figures and
# does not leak into the titration plots drawn in the same process.
STYLE = {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9,
}


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _pct(x, _pos=None):
    if x >= 1:
        return f"{x:.0f}%"
    if x >= 0.001:
        return f"{x:.3f}".rstrip("0").rstrip(".") + "%"
    return f"$10^{{{int(round(np.log10(x)))}}}$%"


def _spread_labels(ax, ys_data, min_gap_pt=11):
    """Nudge end-of-line labels apart vertically so they never overlap.
    Works in display space, so it is scale-agnostic (log axes included)."""
    to_disp = ax.transData.transform
    pts_per_px = 72.0 / ax.figure.dpi
    order = np.argsort(ys_data)
    disp = np.array([to_disp((0, y))[1] for y in np.asarray(ys_data)[order]]) * pts_per_px
    for i in range(1, len(disp)):
        disp[i] = max(disp[i], disp[i - 1] + min_gap_pt)
    offsets = np.empty(len(disp))
    offsets[order] = disp - np.array([to_disp((0, y))[1] for y in np.asarray(ys_data)[order]]) * pts_per_px
    return offsets


def _rates(acc, prop):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(prop > 0, 100.0 * acc / np.maximum(prop, 1), np.nan)


def plot_by_mu(summary_rows, out_path, mu_bulk=None):
    rows = [r for r in summary_rows if r["phase"] == "production"]
    rows.sort(key=lambda r: float(r["mu_kcal_mol"]))
    mu = np.array([float(r["mu_kcal_mol"]) for r in rows])
    n_mean = np.array([float(r["mean_n_waters"]) for r in rows])

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})

    series = {}
    for move in MOVES:
        prop = np.array([float(r[f"{move}_proposed"]) for r in rows])
        acc = np.array([float(r[f"{move}_accepted"]) for r in rows])
        series[move] = (prop, acc, _rates(acc, prop))
    positive = np.concatenate([s[2][s[2] > 0] for s in series.values()])
    lo = 10 ** np.floor(np.log10(positive.min())) if positive.size else 1e-3
    # A rate of exactly 0 cannot sit on a log axis. It gets its own row below
    # the axis floor, labelled "0%", set off by a rule, so "proposed but never
    # accepted" is visible without pretending to be a tiny positive value.
    zero_y = lo / 10 ** 0.8
    ax.set_yscale("log")
    ax.set_ylim(lo / 10 ** 1.15, 100)
    ax.axhspan(lo / 10 ** 1.15, lo / 10 ** 0.45, color=GRID, alpha=0.35, lw=0, zorder=0)

    ends = []
    for move in MOVES:
        prop, acc, rate = series[move]
        ax.plot(mu, np.where(rate > 0, rate, np.nan), color=SERIES[move], lw=2,
                marker="o", ms=4, mec=SURFACE, mew=0.8, label=move.capitalize(), zorder=3)
        zero = (prop > 0) & (acc == 0)
        if zero.any():
            ax.plot(mu[zero], np.full(zero.sum(), zero_y), ls="none", marker="o", ms=5,
                    mfc=SURFACE, mec=SERIES[move], mew=1.2, zorder=3)
        last = np.where(np.isfinite(rate) & (rate > 0))[0]
        if last.size:
            ends.append((move, mu[last[-1]], rate[last[-1]]))

    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ticks = [t for t in ax.get_yticks() if lo <= t <= 100]
    ax.set_yticks([zero_y] + ticks)
    ax.set_yticklabels(["0%"] + [_pct(t) for t in ticks])
    ax.set_ylim(lo / 10 ** 1.15, 100)
    ax.set_ylabel("Accepted / proposed")

    fig.canvas.draw()  # transforms are final only after a draw
    offsets = _spread_labels(ax, [e[2] for e in ends])
    for (move, x, y), dy in zip(ends, offsets):
        ax.annotate(move.capitalize(), (x, y), xytext=(7, dy), textcoords="offset points",
                    va="center", color=TEXT, fontsize=9)

    fig.suptitle("Acceptance rate by chemical potential (production phase)", x=0.125, y=0.995,
                 ha="left", fontsize=13, fontweight="semibold", color=TEXT)
    fig.text(0.125, 0.952, "Shaded bottom row: proposed but never accepted.  Gaps: never "
             "proposed (no water to delete or move).", fontsize=8.5, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2,
              handlelength=1.6, columnspacing=1.4)

    ax_n.plot(mu, n_mean, color=TEXT_2, lw=2, marker="o", ms=3.5, mec=SURFACE, mew=0.6)
    ax_n.set_ylabel("Mean waters ⟨N⟩")
    ax_n.set_xlabel("μ (kcal/mol)")
    ax_n.set_ylim(bottom=0)

    if mu_bulk is not None:
        for a in (ax, ax_n):
            a.axvline(mu_bulk, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"bulk μ = {mu_bulk:g}", (mu_bulk, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", color=TEXT_2, fontsize=8.5)

    ax.set_xlim(mu.min() - 0.5, mu.max() + 3.5)  # room for the end labels
    fig.align_ylabels((ax, ax_n))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _pick_mus(mean_n, mu_values, mu_bulk, n_panels):
    """Indices of up to n_panels mu values spread across the titration.

    Only windows holding at least one water on average are candidates (an
    empty region has nothing to equilibrate), taken evenly from the first to
    the last; the one nearest bulk mu is swapped in when it is not picked."""
    candidates = np.where(mean_n >= 1.0)[0]
    if candidates.size == 0:
        candidates = np.where(mean_n > 0)[0]
    if candidates.size == 0:
        return []
    picks = candidates[np.unique(np.round(np.linspace(0, candidates.size - 1, n_panels)).astype(int))]
    if mu_bulk is not None:
        bulk = candidates[np.argmin(np.abs(mu_values[candidates] - mu_bulk))]
        if bulk not in picks:
            picks[np.argmin(np.abs(mu_values[picks] - mu_bulk))] = bulk
    return sorted(int(i) for i in set(picks))


def _verdict(res, total_steps):
    """A short plain-language verdict per panel, from convergence.analyse_mu."""
    detected, used = res.equilibration_steps_detected, res.equilibration_steps_used
    if detected >= 0.9 * total_steps:
        # The detector's best discard point is the tail end: no stationary
        # stretch was found at all.
        return "! still changing at the end of the run"
    # The detector scans a grid of candidate points, so a detected point just
    # past the configured one is within its resolution; flag only when it
    # would discard a real share of production.
    if detected - used > LATE_EQUILIBRATION_FRACTION * (total_steps - used):
        return f"! stationary only after ~{detected / 1e6:.3g}M steps (equilibration: {used / 1e6:.3g}M)"
    # A drift counts only when it is both a sizeable fraction of <N> and well
    # outside the statistical noise: at small <N> a few % between the halves
    # is just a couple of standard errors.
    drift = res.second_half_drift_fraction
    if np.isfinite(drift) and drift > MAX_DRIFT and drift * abs(res.mean_n_waters) > DRIFT_SEMS * res.sem_n_waters:
        return f"! production halves differ by {100 * drift:.1f}%"
    return "✓ equilibrated"


def plot_n_trace(windows_csv, out_path, mu_bulk=None, n_panels=6):
    trace = convergence.load_trace(windows_csv)
    results = convergence.analyse_run(trace)
    mean_n = np.array([r.mean_n_waters for r in results])
    picks = _pick_mus(mean_n, np.asarray(trace.mu_values), mu_bulk, n_panels)
    if not picks:
        return False

    x = 0.5 * (trace.step_begin + trace.step_end) / 1e6
    t_equil = trace.equilibration_steps / 1e6
    n_cols = 3
    n_rows = int(np.ceil(len(picks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows + 1.2), sharex=True,
                             squeeze=False, gridspec_kw={"hspace": 0.55, "wspace": 0.22})
    for a, i in zip(axes.flat, picks):
        res = results[i]
        series = trace.mean_n_waters[i]
        a.axvspan(0, t_equil, color=GRID, alpha=0.6, lw=0, zorder=0)
        a.plot(x, series, color=SERIES["insertion"], lw=1.1, zorder=2)
        prod = trace.is_production
        a.plot([x[prod][0], x[-1]], [res.mean_n_waters] * 2, color=TEXT, lw=1.2,
               ls=(0, (4, 3)), zorder=3)
        verdict = _verdict(res, trace.step_end[-1])
        if "stationary only" in verdict or "still changing" in verdict:
            a.axvline(res.equilibration_steps_detected / 1e6, color=TEXT, lw=1, ls=":", zorder=3)
        a.set_title(f"μ = {trace.mu_values[i]:g} kcal/mol", pad=30)
        a.text(0, 1.03, f"{verdict}\n"
               f"production ⟨N⟩ = {res.mean_n_waters:.4g} ± {res.sem_n_waters:.2g}",
               transform=a.transAxes, fontsize=8.5, color=TEXT_2, va="bottom", linespacing=1.4)
        lo, hi = np.nanmin(series), np.nanmax(series)
        pad = 0.08 * (hi - lo) if hi > lo else max(1.0, 0.1 * hi)
        a.set_ylim(max(0.0, lo - pad), hi + pad)
        a.set_xlim(0, trace.step_end[-1] / 1e6)
    for a in axes.flat[len(picks):]:
        a.set_visible(False)
    for a in axes[:, 0]:
        a.set_ylabel("Waters N")
    for a in axes[-1, :]:
        a.set_xlabel("MC step (millions)")
    # Header placed in inches from the top edge, so it never meets the panels.
    win = int(trace.n_steps[0])
    h = fig.get_figheight()
    fig.subplots_adjust(top=1.0 - 1.45 / h)
    fig.text(0.09, 1.0 - 0.15 / h, "Water count over the run", ha="left", va="top",
             fontsize=13, fontweight="semibold", color=TEXT)
    fig.text(0.09, 1.0 - 0.5 / h, f"Mean N per diagnostic window ({win:,} steps) at {len(picks)} μ "
             "values across the titration.  Gray band: equilibration phase.  Dashed line: production "
             "mean.\nAn equilibrated trace is flat around the dashed line after the gray band.  A "
             "dotted line marks where the trace actually became stationary, when that is later.",
             fontsize=8.5, color=MUTED, va="top", linespacing=1.4)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def main(save_path, mu_bulk=None):
    """Draw both figures from the CSVs under save_path; returns the paths written."""
    summary = os.path.join(save_path, "acceptance_summary.csv")
    windows = os.path.join(save_path, "acceptance_windows.csv")
    outputs = []
    with plt.rc_context(STYLE):
        if os.path.exists(summary):
            out = os.path.join(save_path, "acceptance_by_mu.png")
            plot_by_mu(_read(summary), out, mu_bulk=mu_bulk)
            outputs.append(out)
        if os.path.exists(windows):
            out = os.path.join(save_path, "n_waters_trace.png")
            if plot_n_trace(windows, out, mu_bulk=mu_bulk):
                outputs.append(out)
    return outputs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("save_path")
    p.add_argument("--mu-bulk", type=float, default=None)
    args = p.parse_args()
    for path in main(args.save_path, mu_bulk=args.mu_bulk):
        print(path)
