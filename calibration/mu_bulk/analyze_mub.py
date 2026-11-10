#!/usr/bin/env python
"""mu_bulk from the calibration titrations: the mu at which the cube holds bulk water.

For each finished run in runs/ (or --runs-dir), <N>(mu) is the mean number of sampled
waters over the production diagnostic windows (acceptance_windows.csv), with a
standard error from 10 blocks. mu_bulk is where <N>(mu) crosses the target, by
linear interpolation between the two bracketing windows. Its error is SEM(N) / slope.
Two targets:

  density   the model's mean MD density (boxes.json) x the cube volume (primary)
  snapshot  the waters whose O was in the cube in that run's snapshot

Convergence: mu_bulk is also computed from the first and from the second half of production
(drift_density = second - first); a run whose halves differ by more than max(0.1, 2 x error) is
flagged as not converged. The empty- and full-start runs of a snapshot should also agree (their
difference is reported as hysteresis), but "full" does not start from bulk density (the prefill
holds 100-160 waters in a 20 A cube that takes ~260), so it is not an upper bracket. Per model, mu_bulk is the mean over the runs, with the standard deviation
across runs as its uncertainty.

mu_bulk_summary.csv has a row per model and cube size, and one with cube_A = inf: the
least-squares extrapolation mu_bulk(L) = mu_inf + c / L over all runs (sd_density = its standard error).
The recommended value is the row with cube_A = ">=20": the mean over the 20 and 26 A cubes, which
agree within the snapshot scatter. The 14 A cubes scatter by snapshot (their snapshot counts differ from
the mean-density target by up to 5%, and 1% density moves mu_bulk by ~0.08) and drive the 1/L fit
in opposite directions for the two models, so the extrapolation is not used.

Writes mu_bulk_per_run.csv, mu_bulk_summary.csv, n_vs_mu.png and mu_bulk_vs_size.png into --out
(results/ by default).

    python analyze_mub.py [--runs-dir runs] [--out results]
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WATER_G_MOL = 18.01528
AVOGADRO = 6.02214076e23
N_BLOCKS = 10
RECOMMENDED_MIN_CUBE = 20.0
C_EMPTY, C_FULL, C_GREY = "#2a78d6", "#eb6834", "#8a8984"
LINESTYLES = {26.0: "-", 20.0: "--", 14.0: ":"}
INK, INK2 = "#0b0b0b", "#52514e"


def n_vs_mu(run, part=None):
    """<N>(mu) over the production windows; part = 0 / 1 for the first / second half only."""
    w = pd.read_csv(os.path.join(run, "acceptance_windows.csv"))
    w = w[w.phase == "production"]
    rows = []
    for mu, g in w.groupby("mu_kcal_mol"):
        n = g.sort_values("window").mean_n_waters.to_numpy()
        if part is not None:
            n = np.array_split(n, 2)[part]
        blocks = np.array([b.mean() for b in np.array_split(n, N_BLOCKS)])
        rows.append({"mu": mu, "N": n.mean(), "N_sem": blocks.std(ddof=1) / np.sqrt(N_BLOCKS)})
    return pd.DataFrame(rows).sort_values("mu").reset_index(drop=True)


def crossing(curve, target):
    mu, n, sem = curve.mu.to_numpy(), curve.N.to_numpy(), curve.N_sem.to_numpy()
    above = np.nonzero(n >= target)[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan, np.nan, "target outside the sampled range"
    i = above[0]
    slope = (n[i] - n[i - 1]) / (mu[i] - mu[i - 1])
    m = mu[i - 1] + (target - n[i - 1]) / slope
    # N(mu) should rise through the crossing: flag a drop beyond 3 SEM within 3 windows of it
    lo, hi = max(i - 3, 0), min(i + 3, len(n) - 1)
    drops = np.diff(n[lo:hi + 1]) < -3 * np.hypot(sem[lo:hi], sem[lo + 1:hi + 1])
    status = "N(mu) not monotonic near the crossing" if drops.any() else "ok"
    return m, max(sem[i - 1], sem[i]) / slope, status


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    boxes = json.load(open(os.path.join(HERE, "boxes.json")))

    rows, curves = [], {}
    for run in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        if not os.path.exists(os.path.join(run, "acceptance_windows.csv")):
            continue
        model, box, region, fill = os.path.basename(run).split("_")
        snap = next(b for b in boxes[model] if b["snapshot"] == int(box[3:]) and f"r{b['region_A']:g}" == region)
        volume = snap["region_A"] ** 3
        rho = np.mean([s["density_g_cm3_last_200ps"] for s in boxes[model] if s["region_A"] == snap["region_A"]])
        targets = {"density": rho * 1e-24 * volume / WATER_G_MOL * AVOGADRO, "snapshot": snap["n_target"]}
        curve = n_vs_mu(run)
        curves[(model, box, snap["region_A"], fill)] = curve
        row = {"model": model, "snapshot": int(box[3:]), "cube_A": snap["region_A"], "start": fill,
               "done": os.path.exists(os.path.join(run, "DONE"))}
        for name, t in targets.items():
            m, err, status = crossing(curve, t)
            row |= {f"n_target_{name}": t, f"mu_bulk_{name}": m, f"err_{name}": err, f"status_{name}": status}
        # convergence: mu_bulk from the first and second half of production
        halves = [crossing(n_vs_mu(run, part), targets["density"])[0] for part in (0, 1)]
        row["drift_density"] = halves[1] - halves[0]
        if np.isfinite(row["drift_density"]) and abs(row["drift_density"]) > max(0.1, 2 * row["err_density"]):
            row["status_density"] = f"not converged: halves differ by {row['drift_density']:+.2f}"
        rows.append(row)
    if not rows:
        print(f"no runs with acceptance_windows.csv in {args.runs_dir}")
        return
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out, "mu_bulk_per_run.csv"), index=False)

    summ = []
    for (model, cube), g in R.groupby(["model", "cube_A"]):
        s = {"model": model, "cube_A": cube, "n_runs": len(g), "n_done": int(g.done.sum())}
        for name in ("density", "snapshot"):
            v = g[f"mu_bulk_{name}"]
            s |= {f"mu_bulk_{name}": v.mean(), f"sd_{name}": v.std(ddof=1)}
        piv = g.pivot_table(index="snapshot", columns="start", values="mu_bulk_density")
        if {"empty", "full"} <= set(piv.columns):
            s["hysteresis_full_minus_empty"] = (piv["full"] - piv["empty"]).mean()
        summ.append(s)
    # extrapolation to an infinite cube: mu_bulk(L) = mu_inf + c / L, least squares over the runs
    fits = {}
    for model, g in R.dropna(subset=["mu_bulk_density"]).groupby("model"):
        if g.cube_A.nunique() < 2:
            continue
        x, y = 1.0 / g.cube_A.to_numpy(), g.mu_bulk_density.to_numpy()
        A = np.vstack([np.ones_like(x), x]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        dof = len(y) - 2
        se = (np.sqrt(np.diag(np.sum((y - A @ coef) ** 2) / dof * np.linalg.inv(A.T @ A)))
              if dof > 0 else np.array([np.nan, np.nan]))
        fits[model] = coef
        summ.append({"model": model, "cube_A": np.inf, "n_runs": len(g), "n_done": int(g.done.sum()),
                     "mu_bulk_density": coef[0], "sd_density": se[0], "slope_per_inverse_A": coef[1]})
    # recommended: mean over the cubes of at least RECOMMENDED_MIN_CUBE (the 14 A cube scatters by
    # snapshot, and drives the 1/L fit), sd over snapshot x cube means
    for model, g in R[R.cube_A >= RECOMMENDED_MIN_CUBE].dropna(subset=["mu_bulk_density"]).groupby("model"):
        per = g.groupby(["snapshot", "cube_A"]).mu_bulk_density.mean()
        summ.append({"model": model, "cube_A": f">={RECOMMENDED_MIN_CUBE:g}", "n_runs": len(g),
                     "n_done": int(g.done.sum()), "mu_bulk_density": per.mean(), "sd_density": per.std(ddof=1)})
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(args.out, "mu_bulk_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    print(R.drop(columns=[c for c in R.columns if c.startswith("status")]).round(3).to_string(index=False))
    bad = R[(R.status_density != "ok") | (R.status_snapshot != "ok")]
    if len(bad):
        print("\nwarnings:\n" + bad[["model", "snapshot", "cube_A", "start", "status_density", "status_snapshot"]]
              .to_string(index=False))
    print("\n" + S.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, len(boxes), figsize=(5 * len(boxes), 4.2), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], boxes):
        g = R[R.model == model]
        for (m, box, cube, fill), c in curves.items():
            if m != model:
                continue
            t = g[g.cube_A == cube].n_target_density.iloc[0]
            ax.errorbar(c.mu, c.N / t, yerr=c.N_sem / t, color=C_EMPTY if fill == "empty" else C_FULL, lw=1,
                        alpha=0.8, ls=LINESTYLES.get(cube, "-."),
                        label=f"{cube:g} A cube, {fill} start" if box == "box0" else None)
        ax.axhline(1.0, color=INK2, lw=0.8, ls=":", label="bulk density")
        for cube, gc in g.groupby("cube_A"):
            mb = gc.mu_bulk_density.mean()
            if np.isfinite(mb):
                ax.axvline(mb, color=C_GREY, lw=0.8, ls=LINESTYLES.get(cube, "-."))
        ax.set_title(model.upper(), loc="left", fontsize=11, color=INK)
        ax.set_xlabel("mu (kcal/mol)", fontsize=9, color=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0][0].set_ylabel("<N> / bulk-density target", fontsize=9, color=INK2)
    fig.suptitle("mu_bulk calibration: titration of a water cube in a fixed bulk-water shell", x=0.06,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "n_vs_mu.png"), dpi=200)
    plt.close(fig)

    # mu_bulk against 1/size, with the extrapolation
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for model, col in zip(sorted(set(R.model)), (C_EMPTY, C_FULL)):
        g = R[R.model == model]
        ax.scatter(1.0 / g.cube_A, g.mu_bulk_density, s=28, color=col, edgecolor="white", lw=0.8, zorder=3,
                   label=model.upper())
        if model in fits:
            xs = np.linspace(0, 1.0 / g.cube_A.min(), 50)
            ax.plot(xs, fits[model][0] + fits[model][1] * xs, color=col, lw=1)
            ax.annotate(f"{fits[model][0]:.2f}", (0, fits[model][0]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=col, va="center")
    ax.set_xlim(left=0)
    ax.set_xlabel("1 / cube size (1/A)", fontsize=9, color=INK2)
    ax.set_ylabel("mu_bulk (kcal/mol)", fontsize=9, color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("mu_bulk vs cube size, extrapolated to 1/size = 0", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mu_bulk_vs_size.png"), dpi=200)
    plt.close(fig)
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
