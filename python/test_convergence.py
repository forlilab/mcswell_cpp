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

"""Self-checks for convergence.py. Run directly: `python test_convergence.py`.

No pytest: the project's C++ suite is hand-rolled for the same reason (see
tests/analysis/harness.hpp), and these checks are few enough that a framework
would be more machinery than test.

The estimators are checked against series whose statistical inefficiency is
known analytically -- white noise (g = 1) and AR(1), where g = (1+phi)/(1-phi)
-- rather than against previously recorded output, so a regression in the
autocorrelation sum cannot quietly redefine what "correct" means.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convergence import (  # noqa: E402
    MOVE_TYPES, WindowTrace, analyse_run, autocorrelation, block_sem,
    convergence_table, detect_equilibration, load_trace, load_windows,
    save_windows_npz, statistical_inefficiency,
)

_failures = 0


def check(cond, what):
    global _failures
    if not cond:
        print(f"[FAIL] {what}")
        _failures += 1


def check_near(a, b, tol, what):
    check(abs(a - b) <= tol, f"{what}  ({a} vs {b}, tol={tol})")


def ar1(phi, n, rng):
    """AR(1) series with lag-1 correlation phi, so g = (1+phi)/(1-phi)."""
    x = np.empty(n)
    noise = rng.normal(size=n)
    x[0] = noise[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_autocorrelation_basics():
    rng = np.random.default_rng(0)
    c = autocorrelation(rng.normal(size=1000))
    check_near(c[0], 1.0, 1e-12, "autocorrelation is normalised to C(0)=1")

    # A constant series has no fluctuations to correlate.
    check(np.all(autocorrelation(np.full(100, 3.0)) == 1.0),
          "constant series gives a degenerate autocorrelation, not a crash")

    # Degenerate inputs must not raise.
    check(autocorrelation(np.array([1.0])).size == 1, "length-1 series handled")


def test_statistical_inefficiency_known_cases():
    rng = np.random.default_rng(1)

    check_near(statistical_inefficiency(rng.normal(size=200_000)), 1.0, 0.1,
               "white noise has g = 1")

    for phi in (0.5, 0.8, 0.9):
        expected = (1 + phi) / (1 - phi)
        g = statistical_inefficiency(ar1(phi, 400_000, rng))
        # 10% tolerance: g itself is a noisy estimate even on 4e5 points.
        check_near(g, expected, 0.1 * expected, f"AR(1) phi={phi} gives g={expected:.1f}")

    check(statistical_inefficiency(np.full(1000, 2.0)) == 1.0,
          "constant series has g = 1")
    check(statistical_inefficiency(np.array([1.0])) == 1.0, "length-1 series has g = 1")
    check(statistical_inefficiency(rng.normal(size=5)) >= 1.0, "g is never below 1")


def test_block_sem_scales_with_correlation():
    rng = np.random.default_rng(2)
    n = 200_000

    white = rng.normal(size=n)
    sem_white = block_sem(white)
    check_near(sem_white, 1.0 / np.sqrt(n), 0.2 / np.sqrt(n),
               "white-noise SEM is sigma/sqrt(N)")

    # A correlated series of the same length and variance has a larger SEM, by
    # sqrt(g). This is the whole point of the correction.
    phi = 0.9
    x = ar1(phi, n, rng)
    sem_corr = block_sem(x)
    naive = x.std(ddof=1) / np.sqrt(n)
    check_near(sem_corr / naive, np.sqrt((1 + phi) / (1 - phi)), 0.4,
               "correlated SEM is inflated by sqrt(g)")


def test_detect_equilibration_finds_a_transient():
    rng = np.random.default_rng(3)
    n = 20_000
    t = np.arange(n)
    # Exponential approach to a plateau plus noise: the equilibrated part
    # starts at a few multiples of the relaxation time.
    series = 10 * np.exp(-t / 1500.0) + rng.normal(size=n) * 0.5
    t0, g, n_eff = detect_equilibration(series)
    check(3000 < t0 < 12000, f"transient discard point is sensible (got {t0})")
    check(n_eff > 1000, f"plenty of samples survive the discard (got {n_eff:.0f})")

    # An already-equilibrated series should not need a large discard.
    flat = rng.normal(size=n)
    t0_flat, _, _ = detect_equilibration(flat)
    check(t0_flat < n // 2, f"stationary series keeps most of its data (got {t0_flat})")

    check(detect_equilibration(np.array([1.0, 2.0]))[0] == 0, "tiny series handled")


def make_trace(n_mu=2, n_windows=20, steps_per_window=100, equil_windows=5, seed=0):
    rng = np.random.default_rng(seed)
    n_steps = np.full(n_windows, steps_per_window)
    step_begin = np.arange(n_windows) * steps_per_window
    is_production = np.arange(n_windows) >= equil_windows
    proposed = np.zeros((n_mu, n_windows, 3), dtype=np.int64)
    accepted = np.zeros((n_mu, n_windows, 3), dtype=np.int64)
    proposed[:, :, 0] = 40
    proposed[:, :, 1] = 30
    proposed[:, :, 2] = 30
    accepted[:, :, 0] = 1
    accepted[:, :, 1] = 1
    accepted[:, :, 2] = 3
    mean_n = np.stack([np.full(n_windows, 5.0) + rng.normal(size=n_windows) * 0.1
                       for _ in range(n_mu)])
    return WindowTrace(
        mu_values=np.array([-10.0, -5.0][:n_mu]),
        step_begin=step_begin,
        step_end=step_begin + steps_per_window,
        n_steps=n_steps,
        is_production=is_production,
        proposed=proposed,
        accepted=accepted,
        capacity_blocked=np.zeros((n_mu, n_windows), dtype=np.int64),
        mean_n_waters=mean_n,
        n_waters_end=np.full((n_mu, n_windows), 5, dtype=np.int64),
        energy_end=np.full((n_mu, n_windows), -100.0),
    )


def test_trace_slicing_and_rates():
    tr = make_trace()
    check(tr.n_mu == 2 and tr.n_windows == 20, "trace reports its own shape")
    check(tr.steps_per_window == 100, "window length read off the trace")
    check(tr.equilibration_steps == 500, "equilibration boundary from the phase labels")

    prod = tr.production()
    check(prod.n_windows == 15, "production slice keeps only production windows")
    check(prod.n_mu == 2, "production slice keeps every mu")
    check(bool(prod.is_production.all()), "production slice is all production")

    # Rates pool counts; they are not means of per-window rates.
    check_near(float(prod.overall_rate_by_mu("insertion")[0]), 15 / 600, 1e-12,
               "insertion rate pools over windows")
    check_near(float(prod.overall_rate_by_mu()[0]), 75 / 1500, 1e-12,
               "overall rate pools over move types")

    per_window = prod.acceptance_rate("displacement")
    check(per_window.shape == (2, 15), "per-window rate has one entry per (mu, window)")
    check_near(float(per_window[0, 0]), 3 / 30, 1e-12, "per-window rate is correct")


def test_analyse_run_shapes_and_fields():
    tr = make_trace()
    results = analyse_run(tr)
    check(len(results) == tr.n_mu, "one result per mu")

    r = results[0]
    check_near(r.mu_value, -10.0, 1e-12, "mu carried through")
    check_near(r.mean_n_waters, 5.0, 0.2, "<N> recovered from the trace")
    check(r.sem_n_waters > 0, "SEM is positive")
    check(r.window_steps == 100, "window length reported in steps")
    # 15 production windows x (40+30+30) proposals, 1 insertion accepted each.
    check(r.accepted_insertions == 15, "accepted insertions summed over production")
    check(r.capacity_blocked == 0, "no capacity blocks in this trace")

    headers, rows = convergence_table(results)
    check(len(rows) == tr.n_mu, "table has one row per mu")
    check(all(len(row) == len(headers) for row in rows),
          "every row matches the header width")


def test_npz_roundtrip():
    tr = make_trace()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        save_windows_npz(tr, path)
        back = load_trace(path)
        check(back.n_mu == tr.n_mu and back.n_windows == tr.n_windows,
              "npz round-trip preserves shape")
        check(np.array_equal(back.mean_n_waters, tr.mean_n_waters),
              "npz round-trip preserves the occupancy trace")
        check(np.array_equal(back.is_production, tr.is_production),
              "npz round-trip preserves the phase labels")
        check(np.array_equal(back.proposed, tr.proposed),
              "npz round-trip preserves the counters")


def test_csv_loader_rejects_truncated_traces():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "windows.csv")
        header = ["mu_index", "mu_kcal_mol", "window", "step_begin", "step_end",
                  "n_steps", "phase"]
        for m in MOVE_TYPES:
            header += [f"{m}_proposed", f"{m}_accepted", f"{m}_rate"]
        header += ["overall_rate", "capacity_blocked", "mean_n_waters",
                   "n_waters_end", "energy_end_kcal_mol"]
        # Two mu but only one window's worth of rows for the second.
        rows = [
            "0,-10,0,0,100,100,production," + "1,1,1," * 3 + "1,0,5,5,-100",
            "0,-10,1,100,200,100,production," + "1,1,1," * 3 + "1,0,5,5,-100",
            "1,-5,0,0,100,100,production," + "1,1,1," * 3 + "1,0,5,5,-100",
        ]
        open(path, "w").write(",".join(header) + "\n" + "\n".join(rows) + "\n")
        try:
            load_windows(path)
            check(False, "truncated trace should raise")
        except ValueError:
            check(True, "truncated trace raises ValueError")


def main():
    test_autocorrelation_basics()
    test_statistical_inefficiency_known_cases()
    test_block_sem_scales_with_correlation()
    test_detect_equilibration_finds_a_transient()
    test_trace_slicing_and_rates()
    test_analyse_run_shapes_and_fields()
    test_npz_roundtrip()
    test_csv_loader_rejects_truncated_traces()

    if _failures:
        print(f"[FAIL] {_failures} check(s) failed")
        return 1
    print("[PASS] all convergence tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
