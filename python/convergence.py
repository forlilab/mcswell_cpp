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

"""Convergence and sampling diagnostics for an MCSwell GCMC titration.

Reads the acceptance_windows.csv / acceptance_summary.csv tables the sampler
writes (see include/analysis/sampler_diagnostics.hpp) and turns the per-window
occupancy trace into the quantities needed to say whether a run sampled
enough: a correlation time, an effective sample size, a block-averaged
standard error on <N>, and an automatically detected equilibration point.

The statistics follow the standard Markov-chain estimators -- normalised
fluctuation autocorrelation for the statistical inefficiency, and Chodera's
reverse-cumulative scan for equilibration detection (J. Chem. Theory Comput.
2016, 12, 1799). They are reimplemented here on numpy rather than pulled in
from pymbar so that the diagnostics stay a dependency-free part of the
package.

All times are reported in MC steps, not in windows: the window length is read
from the trace and multiplied back in, so numbers from runs recorded at
different window resolutions are directly comparable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "WindowTrace",
    "MuConvergence",
    "load_windows",
    "load_windows_npz",
    "load_trace",
    "save_windows_npz",
    "autocorrelation",
    "statistical_inefficiency",
    "integrated_autocorrelation_time",
    "detect_equilibration",
    "block_sem",
    "analyse_mu",
    "analyse_run",
    "convergence_table",
    "MIN_CORRELATION_TIMES",
]

MOVE_TYPES = ("insertion", "deletion", "displacement")

# A statistical-inefficiency estimate is only meaningful once the series spans
# a good number of correlation times. Below this, the autocorrelation sum is
# dominated by its own noise and g is reported but flagged unreliable.
MIN_CORRELATION_TIMES = 50


@dataclass
class WindowTrace:
    """Per-(mu, window) trace loaded from acceptance_windows.csv.

    Every array is shaped (n_mu, n_windows), except `proposed` and `accepted`
    which are (n_mu, n_windows, 3) indexed by MOVE_TYPES.
    """

    mu_values: np.ndarray
    step_begin: np.ndarray  # (n_windows,)
    step_end: np.ndarray  # (n_windows,)
    n_steps: np.ndarray  # (n_windows,)
    is_production: np.ndarray  # (n_windows,) bool
    proposed: np.ndarray
    accepted: np.ndarray
    capacity_blocked: np.ndarray
    mean_n_waters: np.ndarray
    n_waters_end: np.ndarray
    energy_end: np.ndarray

    @property
    def n_mu(self) -> int:
        return len(self.mu_values)

    @property
    def n_windows(self) -> int:
        return len(self.step_begin)

    @property
    def steps_per_window(self) -> float:
        return float(np.median(self.n_steps))

    @property
    def equilibration_steps(self) -> int:
        """First step of the first production window (0 if all production)."""
        prod = np.flatnonzero(self.is_production)
        return int(self.step_begin[prod[0]]) if prod.size else int(self.step_end[-1])

    def production(self) -> "WindowTrace":
        """The same trace restricted to the production windows."""
        keep = self.is_production
        return WindowTrace(
            mu_values=self.mu_values,
            step_begin=self.step_begin[keep],
            step_end=self.step_end[keep],
            n_steps=self.n_steps[keep],
            is_production=self.is_production[keep],
            proposed=self.proposed[:, keep],
            accepted=self.accepted[:, keep],
            capacity_blocked=self.capacity_blocked[:, keep],
            mean_n_waters=self.mean_n_waters[:, keep],
            n_waters_end=self.n_waters_end[:, keep],
            energy_end=self.energy_end[:, keep],
        )

    def acceptance_rate(self, move_type: str | None = None) -> np.ndarray:
        """Per-(mu, window) acceptance rate, overall or for one move type."""
        if move_type is None:
            acc = self.accepted.sum(axis=2)
            prop = self.proposed.sum(axis=2)
        else:
            m = MOVE_TYPES.index(move_type)
            acc = self.accepted[:, :, m]
            prop = self.proposed[:, :, m]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(prop > 0, acc / np.maximum(prop, 1), np.nan)

    def overall_rate_by_mu(self, move_type: str | None = None) -> np.ndarray:
        """Acceptance rate per mu, pooled over all windows in the trace."""
        if move_type is None:
            acc = self.accepted.sum(axis=(1, 2))
            prop = self.proposed.sum(axis=(1, 2))
        else:
            m = MOVE_TYPES.index(move_type)
            acc = self.accepted[:, :, m].sum(axis=1)
            prop = self.proposed[:, :, m].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(prop > 0, acc / np.maximum(prop, 1), np.nan)


def load_windows(path: str) -> WindowTrace:
    """Load an acceptance_windows.csv written by the sampler."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError(f"{path} has no data rows")

    mu_index = np.array([int(r["mu_index"]) for r in rows])
    window = np.array([int(r["window"]) for r in rows])
    n_mu = int(mu_index.max()) + 1
    n_win = int(window.max()) + 1
    if len(rows) != n_mu * n_win:
        raise ValueError(
            f"{path}: expected {n_mu} x {n_win} rows, found {len(rows)}; "
            "the trace looks truncated"
        )

    flat = mu_index * n_win + window

    def col(name, dtype=float):
        out = np.empty(len(rows), dtype=dtype)
        for i, r in enumerate(rows):
            out[flat[i]] = dtype(r[name])
        return out

    mu_values = np.empty(n_mu)
    for r in rows:
        mu_values[int(r["mu_index"])] = float(r["mu_kcal_mol"])

    # Window geometry is shared across mu, so read it off the first mu block.
    per_window = {}
    for name, dtype in (("step_begin", int), ("step_end", int), ("n_steps", int)):
        v = col(name, dtype).reshape(n_mu, n_win)
        per_window[name] = v[0]
    phase = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        phase[flat[i]] = r["phase"]
    is_production = (phase.reshape(n_mu, n_win)[0] == "production")

    proposed = np.stack(
        [col(f"{m}_proposed", int).reshape(n_mu, n_win) for m in MOVE_TYPES], axis=2
    )
    accepted = np.stack(
        [col(f"{m}_accepted", int).reshape(n_mu, n_win) for m in MOVE_TYPES], axis=2
    )

    return WindowTrace(
        mu_values=mu_values,
        step_begin=per_window["step_begin"],
        step_end=per_window["step_end"],
        n_steps=per_window["n_steps"],
        is_production=is_production,
        proposed=proposed,
        accepted=accepted,
        capacity_blocked=col("capacity_blocked", int).reshape(n_mu, n_win),
        mean_n_waters=col("mean_n_waters").reshape(n_mu, n_win),
        n_waters_end=col("n_waters_end", int).reshape(n_mu, n_win),
        energy_end=col("energy_end_kcal_mol").reshape(n_mu, n_win),
    )


def save_windows_npz(trace: WindowTrace, path: str) -> None:
    """Store a trace compactly. A high-resolution run's CSV reaches ~10^6
    rows; the npz round-trips the same arrays at a fraction of the size and
    loads without re-parsing text."""
    np.savez_compressed(
        path,
        mu_values=trace.mu_values,
        step_begin=trace.step_begin,
        step_end=trace.step_end,
        n_steps=trace.n_steps,
        is_production=trace.is_production,
        proposed=trace.proposed,
        accepted=trace.accepted,
        capacity_blocked=trace.capacity_blocked,
        mean_n_waters=trace.mean_n_waters,
        n_waters_end=trace.n_waters_end,
        energy_end=trace.energy_end,
    )


def load_windows_npz(path: str) -> WindowTrace:
    d = np.load(path)
    return WindowTrace(**{k: d[k] for k in (
        "mu_values", "step_begin", "step_end", "n_steps", "is_production",
        "proposed", "accepted", "capacity_blocked", "mean_n_waters",
        "n_waters_end", "energy_end")})


def load_trace(path: str) -> WindowTrace:
    """Load a window trace from either the CSV the sampler writes or an npz."""
    return load_windows_npz(path) if path.endswith(".npz") else load_windows(path)


def autocorrelation(series: np.ndarray) -> np.ndarray:
    """Normalised fluctuation autocorrelation C(t), C(0) == 1.

    FFT-based, so exact rather than the truncated-stride approximation that
    pymbar's "fast" mode uses, and cheap enough to run on every mu.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.ones(1)
    dx = x - x.mean()
    var = np.dot(dx, dx) / n
    if var <= 0:
        return np.ones(1)

    # Zero-pad to >= 2n to turn the circular FFT correlation into a linear one.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(dx, size)
    acov = np.fft.irfft(f * np.conjugate(f), size)[:n]
    # Unbiased: lag t has n - t overlapping pairs.
    acov /= (n - np.arange(n))
    return acov / (acov[0] if acov[0] != 0 else 1.0)


def statistical_inefficiency(series: np.ndarray, mintime: int = 3) -> float:
    """g = 1 + 2*tau, the number of correlated samples per independent one.

    The autocorrelation sum is truncated at the first non-positive C(t), the
    standard guard against the noisy long-lag tail, and each term carries the
    (1 - t/N) finite-sample weight.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 2 or np.ptp(x) == 0:
        return 1.0

    c = autocorrelation(x)
    lags = np.arange(1, n)
    c_lag = c[1:]

    nonpositive = np.flatnonzero(c_lag <= 0)
    cut = n - 1
    for idx in nonpositive:
        if lags[idx] > mintime:
            cut = idx
            break

    if cut == 0:
        return 1.0
    t = lags[:cut]
    g = 1.0 + 2.0 * np.sum(c_lag[:cut] * (1.0 - t / n))
    return float(max(g, 1.0))


def integrated_autocorrelation_time(series: np.ndarray) -> float:
    """tau = (g - 1) / 2, in units of the sampling interval of `series`."""
    return (statistical_inefficiency(series) - 1.0) / 2.0


def detect_equilibration(
    series: np.ndarray, n_candidates: int = 200
) -> tuple[int, float, float]:
    """Chodera's reverse-cumulative scan for the equilibration point.

    Returns (t0, g, n_eff): the discard point maximising the number of
    effectively uncorrelated samples left in series[t0:], the statistical
    inefficiency of that remainder, and the effective sample size. Indices are
    positions in `series`, so multiply by the sampling interval for steps.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0, 1.0, float(n)

    # Scanning every t0 is O(N^2 log N); a subsampled grid is plenty, since the
    # N_eff(t0) curve is smooth and only its argmax matters.
    stride = max(1, n // max(n_candidates, 1))
    candidates = np.arange(0, max(n - 3, 1), stride)

    best = (0, 1.0, -np.inf)
    for t0 in candidates:
        tail = x[t0:]
        if tail.size < 3 or np.ptp(tail) == 0:
            continue
        g = statistical_inefficiency(tail)
        n_eff = tail.size / g
        if n_eff > best[2]:
            best = (int(t0), float(g), float(n_eff))

    if not np.isfinite(best[2]):
        return 0, 1.0, float(n)
    return best


def block_sem(series: np.ndarray, g: float | None = None) -> float:
    """Standard error of the mean, inflated by the statistical inefficiency."""
    x = np.asarray(series, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    if g is None:
        g = statistical_inefficiency(x)
    return float(np.sqrt(x.var(ddof=1) * g / x.size))


@dataclass
class MuConvergence:
    """Convergence summary for one chemical potential."""

    mu_index: int
    mu_value: float

    mean_n_waters: float
    sem_n_waters: float
    statistical_inefficiency: float
    autocorrelation_steps: float
    n_effective_samples: float

    # The trace is a series of window *means*, so correlations shorter than one
    # window are averaged away before they can be measured: when this is False,
    # autocorrelation_steps is an upper bound of order window_steps rather than
    # an estimate, and the run should be re-recorded with shorter windows if the
    # actual value matters.
    autocorrelation_resolved: bool
    window_steps: float

    # An autocorrelation estimate needs the series to span many correlation
    # times; when it spans only a handful, g is both badly biased and very
    # noisy. This is True only when the production phase contains at least
    # MIN_CORRELATION_TIMES correlation times. Where it is False, g, tau,
    # n_effective_samples and sem_n_waters should be read as "the run is too
    # short to say", and second_half_drift_fraction -- which assumes nothing
    # about the correlation structure -- is the statistic to trust.
    autocorrelation_reliable: bool

    equilibration_steps_detected: int
    equilibration_steps_used: int

    insertion_rate: float
    deletion_rate: float
    displacement_rate: float
    overall_rate: float

    accepted_insertions: int
    accepted_deletions: int
    accepted_displacements: int
    capacity_blocked: int

    # Running-mean drift over the second half of production, as a fraction of
    # <N>: a direct "has it stopped moving?" number that does not depend on any
    # autocorrelation estimate.
    second_half_drift_fraction: float

    extras: dict = field(default_factory=dict)


def analyse_mu(trace: WindowTrace, mu_index: int) -> MuConvergence:
    """Convergence statistics for one mu.

    Two different slices of the trace are used, deliberately:

    * The reported <N>, correlation time, effective sample size and standard
      error come from *all* production windows, because that is exactly the
      data the free-energy analyses consume. Re-deriving them from some
      internally chosen sub-slice would report the precision of a quantity
      nobody computes.
    * The equilibration point is detected on the *whole* trace, equilibration
      phase included, so that it answers the question actually being asked --
      was the configured equilibration long enough? Running the detector on
      production alone cannot say that, and on an already-equilibrated series
      its argmax wanders into the tail, where a quiet stretch happens to look
      uncorrelated.
    """
    full_series = trace.mean_n_waters[mu_index]
    interval = trace.steps_per_window

    t0_full, _, _ = detect_equilibration(full_series)
    detected_steps = int(trace.step_begin[min(t0_full, trace.n_windows - 1)])

    prod = trace.production()
    series = prod.mean_n_waters[mu_index]
    g = statistical_inefficiency(series) if series.size > 2 else 1.0
    n_eff = series.size / g if g > 0 else float(series.size)

    # Drift: compare the two halves of production. Independent of any
    # autocorrelation estimate, so it stays meaningful where g is unreliable.
    if series.size >= 4:
        half = series.size // 2
        first_half, second_half = series[:half].mean(), series[half:].mean()
        denom = abs(series.mean())
        drift = abs(second_half - first_half) / denom if denom > 0 else 0.0
    else:
        drift = float("nan")

    acc = prod.accepted[mu_index].sum(axis=0)
    return MuConvergence(
        mu_index=mu_index,
        mu_value=float(trace.mu_values[mu_index]),
        mean_n_waters=float(series.mean()) if series.size else float("nan"),
        sem_n_waters=block_sem(series, g),
        statistical_inefficiency=g,
        autocorrelation_steps=(g - 1.0) / 2.0 * interval,
        n_effective_samples=n_eff,
        # tau >= one window, i.e. g >= 3, is the point below which the window
        # averaging stops hiding the correlation.
        autocorrelation_resolved=bool(g >= 3.0),
        window_steps=interval,
        autocorrelation_reliable=bool(n_eff >= MIN_CORRELATION_TIMES),
        equilibration_steps_detected=detected_steps,
        equilibration_steps_used=trace.equilibration_steps,
        insertion_rate=float(prod.overall_rate_by_mu("insertion")[mu_index]),
        deletion_rate=float(prod.overall_rate_by_mu("deletion")[mu_index]),
        displacement_rate=float(prod.overall_rate_by_mu("displacement")[mu_index]),
        overall_rate=float(prod.overall_rate_by_mu()[mu_index]),
        accepted_insertions=int(acc[0]),
        accepted_deletions=int(acc[1]),
        accepted_displacements=int(acc[2]),
        capacity_blocked=int(prod.capacity_blocked[mu_index].sum()),
        second_half_drift_fraction=float(drift),
    )


def analyse_run(trace: WindowTrace) -> list[MuConvergence]:
    """Convergence statistics for every mu in the trace."""
    return [analyse_mu(trace, i) for i in range(trace.n_mu)]


def convergence_table(results: list[MuConvergence]) -> tuple[list[str], list[list[str]]]:
    """(headers, rows) for writing the convergence summary as CSV."""
    headers = [
        "mu_index",
        "mu_kcal_mol",
        "mean_n_waters",
        "sem_n_waters",
        "statistical_inefficiency",
        "autocorrelation_steps",
        "autocorrelation_resolved",
        "autocorrelation_reliable",
        "window_steps",
        "n_effective_samples",
        "equilibration_steps_detected",
        "equilibration_steps_used",
        "insertion_rate",
        "deletion_rate",
        "displacement_rate",
        "overall_rate",
        "accepted_insertions",
        "accepted_deletions",
        "accepted_displacements",
        "capacity_blocked",
        "second_half_drift_fraction",
    ]
    rows = []
    for r in results:
        rows.append([
            str(r.mu_index),
            f"{r.mu_value:g}",
            f"{r.mean_n_waters:.6g}",
            f"{r.sem_n_waters:.6g}",
            f"{r.statistical_inefficiency:.6g}",
            f"{r.autocorrelation_steps:.6g}",
            str(r.autocorrelation_resolved),
            str(r.autocorrelation_reliable),
            f"{r.window_steps:.6g}",
            f"{r.n_effective_samples:.6g}",
            str(r.equilibration_steps_detected),
            str(r.equilibration_steps_used),
            f"{r.insertion_rate:.6g}",
            f"{r.deletion_rate:.6g}",
            f"{r.displacement_rate:.6g}",
            f"{r.overall_rate:.6g}",
            str(r.accepted_insertions),
            str(r.accepted_deletions),
            str(r.accepted_displacements),
            str(r.capacity_blocked),
            f"{r.second_half_drift_fraction:.6g}",
        ])
    return headers, rows
