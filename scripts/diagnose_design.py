"""Diagnostics for the experiment design itself, not for any one policy.

Two questions decide how tight the reported intervals can be, and neither was
being checked:

1. Is the baseline stationary over the measurement window? If the model is
   still drifting, every seed is measured at a different point on its own
   trajectory, and that spread enters the cross-seed standard deviation as
   pure noise. It also makes the validation table incomparable to the policy
   runs when the two use different spin-up lengths.

2. How long should the measurement window be? Longer windows average away more
   within-run noise, but the two arms of a paired comparison drift apart as
   they run, so the common-random-numbers correlation decays. The standard
   error of the paired effect is minimised somewhere in between, and that
   optimum is worth locating rather than guessing.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import build_and_spinup, fork_with_policy  # noqa: E402
from housing_abm.metrics import (  # noqa: E402
    PRIMARY_METRICS,
    RATIO_METRICS,
    effective_sample_size,
    ratio_series,
    run_window,
)

TRACKED = list(PRIMARY_METRICS) + [
    "ftb_purchase_share",
    "institutional_share_of_rentals",
]


def _series(series, name):
    """Monthly series for a metric, expanding-window for pooled ratios."""
    if name in RATIO_METRICS:
        return ratio_series(series, name)
    return series[name]


def _baseline_series(job):
    seed, households, spinup, months, config = job
    model = build_and_spinup(seed, households, spinup, config)
    return run_window(model, months)


def _paired_series(job):
    seed, households, spinup, months, policy, config = job
    spun = build_and_spinup(seed, households, spinup, config)
    import copy

    base = copy.deepcopy(spun)
    treated = fork_with_policy(spun, [policy])
    return run_window(base, months), run_window(treated, months)


def report_stationarity(all_series, months):
    print("\n=== stationarity of the baseline over the measurement window ===")
    print("OLS of each monthly series on month index, pooled across seeds.")
    print("A slope indistinguishable from zero means the window is usable.\n")
    print(f"{'metric':<32} {'drift/100mo':>12} {'95% CI':>24} {'p':>8}  verdict")
    print("-" * 92)
    x = np.arange(months, dtype=float)
    for m in TRACKED:
        slopes = []
        for series in all_series:
            y = np.asarray(
                [np.nan if v is None else v for v in _series(series, m)], dtype=float
            )
            ok = ~np.isnan(y)
            if ok.sum() < 10:
                continue
            slopes.append(stats.linregress(x[ok], y[ok]).slope)
        if not slopes:
            continue
        slopes = np.asarray(slopes) * 100.0  # per 100 months
        n = slopes.size
        mean = slopes.mean()
        se = slopes.std(ddof=1) / np.sqrt(n)
        crit = stats.t.ppf(0.975, n - 1)
        p = float(2 * stats.t.sf(abs(mean / se), n - 1)) if se > 0 else 0.0
        verdict = "stationary" if p > 0.05 else "DRIFTING"
        ci = f"[{mean - crit * se:+.4f},{mean + crit * se:+.4f}]"
        print(f"{m:<32} {mean:+12.4f} {ci:>24} {p:8.4f}  {verdict}")

    print("\n=== effective sample size of the window average ===")
    for m in TRACKED:
        vals = [effective_sample_size(_series(s, m)) for s in all_series]
        n_eff = float(np.mean(vals))
        print(
            f"  {m:<32} n_eff={n_eff:6.1f} of {months} months  "
            f"-> SE cut ~{np.sqrt(max(n_eff, 1)):.1f}x versus a single-month snapshot"
        )


def report_window_choice(paired, months, windows):
    print("\n=== choosing the measurement window ===")
    print("Window averages recomputed from the same runs at several lengths.")
    print("corr is the arm-to-arm correlation; SE is the standard error of the")
    print("paired effect. Longer windows average away more noise but let the")
    print("arms drift apart, so SE is minimised in between.\n")
    for m in TRACKED:
        print(f"  {m}")
        print(f"    {'window':>8} {'corr':>7} {'sd(diff)':>10} {'SE(effect)':>11}")
        best, best_se = None, np.inf
        for w in windows:
            if w > months:
                continue
            b, p = [], []
            for bs, ps in paired:
                bv = [v for v in _series(bs, m)[:w] if v is not None and not np.isnan(v)]
                pv = [v for v in _series(ps, m)[:w] if v is not None and not np.isnan(v)]
                if not bv or not pv:
                    continue
                b.append(np.mean(bv))
                p.append(np.mean(pv))
            if len(b) < 3:
                continue
            b, p = np.asarray(b), np.asarray(p)
            diff = p - b
            corr = np.corrcoef(b, p)[0, 1]
            se = diff.std(ddof=1) / np.sqrt(diff.size)
            if se < best_se:
                best, best_se = w, se
            print(f"    {w:8d} {corr:+7.3f} {diff.std(ddof=1):10.5f} {se:11.5f}")
        if best:
            print(f"    -> narrowest interval at window = {best} months\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=600)
    parser.add_argument("--spinup", type=int, default=120)
    parser.add_argument("--months", type=int, default=240)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--config", type=str, default="config/baseline_params.yaml")
    parser.add_argument(
        "--policy",
        type=str,
        default="config/policy_scenarios/ownership_cap_hard.yaml",
        help="policy used for the window-length sweep",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    print(
        f"households={args.households}  spinup={args.spinup}mo  "
        f"window={args.months}mo  seeds={args.seeds}"
    )

    jobs = [
        (s, args.households, args.spinup, args.months, args.policy, args.config)
        for s in range(args.seeds)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        paired = list(pool.map(_paired_series, jobs))

    report_stationarity([b for b, _ in paired], args.months)
    report_window_choice(paired, args.months, [24, 48, 72, 120, 180, 240])


if __name__ == "__main__":
    main()
