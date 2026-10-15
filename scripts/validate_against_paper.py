"""Validate the baseline against Atlanta plausibility targets over multiple
seeds, using the same spin-up and window as the policy experiments."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import build_and_spinup  # noqa: E402
from housing_abm.metrics import run_window, window_means  # noqa: E402


def _appreciation_target():
    """Band for price_cagr: long-run real US house price growth, since the
    model has no inflation. Nominal Case-Shiller is shown for context only."""
    note = "long-run real US house price growth (Shiller); model has no inflation"
    try:
        cs = pd.read_csv("atlanta_case_shiller.csv")
        note += (
            f". Nominal Atlanta ATXRSA 2015-2026 for context: "
            f"mean {cs['g'].mean():.3f}, median {cs['g'].median():.3f}"
        )
    except FileNotFoundError:
        pass
    return -0.01, 0.02, note


TARGETS = [
    (
        "homeownership_rate",
        0.48,
        0.60,
        "ACS 5-year 2019 owner-occupancy for the seven Atlanta metro counties "
        "is 52.9%; band allows for small-ABM variation",
    ),
    (
        "rental_vacancy_rate",
        0.05,
        0.15,
        "ACS metro Atlanta rental vacancy 7.3%; wider band for a small ABM",
    ),
    (
        "mean_ltv_owner_occupier",
        0.60,
        0.95,
        "origination LTV across FHA (up to 96.5%) and conventional (up to 80%)",
    ),
    (
        "mean_lti_owner_occupier",
        1.5,
        5.0,
        "typical loan-to-income multiples for US mortgages",
    ),
    (
        "institutional_share_of_rentals",
        0.20,
        0.40,
        "institutional operators hold roughly 30% of metro Atlanta SFR",
    ),
    ("price_cagr", *_appreciation_target()),
    (
        "months_of_inventory",
        1.0,
        8.0,
        "US sale markets typically run 3-6 months of inventory; the EQ 8 "
        "price-reduction rule clears this model's sale market far more slowly, "
        "which is the main known limitation of the current calibration",
    ),
]


def _one(job):
    seed, households, spinup, months, config = job
    model = build_and_spinup(seed, households, spinup, config)
    return window_means(run_window(model, months))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=600)
    parser.add_argument("--spinup", type=int, default=120)
    parser.add_argument("--months", type=int, default=180)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--config", type=str, default="config/baseline_params.yaml")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    jobs = [
        (s, args.households, args.spinup, args.months, args.config)
        for s in range(args.seeds)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_one, jobs))
    df = pd.DataFrame(rows)

    print(
        f"Baseline validation -- {args.households} households, "
        f"{args.spinup}mo spin-up, {args.months}mo window averaged, "
        f"{args.seeds} seeds\n"
    )
    print(f"{'metric':<32} {'mean':>8} {'95% CI':>20} {'target':>16}   status")
    print("-" * 96)
    n_out = 0
    for label, low, high, note in TARGETS:
        vals = df[label].dropna()
        if vals.empty:
            print(f"{label:<32} {'n/a':>8} {'':>20} {f'[{low}, {high}]':>16}   NO DATA")
            continue
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        crit = stats.t.ppf(0.975, len(vals) - 1)
        lo, hi = mean - crit * se, mean + crit * se
        in_range = low <= mean <= high
        overlaps = not (hi < low or lo > high)
        status = "OK" if in_range else ("marginal" if overlaps else "OUT OF RANGE")
        n_out += 0 if in_range else 1
        ci = f"[{lo:.3f}, {hi:.3f}]"
        print(
            f"{label:<32} {mean:8.3f} {ci:>20} {f'[{low}, {high}]':>16}   {status}"
        )
        if not in_range:
            print(f"{'':<32} note: {note}")

    print(f"\n{len(TARGETS) - n_out}/{len(TARGETS)} targets met.")
    print(
        "Spin-up and window match the policy experiments, so these numbers "
        "describe\nthe same model state the policy effects are measured in."
    )


if __name__ == "__main__":
    main()
