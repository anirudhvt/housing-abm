"""Paired comparison: current downpayment_eq17 config vs. a data-fit
calibration of it. See docs/methodology.md Section 10/10a/10b for the
derivation history:

- 10/10a: a first pass fit against HMDA, using loan_type (FHA vs.
  conventional) as a proxy for first-time vs. repeat buyer, since HMDA
  carries no direct first-time-buyer flag.
- 10b: the values below, fit against Fannie Mae's Single-Family Loan
  Performance Data (Atlanta MSA, 2019 Q1 purchase-money loans), which
  carries a genuine First Time Home Buyer Indicator reported at
  origination -- ground truth, not a proxy. This changed the first-time
  buyer numbers in a way the proxy got backwards (see 10b): real
  first-time buyers cluster at the down-payment floor *less* than the
  model assumes, not more.

Same CRN design as run_all_policies.py: one shared spin-up per seed, forked
into two arms that differ only in downpayment_eq17 from the fork point
onward. This treats "adopt the data-fit down-payment calibration" as a
policy switched on at the end of spin-up, which is what makes the two arms
comparable seed-by-seed rather than two independent, decorrelated runs.
"""

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import (  # noqa: E402
    build_and_spinup,
    holm_adjust,
    minimum_detectable_effect,
    paired_summary,
)
from housing_abm.metrics import run_window, window_means  # noqa: E402

# Ground-truth fit from calibrate_downpayment_eq17_fnma.py against Fannie
# Mae's Single-Family Loan Performance Data, Atlanta MSA (12060), 2019 Q1
# purchase-money owner-occupied loans, split on the real First Time Home
# Buyer Indicator (n=1,718 FTB / n=1,777 repeat).
ADOPTED_DOWNPAYMENT_EQ17 = {
    "first_time_buyer": {
        "d_minimum_pct": 0.035,
        "floor_share_p_floor": 0.33469150174621654,
        "lognorm_m": -2.22645358501561,
        "lognorm_s": 0.7017295474760067,
    },
    "repeat_buyer": {
        "d_minimum_pct": 0.20,
        "floor_share_p_floor": 0.7383230163196398,
        "lognorm_m": -0.9921282129710788,
        "lognorm_s": 0.361770237163324,
    },
}

OUTCOMES = [
    ("homeownership_rate", "homeownership rate"),
    ("ftb_purchase_share", "first-time-buyer purchase share"),
    ("repeat_buyer_purchase_share", "repeat-buyer purchase share"),
    ("mean_ltv_owner_occupier", "mean owner-occupier LTV"),
    ("mean_lti_owner_occupier", "mean owner-occupier LTI"),
    ("median_price", "median price"),
    ("institutional_share_of_rentals", "institutional share of rentals (negative control)"),
]


def fork_with_adopted_downpayment(model):
    treated = copy.deepcopy(model)
    treated.params["downpayment_eq17"] = copy.deepcopy(ADOPTED_DOWNPAYMENT_EQ17)
    return treated


def _one_seed(job):
    seed, households, spinup, months, config = job
    spun = build_and_spinup(seed, households, spinup, config)

    baseline_model = copy.deepcopy(spun)
    adopted_model = fork_with_adopted_downpayment(spun)

    baseline_means = window_means(run_window(baseline_model, months))
    adopted_means = window_means(run_window(adopted_model, months))
    return seed, {"baseline": baseline_means, "adopted": adopted_means}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=600)
    parser.add_argument("--spinup", type=int, default=120)
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--config", type=str, default="config/baseline_params.yaml")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    jobs = [
        (s, args.households, args.spinup, args.months, args.config)
        for s in range(args.seeds)
    ]

    print(
        f"households={args.households}  spinup={args.spinup}mo  "
        f"window={args.months}mo  seeds={args.seeds}  workers={args.workers}"
    )
    print("design: one spin-up per seed under current downpayment_eq17,")
    print("        forked into (baseline unchanged) vs (adopted HMDA-fit values)\n")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (seed, arms) in enumerate(pool.map(_one_seed, jobs), 1):
            results[seed] = arms
            print(f"  {i}/{args.seeds} seeds", end="\r", flush=True)
    print(" " * 40, end="\r")

    seeds = sorted(results)
    rows = []
    for seed in seeds:
        for arm, values in results[seed].items():
            rows.append({"seed": seed, "arm": arm, **values})
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(args.outdir, "downpayment_calibration_raw.csv"), index=False)

    base = raw[raw.arm == "baseline"]
    print(f"--- BASELINE (current downpayment_eq17), n={len(seeds)} seeds ---")
    for key, label in OUTCOMES:
        v = base[key].dropna()
        print(f"  {label:<48} {v.mean():9.4f}  (sd {v.std():.4f})")

    summaries = {}
    for key, _ in OUTCOMES:
        b = base.set_index("seed")[key]
        a = raw[raw.arm == "adopted"].set_index("seed")[key]
        s = paired_summary(b.loc[seeds].tolist(), a.loc[seeds].tolist())
        if s is not None:
            summaries[key] = s

    keys = list(summaries)
    adjusted = holm_adjust([summaries[k]["p_value"] for k in keys])
    for k, adj in zip(keys, adjusted):
        summaries[k]["p_holm"] = adj

    print(f"\n--- ADOPTED - BASELINE (Holm-adjusted across {len(keys)} outcomes) ---")
    print(f"  {'outcome':<48} {'effect':>9} {'95% CI':>21} {'p':>8} {'p_holm':>8} {'corr':>6} {'MDE':>8}")
    print("  " + "-" * 100)
    for key, label in OUTCOMES:
        s = summaries[key]
        mde = minimum_detectable_effect(s["sd_diff"], s["n"])
        ci = f"[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]"
        star = (
            "***" if s["p_holm"] < 0.001
            else "**" if s["p_holm"] < 0.01
            else "*" if s["p_holm"] < 0.05
            else ""
        )
        print(
            f"  {label:<48} {s['mean_diff']:+9.4f} {ci:>21} {s['p_value']:8.4f} "
            f"{s['p_holm']:8.4f} {s['arm_correlation']:+6.2f} {mde:8.4f} {star}"
        )

    with open(os.path.join(args.outdir, "downpayment_calibration_summary.json"), "w") as f:
        json.dump(
            {"config": vars(args), "adopted_downpayment_eq17": ADOPTED_DOWNPAYMENT_EQ17,
             "effects": summaries},
            f, indent=2, default=float,
        )
    print(f"\nWrote {args.outdir}/downpayment_calibration_raw.csv and _summary.json")


if __name__ == "__main__":
    main()
