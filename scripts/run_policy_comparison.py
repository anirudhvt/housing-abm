"""Paired Monte Carlo policy comparison.

For each seed the model is spun up once with no policy, then forked: one copy
runs on unchanged, the other has the policy switched on. Both arms therefore
start from a bit-identical microstate, which is what makes the paired
difference a low-variance estimate of the policy effect. Outcomes are averaged
over the measurement window rather than read off the final month.

See housing_abm.experiment for why the design differs from a naive two-arm run.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import (  # noqa: E402
    minimum_detectable_effect,
    paired_summary,
    run_paired_seed,
    seeds_needed_for,
)
from housing_abm.metrics import (  # noqa: E402
    PRIMARY_METRICS,
    effective_sample_size,
)

REPORTED = list(PRIMARY_METRICS) + [
    "ftb_purchase_share",
    "investor_purchase_share",
    "institutional_share_of_rentals",
    "median_price",
    "median_rent",
]


def _one_seed(args):
    seed, households, spinup, months, policy_paths, config_path = args
    baseline, policy, b_series, p_series = run_paired_seed(
        seed, households, spinup, months, policy_paths, config_path
    )
    n_eff = {m: effective_sample_size(b_series[m]) for m in PRIMARY_METRICS}
    return seed, baseline, policy, n_eff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=600)
    parser.add_argument("--months", type=int, default=180, help="measurement window")
    parser.add_argument("--spinup", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/baseline_params.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    jobs = [
        (s, args.households, args.spinup, args.months, [args.policy], args.config)
        for s in seeds
    ]

    print(f"Paired policy comparison: {args.policy}")
    print(
        f"households={args.households}  spinup={args.spinup}mo  "
        f"window={args.months}mo  seeds={len(seeds)}  workers={args.workers}"
    )
    print("design: shared spin-up per seed, forked into baseline/policy arms")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, out in enumerate(pool.map(_one_seed, jobs), 1):
            results.append(out)
            print(f"  {i}/{len(seeds)} seeds done", end="\r", flush=True)
    print(" " * 40, end="\r")
    results.sort(key=lambda r: r[0])

    baseline_df = pd.DataFrame([r[1] for r in results])
    baseline_df.insert(0, "seed", [r[0] for r in results])
    policy_df = pd.DataFrame([r[2] for r in results])
    policy_df.insert(0, "seed", [r[0] for r in results])

    print(f"\n--- BASELINE window means (n={len(seeds)} seeds) ---")
    print(f"{'metric':<32} {'mean':>10} {'sd':>10} {'min':>10} {'max':>10}")
    print("-" * 76)
    for m in REPORTED:
        v = baseline_df[m].dropna()
        if v.empty:
            continue
        print(f"{m:<32} {v.mean():10.4f} {v.std():10.4f} {v.min():10.4f} {v.max():10.4f}")

    print(f"\n--- PAIRED EFFECT (policy - baseline, shared spin-up, n={len(seeds)}) ---")
    print(
        f"{'metric':<32} {'effect':>9} {'95% CI':>22} {'p':>8} "
        f"{'corr':>6} {'VRF':>6} {'MDE':>9}"
    )
    print("-" * 100)
    summaries = {}
    for m in REPORTED:
        s = paired_summary(baseline_df[m].tolist(), policy_df[m].tolist())
        if s is None:
            continue
        summaries[m] = s
        mde = minimum_detectable_effect(s["sd_diff"], s["n"])
        ci = f"[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]"
        print(
            f"{m:<32} {s['mean_diff']:+9.4f} {ci:>22} {s['p_value']:8.4f} "
            f"{s['arm_correlation']:+6.2f} {s['variance_reduction_factor']:6.1f} {mde:9.4f}"
        )

    print("\ncorr = correlation between the two arms across seeds; near 1 means the")
    print("      shared spin-up is doing its job. VRF = variance reduction versus")
    print("      unpaired sampling. MDE = smallest effect detectable at 80% power.")

    print("\n--- effective sample size of the window average (baseline arm) ---")
    for m in PRIMARY_METRICS:
        vals = [r[3][m] for r in results]
        print(
            f"  {m:<30} n_eff = {np.mean(vals):6.1f} independent months "
            f"of {args.months}"
        )

    for m, s in summaries.items():
        if s["p_value"] > 0.05:
            need = seeds_needed_for(s["mean_diff"], s["sd_diff"])
            if np.isfinite(need):
                print(
                    f"\nnull on {m}: point estimate {s['mean_diff']:+.4f}, "
                    f"would need ~{int(need)} seeds to resolve at 80% power"
                )

    if args.output:
        baseline_df.insert(1, "arm", "baseline")
        policy_df.insert(1, "arm", "policy")
        pd.concat([baseline_df, policy_df]).to_csv(args.output, index=False)
        meta = {
            "policy": args.policy,
            "households": args.households,
            "spinup": args.spinup,
            "months": args.months,
            "seeds": len(seeds),
            "design": "shared-spinup fork, window-averaged outcomes",
            "summaries": summaries,
        }
        with open(args.output.replace(".csv", "_summary.json"), "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"\nPer-seed results -> {args.output}")


if __name__ == "__main__":
    main()
