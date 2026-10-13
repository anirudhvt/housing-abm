"""Run every policy scenario against a shared baseline and compare them.

Each seed is spun up once with no policy and forked into the two arms, so the
comparison is paired on an identical microstate. The same spun-up seeds are
reused across all six policies, which means the policies are compared to each
other on common random numbers too, not just each to its own baseline.

Family-wise error is controlled with Holm-Bonferroni across the full grid of
policies x outcomes, and every null is reported with the effect size it was
powered to detect, so "no effect" can be read as a bound rather than as an
absence of evidence.
"""

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import (  # noqa: E402
    build_and_spinup,
    fork_with_policy,
    holm_adjust,
    minimum_detectable_effect,
    paired_summary,
)
from housing_abm.metrics import run_window, window_means  # noqa: E402

POLICIES = [
    ("waiting_period", "config/policy_scenarios/waiting_period.yaml"),
    ("ownership_cap_soft", "config/policy_scenarios/ownership_cap_soft.yaml"),
    ("ownership_cap_hard", "config/policy_scenarios/ownership_cap_hard.yaml"),
    ("purchase_tax", "config/policy_scenarios/purchase_tax.yaml"),
    ("vacancy_tax", "config/policy_scenarios/vacancy_tax.yaml"),
    ("portfolio_tax", "config/policy_scenarios/portfolio_tax.yaml"),
]

OUTCOMES = [
    ("homeownership_rate", "homeownership rate"),
    ("rental_vacancy_rate", "rental vacancy rate"),
    ("annual_appreciation_g", "annual appreciation"),
    ("ftb_purchase_share", "first-time-buyer purchase share"),
    ("institutional_share_of_rentals", "institutional share of rentals"),
]

# who ends up buying the homes the policy diverts from institutional investors
MECHANISM = [
    ("institutional_purchase_share", "institutional share of purchases"),
    ("small_landlord_purchase_share", "small-landlord share of purchases"),
    ("repeat_buyer_purchase_share", "repeat-buyer share of purchases"),
    ("ftb_purchase_share", "first-time-buyer share of purchases"),
]


def _one_seed(job):
    """All arms for one seed, from a single shared spin-up."""
    seed, households, spinup, months, config, policy_paths = job
    spun = build_and_spinup(seed, households, spinup, config)

    out = {"baseline": window_means(run_window(copy.deepcopy(spun), months))}
    for name, path in policy_paths:
        arm = fork_with_policy(spun, [path])
        out[name] = window_means(run_window(arm, months))
    return seed, out


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
        (s, args.households, args.spinup, args.months, args.config, POLICIES)
        for s in range(args.seeds)
    ]

    print(
        f"households={args.households}  spinup={args.spinup}mo  "
        f"window={args.months}mo  seeds={args.seeds}  workers={args.workers}"
    )
    print("design: one spin-up per seed, forked into baseline + 6 policy arms")
    print("        (all policies share the same baseline draws)\n")

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
    raw.to_csv(os.path.join(args.outdir, "all_policies_raw.csv"), index=False)

    # ---- baseline ----
    base = raw[raw.arm == "baseline"]
    print(f"--- BASELINE (n={len(seeds)} seeds, window means) ---")
    print(f"{'outcome':<34} {'mean':>9} {'sd across seeds':>17}")
    print("-" * 64)
    for key, label in OUTCOMES:
        v = base[key].dropna()
        print(f"{label:<34} {v.mean():9.4f} {v.std():17.4f}")

    # ---- effects ----
    summaries = {}
    for key, _ in OUTCOMES:
        b = base.set_index("seed")[key]
        for name, _path in POLICIES:
            p = raw[raw.arm == name].set_index("seed")[key]
            s = paired_summary(b.loc[seeds].tolist(), p.loc[seeds].tolist())
            if s is not None:
                summaries[(name, key)] = s

    keys = list(summaries)
    adjusted = holm_adjust([summaries[k]["p_value"] for k in keys])
    for k, adj in zip(keys, adjusted):
        summaries[k]["p_holm"] = adj

    print(
        f"\n--- PAIRED POLICY EFFECTS "
        f"(Holm-adjusted across {len(keys)} policy x outcome tests) ---"
    )
    for key, label in OUTCOMES:
        print(f"\n{label}")
        print(
            f"  {'policy':<20} {'effect':>9} {'95% CI':>21} "
            f"{'p':>8} {'p_holm':>8} {'corr':>6} {'MDE':>8}"
        )
        print("  " + "-" * 88)
        ordered = sorted(
            (n for n, _ in POLICIES),
            key=lambda n: -abs(summaries[(n, key)]["mean_diff"]),
        )
        for name in ordered:
            s = summaries[(name, key)]
            mde = minimum_detectable_effect(s["sd_diff"], s["n"])
            ci = f"[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]"
            star = (
                "***" if s["p_holm"] < 0.001
                else "**" if s["p_holm"] < 0.01
                else "*" if s["p_holm"] < 0.05
                else ""
            )
            print(
                f"  {name:<20} {s['mean_diff']:+9.4f} {ci:>21} {s['p_value']:8.4f} "
                f"{s['p_holm']:8.4f} {s['arm_correlation']:+6.2f} {mde:8.4f} {star}"
            )

    # ---- where the diverted purchases actually go ----
    print("\n--- INCIDENCE: share of all purchases, by buyer class ---")
    print("Who buys the homes a policy stops institutional investors from buying.")
    print(f"\n  {'arm':<22}" + "".join(f"{lbl.split()[0][:12]:>14}" for _k, lbl in MECHANISM))
    for arm in ["baseline"] + [n for n, _ in POLICIES]:
        sub = raw[raw.arm == arm]
        cells = "".join(f"{sub[k].mean():14.4f}" for k, _ in MECHANISM)
        print(f"  {arm:<22}{cells}")

    print("\n* / ** / *** = Holm-adjusted p < 0.05 / 0.01 / 0.001.")
    print("MDE is the smallest effect this many seeds could detect at 80% power;")
    print("a null with |effect| < MDE bounds the effect rather than ruling one out.")

    with open(os.path.join(args.outdir, "all_policies_summary.json"), "w") as f:
        json.dump(
            {
                "config": vars(args),
                "design": "shared spin-up per seed, forked arms, window-averaged",
                "effects": {f"{n}|{m}": s for (n, m), s in summaries.items()},
            },
            f,
            indent=2,
            default=float,
        )
    print(f"\nWrote {args.outdir}/all_policies_raw.csv and all_policies_summary.json")


if __name__ == "__main__":
    main()
