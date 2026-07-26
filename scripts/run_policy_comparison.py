"""Monte Carlo policy comparison harness.

Runs the model N times with a baseline config and N times with a policy
overlay (same seeds, paired), each with a spin-up period

Track relevant data
"""

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from housing_abm.model import AtlantaHousingModel
from validate_against_paper import compute_snapshot

METRICS = [
    "homeownership_rate",
    "rental_vacancy_rate",
    "mean_ltv_owner_occupier",
    "mean_lti_owner_occupier",
    "annual_appreciation_g",
]


def _lti_compliance(model, policy_docs):
    """Share of outstanding owner-occupier loans currently above each
    policy's hard_limit, by loan_type """
    from housing_abm.agents.first_time_buyer import FirstTimeBuyer
    from housing_abm.agents.repeat_buyer import RepeatBuyer

    lti_policies = [p for p in policy_docs if p.get("type") == "lti_limit"]
    if not lti_policies:
        return {}

    out = {}
    for policy in lti_policies:
        loan_type = policy["loan_type"]
        hard_limit = policy["hard_limit"]
        agent_cls = FirstTimeBuyer if loan_type == "fha" else RepeatBuyer
        ltis = []
        for a in model.agents:
            if (
                isinstance(a, agent_cls)
                and a.house is not None
                and a.house.mortgage_principal
                and a.income > 0
            ):
                ltis.append(a.house.mortgage_principal / (a.income * 12))
        if ltis:
            out[f"share_lti_above_{hard_limit}_{loan_type}"] = float(
                np.mean(np.array(ltis) > hard_limit)
            )
    return out


def run_arm(seed, households, spinup, months, policy_paths, config_path):
    model = AtlantaHousingModel(
        config_path=config_path,
        n_households=households,
        seed=seed,
        policy_paths=policy_paths,
    )
    if spinup:
        model.run_spinup(spinup)
    for _ in range(months):
        model.step()

    row = compute_snapshot(model)
    if policy_paths:
        import yaml

        docs = []
        for p in policy_paths:
            with open(p) as f:
                docs.extend(yaml.safe_load(f))
        row.update(_lti_compliance(model, docs))
    return row


def summarize(df, label):
    print(f"\n--- {label} (n={len(df)} seeds) ---")
    print(f"{'metric':<32} {'mean':>8} {'std':>8} {'min':>8} {'p25':>8} {'median':>8} {'p75':>8} {'max':>8}")
    print("-" * 96)
    for col in df.columns:
        if col in ("seed", "arm"):
            continue
        vals = df[col].dropna()
        if vals.empty:
            continue
        print(
            f"{col:<32} {vals.mean():8.4f} {vals.std():8.4f} {vals.min():8.4f} "
            f"{vals.quantile(0.25):8.4f} {vals.median():8.4f} {vals.quantile(0.75):8.4f} {vals.max():8.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=300)
    parser.add_argument("--months", type=int, default=150)
    parser.add_argument("--spinup", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds to run (seeds 0..n-1)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--policy", type=str, required=True, help="path to policy yaml to compare against baseline")
    parser.add_argument("--config", type=str, default="config/baseline_params.yaml")
    parser.add_argument("--output", type=str, default=None, help="optional csv path for raw per-seed results")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    baseline_rows, policy_rows = [], []
    for i, seed in enumerate(seeds):
        print(f"  seed {seed} ({i + 1}/{len(seeds)})...", end="\r", flush=True)
        baseline_rows.append(
            run_arm(seed, args.households, args.spinup, args.months, None, args.config)
        )
        policy_rows.append(
            run_arm(seed, args.households, args.spinup, args.months, [args.policy], args.config)
        )
    print(" " * 40, end="\r")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.insert(0, "seed", seeds)
    policy_df = pd.DataFrame(policy_rows)
    policy_df.insert(0, "seed", seeds)

    print(f"\nMonte Carlo policy comparison: {args.policy}")
    print(f"households={args.households}  spinup={args.spinup}mo  reported={args.months}mo  seeds={seeds[0]}..{seeds[-1]}")

    summarize(baseline_df, "BASELINE")
    summarize(policy_df, "WITH POLICY")

    # paired differences (policy - baseline, same seed) 
    print(f"\n--- PAIRED DIFFERENCE (policy - baseline, same seed, n={len(seeds)}) ---")
    common_metrics = [m for m in METRICS if m in baseline_df.columns and m in policy_df.columns]
    print(f"{'metric':<32} {'mean diff':>10} {'std':>8} {'seeds moved same direction':>28}")
    print("-" * 84)
    for m in common_metrics:
        b = baseline_df[m].astype(float)
        p = policy_df[m].astype(float)
        diff = (p - b).dropna()
        if diff.empty:
            continue
        same_dir = max((diff > 0).sum(), (diff < 0).sum())
        print(
            f"{m:<32} {diff.mean():10.4f} {diff.std():8.4f} {same_dir}/{len(diff)}"
        )

    if args.output:
        baseline_df.insert(1, "arm", "baseline")
        policy_df.insert(1, "arm", "policy")
        pd.concat([baseline_df, policy_df]).to_csv(args.output, index=False)
        print(f"\nRaw per-seed results written to {args.output}")


if __name__ == "__main__":
    main()