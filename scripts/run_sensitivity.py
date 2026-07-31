"""Sensitivity analysis for key uncertain parameters.

For each parameter we sweep a range of values, running the full
Monte Carlo (baseline vs. each policy) at every value. This tells
us whether policy rankings are robust to parameter uncertainty.

 parameters swept:
  1. beta_institutional (Eq 10/13) -- investor buy/sell sensitivity
  2. beta_small_landlord (Eq 10/13) -- small landlord sensitivity  
  3. consumption_alpha (Eq 2) -- savings accumulation speed

"""

import argparse
import copy
import io
import sys
import tempfile
import os

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from housing_abm.model import AtlantaHousingModel
from validate_against_paper import compute_snapshot

SWEEP_DEFINITIONS = {
    "beta_institutional": {
        "key_path": ["investor_probs_eq10_eq13", "beta_institutional"],
        "values": [10.0, 25.0, 50.0, 75.0, 100.0],
        "label": "β_institutional (Eq 10/13)",
    },
    "beta_small_landlord": {
        "key_path": ["investor_probs_eq10_eq13", "beta_small_landlord"],
        "values": [10.0, 25.0, 50.0, 75.0, 100.0],
        "label": "β_small_landlord (Eq 10/13)",
    },
    "consumption_alpha": {
        "key_path": ["consumption_eq2", "alpha"],
        "values": [0.2, 0.35, 0.5, 0.65, 0.8],
        "label": "α_consumption (Eq 2)",
    },
}

# Policies to compare at each parameter value
# Maps a short name to the policy yaml path
POLICIES = {
    "waiting_period":       "config/policy_scenarios/waiting_period.yaml",
    "ownership_cap_soft":   "config/policy_scenarios/ownership_cap_soft.yaml",
    "ownership_cap_hard":   "config/policy_scenarios/ownership_cap_hard.yaml",
    "purchase_tax":         "config/policy_scenarios/purchase_tax.yaml",
    "vacancy_tax":          "config/policy_scenarios/vacancy_tax.yaml",
    "portfolio_tax":        "config/policy_scenarios/portfolio_tax.yaml",
    "geographic":           "config/policy_scenarios/geographic_restriction.yaml",
}

# Metrics we care about for sensitivity
METRICS = [
    "homeownership_rate",
    "rental_vacancy_rate",
    "annual_appreciation_g",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def set_nested(d: dict, key_path: list, value):
    """Set a nested key in a dict given a list of keys."""
    for key in key_path[:-1]:
        d = d[key]
    d[key_path[-1]] = value


def make_temp_config(base_params: dict, key_path: list, value: float) -> str:
    """Write a temporary config yaml with one parameter overridden.
    Returns the path to the temp file."""
    params = copy.deepcopy(base_params)
    set_nested(params, key_path, value)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="sensitivity_cfg_"
    )
    yaml.dump(params, tmp)
    tmp.close()
    return tmp.name


def run_arm(seed, households, spinup, months, config_path, policy_path=None):
    """Run one arm (baseline or policy) and return metric snapshot."""
    policy_paths = [policy_path] if policy_path else None
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
    return compute_snapshot(model)


def run_sweep_for_param(
    param_name: str,
    base_params: dict,
    seeds: list,
    households: int,
    spinup: int,
    months: int,
) -> pd.DataFrame:
    """Run the full sweep for one parameter. Returns a DataFrame with columns:
    param_name, param_value, policy, seed, metric, value, diff_from_baseline
    """
    defn = SWEEP_DEFINITIONS[param_name]
    key_path = defn["key_path"]
    values = defn["values"]

    rows = []
    total = len(values) * (1 + len(POLICIES)) * len(seeds)
    done = 0

    for param_value in values:
        # Write temp config with this parameter value
        cfg_path = make_temp_config(base_params, key_path, param_value)

        try:
            for seed in seeds:
                # ── Baseline arm ──────────────────────────────────────────
                baseline = run_arm(seed, households, spinup, months, cfg_path)
                done += 1
                print(
                    f"  [{done}/{total}] param={param_value} seed={seed} arm=baseline",
                    end="\r", flush=True,
                )

                for metric in METRICS:
                    rows.append({
                        "param": param_name,
                        "param_value": param_value,
                        "policy": "baseline",
                        "seed": seed,
                        "metric": metric,
                        "value": baseline.get(metric),
                        "diff_from_baseline": 0.0,
                    })

                # ── Policy arms ───────────────────────────────────────────
                for policy_name, policy_path in POLICIES.items():
                    if not os.path.exists(policy_path):
                        print(f"\n  WARNING: {policy_path} not found, skipping")
                        done += 1
                        continue

                    policy_snap = run_arm(
                        seed, households, spinup, months, cfg_path, policy_path
                    )
                    done += 1
                    print(
                        f"  [{done}/{total}] param={param_value} seed={seed} arm={policy_name}",
                        end="\r", flush=True,
                    )

                    for metric in METRICS:
                        b_val = baseline.get(metric)
                        p_val = policy_snap.get(metric)
                        diff = (p_val - b_val) if (p_val is not None and b_val is not None) else None
                        rows.append({
                            "param": param_name,
                            "param_value": param_value,
                            "policy": policy_name,
                            "seed": seed,
                            "metric": metric,
                            "value": p_val,
                            "diff_from_baseline": diff,
                        })

        finally:
            os.unlink(cfg_path)

    print(" " * 80, end="\r")
    return pd.DataFrame(rows)


def summarize_sweep(df: pd.DataFrame, param_name: str):
    """Print a summary table: for each (param_value, policy), show mean diff
    across seeds for each metric."""
    defn = SWEEP_DEFINITIONS[param_name]
    label = defn["label"]

    print(f"\n{'='*80}")
    print(f"SENSITIVITY: {label}")
    print(f"{'='*80}")

    policy_df = df[df["policy"] != "baseline"]

    for metric in METRICS:
        m_df = policy_df[policy_df["metric"] == metric]
        pivot = (
            m_df.groupby(["policy", "param_value"])["diff_from_baseline"]
            .mean()
            .unstack("param_value")
        )
        print(f"\nMetric: {metric}  (mean diff vs baseline across seeds)")
        print(f"{'Policy':<24}", end="")
        for v in defn["values"]:
            print(f"  {label.split('(')[0].strip()}={v:<6}", end="")
        print()
        print("-" * (24 + 14 * len(defn["values"])))
        for policy in pivot.index:
            print(f"{policy:<24}", end="")
            for v in defn["values"]:
                val = pivot.loc[policy, v] if v in pivot.columns else None
                if val is None or np.isnan(val):
                    print(f"  {'n/a':>12}", end="")
                else:
                    print(f"  {val:>12.4f}", end="")
            print()

    # Ranking stability check
    print(f"\n--- RANKING STABILITY: homeownership_rate ---")
    print("Does the best-performing policy stay consistent across parameter values?")
    hr_df = policy_df[policy_df["metric"] == "homeownership_rate"]
    for v in defn["values"]:
        v_df = hr_df[hr_df["param_value"] == v]
        mean_diffs = v_df.groupby("policy")["diff_from_baseline"].mean().sort_values(ascending=False)
        best = mean_diffs.index[0]
        best_val = mean_diffs.iloc[0]
        print(f"  {label.split('(')[0].strip()}={v:<8}  best policy: {best:<24} (mean diff={best_val:+.4f})")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for housing ABM")
    parser.add_argument(
        "--param",
        choices=list(SWEEP_DEFINITIONS.keys()) + ["all"],
        required=True,
        help="Which parameter to sweep, or 'all' to run all three",
    )
    parser.add_argument("--households", type=int, default=300)
    parser.add_argument("--months", type=int, default=150)
    parser.add_argument("--spinup", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (0..n-1)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--config", type=str, default="config/baseline_params.yaml")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path for raw results. If running 'all', will append param name before .csv",
    )
    args = parser.parse_args()

    # Load base config once
    with open(args.config) as f:
        base_params = yaml.safe_load(f)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    params_to_run = (
        list(SWEEP_DEFINITIONS.keys()) if args.param == "all" else [args.param]
    )

    for param_name in params_to_run:
        defn = SWEEP_DEFINITIONS[param_name]
        print(f"\nRunning sensitivity sweep: {defn['label']}")
        print(f"Values: {defn['values']}")
        print(f"Policies: {list(POLICIES.keys())}")
        print(f"Seeds: {seeds}  |  spinup={args.spinup}mo  reported={args.months}mo")
        print()

        df = run_sweep_for_param(
            param_name=param_name,
            base_params=base_params,
            seeds=seeds,
            households=args.households,
            spinup=args.spinup,
            months=args.months,
        )

        summarize_sweep(df, param_name)

        if args.output:
            if args.param == "all":
                # Insert param name before extension
                base, ext = os.path.splitext(args.output)
                out_path = f"{base}_{param_name}{ext}"
            else:
                out_path = args.output
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"\nRaw results written to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()