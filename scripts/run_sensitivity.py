"""Sensitivity of the policy ranking to uncertain parameters.

Sweeps parameters that could not be estimated from data and re-runs the full
paired comparison at each value, so the question answered is whether the
*ranking* of policies survives parameter uncertainty, not just whether any one
effect changes size.

The sweep now uses the same paired design as the main study: one spin-up per
(parameter value, seed), forked into baseline and policy arms, with outcomes
averaged over the measurement window. Running each arm independently -- as the
previous version did -- meant every cell of the sweep carried the full
unpaired Monte Carlo noise, which at these effect sizes was larger than the
parameter sensitivity being measured.

beta_institutional is the parameter that most deserves this treatment: it is
the shape parameter of the EQ 10/13 logistic, and how much any financial
penalty can move investor behaviour depends directly on where on that logistic
the market sits.
"""

import argparse
import copy
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import (  # noqa: E402
    build_and_spinup,
    fork_with_policy,
    paired_summary,
)
from housing_abm.metrics import run_window, window_means  # noqa: E402

SWEEPS = {
    "beta_institutional": {
        "key_path": ["investor_probs_eq10_eq13", "beta_institutional"],
        "values": [10.0, 25.0, 50.0, 75.0, 100.0],
        "label": "beta_institutional (EQ 10/13)",
    },
    "beta_small_landlord": {
        "key_path": ["investor_probs_eq10_eq13", "beta_small_landlord"],
        "values": [10.0, 25.0, 50.0, 75.0, 100.0],
        "label": "beta_small_landlord (EQ 10/13)",
    },
    "consumption_alpha": {
        "key_path": ["consumption_eq2", "alpha"],
        "values": [0.2, 0.35, 0.5, 0.65, 0.8],
        "label": "alpha_consumption (EQ 2)",
    },
    "institutional_delta": {
        "key_path": ["investor_yield_eq9_eq12", "delta_institutional"],
        "values": [0.3, 0.45, 0.6, 0.75, 0.9],
        "label": "delta_institutional (weight on capital gain)",
    },
}

POLICIES = [
    ("waiting_period", "config/policy_scenarios/waiting_period.yaml"),
    ("ownership_cap_soft", "config/policy_scenarios/ownership_cap_soft.yaml"),
    ("ownership_cap_hard", "config/policy_scenarios/ownership_cap_hard.yaml"),
    ("purchase_tax", "config/policy_scenarios/purchase_tax.yaml"),
    ("vacancy_tax", "config/policy_scenarios/vacancy_tax.yaml"),
    ("portfolio_tax", "config/policy_scenarios/portfolio_tax.yaml"),
]

OUTCOMES = ["homeownership_rate", "ftb_purchase_share", "rental_vacancy_rate"]


def _write_variant(base_config, key_path, value):
    """Config file with one parameter overridden."""
    with open(base_config) as f:
        params = yaml.safe_load(f)
    node = params
    for key in key_path[:-1]:
        node = node[key]
    node[key_path[-1]] = value
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, prefix="sweep_"
    )
    yaml.safe_dump(params, handle)
    handle.close()
    return handle.name


def _one_cell(job):
    seed, households, spinup, months, config = job
    spun = build_and_spinup(seed, households, spinup, config)
    out = {"baseline": window_means(run_window(copy.deepcopy(spun), months))}
    for name, path in POLICIES:
        out[name] = window_means(run_window(fork_with_policy(spun, [path]), months))
    return seed, out


def run_sweep(name, spec, args):
    print(f"\n{'=' * 78}\n{spec['label']}\n{'=' * 78}")
    rows = []
    for value in spec["values"]:
        config = _write_variant(args.config, spec["key_path"], value)
        try:
            jobs = [
                (s, args.households, args.spinup, args.months, config)
                for s in range(args.seeds)
            ]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = dict(pool.map(_one_cell, jobs))
        finally:
            os.unlink(config)

        seeds = sorted(results)
        print(f"\n  {name} = {value}")
        for outcome in OUTCOMES:
            base = [results[s]["baseline"][outcome] for s in seeds]
            print(f"    {outcome}  (baseline {pd.Series(base).mean():.4f})")
            for policy, _path in POLICIES:
                arm = [results[s][policy][outcome] for s in seeds]
                summary = paired_summary(base, arm, n_boot=2000)
                if summary is None:
                    continue
                rows.append(
                    {
                        "parameter": name,
                        "value": value,
                        "policy": policy,
                        "outcome": outcome,
                        **summary,
                    }
                )
                marker = "*" if summary["p_value"] < 0.05 else " "
                print(
                    f"      {policy:<20} {summary['mean_diff']:+8.4f} "
                    f"[{summary['ci_lo']:+.4f},{summary['ci_hi']:+.4f}] "
                    f"p={summary['p_value']:.3f} {marker}"
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="all", choices=list(SWEEPS) + ["all"])
    parser.add_argument("--households", type=int, default=600)
    parser.add_argument("--spinup", type=int, default=120)
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--config", default="config/baseline_params.yaml")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = list(SWEEPS) if args.param == "all" else [args.param]
    for name in names:
        df = run_sweep(name, SWEEPS[name], args)
        path = os.path.join(args.outdir, f"sensitivity_{name}.csv")
        df.to_csv(path, index=False)
        print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
