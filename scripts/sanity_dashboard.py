"""Sanity dashboard: run the model for N months and check for bugs

Exits non-zero if any check fails, 
"""

import argparse
import sys

sys.path.insert(0, "src")

from housing_abm.model import AtlantaHousingModel
from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.agents.small_landlord import SmallLandlord
from housing_abm.agents.institutional_investor import InstitutionalInvestor


def run_checks(n_months: int, n_households: int, seed: int) -> dict:
    model = AtlantaHousingModel(n_households=n_households, seed=seed)

    issues = []
    stats = {
        "worst_bank_balance": 0.0,
        "negative_balance_agent_months": 0,
        "negative_price_events": 0,
        "negative_mortgage_principal_events": 0,
        "months_with_zero_ownership_sales": 0,
        "months_with_zero_rental_matches": 0,
        "max_owning_plus_renting_gap": 0,
    }

    for month in range(1, n_months + 1):
        model.step()

        households = [
            a
            for a in model.agents
            if hasattr(a, "income") and getattr(a, "properties", None) is None
        ]
        investors = [
            a
            for a in model.agents
            if isinstance(a, (SmallLandlord, InstitutionalInvestor))
        ]
        units = [a for a in model.agents if isinstance(a, HousingUnit)]

        for a in households + investors:
            if a.bank_balance < stats["worst_bank_balance"]:
                stats["worst_bank_balance"] = a.bank_balance
            if a.bank_balance < 0:
                stats["negative_balance_agent_months"] += 1

        for u in units:
            if u.price is not None and u.price < 0:
                stats["negative_price_events"] += 1
            if u.mortgage_principal is not None and u.mortgage_principal < 0:
                stats["negative_mortgage_principal_events"] += 1

        n_owning = sum(1 for a in households if a.status == "owning")
        n_renting = sum(1 for a in households if a.status == "renting")
        n_social = len(households) - n_owning - n_renting
        stats["max_owning_plus_renting_gap"] = max(
            stats["max_owning_plus_renting_gap"], n_social
        )

    if stats["worst_bank_balance"] < -0.01:
        issues.append(
            f"Negative bank balance seen: worst={stats['worst_bank_balance']:,.2f}"
        )
    if stats["negative_price_events"] > 0:
        issues.append(
            f"Negative unit price seen {stats['negative_price_events']} times"
        )
    if stats["negative_mortgage_principal_events"] > 0:
        issues.append(
            f"Negative mortgage principal seen {stats['negative_mortgage_principal_events']} times"
        )

    stats["final_n_agents"] = len(model.agents)
    stats["final_n_households"] = sum(
        1
        for a in model.agents
        if hasattr(a, "income") and getattr(a, "properties", None) is None
    )
    stats["final_n_owning"] = sum(
        1 for a in model.agents if getattr(a, "status", None) == "owning"
    )
    stats["final_n_renting"] = sum(
        1 for a in model.agents if getattr(a, "status", None) == "renting"
    )
    stats["final_n_investor_units"] = sum(
        len(a.properties)
        for a in model.agents
        if isinstance(a, (SmallLandlord, InstitutionalInvestor))
    )
    stats["final_mortgage_rate"] = model.mortgage_rate_annual
    stats["issues"] = issues
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=150)
    parser.add_argument("--households", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="1,7,42,99,123")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    any_failed = False

    print(
        f"Sanity dashboard: {args.months} months x {len(seeds)} seeds, n_households={args.households}\n"
    )
    header = f"{'seed':>6} {'worst_balance':>16} {'neg_bal_agent_mo':>18} {'neg_price':>10} {'neg_principal':>14} {'final_n_hh':>11} {'owning':>8} {'renting':>8} {'inv_units':>10} {'mort_rate':>10}"
    print(header)
    print("-" * len(header))

    for seed in seeds:
        stats = run_checks(args.months, args.households, seed)
        status = "OK" if not stats["issues"] else "FAIL"
        print(
            f"{seed:>6} {stats['worst_bank_balance']:>16,.2f} {stats['negative_balance_agent_months']:>18} "
            f"{stats['negative_price_events']:>10} {stats['negative_mortgage_principal_events']:>14} "
            f"{stats['final_n_households']:>11} {stats['final_n_owning']:>8} {stats['final_n_renting']:>8} "
            f"{stats['final_n_investor_units']:>10} {stats['final_mortgage_rate']*100:>9.2f}% "
            f"  [{status}]"
        )
        if stats["issues"]:
            any_failed = True
            for issue in stats["issues"]:
                print(f"         -> {issue}")

    print()
    if any_failed:
        print("SANITY CHECK FAILED - see issues above.")
        sys.exit(1)
    else:
        print("All sanity checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
