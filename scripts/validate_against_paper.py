"""Loose reproduction of the paper's Section 5 validation approach
"""

import argparse
import sys
import pandas as pd
sys.path.insert(0, "src")

from housing_abm.model import AtlantaHousingModel
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.first_time_buyer import FirstTimeBuyer

def _appreciation_target():
    """Real Atlanta appreciation interquartile range from 
    atlanta_case_shiller_2020_2023.csv
    Falls back to placeholder if pull not run yet"""
    try:
        cs = pd.read_csv("atlanta_case_shiller_2020_2023.csv")
        low = cs["g"].quantile(0.25)
        high = cs["g"].quantile(0.75)
        note = (
            f"25th-75th pctile of real Atlanta YoY appreciation, "
            f"2020-2023 (Case-Shiller ATXRSA); full-period mean was "
            f"{cs['g'].mean():.3f}"
        )
        return low, high, note
    except FileNotFoundError:
        return (
            -0.05,
            0.20,
            "PLACEHOLDER band -- run pull_case_shiller_data.py to replace "
            "with real Atlanta 2020-2023 appreciation",
        )

# (label, low, high, note)
TARGETS = [
    (
        "homeownership_rate",
        0.55,
        0.75,
        "US national range is ~63-69%; Atlanta metro tends lower, ~60-63%",
    ),
    (
        "rental_vacancy_rate",
        0.05,
        0.15,
        "US national rental vacancy is typically 5-8%; wider band for a small ABM",
    ),
    (
        "mean_ltv_owner_occupier",
        0.60,
        0.95,
        "typical LTV at origination across FHA (up to 96.5%) and conventional (up to 80%)",
    ),
    (
        "mean_lti_owner_occupier",
        1.5,
        5.0,
        "typical loan-to-income multiples for US mortgages",
    ),
    ("annual_appreciation_g", *_appreciation_target())
]


def compute_snapshot(model):
    owners_with_mortgage = [
        a
        for a in model.agents
        if isinstance(a, (RepeatBuyer, FirstTimeBuyer))
        and a.house is not None
        and a.house.mortgage_principal > 0
    ]

    ltvs, ltis = [], []
    for a in owners_with_mortgage:
        price = a.house.mortgage_principal + (
            a.house.price - a.house.mortgage_principal
        )  # == a.house.price at purchase-adjacent state; use current price as proxy
        price = a.house.price if a.house.price else None
        if price and price > 0:
            ltvs.append(a.house.mortgage_principal / price)
        annual_income = a.income * 12
        if annual_income > 0:
            ltis.append(a.house.mortgage_principal / annual_income)

    return {
        "homeownership_rate": model._homeownership_rate(),
        "rental_vacancy_rate": model._rental_vacancy_rate(),
        "mean_ltv_owner_occupier": sum(ltvs) / len(ltvs) if ltvs else None,
        "mean_lti_owner_occupier": sum(ltis) / len(ltis) if ltis else None,
        "annual_appreciation_g": model._appreciation_g()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=150)
    parser.add_argument("--households", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = AtlantaHousingModel(n_households=args.households, seed=args.seed)
    for _ in range(args.months):
        model.step()

    snapshot = compute_snapshot(model)

    print(
        f"Validation against plausible ranges (n={args.households}, {args.months} months, seed={args.seed})\n"
    )
    print(f"{'metric':<28} {'value':>10} {'target range':>16}   status")
    print("-" * 78)
    for label, low, high, note in TARGETS:
        value = snapshot[label]
        if value is None:
            print(
                f"{label:<28} {'n/a':>10} {f'[{low}, {high}]':>16}   NO DATA - {note}"
            )
            continue
        in_range = low <= value <= high
        status = "OK" if in_range else "OUT OF RANGE"
        print(f"{label:<28} {value:>10.3f} {f'[{low}, {high}]':>16}   {status}")
        if not in_range:
            print(f"{'':<28} note: {note}")

    print("\nReminder: target ranges are generic US plausibility bands, not")
    print("Atlanta-specific targets - replace with real ACS/HMDA figures once sourced.")


if __name__ == "__main__":
    main()
