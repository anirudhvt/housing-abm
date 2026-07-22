"""Loose reproduction of the paper's Section 5 validation approach: run the
model and check whether a handful of target statistics land in a plausible
range, the way the paper checks against the FPC's nine housing core
indicators (Figure 3). This is NOT a pass/fail gate the way the sanity
dashboard is - it's a sense-check that the model's behavior is at least in
the right ballpark before trusting any experiment built on top of it.

Target ranges here are generic US housing-market plausibility bands, not
Atlanta-specific - real Atlanta ACS/Census/HMDA targets should replace these
once that data is sourced (see the calibration step in the project roadmap).

Usage: python scripts/validate_against_paper.py [--months 150] [--households 300]
"""

import argparse
import sys

sys.path.insert(0, "src")

from housing_abm.model import AtlantaHousingModel
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.first_time_buyer import FirstTimeBuyer

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
