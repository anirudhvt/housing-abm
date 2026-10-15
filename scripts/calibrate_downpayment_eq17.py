"""Calibrate EQ 17 (owner down payment) from atlanta_hmda_2019.csv, using FHA
vs. conventional loans as a proxy for first-time vs. repeat buyers (HMDA has no
first-time-buyer flag). Prints fits only; update baseline_params.yaml by hand.
"""

import sys

import pandas as pd

sys.path.insert(0, "scripts")
from pull_hmda_data import clean_hmda, add_derived_fields  # noqa: E402

from housing_abm.equations.mortgage import estimate_floor_share_and_fit  # noqa: E402

FLOOR_BAND = 0.05  # matches the FHA floor's 3.5% minimum down payment, rounded up


def main():
    df = pd.read_csv("atlanta_hmda_2019.csv", low_memory=False)
    df = clean_hmda(df)
    df = add_derived_fields(df)
    df = df[(df["ltv_derived"] >= 40) & (df["ltv_derived"] <= 110)]
    df["downpayment_frac"] = (100 - df["ltv_derived"]) / 100.0

    owner = df[~df["is_investor_occupied"]]

    fha = owner.loc[owner["loan_type"] == 2, "downpayment_frac"].dropna().to_numpy()
    conv = owner.loc[owner["loan_type"] == 1, "downpayment_frac"].dropna().to_numpy()

    print(f"FHA / proxy first-time-buyer fit (n={len(fha)}):")
    print(estimate_floor_share_and_fit(fha, floor_band=FLOOR_BAND))
    print(f"\nConventional / proxy repeat-buyer fit (n={len(conv)}):")
    print(estimate_floor_share_and_fit(conv, floor_band=FLOOR_BAND))


if __name__ == "__main__":
    main()
