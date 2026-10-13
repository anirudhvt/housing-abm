"""Calibrate EQ17 (owner-occupier down payment) against the Atlanta HMDA
loan-level data already checked into the repo (atlanta_hmda_2019.csv).

housing_abm.equations.mortgage.estimate_floor_share_and_fit existed for
exactly this purpose but was never actually invoked against real data --
downpayment_eq17's first_time_buyer/repeat_buyer blocks in
config/baseline_params.yaml carried no provenance comment, unlike
tract_calibration/market_smoothing.

HMDA has no first-time-vs-repeat-buyer flag, so this uses loan_type as a
proxy: FHA (loan_type == 2) for first-time buyers, conventional
(loan_type == 1) for repeat buyers -- FHA's 3.5% minimum down payment and
looser credit terms make it the buyer-type split that actually exists in
the data, but it is a proxy, not a direct field. A repeat buyer can use FHA
and a first-time buyer can use a low-down-payment conventional program
(HomeReady/Home Possible), so this should be read as the best available
split, not ground truth. Print output only -- baseline_params.yaml is
updated by hand once the numbers are reviewed, not by this script.
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
