"""Pull the S&P/Case-Shiller Atlanta metro HPI from FRED and compute 
Eq (4) trailing appreciation estimate 

    g_t = alpha * ( (h_{t-1}+h_{t-2}+h_{t-3}) / (h_{t-13}+h_{t-14}+h_{t-15}) - 1 )

metro-level, not tract-level

Series: ATXRSA (S&P/Case-Shiller GA-Atlanta Home Price Index, NSA, monthly,
Jan 2000 = 100).
"""

import argparse
import os
import sys

import pandas as pd
from fredapi import Fred

SERIES_ID = "ATXRSA"


def pull_series() -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY")
    fred = Fred(api_key=api_key)
    return fred.get_series(SERIES_ID)


def compute_trailing_appreciation(hpi: pd.Series, alpha: float) -> pd.DataFrame:
    df = hpi.rename("hpi").to_frame()
    df["trail_1_3"] = df["hpi"].shift(1) + df["hpi"].shift(2) + df["hpi"].shift(3)
    df["trail_13_15"] = (
        df["hpi"].shift(13) + df["hpi"].shift(14) + df["hpi"].shift(15)
    )
    df["g"] = alpha * (df["trail_1_3"] / df["trail_13_15"] - 1)
    return df.dropna(subset=["g"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0,
                     help="Eq (4) alpha, placeholder until calibrated")
    ap.add_argument("--start", default="2015-01-01",
                     help="trim start date for the output window")
    args = ap.parse_args()

    hpi = pull_series()
    df = compute_trailing_appreciation(hpi, args.alpha)
    df = df[df.index >= args.start]

    df.to_csv("atlanta_case_shiller.csv", index_label="date")
    print(f"Wrote {len(df)} monthly obs to atlanta_case_shiller.csv")
    print(df[["hpi", "g"]].tail(12))

    print("\nMonthly g stats, full window:")
    print(df["g"].describe())

    print(
        "\n2020-2023 window (paper's out-of-sample validation period):"
    )
    val = df[(df.index >= "2020-01-01") & (df.index <= "2023-12-31")]
    print(val["g"].describe())
    val.to_csv("atlanta_case_shiller_2020_2023.csv", index_label="date")


if __name__ == "__main__":
    main()