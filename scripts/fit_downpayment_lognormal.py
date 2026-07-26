"""Fit Eq (17) downpayment log-normal params (m, s) from atlanta_hmda_2019.csv.
 
    d = max(d_minimum, HPI * F^-1((income_percentile - 0.3) / 0.7))
 
F is a log-normal CDF fit to d/HPI for households above the 30th income
percentile. Below  30th percentile, downpayment is assumed to be at
the floor (d_minimum)

owner-occupier segment here is FTB + repeat buyers combined 
 
Usage: python scripts/fit_downpayment_lognormal.py atlanta_hmda_2019.csv
"""

 
import sys
 
import numpy as np
import pandas as pd
from scipy import stats
 
INCOME_PERCENTILE_FLOOR = 0.3
 
 
def fit_segment(df: pd.DataFrame, label: str, median_price: float) -> dict:
    d = df.dropna(subset=["income", "downpayment_pct", "property_value"]).copy()
 


    d["downpayment_amount"] = d["property_value"] * d["downpayment_pct"] / 100
    d["d_norm"] = d["downpayment_amount"] / median_price

    #sort into people who will fit lognormal vs people who pay minimum
 
    above_floor = d[d["income_percentile"] >= INCOME_PERCENTILE_FLOOR]
    below_floor = d[d["income_percentile"] < INCOME_PERCENTILE_FLOOR]
 
    d_minimum = below_floor["downpayment_amount"].median()
    if pd.isna(d_minimum):
        d_minimum = 0.0
 
    fit_vals = above_floor.loc[above_floor["d_norm"] > 0, "d_norm"]
    if len(fit_vals) < 20: #too few values, print warning
        print(f"WARNING: only {len(fit_vals)} usable obs for {label}, "
              "fit will be noisy", file=sys.stderr)
 
    # scipy's lognorm is parameterized as shape=s, scale=exp(m); floc=0
    shape, _, scale = stats.lognorm.fit(fit_vals, floc=0)
    m = np.log(scale)
    s = shape
 
    return {
        "segment": label,
        "m": m,
        "s": s,
        "d_minimum": d_minimum,
        "n_above_floor": len(above_floor),
        "n_below_floor": len(below_floor),
    }
 
 
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "atlanta_hmda_2019.csv"
    df = pd.read_csv(path)
 
    if "downpayment_pct" not in df.columns:
        df["downpayment_pct"] = 100 - df["ltv_derived"]
 
    df = df.dropna(subset=["income"]).copy()
    # income percentile across the full buyer population (not per-segment) -
    df["income_percentile"] = df["income"].rank(pct=True)

    #use real 2019 case-shiller HPI average 
    #falls back to median property_value when pull hasn't been run 

    try:
        cs = pd.read_csv("atlanta_case_shiller.csv", parse_dates=["date"])
        hpi_2019 = cs[cs["date"].dt.year == 2019]["hpi"].mean()
    except FileNotFoundError:
        hpi_2019 = None
 
    if hpi_2019 is None or pd.isna(hpi_2019):
        print("atlanta_case_shiller.csv not found or has no 2019 rows -- "
              "falling back to median property_value.", file=sys.stderr)
        median_price = df["property_value"].median()
    else:
        median_price = hpi_2019

 
 
    results = []
    for occ, group in df.groupby("is_investor_occupied"):
        label = "investor" if occ else "owner_occupier"
        results.append(fit_segment(group, label, median_price))
 
    out = pd.DataFrame(results)
    out.to_csv("downpayment_lognormal_params.csv", index=False)
    print(out.to_string(index=False))
    print(
        f"\nmedian_price (HPI proxy) used for normalization: {median_price:,.0f}"
    )
    print(
        "\nDrop m/s/d_minimum into Eq (17) params in baseline_params.yaml. "
        "Re-fit once you have real monthly/tract HPI instead of the "
        "single-snapshot median-price proxy."
    )
 
 
if __name__ == "__main__":
    main()