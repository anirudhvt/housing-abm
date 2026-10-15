"""Calibrate age, income-by-age, EQ 1 wealth-by-income and small-landlord
propensity from the SCF Summary Extract (implicate 1 only).
See docs/methodology.md Section 11.

Usage:
    python scripts/calibrate_scf.py path/to/SCFP2022.csv
"""

import sys

import numpy as np
import pandas as pd

AGE_BINS = [18, 25, 35, 45, 55, 65, 75, 96]


def fit_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(len(X)), X])
    coefs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    resid_std = float(np.std(y - fitted, ddof=2))
    return coefs, r_squared, resid_std


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cum_weights = np.cumsum(weights) - 0.5 * weights
    cum_weights /= weights.sum()
    return np.interp(q, cum_weights, values)


def main():
    if len(sys.argv) != 2:
        print("usage: calibrate_scf.py <SCFP2022.csv>", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df = df[df["Y1"] % 10 == 1].copy()  # implicate 1 only
    print(f"{len(df)} households (implicate 1), weighted N = {df['WGT'].sum():,.0f}\n")

    # --- 1. Age distribution ---------------------------------------------
    print("=== Age distribution (weighted) ===")
    df["age_bin"] = pd.cut(df["AGE"], AGE_BINS, right=False)
    age_dist = df.groupby("age_bin", observed=True)["WGT"].sum()
    age_dist = age_dist / age_dist.sum()
    print(age_dist.to_string())

    # --- 2. Income given age -----------------------------------------------
    print("\n=== Income given age (weighted mean/median, $) ===")
    for b, group in df.groupby("age_bin", observed=True):
        if group["WGT"].sum() == 0:
            continue
        mean_inc = np.average(group["INCOME"], weights=group["WGT"])
        med_inc = weighted_quantile(group["INCOME"].to_numpy(), group["WGT"].to_numpy(), 0.5)
        print(f"  {b}: mean=${mean_inc:,.0f}  median=${med_inc:,.0f}  n={len(group)}")

    # --- 3. Wealth given income (EQ1: ln(bank_balance) = alpha + beta*ln(income)) ---
    # FIN (liquid financial assets) stands in for bank_balance
    print("\n=== EQ1 fit: ln(FIN) ~ alpha + beta*ln(INCOME) ===")
    print("(SCF has no first-time-vs-repeat-buyer flag, same limitation as HMDA --")
    print(" owner-occupiers reported as one combined group, not split like config's")
    print(" first_time_buyer/repeat_buyer wealth_eq1 blocks)\n")

    groups = {
        "renter (HOUSECL==2)": df[df["HOUSECL"] == 2],
        "owner, no other RE (proxy: FTB+repeat combined)": df[(df["HOUSECL"] == 1) & (df["HORESRE"] == 0)],
        "small_landlord (HORESRE==1)": df[df["HORESRE"] == 1],
    }
    for label, g in groups.items():
        g = g[(g["FIN"] > 0) & (g["INCOME"] > 0)]
        if len(g) < 10:
            print(f"{label}: n={len(g)}, too few observations, skipped")
            continue
        y = np.log(g["FIN"].to_numpy())
        X = np.log(g["INCOME"].to_numpy())
        coefs, r2, resid_std = fit_ols(y, X)
        print(
            f"{label} (n={len(g)}): alpha={coefs[0]:.4f}  beta={coefs[1]:.4f}  "
            f"epsilon_std={resid_std:.4f}  R2={r2:.4f}"
        )

    # --- 4. Investor propensity by income percentile -----------------------
    # HORESRE==1 (owns other residential real estate); households only, so
    # this covers small landlords, not institutional investors
    print("\n=== Investor (small-landlord) propensity by income decile ===")
    df["income_decile"] = pd.qcut(df["INCOME"].rank(method="first"), 10, labels=False)
    for decile, g in df.groupby("income_decile", observed=True):
        propensity = np.average(g["HORESRE"], weights=g["WGT"])
        print(f"  decile {decile + 1}: P(owns other residential RE) = {propensity:.4f}  (n={len(g)})")


if __name__ == "__main__":
    main()
