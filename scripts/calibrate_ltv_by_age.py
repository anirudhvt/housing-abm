"""Calibrate an age-conditioned LTV/LTI curve from real Atlanta HMDA data --
the Atlanta equivalent of the reference Java model's decideLTV(), which the
model currently has no analogue for at all: mortgage_terms.yaml uses flat
regulatory LTV/DTI caps (FHA 96.5%, conventional 80%) with no age term.

HMDA carries no first-time-vs-repeat-buyer flag (see
calibrate_downpayment_eq17.py and docs/methodology.md Section 10a/10b for
where that split is handled instead, via Fannie Mae's genuine FTB flag).
This fits a single pooled regression across all owner-occupied purchase
loans instead of splitting by buyer type.

Usage:
    python scripts/calibrate_ltv_by_age.py path/to/atlanta_hmda_2019.csv
"""

import sys

import numpy as np
import pandas as pd

# HMDA's bucketed age field; midpoints used as the regression's continuous
# age predictor. "8888" means not applicable/unknown and is dropped.
AGE_MIDPOINTS = {
    "<25": 22.0,
    "25-34": 29.5,
    "35-44": 39.5,
    "45-54": 49.5,
    "55-64": 59.5,
    "65-74": 69.5,
    ">74": 80.0,
}


def fit_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Plain OLS via least squares; returns (coefficients incl. intercept, r_squared)."""
    design = np.column_stack([np.ones(len(X)), X])
    coefs, residuals, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coefs, r_squared


def main():
    if len(sys.argv) != 2:
        print("usage: calibrate_ltv_by_age.py <atlanta_hmda_2019.csv>", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sys.argv[1], low_memory=False)
    if "applicant_age" not in df.columns:
        print("applicant_age column not found -- re-pull with the updated FIELDS list", file=sys.stderr)
        sys.exit(1)

    owner = df[~df["is_investor_occupied"]].copy()
    owner = owner[owner["applicant_age"].isin(AGE_MIDPOINTS)]
    owner["age_midpoint"] = owner["applicant_age"].map(AGE_MIDPOINTS)
    owner = owner.dropna(subset=["ltv_derived", "lti_derived", "income", "age_midpoint"])
    owner = owner[owner["income"] > 0]

    print(f"{len(owner)} owner-occupied purchase loans with known age, after cleaning\n")

    print("=== Descriptive: LTV and LTI by age bin ===")
    desc = owner.groupby("applicant_age", observed=True).agg(
        n=("ltv_derived", "size"),
        mean_ltv=("ltv_derived", "mean"),
        mean_lti=("lti_derived", "mean"),
        mean_income=("income", "mean"),
    )
    # order by the natural age progression, not alphabetically
    order = list(AGE_MIDPOINTS.keys())
    print(desc.reindex(order).to_string())

    log_income = np.log(owner["income"].to_numpy())
    age = owner["age_midpoint"].to_numpy()
    X = np.column_stack([log_income, age])

    print("\n=== Regression: LTV ~ log(income) + age (Atlanta equivalent of decideLTV) ===")
    coefs, r2 = fit_ols(owner["ltv_derived"].to_numpy(), X)
    print(f"intercept={coefs[0]:.4f}  log_income_coef={coefs[1]:.4f}  age_coef={coefs[2]:.4f}  R2={r2:.4f}")

    print("\n=== Regression: LTI ~ log(income) + age ===")
    coefs, r2 = fit_ols(owner["lti_derived"].to_numpy(), X)
    print(f"intercept={coefs[0]:.4f}  log_income_coef={coefs[1]:.4f}  age_coef={coefs[2]:.4f}  R2={r2:.4f}")


if __name__ == "__main__":
    main()
