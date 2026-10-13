"""Calibrate EQ17 (owner-occupier down payment) against Fannie Mae's
Single-Family Loan Performance Data, filtered to the Atlanta MSA.

Unlike scripts/calibrate_downpayment_eq17.py (which proxies first-time vs.
repeat buyer with HMDA's loan_type field, since HMDA carries no direct
first-time-buyer flag), this dataset has a genuine First Time Home Buyer
Indicator (Y/N) reported at origination -- see docs/methodology.md Section
10a for why that proxy mattered enough to verify against the real thing.

Schema: pipe-delimited, no header, one row per loan per reporting month.
Field positions below were verified against Fannie Mae's own "Single-Family
Loan Performance Dataset and Credit Risk Transfer - Glossary and File
Layout" (current as of the 5/26/2026 revision) and cross-checked field by
field against a real sample row -- see the session's diagnostic output.
Position N in that document is FIELDS[N-1] here (0-indexed).

Usage:
    python scripts/calibrate_downpayment_eq17_fnma.py path/to/quarter_file.csv [...]

Accepts one or more quarterly files (or the sample file, for a dry run).
"""

import sys

import pandas as pd

sys.path.insert(0, "src")
from housing_abm.equations.mortgage import estimate_floor_share_and_fit  # noqa: E402

ATLANTA_MSA_CODE = "12060"

# 0-indexed position = glossary Field Position - 1. Trailing fields (109+)
# are omitted since nothing here needs them and some file vintages don't
# carry that many columns.
FIELDS = [
    "reference_pool_id", "loan_id", "reporting_period", "channel",       # 1-4
    "seller_name", "servicer_name", "master_servicer",                    # 5-7
    "original_interest_rate", "current_interest_rate", "original_upb",    # 8-10
    "upb_at_issuance", "current_actual_upb", "original_loan_term",        # 11-13
    "origination_date", "first_payment_date", "loan_age",                 # 14-16
    "remaining_months_legal_maturity", "remaining_months_maturity",       # 17-18
    "maturity_date", "original_ltv", "original_cltv", "num_borrowers",    # 19-22
    "dti", "borrower_credit_score", "coborrower_credit_score",            # 23-25
    "first_time_homebuyer", "loan_purpose", "property_type",              # 26-28
    "num_units", "occupancy_status", "property_state", "msa",             # 29-32
    "zip_short",                                                          # 33
]


def load(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep="|", header=None, dtype=str, low_memory=False)
        df = df.iloc[:, : len(FIELDS)]
        df.columns = FIELDS
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def one_row_per_loan(df: pd.DataFrame) -> pd.DataFrame:
    """Origination attributes don't change month to month; keep the
    earliest reporting period per loan so a long-lived loan isn't
    overweighted relative to one with fewer performance-history rows."""
    df = df.sort_values("reporting_period")
    return df.drop_duplicates(subset="loan_id", keep="first")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: calibrate_downpayment_eq17_fnma.py <file.csv> [...]", file=sys.stderr)
        sys.exit(1)

    df = load(paths)
    df = one_row_per_loan(df)
    print(f"{len(df)} unique loans across {len(paths)} file(s)")

    atlanta = df[df["msa"] == ATLANTA_MSA_CODE]
    print(f"{len(atlanta)} loans in the Atlanta MSA ({ATLANTA_MSA_CODE})")

    # purchase money mortgages on an owner-occupied principal residence --
    # matches downpayment_eq17's own context (buying a home to live in)
    purchase = atlanta[
        (atlanta["loan_purpose"] == "P") & (atlanta["occupancy_status"] == "P")
    ].copy()
    print(f"{len(purchase)} of those are owner-occupied purchase loans")

    purchase["original_ltv"] = pd.to_numeric(purchase["original_ltv"], errors="coerce")
    purchase["downpayment_frac"] = 1.0 - purchase["original_ltv"] / 100.0
    purchase = purchase.dropna(subset=["downpayment_frac"])

    ftb = purchase.loc[purchase["first_time_homebuyer"] == "Y", "downpayment_frac"].to_numpy()
    repeat = purchase.loc[purchase["first_time_homebuyer"] == "N", "downpayment_frac"].to_numpy()

    print(f"\nFirst-time buyers (n={len(ftb)}), floor_band=0.035 (FHA-equivalent floor):")
    if len(ftb):
        print(estimate_floor_share_and_fit(ftb, floor_band=0.035))
    else:
        print("  no observations")

    print(f"\nRepeat buyers (n={len(repeat)}), floor_band=0.20 (conventional floor):")
    if len(repeat):
        print(estimate_floor_share_and_fit(repeat, floor_band=0.20))
    else:
        print("  no observations")


if __name__ == "__main__":
    main()
