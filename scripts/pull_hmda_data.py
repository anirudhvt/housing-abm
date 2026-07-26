"""Pull Atlanta HMDA loan-level data via public API 
Feeds into: 
EQ14: mortgage constraints
EQ 17: FTB vs Repeat Buyer downpayment """

import argparse
import io
import sys
 
import pandas as pd
import requests


API_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

#areas we are focused on 
ATLANTA_COUNTIES = ["067", "089", "121", "135", "063", "151", "223"]
GA_STATE_FIPS = "13"
COUNTY_FIPS = [GA_STATE_FIPS + c for c in ATLANTA_COUNTIES]

#data we want

FIELDS = [
    "activity_year", "lei", "county_code", "census_tract",
    "loan_type", "loan_purpose", "action_taken", "occupancy_type",
    "loan_amount", "loan_to_value_ratio", "property_value",
    "income", "debt_to_income_ratio", "combined_loan_to_value_ratio",
    "total_units", "purchaser_type", "applicant_credit_score_type",
    "interest_rate",
]


def pull_year(year: int) -> pd.DataFrame:
    params = {
        "years": str(year),
        "counties": ",".join(COUNTY_FIPS),
        "loan_purposes": "1",       # home purchase
        "actions_taken": "1",       # originated
    }
    resp = requests.get(API_URL, params=params, timeout=120)
    if resp.status_code != 200:
        print(resp.text)
    resp.raise_for_status()
    
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    keep = [c for c in FIELDS if c in df.columns]
    return df[keep]

def clean_hmda(df: pd.DataFrame) -> pd.DataFrame:
    #clean the data
    df = df.copy()

    numeric_cols = [
        "loan_amount",
        "property_value",
        "income",
        "loan_to_value_ratio",
        "interest_rate",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove obviously invalid observations
    if "loan_to_value_ratio" in df.columns:
        df.loc[
            (df["loan_to_value_ratio"] < 20)
            | (df["loan_to_value_ratio"] > 150),
            "loan_to_value_ratio"
        ] = pd.NA

    if "property_value" in df.columns:
        df.loc[
            (df["property_value"] < 25000)
            | (df["property_value"] > 1e8),
            "property_value"
        ] = pd.NA

    if "loan_amount" in df.columns:
        df.loc[
            (df["loan_amount"] < 5000)
            | (df["loan_amount"] > 1e8),
            "loan_amount"
        ] = pd.NA

    if "income" in df.columns:
        # HMDA income is reported in thousands
        df.loc[
            (df["income"] < 5)
            | (df["income"] > 10000),
            "income"
        ] = pd.NA

    return df
 
def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["loan_amount", "property_value", "income", "loan_to_value_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # LTV: prefer reported field, fall back to loan_amount / property_value
    df["ltv_derived"] = df["loan_to_value_ratio"]
    fallback = df["loan_amount"] / df["property_value"] * 100
    df["ltv_derived"] = df["ltv_derived"].fillna(fallback)
 
    # LTI: income reported in thousands
    df["lti_derived"] = df["loan_amount"] / (df["income"] * 1000)
 
    df["is_first_lien_purchase"] = df["loan_purpose"] == 1
    df["is_investor_occupied"] = df["occupancy_type"] == 3
    return df
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--out", default="atlanta_hmda_{year}.csv")
    args = ap.parse_args()
 
    try:
        df = pull_year(args.year)
    except requests.RequestException as e:
        print(f"HMDA API request failed: {e}", file=sys.stderr)
        print(
            "If the API schema/params have changed, check "
            "https://ffiec.cfpb.gov/data-browser/ and adjust FIELDS/params.",
            file=sys.stderr,
        )
        sys.exit(1)
    df = clean_hmda(df)
 
    df = add_derived_fields(df)


    #remove impossible loan terms    
    df = df[
    (df["ltv_derived"] >= 40)
    &
    (df["ltv_derived"] <= 110)
    ]
    

    #grab the quantile distrbution of the data

    for occ, group in df.groupby("is_investor_occupied"):
        print("\n", occ)

        print(
            group["ltv_derived"].quantile(
                [0.1,0.25,0.5,0.75,0.9]
            )
        )
    df["downpayment_pct"] = 100 - df["ltv_derived"]
    for occ, group in df.groupby("is_investor_occupied"):
        print(
            group["downpayment_pct"].quantile(
                [0.1,0.25,0.5,0.75,0.9]
            )
        )


    out_path = args.out.format(year=args.year)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} loans to {out_path}")
    print(
        df.groupby("is_investor_occupied")[["ltv_derived", "lti_derived"]]
        .describe()
    )
 
 
if __name__ == "__main__":
    main()