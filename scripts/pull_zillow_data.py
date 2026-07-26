"""Pull Zillow ZHVI for Atlanta metro and compute EQ 4 trailing appreciatoin estimate

Metro Level"""
import argparse
import pandas as pd

ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv"
)

METRO_NAME = "Atlanta, GA"

def pull_series() -> pd.Series:
    wide = pd.read_csv(ZHVI_URL)
    row = wide[wide["RegionName"] == METRO_NAME]
    # everything after the metadata columns is a monthly date column
    date_cols = [c for c in wide.columns if c[:2] in ("19", "20")]
    series = row[date_cols].iloc[0]
    series.index = pd.to_datetime(series.index)
    return series.astype(float).sort_index()

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
    ap.add_argument(
        "--alpha", type=float, default=1.0, help="Eq (4) alpha, placeholder until calibrated"
    )
    ap.add_argument(
        "--start", default="2015-01-01", help="trim start date for the output window"
    )
    args = ap.parse_args()
 
    hpi = pull_series()
    df = compute_trailing_appreciation(hpi, args.alpha)
    df = df[df.index >= args.start]
 
    df.to_csv("atlanta_zillow_zori.csv", index_label="date")
    print(f"Wrote {len(df)} monthly obs to atlanta_zillow_zori.csv")
    print(df[["hpi", "g"]].tail(12))
 
    print("\nMonthly g stats, full window:")
    print(df["g"].describe())
 
    print("\n2020-2023 window (for comparison against  Case-Shiller pull):")
    val = df[(df.index >= "2020-01-01") & (df.index <= "2023-12-31")]
    print(val["g"].describe())
    val.to_csv("atlanta_zillow_zori_2020_2023.csv", index_label="date")
 
 
if __name__ == "__main__":
    main()