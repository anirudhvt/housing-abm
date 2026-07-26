"""Load external (Zillow ZHVI/ZORI, or Case-Shiller) monthly series to
replace internally-derived appreciation/rent-growth signals.
"""

import csv
 
 
def load_g_series(csv_path: str) -> list[float]:
    """Load the 'g' column from a csv file."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [float(row["g"]) for row in reader]
 
 
def load_monthly_growth_series(csv_path: str) -> list[float]:
    """Derive month-over-month growth rates from the raw index ('hpi')
    column of a csv."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        levels = [float(row["hpi"]) for row in reader]
    growth = []
    for prev, cur in zip(levels, levels[1:]):
        growth.append((cur / prev) - 1.0 if prev else 0.0)
    return growth