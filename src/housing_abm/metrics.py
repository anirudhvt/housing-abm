"""Outcome metrics for the policy experiments.

The original harness read every outcome from a single terminal snapshot: one
month's homeownership rate, one month's rental vacancy rate. With a few hundred
agents that snapshot is close to a single binomial draw, so its cross-seed
standard deviation is dominated by sampling noise that has nothing to do with
the policy under test. Averaging the same quantity over the measurement window
uses all the information the run already produced, and cuts the standard error
by roughly the square root of the number of effectively independent months.

Everything here reads from a per-month record collected during the run, so a
single simulation yields both the window mean and the within-run series needed
for stationarity diagnostics.
"""

from __future__ import annotations

import numpy as np

from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord

# outcomes reported for every policy comparison
PRIMARY_METRICS = (
    "homeownership_rate",
    "rental_vacancy_rate",
    "annual_appreciation_g",
)

# additional series used for validation and for reading the mechanism
SECONDARY_METRICS = (
    "mean_ltv_owner_occupier",
    "mean_lti_owner_occupier",
    "institutional_share_of_rentals",
    "investor_share_of_stock",
    "median_price",
    "median_rent",
    "price_per_quality",
    "n_households",
    "months_of_inventory",
    "median_months_on_market",
)

# Metrics that are ratios of monthly counts.
#
# A handful of houses change hands each month, so the monthly first-time-buyer
# share is a ratio of very small integers -- 0/2, 1/3, undefined -- and the
# mean of those ratios is both extremely noisy and biased by the months with
# few transactions. Aggregating numerator and denominator over the window
# first gives the quantity actually of interest: the share of transactions in
# the window won by first-time buyers.
RATIO_METRICS = {
    "ftb_purchase_share": ("n_purchases_ftb", "n_purchases_total"),
    "repeat_buyer_purchase_share": ("n_purchases_repeat", "n_purchases_total"),
    "investor_purchase_share": ("n_purchases_investor", "n_purchases_total"),
    "institutional_purchase_share": ("n_purchases_institutional", "n_purchases_total"),
    "small_landlord_purchase_share": ("n_purchases_small_landlord", "n_purchases_total"),
}

COUNT_METRICS = (
    "n_purchases_ftb",
    "n_purchases_repeat",
    "n_purchases_investor",
    "n_purchases_institutional",
    "n_purchases_small_landlord",
    "n_purchases_total",
)

ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS + COUNT_METRICS


def observe(model) -> dict:
    """Snapshot every tracked outcome for the current month."""
    owner_occupiers = [
        a
        for a in model.agents
        if isinstance(a, (RepeatBuyer, FirstTimeBuyer))
        and a.house is not None
        and a.house.mortgage_principal > 0
    ]
    ltvs, ltis = [], []
    for a in owner_occupiers:
        price = a.house.price
        if price and price > 0:
            ltvs.append(a.house.mortgage_principal / price)
        annual_income = a.income * 12
        if annual_income > 0:
            ltis.append(a.house.mortgage_principal / annual_income)

    rental_stock = len(model.rental_units)
    institutional_units = sum(
        len(a.properties) for a in model.agents if isinstance(a, InstitutionalInvestor)
    )
    investor_units = institutional_units + sum(
        len(a.properties) for a in model.agents if isinstance(a, SmallLandlord)
    )
    stock = model.total_housing_stock()

    prices = [u.price for u in model.housing_units if u.price]
    rents = [u.rent for u in model.rental_units if u.rent]

    # sale-market liquidity. A healthy US market runs 3-6 months of inventory;
    # this model clears far more slowly, which is a known limitation of the
    # EQ 8 price-reduction rule at this population size, so it is measured and
    # reported rather than left implicit.
    listings = model.active_for_sale()
    sales = sum(model.purchases_this_month.values())
    months_on_market = [u.days_on_market for u in listings]

    return {
        "homeownership_rate": model._homeownership_rate(),
        "rental_vacancy_rate": model._rental_vacancy_rate(),
        "annual_appreciation_g": model._appreciation_g(),
        "mean_ltv_owner_occupier": float(np.mean(ltvs)) if ltvs else None,
        "mean_lti_owner_occupier": float(np.mean(ltis)) if ltis else None,
        "institutional_share_of_rentals": (
            institutional_units / rental_stock if rental_stock else None
        ),
        "investor_share_of_stock": investor_units / stock if stock else None,
        "n_purchases_ftb": float(model.purchases_this_month["ftb"]),
        "n_purchases_repeat": float(model.purchases_this_month["repeat"]),
        "n_purchases_institutional": float(model.purchases_this_month["institutional"]),
        "n_purchases_small_landlord": float(
            model.purchases_this_month["small_landlord"]
        ),
        "n_purchases_investor": float(
            model.purchases_this_month["small_landlord"]
            + model.purchases_this_month["institutional"]
        ),
        "n_purchases_total": float(sum(model.purchases_this_month.values())),
        "median_price": float(np.median(prices)) if prices else None,
        "median_rent": float(np.median(rents)) if rents else None,
        # the tract's own smoothed price index -- the cleanest series to read
        # a price *trend* off, since it isn't re-sampled from whichever units
        # happen to be listed this month (see price_cagr in window_means)
        "price_per_quality": float(model.tracts["tract_001"].price_per_quality),
        "n_households": float(model.n_household_agents()),
        "months_of_inventory": (len(listings) / sales) if sales else None,
        "median_months_on_market": (
            float(np.median(months_on_market)) if months_on_market else None
        ),
    }


def run_window(model, months: int) -> dict[str, list]:
    """Step the model `months` times, recording every outcome each month."""
    series: dict[str, list] = {name: [] for name in ALL_METRICS}
    for _ in range(months):
        model.step()
        row = observe(model)
        for name in ALL_METRICS:
            series[name].append(row[name])
    return series


def price_trend_cagr(prices: list) -> float | None:
    """Annualised price growth, as the OLS slope of log(price) on month index.

    `annual_appreciation_g` is the *mean over months* of EQ4's ratio-based
    trailing growth, which is the right quantity for agent behaviour (it is
    what households actually perceive and act on) but a biased estimator of
    the underlying trend. On a volatile mean-reverting series, growth ratios
    are asymmetric: a rise from A to B reads as +(B-A)/A, while the matching
    fall back to A reads as -(B-A)/B, which is smaller in magnitude. Averaged
    over up-and-down cycles that yields a positive number even at zero net
    trend, and the bias grows with volatility.

    Measured at 300 households over a 180-month window, the gap is large and
    one-directional: the mean of monthly EQ4 g came to +3.75%/yr against an
    actual price CAGR of +0.99%/yr, overstating by +2.76%/yr in every one of
    8 seeds (in two of them the price level *fell* over the window while the
    metric still reported ~+3%/yr). So a trend target has to be validated
    against a trend estimator, not against the behavioural signal.

    Uses the log-OLS slope rather than an endpoint-to-endpoint CAGR, since it
    uses the whole window instead of two noisy endpoint observations.
    """
    clean = [p for p in prices if p is not None and not np.isnan(p) and p > 0]
    if len(clean) < 24:
        return None
    y = np.log(np.asarray(clean, dtype=float))
    x = np.arange(len(y), dtype=float)
    monthly_slope = float(np.polyfit(x, y, 1)[0])
    return float(np.exp(monthly_slope * 12) - 1)


def window_means(series: dict[str, list]) -> dict[str, float | None]:
    """Window summary: means for levels, pooled ratios for count ratios, and
    a log-OLS trend for price growth (see price_trend_cagr)."""
    out: dict[str, float | None] = {}
    for name, values in series.items():
        clean = [v for v in values if v is not None and not np.isnan(v)]
        out[name] = float(np.mean(clean)) if clean else None

    for name, (num_key, den_key) in RATIO_METRICS.items():
        numerator = float(np.nansum(np.asarray(series[num_key], dtype=float)))
        denominator = float(np.nansum(np.asarray(series[den_key], dtype=float)))
        out[name] = numerator / denominator if denominator > 0 else None

    out["price_cagr"] = price_trend_cagr(series.get("price_per_quality", []))
    return out


def ratio_series(series: dict[str, list], name: str) -> list:
    """Expanding-window ratio, for diagnostics that need a time series."""
    num_key, den_key = RATIO_METRICS[name]
    num = np.nan_to_num(np.asarray(series[num_key], dtype=float))
    den = np.nan_to_num(np.asarray(series[den_key], dtype=float))
    cum_num, cum_den = np.cumsum(num), np.cumsum(den)
    return [
        float(n / d) if d > 0 else None for n, d in zip(cum_num, cum_den)
    ]


def effective_sample_size(values) -> float:
    """Months of independent information in an autocorrelated series.

    n_eff = n * (1 - rho) / (1 + rho) for an AR(1) with lag-1 autocorrelation
    rho. Used to report honestly how much a window average actually buys over
    a single snapshot, rather than claiming the full month count.
    """
    x = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    n = x.size
    if n < 3:
        return float(n)
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return float(n)
    rho = float(x[:-1] @ x[1:]) / denom
    rho = min(max(rho, -0.99), 0.99)
    return float(n * (1.0 - rho) / (1.0 + rho))
