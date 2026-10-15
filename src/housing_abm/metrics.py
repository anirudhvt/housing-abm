"""Outcome metrics: recorded every month, then summarised as averages over the
measurement window rather than read from the final month."""

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

# Share metrics pooled over the window (sum numerator / sum denominator), since
# monthly ratios of a handful of sales are noisy and often undefined.
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

    # sale-market liquidity (real US markets run 3-6 months of inventory)
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
        # smoothed tract price, used for the price_cagr trend
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
    """Annualised price trend from the OLS slope of log(price) on month.

    Use this, not the mean of EQ 4's g, to measure the trend: averaging growth
    ratios over a volatile series biases upward.
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
    """AR(1) effective sample size: n * (1 - rho) / (1 + rho)."""
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
