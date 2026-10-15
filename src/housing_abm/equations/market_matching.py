"""Helpers for the multi-round double auction used by both markets. Each round,
bids claim their preferred affordable offer; contested offers are bid up and
go to a random bidder who can still afford them."""

import numpy as np


def expected_gross_rental_yield(
    monthly_rent: float, price: float, avg_days_on_market: float
) -> float:
    """EQ19/20: E(y_q) = 12 * r_q * E(o_q) / p, E(o_q) = 547 / (547 + D_q)
    monthly_rent/price: r_q, p for a house of this quality
    avg_days_on_market: D_q, exponential moving average days-to-let for this quality"""
    if price <= 0:
        return 0.0
    expected_occupancy = 547.0 / (547.0 + max(avg_days_on_market, 0.0))
    return (12.0 * monthly_rent * expected_occupancy) / price


def sample_bid_up_multiplier(
    rng: np.random.Generator,
    n_bids: int,
    bid_up_pct: float = 0.0075,
    arrival_window_days: float = 7.0,
    month_days: float = 30.0,
    max_multiplier: float = 3.0,
) -> float:
    """EQ 21: price multiplier 1.0075^(k-1), with k geometric in n_bids,
    capped at max_multiplier."""
    if n_bids <= 1:
        return 1.0
    p = np.exp(-arrival_window_days * n_bids / month_days)
    p = min(max(p, 1e-12), 1.0)
    k = rng.geometric(p)
    max_k = np.log(max_multiplier) / np.log(1.0 + bid_up_pct)
    k = min(k, max_k)
    return (1.0 + bid_up_pct) ** (k - 1)


def max_rounds(n_bids: int, n_offers: int, n_households: int, round_floor: int) -> int:
    """The paper's round count, floored at round_floor."""
    n_orders = n_bids + n_offers
    paper_formula = min(n_households / 1000.0, 1.0 + n_orders / 5_000_000.0)
    return max(round_floor, round(paper_formula))
