""" "Multi round double auction market clearing, shared by both market
Per round:
    Phase 1: every remaining bid matched to best offer it can afford -
    highest quality/highest expected gross rental yield
    Phase 2: offer matched to one bid sells at asking price
    offer matched to multiple bids gets bid up, people who can still afford are randomly chosen

Unmatched bids return to pool for another round, up to max_rounds cap"""

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
    """EQ21: multiply asking price by 1.0075^k, k drawn from a geometric
    distribution
    b = n_bids.
    Approximates bids arriving on random days within the month, each new bid
    within a 7-day window of the previous one bidding the price up 0.75%.
    max_multiplier caps how far a single round can bid a price up
    """
    if n_bids <= 1:
        return 1.0
    p = np.exp(-arrival_window_days * n_bids / month_days)
    p = min(max(p, 1e-12), 1.0)
    k = rng.geometric(p)
    max_k = np.log(max_multiplier) / np.log(1.0 + bid_up_pct)
    k = min(k, max_k)
    return (1.0 + bid_up_pct) ** (k - 1)


def max_rounds(n_bids: int, n_offers: int, n_households: int, round_floor: int) -> int:
    """max rounds formula, floored with placeholder param"""
    n_orders = n_bids + n_offers
    paper_formula = min(n_households / 1000.0, 1.0 + n_orders / 5_000_000.0)
    return max(round_floor, round(paper_formula))  # use paper's formula if big enough

def pick_preferred(rng: np.random.Generator, candidates: list, key_fn) -> object:
    """Pick highest key candidate, breaking ties randomly"""
    best_key = max(key_fn(c) for c in candidates)
    tied = [ c for c in candidates if key_fn(c) == best_key]
    if len(tied) == 1:
        return tied[0]
    return tied[rng.integers(len(tied))]
