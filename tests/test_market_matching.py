import numpy as np
from housing_abm.equations.market_matching import (
    expected_gross_rental_yield,
    sample_bid_up_multiplier,
    max_rounds,
)

#expected gross rental yield 

def test_expected_gross_rental_yield_zero_price_returns_zero():
    y = expected_gross_rental_yield(monthly_rent=1500, price=0, avg_days_on_market=20)
    assert y == 0.0


def test_expected_gross_rental_yield_decreases_with_more_days_on_market():
    # longer expected vacancy -> lower expected occupancy -> lower expected yield
    quick = expected_gross_rental_yield(
        monthly_rent=1500, price=200_000, avg_days_on_market=5
    )
    slow = expected_gross_rental_yield(
        monthly_rent=1500, price=200_000, avg_days_on_market=200
    )
    assert quick > slow

#sample bid up multiplier


def test_bid_up_multiplier_no_bids_returns_one():
    rng = np.random.default_rng(0)
    assert sample_bid_up_multiplier(rng, n_bids=0) == 1.0
    assert sample_bid_up_multiplier(rng, n_bids=1) == 1.0


def test_bid_up_multiplier_never_overflows_with_huge_bid_counts():
    rng = np.random.default_rng(0)
    for n_bids in (2, 10, 50, 200, 1000, 10_000):
        for _ in range(20):
            multiplier = sample_bid_up_multiplier(rng, n_bids=n_bids)
            assert np.isfinite(multiplier)
            assert multiplier >= 1.0


def test_bid_up_multiplier_respects_max_multiplier_ceiling():
    rng = np.random.default_rng(0)
    for _ in range(200):
        multiplier = sample_bid_up_multiplier(rng, n_bids=5000, max_multiplier=2.0)
        assert multiplier <= 2.0 + 1e-9


def test_bid_up_multiplier_more_bidders_tends_to_bid_up_more(rng):
    # average multiplier across many draws should be non-decreasing in n_bids
    def avg_multiplier(n_bids, n_draws=300):
        return np.mean([sample_bid_up_multiplier(rng, n_bids) for _ in range(n_draws)])

    low = avg_multiplier(2)
    high = avg_multiplier(20)
    assert high >= low


#max_rounds

def test_max_rounds_respects_floor_at_small_population():
    # the floor should always dominate
    rounds = max_rounds(n_bids=10, n_offers=10, n_households=300, round_floor=5)
    assert rounds == 5


def test_max_rounds_never_below_floor_even_with_many_orders():
    rounds = max_rounds(
        n_bids=100_000, n_offers=100_000, n_households=300, round_floor=5
    )
    assert rounds >= 5
