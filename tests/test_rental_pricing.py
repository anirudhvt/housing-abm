import numpy as np
from collections import Counter
from housing_abm.equations.rental_pricing import (
    small_landlord_rent,
    institutional_rent,
    sample_lease_length,
)


def test_small_landlord_rent_sticky_when_not_repriced(never_trigger_rng):
    # reprice_prob check uses rng.random() >= reprice_prob to decide stickiness;
    # never_trigger_rng always returns 1.0, so it should always stay sticky
    # whenever previous_rent is given (1.0 >= any reprice_prob < 1.0)
    rent = small_landlord_rent(
        r_bar_tract=1500,
        f_bar_tract=20,
        alpha=0,
        beta=0,
        zeta=1.0,
        epsilon_std=0.05,
        reprice_prob=0.3,
        previous_rent=1400,
        rng=never_trigger_rng,
    )
    assert rent == 1400


def test_small_landlord_rent_reprices_with_no_previous_rent(zero_rng):
    rent = small_landlord_rent(
        r_bar_tract=1500,
        f_bar_tract=20,
        alpha=0,
        beta=0,
        zeta=1.0,
        epsilon_std=0.05,
        reprice_prob=0.3,
        previous_rent=None,
        rng=zero_rng,
    )
    assert rent > 0
    assert np.isclose(rent, 1500)  # alpha=beta=epsilon=0, f_bar irrelevant since beta=0


def test_institutional_rent_applies_premium_exactly():
    rent = institutional_rent(market_rate=1500, premium=1.04)
    assert np.isclose(rent, 1560)


def test_sample_lease_length_only_returns_valid_terms(rng):
    valid_terms = {1, 6, 12, 13, 24}
    draws = [sample_lease_length(rng) for _ in range(2000)]
    assert set(draws).issubset(valid_terms)


def test_sample_lease_length_roughly_matches_documented_proportions(rng):
    # 12-month should be the dominant term (~60%), month-to-month next (~32%)
    draws = [sample_lease_length(rng) for _ in range(5000)]
    counts = Counter(draws)
    share_12mo = counts[12] / len(draws)
    share_1mo = counts[1] / len(draws)
    assert 0.55 < share_12mo < 0.70
    assert 0.25 < share_1mo < 0.38
