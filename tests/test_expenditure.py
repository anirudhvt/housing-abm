import numpy as np
import pytest
from housing_abm.equations.expenditure import (
    desired_expenditure,
    price_appreciation_expectation,
)

#desired expenditure (EQ 3)

def test_desired_expenditure_matches_formula_with_zero_noise(zero_rng):
    income, g, alpha, beta = 60_000, 0.02, 4.5, 0.08
    p = desired_expenditure(income, g, alpha, beta, epsilon_std=0.5, rng=zero_rng)
    expected = alpha * income / (1 - beta * g)
    assert np.isclose(p, expected)


def test_desired_expenditure_capped_by_mortgage_cap(zero_rng):
    # a huge income would otherwise blow past a small mortgage cap
    p = desired_expenditure(
        income_or_capital=10_000_000,
        g=0.02,
        alpha=4.5,
        beta=0.08,
        epsilon_std=0.5,
        rng=zero_rng,
        mortgage_cap=250_000,
    )
    assert p == 250_000


def test_desired_expenditure_uncapped_when_below_mortgage_cap(zero_rng):
    p = desired_expenditure(
        income_or_capital=1000,
        g=0.02,
        alpha=4.5,
        beta=0.08,
        epsilon_std=0.5,
        rng=zero_rng,
        mortgage_cap=10_000_000,
    )
    assert p < 10_000_000


#price appreciatoin 

def test_price_appreciation_raises_on_insufficient_history():
    with pytest.raises(ValueError):
        price_appreciation_expectation([100.0] * 10, alpha=0.5)


def test_price_appreciation_zero_when_flat_hpi():
    # no growth at all -> g should be exactly zero
    g = price_appreciation_expectation([100.0] * 15, alpha=0.5)
    assert np.isclose(g, 0.0)


def test_price_appreciation_positive_when_hpi_rising():
    # index doubled over the year -> recent-3 average >> year-ago-3 average
    hpi = [100.0] * 12 + [190.0, 195.0, 200.0]  # indices -15..-1, most recent last
    g = price_appreciation_expectation(hpi, alpha=0.5)
    assert g > 0
