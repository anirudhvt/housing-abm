"""Tests for the SCF-grounded small-landlord selection mechanism.

See docs/methodology.md Section 11/12: replaces a disconnected, hand-picked
"landlords skew higher-income" lognormal with landlord incomes drawn from
the same household distribution as everyone else, selected without
replacement weighted by the real income-percentile ownership-propensity
curve fit from the 2022 SCF.
"""

import numpy as np
import pytest

from housing_abm.equations.investor_propensity import (
    landlord_selection_weight,
    sample_landlord_incomes,
)

MEAN, SIGMA = 8.6, 0.65
FLAT_PROBS = [0.1] * 10
REAL_PROBS = [0.0297, 0.0379, 0.0745, 0.0811, 0.1375, 0.1308, 0.1294, 0.3209, 0.4298, 0.5987]


def test_landlord_selection_weight_uses_the_correct_decile():
    # median income (50th percentile) should land in decile index 4 or 5
    median_income = np.exp(MEAN)
    weight = landlord_selection_weight(median_income, REAL_PROBS, MEAN, SIGMA)
    assert weight in (REAL_PROBS[4], REAL_PROBS[5])


def test_landlord_selection_weight_low_income_gets_bottom_decile_probability():
    tiny_income = np.exp(MEAN) * 0.01  # far below the distribution
    weight = landlord_selection_weight(tiny_income, REAL_PROBS, MEAN, SIGMA)
    assert weight == REAL_PROBS[0]


def test_landlord_selection_weight_high_income_gets_top_decile_probability():
    huge_income = np.exp(MEAN) * 100  # far above the distribution
    weight = landlord_selection_weight(huge_income, REAL_PROBS, MEAN, SIGMA)
    assert weight == REAL_PROBS[-1]


def test_sample_landlord_incomes_returns_exactly_n_landlords():
    rng = np.random.default_rng(0)
    incomes = sample_landlord_incomes(rng, n_landlords=15, pool_size=300,
                                       decile_probs=REAL_PROBS,
                                       income_lognormal_mean=MEAN, income_lognormal_sigma=SIGMA)
    assert len(incomes) == 15
    assert all(i > 0 for i in incomes)


def test_sample_landlord_incomes_returns_empty_for_zero_landlords():
    rng = np.random.default_rng(0)
    incomes = sample_landlord_incomes(rng, n_landlords=0, pool_size=100,
                                       decile_probs=REAL_PROBS,
                                       income_lognormal_mean=MEAN, income_lognormal_sigma=SIGMA)
    assert len(incomes) == 0


def test_sample_landlord_incomes_handles_pool_smaller_than_requested():
    # pool_size < n_landlords should still return exactly n_landlords,
    # not raise from np.random.Generator.choice's replace=False constraint
    rng = np.random.default_rng(0)
    incomes = sample_landlord_incomes(rng, n_landlords=20, pool_size=5,
                                       decile_probs=REAL_PROBS,
                                       income_lognormal_mean=MEAN, income_lognormal_sigma=SIGMA)
    assert len(incomes) == 20


def test_real_propensity_curve_skews_landlord_incomes_above_flat_selection():
    # a monotonically increasing propensity curve should pull the selected
    # sample's mean income above what uniform (flat-weight) selection from
    # the same pool would give -- this is the entire point of the mechanism
    rng_real = np.random.default_rng(42)
    rng_flat = np.random.default_rng(42)
    real_incomes = sample_landlord_incomes(rng_real, n_landlords=200, pool_size=2000,
                                            decile_probs=REAL_PROBS,
                                            income_lognormal_mean=MEAN, income_lognormal_sigma=SIGMA)
    flat_incomes = sample_landlord_incomes(rng_flat, n_landlords=200, pool_size=2000,
                                            decile_probs=FLAT_PROBS,
                                            income_lognormal_mean=MEAN, income_lognormal_sigma=SIGMA)
    assert real_incomes.mean() > flat_incomes.mean()


def test_flat_propensity_curve_gives_uniform_weight_across_deciles():
    # sanity check on the weighting function itself, independent of sampling
    low = landlord_selection_weight(np.exp(MEAN) * 0.1, FLAT_PROBS, MEAN, SIGMA)
    high = landlord_selection_weight(np.exp(MEAN) * 10, FLAT_PROBS, MEAN, SIGMA)
    assert low == pytest.approx(high)
