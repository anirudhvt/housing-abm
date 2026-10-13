"""Tests for price_trend_cagr, the trend estimator used to validate the
model's price growth.

Exists because `annual_appreciation_g` -- the mean over months of EQ4's
ratio-based trailing growth -- is a biased estimator of the underlying
trend on a volatile mean-reverting series, and was failing the validation
band for that reason rather than because of the model's actual price
behaviour. See housing_abm.metrics.price_trend_cagr for the measurement.
"""

import numpy as np
import pytest

from housing_abm.metrics import price_trend_cagr


def test_flat_series_has_zero_trend():
    assert price_trend_cagr([250_000.0] * 120) == pytest.approx(0.0, abs=1e-9)


def test_recovers_a_known_constant_growth_rate():
    monthly = (1.03) ** (1 / 12)  # 3%/yr compounded monthly
    prices = [250_000.0 * monthly**i for i in range(120)]
    assert price_trend_cagr(prices) == pytest.approx(0.03, abs=1e-6)


def test_recovers_a_known_decline():
    monthly = (0.98) ** (1 / 12)
    prices = [250_000.0 * monthly**i for i in range(120)]
    assert price_trend_cagr(prices) == pytest.approx(-0.02, abs=1e-6)


def test_returns_none_for_too_short_a_window():
    assert price_trend_cagr([250_000.0] * 12) is None


def test_ignores_none_and_nonpositive_entries():
    prices = [None, 250_000.0, float("nan"), 0.0, -5.0] + [250_000.0] * 40
    assert price_trend_cagr(prices) == pytest.approx(0.0, abs=1e-9)


def test_mean_reverting_noise_does_not_manufacture_a_trend():
    # the exact failure mode this metric exists to avoid: a volatile series
    # with no underlying trend must not read as positive growth
    rng = np.random.default_rng(0)
    prices = 250_000.0 * np.exp(rng.normal(0, 0.08, size=180))
    trend = price_trend_cagr(list(prices))
    assert abs(trend) < 0.02


def test_trend_survives_noise_on_top_of_real_growth():
    rng = np.random.default_rng(1)
    monthly = (1.03) ** (1 / 12)
    base = np.array([250_000.0 * monthly**i for i in range(180)])
    noisy = base * np.exp(rng.normal(0, 0.05, size=180))
    assert price_trend_cagr(list(noisy)) == pytest.approx(0.03, abs=0.01)
