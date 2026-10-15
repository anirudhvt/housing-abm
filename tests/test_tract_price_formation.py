"""Tests for Tract's two-stage price formation (EMA + reversion to reference)."""

import pytest

from housing_abm.tract import Tract


def make_tract(**overrides):
    kwargs = dict(
        tract_id="t",
        price_per_quality=200_000.0,
        reference_price_per_quality=200_000.0,
        smoothing_factor=0.2,
        price_decay=0.5,
    )
    kwargs.update(overrides)
    return Tract(**kwargs)


def test_no_sales_this_month_reverts_toward_reference_only():
    # price starts above reference; with no sales, stage 1 is skipped and
    # stage 2 alone should pull it exactly halfway to reference * HPI (HPI=1)
    t = make_tract(price_per_quality=300_000.0)
    t.update_hpi_history()
    assert t.price_per_quality == pytest.approx(0.5 * 300_000.0 + 0.5 * 200_000.0)


def test_no_sales_never_changes_house_price_index():
    t = make_tract()
    t.house_price_index = 1.3
    t.update_hpi_history()
    assert t.house_price_index == 1.3  # only stage 1 (gated on sales) can move it


def test_reference_price_is_immutable_across_updates():
    t = make_tract()
    ref = t.reference_price_per_quality
    t.record_sale(price=500_000.0, quality=1.0, days_on_market=10)
    t.update_hpi_history()
    t.record_sale(price=50_000.0, quality=1.0, days_on_market=10)
    t.update_hpi_history()
    assert t.reference_price_per_quality == ref


def test_single_sale_at_reference_price_leaves_price_unchanged():
    # a sale at the reference level is a no-op for both stages
    t = make_tract()
    t.record_sale(price=200_000.0, quality=1.0, days_on_market=10)
    t.update_hpi_history()
    assert t.price_per_quality == pytest.approx(200_000.0)
    assert t.house_price_index == pytest.approx(1.0)


def test_one_outlier_sale_moves_price_by_less_than_the_raw_jump():
    # one sale at 3x reference must be heavily damped
    t = make_tract(smoothing_factor=0.1091, price_decay=0.5)
    t.record_sale(price=600_000.0, quality=1.0, days_on_market=10)
    t.update_hpi_history()
    raw_jump_target = 600_000.0
    assert t.price_per_quality < 0.5 * raw_jump_target
    assert t.price_per_quality > 200_000.0  # still moved up, just damped


def test_house_price_index_is_smoothed_not_overwritten():
    # one outlier month must not set the HPI outright
    t = make_tract(smoothing_factor=0.1091, price_decay=0.5)
    t.house_price_index = 1.0
    t.record_sale(price=600_000.0, quality=1.0, days_on_market=10)  # 3x reference
    t.update_hpi_history()
    assert t.house_price_index < 1.3  # nowhere near the raw 3.0 ratio


def test_repeated_sales_at_elevated_price_converge_hpi_toward_that_level():
    # persistent moves still pull the series over time; only noise is damped
    t = make_tract(smoothing_factor=0.1091, price_decay=0.5)
    for _ in range(60):
        t.record_sale(price=400_000.0, quality=1.0, days_on_market=10)
        t.update_hpi_history()
    assert t.price_per_quality == pytest.approx(400_000.0, rel=0.05)
    assert t.house_price_index == pytest.approx(2.0, rel=0.05)


def test_avg_sold_price_reads_the_smoothed_series():
    t = make_tract(price_per_quality=250_000.0)
    assert t.avg_sold_price(quality=2.0) == pytest.approx(500_000.0)
    assert t.avg_sold_price(quality=0.5) == pytest.approx(125_000.0)


def test_avg_sold_price_moves_with_price_per_quality_after_an_update():
    t = make_tract(price_per_quality=200_000.0)
    t.record_sale(price=200_000.0, quality=1.0, days_on_market=10)
    t.update_hpi_history()
    t.record_sale(price=200_000.0, quality=1.0, days_on_market=10)
    t.update_hpi_history()
    # no movement expected since sales are at the reference level throughout
    assert t.avg_sold_price(quality=1.0) == pytest.approx(t.price_per_quality)


def test_smoothing_factor_derivation_matches_reference_constant():
    # pin 1 - 0.25**(1/12), the reference model's CUMULATIVE_WEIGHT_BEYOND_YEAR
    cumulative_weight_beyond_year = 0.25
    smoothing_factor = 1.0 - cumulative_weight_beyond_year ** (1.0 / 12.0)
    assert smoothing_factor == pytest.approx(0.10910, abs=1e-4)


def test_days_on_market_tracking_unaffected_by_price_formation_change():
    # record_sale's second responsibility (days-on-market window for EQ7/
    # EQ19-20) must survive independent of anything to do with price
    t = make_tract()
    t.record_sale(price=200_000.0, quality=1.0, days_on_market=15)
    t.record_sale(price=200_000.0, quality=1.0, days_on_market=45)
    assert t.avg_days_on_market() == pytest.approx(30.0)


def test_monthly_sales_buffer_clears_after_each_update():
    t = make_tract()
    t.record_sale(price=200_000.0, quality=1.0, days_on_market=10)
    assert len(t._monthly_sales) == 1
    t.update_hpi_history()
    assert t._monthly_sales == []
