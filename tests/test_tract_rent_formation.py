"""Tests for Tract's rent formation -- the rent-side mirror of
test_tract_price_formation.py.

Before this mechanism existed, rent_per_quality was set once from config and
never updated by the rental market (measured: exactly one value across 240
simulated months). Since that number is the market signal read by EQ5's
buy-vs-rent decision, EQ9/12's investor yield, EQ11's rent setting and the
investor yield ranking, the reference's stabilising loop could not operate.
See docs/reference_gap_analysis.md Problem 1.

These tests pin the two stages' arithmetic directly, independent of the full
model, and guard the specific failure mode that motivated the work: a rent
level that cannot respond to the rental market at all.
"""

import pytest

from housing_abm.tract import Tract


def make_tract(**overrides):
    kwargs = dict(
        tract_id="t",
        price_per_quality=200_000.0,
        rent_per_quality=1000.0,
        reference_rent_per_quality=1000.0,
        smoothing_factor=0.2,
        rent_decay=0.5,
    )
    kwargs.update(overrides)
    return Tract(**kwargs)


def test_rent_responds_to_lettings_at_all():
    """The regression this whole mechanism exists to prevent: rent frozen."""
    t = make_tract()
    start = t.rent_per_quality
    for _ in range(12):
        t.record_letting(rent=2000.0, quality=1.0, days_vacant=10)
        t.update_rent_history()
    assert t.rent_per_quality != start
    assert t.rent_per_quality > start  # lettings above the anchor pull rent up


def test_no_lettings_this_month_reverts_toward_reference_only():
    t = make_tract(rent_per_quality=1500.0)
    t.update_rent_history()
    # stage 1 skipped, stage 2 alone pulls halfway to reference * index (1.0)
    assert t.rent_per_quality == pytest.approx(0.5 * 1500.0 + 0.5 * 1000.0)


def test_no_lettings_never_changes_rent_index():
    t = make_tract()
    t.rent_index = 1.3
    t.update_rent_history()
    assert t.rent_index == 1.3  # only stage 1 (gated on lettings) can move it


def test_reference_rent_is_immutable_across_updates():
    t = make_tract()
    ref = t.reference_rent_per_quality
    t.record_letting(rent=4000.0, quality=1.0, days_vacant=5)
    t.update_rent_history()
    t.record_letting(rent=400.0, quality=1.0, days_vacant=5)
    t.update_rent_history()
    assert t.reference_rent_per_quality == ref


def test_letting_at_reference_rent_leaves_rent_unchanged():
    t = make_tract()
    t.record_letting(rent=1000.0, quality=1.0, days_vacant=10)
    t.update_rent_history()
    assert t.rent_per_quality == pytest.approx(1000.0)
    assert t.rent_index == pytest.approx(1.0)


def test_one_outlier_letting_moves_rent_less_than_the_raw_jump():
    t = make_tract(smoothing_factor=0.1091, rent_decay=0.5)
    t.record_letting(rent=3000.0, quality=1.0, days_vacant=10)
    t.update_rent_history()
    assert t.rent_per_quality < 0.5 * 3000.0
    assert t.rent_per_quality > 1000.0  # still moved up, just damped


def test_rent_index_is_smoothed_not_overwritten():
    """Same double-counting guard as the price side's HPI test."""
    t = make_tract(smoothing_factor=0.1091, rent_decay=0.5)
    t.record_letting(rent=3000.0, quality=1.0, days_vacant=10)  # 3x reference
    t.update_rent_history()
    assert t.rent_index < 1.3  # nowhere near the raw 3.0 ratio


def test_sustained_lettings_converge_toward_that_level():
    t = make_tract(smoothing_factor=0.1091, rent_decay=0.5)
    for _ in range(60):
        t.record_letting(rent=2000.0, quality=1.0, days_vacant=10)
        t.update_rent_history()
    assert t.rent_per_quality == pytest.approx(2000.0, rel=0.05)
    assert t.rent_index == pytest.approx(2.0, rel=0.05)


def test_market_rent_scales_with_quality():
    t = make_tract(rent_per_quality=1200.0)
    assert t.market_rent(quality=2.0) == pytest.approx(2400.0)
    assert t.market_rent(quality=0.5) == pytest.approx(600.0)


def test_avg_days_vacant_tracks_lettings_not_sales():
    """EQ11's f_bar must be rental-market days, not ownership-market days."""
    t = make_tract()
    t.record_sale(price=200_000.0, quality=1.0, days_on_market=90)
    t.record_letting(rent=1000.0, quality=1.0, days_vacant=10)
    t.record_letting(rent=1000.0, quality=1.0, days_vacant=20)
    assert t.avg_days_vacant() == pytest.approx(15.0)
    assert t.avg_days_on_market() == pytest.approx(90.0)  # kept separate


def test_monthly_lettings_buffer_clears_after_each_update():
    t = make_tract()
    t.record_letting(rent=1000.0, quality=1.0, days_vacant=10)
    assert len(t._monthly_lettings) == 1
    t.update_rent_history()
    assert t._monthly_lettings == []


def test_external_rent_series_overrides_endogenous_formation():
    """Same override relationship external_g_series has with appreciation_g."""
    t = make_tract(external_rent_growth_series=[0.10])
    t.record_letting(rent=5000.0, quality=1.0, days_vacant=10)
    t.update_rent_history()
    # driven by the external +10%, not by the 5000 letting
    assert t.rent_per_quality == pytest.approx(1100.0)
    assert t._monthly_lettings == []
