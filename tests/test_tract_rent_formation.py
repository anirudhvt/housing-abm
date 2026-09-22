"""Tests for Tract rent price discovery and related methods."""

import numpy as np
from housing_abm.tract import Tract


def test_record_letting_stores_rent_quality_pair():
    t = Tract("t1")
    t.record_letting(rent=1500.0, quality=1.0, days_vacant=10)
    assert len(t._monthly_lettings) == 1
    assert t._monthly_lettings[0] == (1500.0, 1.0)


def test_record_letting_stores_days_vacant():
    t = Tract("t1")
    t.record_letting(rent=1500.0, quality=1.0, days_vacant=15)
    assert t.recent_days_vacant == [15]


def test_market_rent_returns_median():
    t = Tract("t1")
    for r in [1200, 1400, 1600]:
        t.record_letting(rent=float(r), quality=1.0, days_vacant=5)
    assert np.isclose(t.market_rent(1.0), 1400.0)


def test_market_rent_falls_back_to_config_when_no_lettings():
    t = Tract("t1", rent_per_quality=1500.0)
    assert np.isclose(t.market_rent(1.0), 1500.0)


def test_avg_days_vacant_default():
    t = Tract("t1")
    assert t.avg_days_vacant() == 30.0


def test_avg_days_vacant_computed():
    t = Tract("t1")
    t.record_letting(rent=1000, quality=1.0, days_vacant=10)
    t.record_letting(rent=1000, quality=1.0, days_vacant=20)
    assert np.isclose(t.avg_days_vacant(), 15.0)


def test_update_rent_history_smooths_toward_realized():
    t = Tract("t1", rent_per_quality=1400.0, rent_decay=0.5)
    for _ in range(5):
        t.record_letting(rent=1600.0, quality=1.0, days_vacant=5)
    old_rent = t.rent_per_quality
    t.update_rent_history()
    assert t.rent_per_quality != old_rent
    assert len(t.rent_history) == 16


def test_update_rent_history_reverts_toward_reference():
    t = Tract("t1", rent_per_quality=2000.0, rent_decay=1.0)
    t.record_letting(rent=2000.0, quality=1.0, days_vacant=5)
    t.update_rent_history()
    # reference_rent_per_quality = 2000, realized = 2000, so reversion keeps it at 2000
    assert np.isclose(t.rent_per_quality, 2000.0)
    # now test actual reversion: rent above reference should pull down
    t2 = Tract("t1", rent_per_quality=1400.0, rent_decay=1.0)
    t2.rent_per_quality = 2000.0  # artificially inflate above reference
    t2.record_letting(rent=2000.0, quality=1.0, days_vacant=5)
    t2.update_rent_history()
    assert t2.rent_per_quality < 2000.0


def test_update_rent_history_clears_monthly_lettings():
    t = Tract("t1")
    t.record_letting(rent=1500.0, quality=1.0, days_vacant=5)
    assert len(t._monthly_lettings) == 1
    t.update_rent_history()
    assert len(t._monthly_lettings) == 0


def test_no_duplicate_gross_rental_yield():
    import inspect
    source = inspect.getsource(Tract)
    assert source.count("def gross_rental_yield") == 1


def test_gross_rental_yield_correct():
    t = Tract("t1", price_per_quality=250_000, rent_per_quality=1400)
    expected = (1400 * 12) / 250_000
    assert np.isclose(t.gross_rental_yield(), expected)
