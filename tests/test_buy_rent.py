import numpy as np
from housing_abm.equations.buy_rent import sigmoid, p_buy


def test_sigmoid_midpoint_is_half():
    assert np.isclose(sigmoid(0.0), 0.5)


def test_sigmoid_bounds():
    assert 0.0 < sigmoid(-50) < 0.001
    assert 0.999 < sigmoid(50) <= 1.0


def test_p_buy_favors_buying_when_renting_is_much_more_expensive():
    p = p_buy(
        rent_q=30_000,
        tau=1.1 / 12,
        monthly_mortgage=1000,
        price=200_000,
        g=0.02,
        beta=1 / 3500,
    )
    assert p > 0.5


def test_p_buy_favors_renting_when_buying_is_much_more_expensive():
    p = p_buy(
        rent_q=6_000,
        tau=1.1 / 12,
        monthly_mortgage=10_000,
        price=200_000,
        g=0.0,
        beta=1 / 3500,
    )
    assert p < 0.5
