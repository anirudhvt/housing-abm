import numpy as np
from housing_abm.equations.investor_yield import (
    expected_yield_buy,
    effective_yield_sell,
)


def test_expected_yield_buy_cash_purchase_has_no_mortgage_drag():
    # all-cash: down_payment == price, monthly_mortgage == 0
    omega = expected_yield_buy(
        price=200_000,
        down_payment=200_000,
        delta=0.3,
        g=0.02,
        kappa=0.05,
        r_bar=0.06,
        monthly_mortgage=0.0,
    )
    leverage = 1.0  # price/down_payment
    expected = leverage * (0.3 * (0.02 + 0.05) + 0.7 * 0.06)
    assert np.isclose(omega, expected)


def test_expected_yield_buy_leverage_amplifies_yield():
    # smaller down payment (more leverage) on the same house should raise omega
    # when the underlying (g+kappa)/r_bar terms are positive
    low_leverage = expected_yield_buy(
        price=200_000,
        down_payment=200_000,
        delta=0.3,
        g=0.02,
        kappa=0.05,
        r_bar=0.06,
        monthly_mortgage=0.0,
    )
    high_leverage = expected_yield_buy(
        price=200_000,
        down_payment=40_000,
        delta=0.3,
        g=0.02,
        kappa=0.05,
        r_bar=0.06,
        monthly_mortgage=500.0,
    )
    assert high_leverage > low_leverage


def test_expected_yield_buy_policy_cost_subtracts_directly():
    base = expected_yield_buy(
        price=200_000,
        down_payment=100_000,
        delta=0.3,
        g=0.02,
        kappa=0.05,
        r_bar=0.06,
        monthly_mortgage=500.0,
    )
    with_cost = expected_yield_buy(
        price=200_000,
        down_payment=100_000,
        delta=0.3,
        g=0.02,
        kappa=0.05,
        r_bar=0.06,
        monthly_mortgage=500.0,
        policy_cost=0.01,
    )
    assert np.isclose(base - with_cost, 0.01)


def test_effective_yield_sell_equity_floored_above_zero():
    # zero or negative equity shouldn't blow up with a division by zero
    psi = effective_yield_sell(
        price=200_000,
        equity=0.0,
        delta=0.3,
        g=0.02,
        kappa=0.05,
        r_bar=0.06,
        monthly_mortgage=500.0,
    )
    assert np.isfinite(psi)
    psi_negative_equity = effective_yield_sell(
        price=200_000,
        equity=-50_000,
        delta=0.3,
        g=0.02,
        kappa=0.05,
        r_bar=0.06,
        monthly_mortgage=500.0,
    )
    assert np.isfinite(psi_negative_equity)
