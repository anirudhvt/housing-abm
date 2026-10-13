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
    # EQ 9 is defined on a monthly basis; g/kappa/r_bar are supplied annualized
    expected = leverage * (0.3 * (0.02 + 0.05) + 0.7 * 0.06) / 12.0
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
    # policy_cost is an annual rate, applied on the equation's monthly basis
    assert np.isclose(base - with_cost, 0.01 / 12.0)


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


def test_omega_stays_inside_the_logistic_response_region():
    """Regression: EQ 9/12 must not saturate the EQ 10/13 logistic.

    beta = 50 is Baptista's calibration for a monthly-scale Omega. If Omega is
    computed on an annual scale instead, beta*Omega lands around 10-12, where
    the logistic is flat to ~1e-5 and no policy can move investor behaviour.
    Across realistic market states |beta*Omega| must stay small enough that the
    logistic still responds.
    """
    from housing_abm.equations.investor_probs import p_buy_investor

    beta = 50.0
    for g in (-0.05, 0.0, 0.02, 0.05, 0.10):
        omega = expected_yield_buy(
            price=250_000,
            down_payment=250_000 * 0.25,
            delta=0.3,
            g=g,
            kappa=0.0631,
            r_bar=0.0672,
            monthly_mortgage=900.0,
        )
        assert abs(beta * omega) < 2.0, f"logistic saturated at g={g}: beta*omega={beta * omega}"
        # and a 5% purchase tax must produce a visible change in P(buy)
        taxed = expected_yield_buy(
            price=250_000,
            down_payment=250_000 * 0.25,
            delta=0.3,
            g=g,
            kappa=0.0631,
            r_bar=0.0672,
            monthly_mortgage=900.0,
            policy_cost=0.05,
        )
        response = p_buy_investor(omega, beta) - p_buy_investor(taxed, beta)
        assert response > 1e-3, f"purchase tax has no bite at g={g}: dP={response}"
