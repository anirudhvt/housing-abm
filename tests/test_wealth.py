import numpy as np
from housing_abm.equations.wealth import desired_bank_balance, monthly_consumption


def test_desired_bank_balance_matches_formula_with_zero_noise(zero_rng):
    alpha, beta, income = -30.0, 4.0, 3000.0
    w = desired_bank_balance(income, alpha, beta, epsilon_std=0.3, rng=zero_rng)
    expected = np.exp(alpha + beta * np.log(income))
    assert np.isclose(w, expected)


def test_desired_bank_balance_increases_with_income(zero_rng):
    alpha, beta = -30.0, 4.0
    low = desired_bank_balance(1000.0, alpha, beta, 0.3, zero_rng)
    high = desired_bank_balance(5000.0, alpha, beta, 0.3, zero_rng)
    assert high > low  # beta > 0 => higher income => higher target balance


def test_monthly_consumption_zero_when_below_target():
    # balance below desired target -> no discretionary consumption
    assert (
        monthly_consumption(current_balance=1000, desired_balance=5000, alpha=0.5)
        == 0.0
    )


def test_monthly_consumption_positive_when_above_target():
    c = monthly_consumption(current_balance=10000, desired_balance=5000, alpha=0.5)
    assert c == 0.5 * (10000 - 5000)


def test_monthly_consumption_never_negative(rng):
    for _ in range(200):
        balance = rng.uniform(-1000, 20000)
        desired = rng.uniform(0, 20000)
        c = monthly_consumption(balance, desired, alpha=rng.uniform(0, 1))
        assert c >= 0.0
