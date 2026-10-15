"""EQ 9 (expected yield on a purchase) and EQ 12 (effective yield on a holding).

Inputs g, kappa, r_bar and policy_cost are annual rates; they are converted
here to the monthly basis the equations (and beta in EQ 10/13) are defined on.
"""

MONTHS_PER_YEAR = 12.0


def _monthly_yield_core(delta: float, g: float, kappa: float, r_bar: float) -> float:
    """delta*(g+kappa) + (1-delta)*r_bar, converted from annual to monthly."""
    capital_yield = (g + kappa) / MONTHS_PER_YEAR
    rental_yield = r_bar / MONTHS_PER_YEAR
    return delta * capital_yield + (1.0 - delta) * rental_yield


def expected_yield_buy(
    price: float,
    down_payment: float,
    delta: float,
    g: float,
    kappa: float,
    r_bar: float,
    monthly_mortgage: float,
    policy_cost: float = 0.0,
) -> float:
    """EQ 9: omega = (p/d)*(delta*(g+kappa)+(1-delta)*r_bar) - m/d - policy_cost

    down_payment == price when purchase is all cash (m is zero)
    p/d: leverage (price/down_payment)
    delta: weight on capital yield
    g: annualized house price growth expectation (EQ 4)
    kappa: long-term average annual gross yield
    r_bar: current average annual gross yield
    m/d: monthly mortgage payment over down payment
    policy_cost: annual policy cost rate, converted to monthly here
    """
    if down_payment == 0:
        return 0.0
    leverage = price / down_payment
    omega = leverage * _monthly_yield_core(delta, g, kappa, r_bar)
    omega -= monthly_mortgage / down_payment
    omega -= policy_cost / MONTHS_PER_YEAR
    return omega


def effective_yield_sell(
    price: float,
    equity: float,
    delta: float,
    g: float,
    kappa: float,
    r_bar: float,
    monthly_mortgage: float,
    policy_cost: float = 0.0,
) -> float:
    """EQ 12: EQ 9 with equity in place of the down payment."""
    equity = max(equity, 1e-6)
    leverage = price / equity
    psi = leverage * _monthly_yield_core(delta, g, kappa, r_bar)
    psi -= monthly_mortgage / equity
    psi -= policy_cost / MONTHS_PER_YEAR
    return psi
