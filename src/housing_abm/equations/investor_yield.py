"""EQ 9 (expected yield, buy), EQ 12: (effective yield, sell)
delta: small_landlord delta = 0.3, institutional investor delta = 0.6 - investors value long term yield
Policy costs subtract directly from each yield

UNITS. Baptista et al. define Omega and Psi in *monthly* terms: g is a monthly
house-price growth expectation and m/d is a monthly mortgage payment over the
downpayment. The shape parameter beta = 50 in EQ 10/13 was calibrated against
that monthly scale.

This model computes g, kappa and r_bar as annualized rates (EQ 4 compares a
3-month price average against the same average 12 months earlier; kappa and
r_bar are annual gross yields). Feeding those straight in alongside a monthly
m/d made Omega roughly an order of magnitude too large and inconsistent
internally. At beta = 50 that put the logistic in EQ 10/13 fully into
saturation: P(buy) = 1.0000 and P(sell) ~ 1e-5 in every market state, so
landlord portfolios only ever grew, and no financial penalty of any plausible
size could shift investor behaviour at all -- a 5% purchase tax moved P(buy) by
less than 1e-4.

These functions therefore take annualized g, kappa and r_bar and convert them
to the monthly basis the equations are defined on. policy_cost is likewise
supplied as an annual rate and converted here, so a "5% purchase tax" means 5%
per year of holding, not 5% per month.
"""

MONTHS_PER_YEAR = 12.0


def _monthly_yield_core(delta: float, g: float, kappa: float, r_bar: float) -> float:
    """delta*(g+kappa) + (1-delta)*r_bar, on a monthly basis.

    Inputs are annualized rates; the division puts them on the monthly scale
    that EQ 9/12 and the beta calibration assume.
    """
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
    """EQ 12: same as eq 9 but with equity instead of down_payment"""
    equity = max(equity, 1e-6)  # equity > 0
    leverage = price / equity
    psi = leverage * _monthly_yield_core(delta, g, kappa, r_bar)
    psi -= monthly_mortgage / equity
    psi -= policy_cost / MONTHS_PER_YEAR
    return psi
