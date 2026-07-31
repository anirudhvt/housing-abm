"""EQ 5: buy vs rent decision"""

import numpy as np


def sigmoid(x: float) -> float:
    # clip to avoid overflow
    x = max(min(x, 500), -500)
    return 1 / (1 + np.exp(-x))


def p_buy(
    rent_q: float,
    tau: float,
    monthly_mortgage: float,
    price: float,
    g: float,
    beta: float,
    annual_income: float = 1.0,
) -> float:
    """EQ 5: P(buy) = sigma(beta * (rQ(1+tau) - 12(m-pg)) / income)

    rent_q: annual rent for house of quality Q
    tau: psychological cost-of-renting premium
    monthly_mortgage: m
    price: p, desired expenditure
    g: expected appreciation (eq 4) — clamped internally
    beta: sensitivity parameter 
    annual_income: normalizer; defaults to 1 (raw dollars)"""
    g_safe = g if g is not None else 0.0
    g_upside = max(g_safe, 0.0)   # only positive appreciation enters buying cost
    renting_cost = rent_q * (1 + tau)
    buying_cost = 12 * (monthly_mortgage - price * g_upside)
    normalizer = max(annual_income, 1.0)
    return sigmoid(beta * (renting_cost - buying_cost) / normalizer)