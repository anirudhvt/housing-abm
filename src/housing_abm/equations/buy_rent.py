"""EQ 5: buy vs rent decision."""

import numpy as np


def sigmoid(x: float) -> float:
    x = max(min(x, 500), -500)  # avoid exp overflow
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
    g: expected appreciation (EQ 4); None is treated as 0
    beta: sensitivity parameter
    annual_income: normalizer; defaults to 1 (raw dollars)"""
    g_safe = g if g is not None else 0.0
    renting_cost = rent_q * (1 + tau)
    # g is signed: expected price falls make owning more expensive
    buying_cost = 12 * (monthly_mortgage - price * g_safe)
    normalizer = max(annual_income, 1.0)
    return sigmoid(beta * (renting_cost - buying_cost) / normalizer)