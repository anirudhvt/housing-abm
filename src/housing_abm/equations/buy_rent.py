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
    renting_cost = rent_q * (1 + tau)
    # g enters signed, as in the reference. This previously used
    # max(g, 0), which discarded falling prices entirely: when prices fall,
    # owning should look *worse* (the expected capital loss adds to the cost
    # of owning) and the bust should deepen. Truncating g removed the whole
    # downside half of the expectations channel -- the mechanism the
    # reference identifies as the cycle driver. See
    # docs/reference_gap_analysis.md Problem 5.
    buying_cost = 12 * (monthly_mortgage - price * g_safe)
    normalizer = max(annual_income, 1.0)
    return sigmoid(beta * (renting_cost - buying_cost) / normalizer)