"""EQ 5: buy vs rent decision"""

import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def p_buy(
    rent_q: float,
    tau: float,
    monthly_mortgage: float,
    price: float,
    g: float,
    beta: float,
) -> float:
    """EQ 5: P(buy) = sigma(beta * (rQ(1+tau) - 12(m-pg)))

    rent_q: annual rent for house of quality Q
    tau: psychologist cost-of-renting premium (recalibrated)
    monthly_mortgage: m, from equations/morgage.py for fha and conventional loans
    price: p, households desired expenditure
    g: expected appreciation (eq 4)
    beta: sensitivity parameter"""

    renting_cost = rent_q * (1 + tau)
    buying_cost = 12 * monthly_mortgage - price * g
    return sigmoid(beta * (renting_cost - buying_cost))
