"""Eq 10 (Investor/landlord purchase probability) and EQ 13 (investor/landlord sell probability

beta: beta_small_landlord and beta_institutional matched against purchase share (atlanta)

"""

from .buy_rent import sigmoid


def p_buy_investor(omega: float, beta: float, policy_blocked: bool = False) -> float:
    """EQ 10: P(buy) = sigmoid(beta * omega)^(1/12)
    policy_blocked: True to force P(0) for hard policies: waiting period,
    ownership cap, geographic restriction - BEFORE equatoin runs
    omega: expected yield (EQ 9)"""

    if policy_blocked:
        return 0.0
    return sigmoid(beta * omega) ** (1 / 12)


def p_sell_investor(psi: float, beta: float, forced_divesture: bool = False) -> float:
    """EQ 13: P(sell) = sigmoid(beta * psi)^(1/12)
    forced_divesture: True for ownership caps to force P = 1
    psi: effective yield (EQ 12)"""

    if forced_divesture:
        return 1.0
    return 1 - sigmoid(beta * psi) ** (1 / 12)
