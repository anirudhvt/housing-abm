"""EQ 10 (investor purchase probability) and EQ 13 (investor sell probability).
beta is set separately for small landlords and institutional investors."""

from .buy_rent import sigmoid


def p_buy_investor(omega: float, beta: float, policy_blocked: bool = False) -> float:
    """EQ 10: P(buy) = sigmoid(beta * omega)^(1/12), omega from EQ 9.
    policy_blocked forces P = 0."""

    if policy_blocked:
        return 0.0
    return sigmoid(beta * omega) ** (1 / 12)


def p_sell_investor(psi: float, beta: float, forced_divesture: bool = False) -> float:
    """EQ 13: P(sell) = 1 - sigmoid(beta * psi)^(1/12), psi from EQ 12.
    forced_divesture forces P = 1."""

    if forced_divesture:
        return 1.0
    return 1 - sigmoid(beta * psi) ** (1 / 12)
