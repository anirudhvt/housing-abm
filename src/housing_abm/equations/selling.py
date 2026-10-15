"""EQ 6 (sell probability), EQ 7 (asking price), EQ 8 (price reduction)."""

import numpy as np


def p_sell(
    tenure_years: float,
    n_h: float,
    n_h_avg: float,
    i_current: float,
    i_avg: float,
    alpha: float,
    beta: float,
    i_mortgage: float | None = None,
    gamma: float = 0.0,
) -> float:
    """EQ 6: P(sell) = 1/12 * max((1 + alpha(n_h_avg - n_h) + beta(i_avg - i_current))
    / tenure_years, 0), with an added rate lock-in term gamma(i_current - i_mortgage)
    that is not in the paper (pass i_mortgage=None to disable)."""
    base = 1 + alpha * (n_h_avg - n_h) + beta * (i_avg - i_current)
    if i_mortgage is not None:
        base -= gamma * (i_current - i_mortgage)
    return 1 / 12 * max((1 / tenure_years) * base, 0.0)


def asking_price(
    p_bar_tract: float,
    f_bar_tract: float,
    alpha: float,
    beta: float,
    zeta: float,
    epsilon_std: float,
    rng: np.random.Generator,
) -> float:
    """EQ 7: ln(p_s) = alpha + ln(p_bar) - beta*ln(zeta*(1+f_bar)) + epsilon,
    with p_bar and f_bar measured per census tract."""
    epsilon = rng.normal(0, epsilon_std)
    ln_ps = (
        alpha + np.log(p_bar_tract) - beta * np.log(zeta * (1 + f_bar_tract)) + epsilon
    )
    return np.exp(ln_ps)


def price_reduction(
    current_price: float,
    reduction_prob: float,
    epsilon_std: float,
    rng: np.random.Generator,
    vacancy_tax_multiplier: float = 1.0,
) -> float:
    """EQ 8: cut the price with probability reduction_prob (scaled up by
    vacancy_tax_multiplier for taxed investor listings)."""
    prob = min(reduction_prob * vacancy_tax_multiplier, 1.0)
    if rng.random() < prob:
        epsilon = -abs(rng.normal(0, epsilon_std))
        return max(current_price * np.exp(epsilon), 0)
    return current_price
