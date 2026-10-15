"""EQ 3 (desired expenditure) and EQ 4 (price appreciation expectation).
Adaptations from Baptista et al. (2016): investors substitute capital for
income, and g is computed per census tract."""

import numpy as np


def desired_expenditure(
    income_or_capital: float,
    g: float,
    alpha: float,
    beta: float,
    epsilon_std: float,
    rng: np.random.Generator,
    mortgage_cap: float | None = None,
) -> float:
    """Eq 3: p_desired = alpha * y * exp(epsilon) / (1-beta*g)

    income_or_capital: household income, or fund capital for investors
    mortgage_cap: if given, caps the result at max loan + down payment (EQ 14)
    """
    epsilon = rng.normal(0, epsilon_std)
    p_desired = alpha * income_or_capital * np.exp(epsilon) / (1 - beta * g)
    if mortgage_cap is not None:
        p_desired = min(p_desired, mortgage_cap)
    return p_desired


def price_appreciation_expectation(hpi_history: list[float], alpha: float) -> float:
    """EQ 4: g_t = alpha*[(h_t-1+h_t-2+h_t-3)/(h_t-13+h_t-14+h_t-15)-1],
    clamped to [-0.03, 0.256]. hpi_history needs >= 15 values, most recent last."""
    if len(hpi_history) < 15:
        raise ValueError("hpi_history must contain at least 15 values")
    h_t_minus_1 = hpi_history[-1]
    h_t_minus_2 = hpi_history[-2]
    h_t_minus_3 = hpi_history[-3]
    h_t_minus_13 = hpi_history[-13]
    h_t_minus_14 = hpi_history[-14]
    h_t_minus_15 = hpi_history[-15]
    g_t = alpha * (
        (h_t_minus_1 + h_t_minus_2 + h_t_minus_3)
        / (h_t_minus_13 + h_t_minus_14 + h_t_minus_15)
        - 1
    )
    return max(min(g_t, 0.256), -0.03)
