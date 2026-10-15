"""EQ 11 rental pricing, differentiated by landlord type, and lease lengths."""

import numpy as np


def small_landlord_rent(
    r_bar_tract: float,
    f_bar_tract: float,
    alpha: float,
    beta: float,
    zeta: float,
    epsilon_std: float,
    reprice_prob: float,
    previous_rent: float | None,
    rng: np.random.Generator,
) -> float:
    """EQ 11 with stickiness: reprice with probability reprice_prob, otherwise
    keep the previous rent."""
    if previous_rent is not None and rng.random() >= reprice_prob:
        return previous_rent
    epsilon = rng.normal(0, epsilon_std)
    ln_q = (
        alpha + np.log(r_bar_tract) - beta * np.log(zeta * (1 + f_bar_tract)) + epsilon
    )
    return np.exp(ln_q)


def institutional_rent(market_rate: float, premium: float = 1.04) -> float:
    """Market rate plus a premium, no noise or stickiness (algorithmic pricing)."""
    return market_rate * premium


def sample_lease_length(rng: np.random.Generator) -> int:
    """Lease term in months from the national BLS CPI Housing Survey mix
    (month-to-month = 1). TODO: Atlanta-specific data if available."""

    r = rng.random()
    if r < 0.318:
        return 1  # month-to-month (31.8%)
    elif r < 0.318 + 0.596:
        return 12  # standard 12-month (59.6%)
    elif r < 0.318 + 0.596 + 0.026:
        return 24  # 24-month (~2.6% = 8.6% * 29.9%)
    elif r < 0.318 + 0.596 + 0.026 + 0.013:
        return 13  # 13-month (~1.3% = 8.6% * 14.8%)
    elif r < 0.318 + 0.596 + 0.026 + 0.013 + 0.011:
        return 6  # 6-month (~1.1% = 8.6% * 12.4%)
    else:
        return 12  # remaining ~3.6% of "other" terms, folded in
