"""Small-landlord selection weighted by income percentile, using the 2022 SCF
curve for P(owns other residential real estate) (~3% bottom decile to ~60% top).
See docs/methodology.md Section 11."""

import numpy as np
from scipy.stats import lognorm


def landlord_selection_weight(
    income: float,
    decile_probs: list[float],
    income_lognormal_mean: float,
    income_lognormal_sigma: float,
) -> float:
    """SCF propensity for the income's percentile in the household distribution.
    decile_probs has one entry per equal-width percentile bin."""
    percentile = lognorm.cdf(
        income, income_lognormal_sigma, scale=np.exp(income_lognormal_mean)
    )
    idx = min(int(percentile * len(decile_probs)), len(decile_probs) - 1)
    return decile_probs[idx]


def landlord_birth_probability(
    income: float,
    target_fraction: float,
    decile_probs: list[float],
    income_lognormal_mean: float,
    income_lognormal_sigma: float,
) -> float:
    """Probability a newborn household becomes a SmallLandlord: the SCF curve
    rescaled so the expected population share equals target_fraction."""
    raw_weight = landlord_selection_weight(
        income, decile_probs, income_lognormal_mean, income_lognormal_sigma
    )
    mean_weight = sum(decile_probs) / len(decile_probs)
    if mean_weight <= 0:
        return 0.0
    return min(target_fraction / mean_weight * raw_weight, 1.0)


def sample_landlord_incomes(
    rng: np.random.Generator,
    n_landlords: int,
    pool_size: int,
    decile_probs: list[float],
    income_lognormal_mean: float,
    income_lognormal_sigma: float,
) -> np.ndarray:
    """Draw pool_size candidate incomes and select n_landlords of them,
    weighted by the SCF curve (pool_size is raised to n_landlords if smaller)."""
    if n_landlords <= 0:
        return np.array([])
    pool_size = max(pool_size, n_landlords)
    candidates = rng.lognormal(
        mean=income_lognormal_mean, sigma=income_lognormal_sigma, size=pool_size
    )
    weights = np.array(
        [
            landlord_selection_weight(
                c, decile_probs, income_lognormal_mean, income_lognormal_sigma
            )
            for c in candidates
        ]
    )
    weights = weights / weights.sum()
    idx = rng.choice(pool_size, size=n_landlords, replace=False, p=weights)
    return candidates[idx]
