"""Small-landlord selection weighted by real income-percentile propensity.

Grounded in the 2022 Survey of Consumer Finances (docs/methodology.md
Section 11, scripts/calibrate_scf.py): P(household owns other residential
real estate) rises monotonically from ~3% at the bottom income decile to
~60% at the top -- the direct US analogue of the reference Java model's
WAS-derived BTL propensity curve. Unlike decideLTV (Section 10c), the
reference's equivalent mechanism is genuinely live:
BTLProbability.getBinAt(incomePercentile) runs in every household's
constructor in HouseholdBehaviour.java, not commented out.

Institutional investors are out of scope for this module -- SCF surveys
individual households, not LLCs/funds, so it can only ground "mom and pop"
landlord selection (SmallLandlord), not institutional counts.

This replaces the previous mechanism -- landlord income drawn from its own
disconnected, hand-picked lognormal ("landlords skew higher-income than
renters") -- with landlord incomes drawn from the same household income
distribution as everyone else, then *selected* with probability weighted
by the real income-percentile curve. The realistic income skew now falls
out of who gets selected, rather than being assumed as a separate
distribution.
"""

import numpy as np
from scipy.stats import lognorm


def landlord_selection_weight(
    income: float,
    decile_probs: list[float],
    income_lognormal_mean: float,
    income_lognormal_sigma: float,
) -> float:
    """P(owns other residential real estate) for a household at this income's
    percentile within the household income distribution, per the SCF decile
    curve. decile_probs must have one entry per equal-width percentile bin
    (10 for deciles)."""
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
    """Probability that a newborn household with this income gets the
    SmallLandlord gene, calibrated so the population-level fraction
    matches target_fraction in expectation.

    Uses the same SCF curve as the init-time weighted selection, scaled
    by target_fraction / mean(decile_probs) so the integral over the
    income distribution equals the target.
    """
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
    """Draw n_landlords incomes from the household income distribution,
    selected without replacement and weighted by the real income-percentile
    landlord-propensity curve.

    pool_size is the number of candidate incomes drawn before selection;
    must be >= n_landlords (raised to n_landlords if given smaller).
    """
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
