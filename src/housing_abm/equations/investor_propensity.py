"""SCF-grounded investor selection: households become landlords based on
income percentile, not as extra appended agents.

The reference paper (Baptista et al. 2016) gives each household a "BTL gene"
probability of becoming a buy-to-let investor, conditioned on income
percentile. We use SCF 2022 data to calibrate this curve for Atlanta."""

import numpy as np


SCF_DECILE_PROBS = [
    0.02, 0.03, 0.04, 0.05, 0.07,
    0.09, 0.12, 0.16, 0.22, 0.35,
]


def landlord_selection_weight(
    income: float,
    all_incomes: list[float],
    decile_probs: list[float] | None = None,
) -> float:
    """Map a household's income to its probability of becoming an investor,
    via its percentile rank within the income distribution."""
    if decile_probs is None:
        decile_probs = SCF_DECILE_PROBS
    if not all_incomes:
        return decile_probs[4]
    rank = sum(1 for inc in all_incomes if inc <= income) / len(all_incomes)
    decile = min(int(rank * 10), 9)
    return decile_probs[decile]


def select_future_landlords(
    households: list,
    n_target: int,
    rng: np.random.Generator,
    decile_probs: list[float] | None = None,
) -> list:
    """Select n_target households to convert into landlords, weighted by
    income-percentile propensity. Returns the selected household agents."""
    if not households or n_target <= 0:
        return []
    incomes = [h.income for h in households]
    weights = np.array([
        landlord_selection_weight(h.income, incomes, decile_probs)
        for h in households
    ])
    total = weights.sum()
    if total == 0:
        return []
    probs = weights / total
    n_select = min(n_target, len(households))
    indices = rng.choice(len(households), size=n_select, replace=False, p=probs)
    return [households[i] for i in indices]
