"""EQ 14 (owner-occupier max loan), EQ 15 (investor DSCR/ICR),
EQ 17 (owner down payment), EQ 18 (investor down payment)."""

import numpy as np
from scipy.stats import lognorm


def max_loan_owner_occupier(
    bank_balance: float,
    disposable_income: float,
    chi_max_ltv: float,
    dti_front: float,
    i_r_monthly: float,
    term_months: int,
    lti_max: float | None = None,
    income: float | None = None,
) -> float:
    """EQ 14: min(LTV bound, DTI affordability bound, optional LTI bound).
    Pass both lti_max and income to apply the LTI bound."""

    ltv_constraint = bank_balance * chi_max_ltv / (1 - chi_max_ltv)
    annuity_factor = (1 - (1 + i_r_monthly) ** (-term_months)) / i_r_monthly
    affordability_constraint = disposable_income * dti_front * annuity_factor

    constraints = [ltv_constraint, affordability_constraint]
    if lti_max is not None and income is not None:
        lti_constraint = lti_max * income
        constraints.append(lti_constraint)

    return min(constraints)


def passes_investor_dscr(
    bank_balance: float,
    expected_annual_rent_yield: float,
    xi_icr: float,
    i_btl_monthly: float,
    proposed_loan: float,
    chi_max_ltv: float,
) -> bool:
    """EQ 15 (US DSCR analogue): True if the loan passes both the ICR buffer
    (xi_icr) and the investor LTV cap."""

    icr_bound = bank_balance / (
        1 - expected_annual_rent_yield / (xi_icr * i_btl_monthly * 12)
    )
    passes_icr = proposed_loan <= icr_bound
    passes_ltv = proposed_loan <= bank_balance * chi_max_ltv / (1 - chi_max_ltv)
    return passes_icr and passes_ltv


def down_payment_owner(
    price: float,
    income_rank: float,
    income_cutoff: float,
    d_minimum_pct: float,
    lognorm_m: float,
    lognorm_s: float,
    rng: np.random.Generator,
) -> float:
    """EQ 17: buyers at or below income_cutoff pay the minimum; others draw
    from a log-normal fit to above-floor loans."""

    if income_rank <= income_cutoff:
        return d_minimum_pct * price
    draw = lognorm.rvs(lognorm_s, scale=np.exp(lognorm_m), random_state=rng)
    return max(d_minimum_pct * price, draw * price)


def down_payment_investor(
    price: float,
    wealth: float,
    agent_type: str,
    mu: float,
    sigma: float,
    p_cash: float,
    d_minimum_pct: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    """EQ 18. Returns (down_payment, is_cash). Small landlords pay cash if
    wealth > 2x price; institutional investors pay cash with probability p_cash."""

    if agent_type == "small_landlord":
        if wealth > 2 * price:
            return price, True
        draw = rng.normal(mu, sigma)
        return max(d_minimum_pct * price, draw * price), False
    elif agent_type == "institutional":
        if rng.random() < p_cash:
            return price, True
        draw = rng.normal(mu, sigma)
        return max(d_minimum_pct * price, draw * price), False


def estimate_floor_share_and_fit(
    down_payment_pcts: np.ndarray, floor_band: float = 0.05
) -> dict:
    """Calibration helper: share of loans at the floor, plus a log-normal fit
    to the rest. Returns {p_floor, lognorm_m, lognorm_s}."""
    at_floor = down_payment_pcts <= floor_band
    p_floor = at_floor.mean()
    above_floor = down_payment_pcts[~at_floor]
    shape, loc, scale = lognorm.fit(above_floor, floc=0)

    return {
        "p_floor": float(p_floor),
        "lognorm_m": float(np.log(scale)),
        "lognorm_s": float(shape),
    }
