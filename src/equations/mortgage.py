"""Eq 14 (owner-occupier mortgage), EQ 15 (investor DSCR/ICR),
 EQ 17 (owner down payment), EQ 18 (investor down payment)
 
 """

import numpy as np
from scipy.stats import lognorm

def max_loan_owner_occupier(bank_balance: float, disposable_income: float,
                            chi_max_ltv: float, dti_front: float,
                            i_r_monthly: float, term_months: int,
                            lti_max: float | None = None,
                            income: float | None = None) -> float:
    """EQ 14: q = min(b*chi/(1-chi), y*psi (optional), yd * nu * annuity_factor)
    chi_max_ltv is LTV Cap (0.965 for FHA, for example)
    lti_max/income: pass both to reinstate lti constraint"""

    ltv_constraint = bank_balance * chi_max_ltv/(1-chi_max_ltv)
    annuity_factor = (1-(1+i_r_monthly)**(-term_months)) / i_r_monthly
    affordability_constraint = disposable_income * dti_front * annuity_factor

    constraints = [ltv_constraint, affordability_constraint]
    if lti_max is not None and income is not None: #using lti constraint
        lti_constraint = lti_max * income
        constraints.append(lti_constraint)

    return min(constraints)

def passes_investor_dscr(bank_balance: float, expected_annual_rent_yield: float,
                          xi_icr: float, i_btl_monthly: float,
                          proposed_loan: float, chi_max_ltv: float) -> bool:
    """EQ 15 analog: Us DSCR for investor mortgages
    expected_annual_rent_yield: gross annual rental yield (annual rent/price)
    returns True if loan passes DSCR/ICR buffer and LTV cap for investor loans
    
    xi_icr: ICR constraint applied by the bank
    i_btl_monthly: mortgage interest rate
    - provides buffer so that borrowers can still afford mortgage if costs increase    

    """

    icr_bound = bank_balance/(1-expected_annual_rent_yield/(xi_icr*i_btl_monthly))
    passes_icr = proposed_loan <= icr_bound
    passes_ltv = proposed_loan <= bank_balance * chi_max_ltv / (1 - chi_max_ltv)
    return passes_icr and passes_ltv

def down_payment_owner(price: float, income_rank: float, income_cutoff: float,
                        d_minimum_pct: float, lognorm_m: float, lognorm_s: float,
                        rng: np.random.Generator) -> float:
    """EQ 17: two-part mixture - down payments cluster at FHA floor
    income_rank and income_cutoff on same scale - below cutoff gets d_minimum
    agents above cutoff draw from log-normal fit to ABOVE FLOOR HMDA observations
    
    d_minimum_pct: minimum down payment percentage
    """

    if income_rank <= income_cutoff:
        return d_minimum_pct * price
    draw = lognorm.rvs(lognorm_s, scale = np.exp(lognorm_m), random_state = rng)
    return max(d_minimum_pct * price, draw * price)

def down_payment_investor(price: float, wealth: float, agent_type: str,
                           mu: float, sigma: float, p_cash: float,
                           d_minimum_pct: float,
                           rng: np.random.Generator) -> tuple[float, bool]:
    """EQ 18 split by agent type. Returns (down_payment, is cash purchase)
    
    small_landlord - Baptista et. al: cash if wealth > 2x price
    institutional: empirical p_cash from Redfin data"""

    if agent_type == "small_landlord":
        if wealth > 2 * price:
            return price, True #paying full price in cash
        draw = rng.normal(mu, sigma)
        return max(d_minimum_pct * price, draw * price), False #otherwise draw from distribution above min
    elif agent_type == "institutional":
        if rng.random() < p_cash:
            return price, True
        draw = rng.normal(mu, sigma)
        return max(d_minimum_pct * price, draw * price), False
    
def estimate_floor_share_and_fit(down_payment_pcts: np.ndarray, 
                                 floor_band: float = 0.05) -> dict:
    """helper for scripts: given array of down-payment percentages, estimate p_floor (share at minimum),
    fit a log-normal to the above floor sample. returns dict with p_floor, lognorm_m, lognorm_s"""

    at_floor = down_payment_pcts <= floor_band #true when < 5%
    p_floor = at_floor.mean() #fraction of loans near the floor
    above_floor = down_payment_pcts[~at_floor] #restricted subsample
    shape, loc, scale = lognorm.fit(above_floor, floc=0)

    return {"p_floor": float(p_floor), "lognorm_m": float(np.log(scale)), "lognorm_s": float(shape)}
