import numpy as np
from housing_abm.equations.mortgage import (
    max_loan_owner_occupier,
    passes_investor_dscr,
    down_payment_owner,
    down_payment_investor,
)

#max loan for owner occupier 

def test_max_loan_is_the_binding_ltv_constraint_when_lowest():
    # tiny bank balance -> LTV constraint should bind vs a generous income
    loan = max_loan_owner_occupier(
        bank_balance=1000,
        disposable_income=50000,
        chi_max_ltv=0.9,
        dti_front=0.31,
        i_r_monthly=0.03 / 12,
        term_months=360,
    )
    ltv_constraint = 1000 * 0.9 / (1 - 0.9)
    assert np.isclose(loan, ltv_constraint)


def test_max_loan_is_the_binding_affordability_constraint_when_lowest():
    # huge bank balance, tiny income -> affordability constraint should bind
    loan = max_loan_owner_occupier(
        bank_balance=10_000_000,
        disposable_income=100,
        chi_max_ltv=0.9,
        dti_front=0.31,
        i_r_monthly=0.03 / 12,
        term_months=360,
    )
    annuity_factor = (1 - (1 + 0.03 / 12) ** -360) / (0.03 / 12)
    affordability_constraint = 100 * 0.31 * annuity_factor
    assert np.isclose(loan, affordability_constraint)


def test_max_loan_respects_optional_lti_cap():
    loan_uncapped = max_loan_owner_occupier(
        bank_balance=10_000_000,
        disposable_income=100_000,
        chi_max_ltv=0.9,
        dti_front=0.9,
        i_r_monthly=0.03 / 12,
        term_months=360,
    )
    loan_capped = max_loan_owner_occupier(
        bank_balance=10_000_000,
        disposable_income=100_000,
        chi_max_ltv=0.9,
        dti_front=0.9,
        i_r_monthly=0.03 / 12,
        term_months=360,
        lti_max=3.5,
        income=50_000,
    )
    assert loan_capped < loan_uncapped
    assert np.isclose(loan_capped, 3.5 * 50_000)


def test_max_loan_ignores_lti_cap_if_income_not_given():
    # lti_max alone (without income) shouldn't blow up 
    loan = max_loan_owner_occupier(
        bank_balance=10_000_000,
        disposable_income=100_000,
        chi_max_ltv=0.9,
        dti_front=0.9,
        i_r_monthly=0.03 / 12,
        term_months=360,
        lti_max=3.5,
        income=None,
    )
    loan_no_lti = max_loan_owner_occupier(
        bank_balance=10_000_000,
        disposable_income=100_000,
        chi_max_ltv=0.9,
        dti_front=0.9,
        i_r_monthly=0.03 / 12,
        term_months=360,
    )
    assert np.isclose(loan, loan_no_lti)


#  passes_investor_dscr (EQ15)


def test_dscr_fails_when_loan_exceeds_ltv_cap():
    passes = passes_investor_dscr(
        bank_balance=10_000,
        expected_annual_rent_yield=0.06,
        xi_icr=1.25,
        i_btl_monthly=0.005,
        proposed_loan=1_000_000,
        chi_max_ltv=0.75,
    )
    assert passes is False


def test_dscr_passes_for_a_small_conservative_loan():
    #
    passes = passes_investor_dscr(
        bank_balance=100_000,
        expected_annual_rent_yield=0.05,
        xi_icr=1.25,
        i_btl_monthly=0.005,
        proposed_loan=50_000,
        chi_max_ltv=0.75,
    )
    assert passes is True


def test_dscr_fails_when_yield_too_thin_to_cover_icr_buffer():
    # 
    passes = passes_investor_dscr(
        bank_balance=100_000,
        expected_annual_rent_yield=0.08,
        xi_icr=1.25,
        i_btl_monthly=0.005,
        proposed_loan=10_000,
        chi_max_ltv=0.75,
    )
    assert passes is False




def test_down_payment_owner_below_cutoff_returns_exact_minimum(rng):
    price = 200_000
    d = down_payment_owner(
        price=price,
        income_rank=10,
        income_cutoff=50,
        d_minimum_pct=0.035,
        lognorm_m=-1.6,
        lognorm_s=0.5,
        rng=rng,
    )
    assert np.isclose(d, 0.035 * price)


def test_down_payment_owner_above_cutoff_is_at_least_the_minimum(rng):
    price = 300_000
    for _ in range(50):
        d = down_payment_owner(
            price=price,
            income_rank=90,
            income_cutoff=50,
            d_minimum_pct=0.2,
            lognorm_m=-1.2,
            lognorm_s=0.5,
            rng=rng,
        )
        assert d >= 0.2 * price - 1e-9


#down pamynet investor EQ 18

def test_investor_down_payment_small_landlord_pays_cash_when_wealthy(rng):
    price = 200_000
    d, is_cash = down_payment_investor(
        price=price,
        wealth=500_000,
        agent_type="small_landlord",
        mu=0.25,
        sigma=0.05,
        p_cash=0.0,
        d_minimum_pct=0.2,
        rng=rng,
    )
    assert is_cash is True
    assert d == price


def test_investor_down_payment_small_landlord_finances_when_not_wealthy(rng):
    price = 400_000
    d, is_cash = down_payment_investor(
        price=price,
        wealth=100_000,
        agent_type="small_landlord",
        mu=0.25,
        sigma=0.05,
        p_cash=0.0,
        d_minimum_pct=0.2,
        rng=rng,
    )
    assert is_cash is False
    assert d >= 0.2 * price - 1e-9


def test_investor_down_payment_institutional_respects_p_cash(
    always_trigger_rng, never_trigger_rng
):
    price = 300_000
    _, is_cash_always = down_payment_investor(
        price=price,
        wealth=1_000_000,
        agent_type="institutional",
        mu=0.3,
        sigma=0.05,
        p_cash=0.5,
        d_minimum_pct=0.25,
        rng=always_trigger_rng,
    )
    _, is_cash_never = down_payment_investor(
        price=price,
        wealth=1_000_000,
        agent_type="institutional",
        mu=0.3,
        sigma=0.05,
        p_cash=0.5,
        d_minimum_pct=0.25,
        rng=never_trigger_rng,
    )
    assert is_cash_always is True
    assert is_cash_never is False
