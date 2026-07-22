from housing_abm.demographics import monthly_death_probability

CFG = {
    "base_annual_rate": 0.0005,
    "age_midpoint": 95,
    "age_scale": 6,
    "mortality_scale": 1.0,
}


def annualized(age):
    p_monthly = monthly_death_probability(age, CFG)
    return 1 - (1 - p_monthly) ** 12


def test_mortality_increases_monotonically_with_age():
    ages = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    probs = [annualized(a) for a in ages]
    assert probs == sorted(probs)


def test_mortality_stays_in_plausible_range_for_working_age():
    # REGRESSION TEST: for age mortality 
    assert annualized(65) < 0.03
    assert annualized(30) < 0.01


def test_mortality_rises_substantially_at_advanced_age():
    # the hazard should still meaningfully increase in old age, or the
    # population will never turn over
    assert annualized(90) > 0.10


def test_mortality_probability_bounded():
    for age in range(0, 130, 5):
        p = monthly_death_probability(age, CFG)
        assert 0.0 <= p <= 1.0
