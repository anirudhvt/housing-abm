import numpy as np
from housing_abm.equations.selling import p_sell, asking_price, price_reduction

#p_sell EQ6

def test_p_sell_matches_long_run_average_at_baseline_conditions():
    # when n_h == n_h_avg and i_current == i_avg, base term is exactly 1,
    # so P(sell) should be exactly 1/(12*tenure_years) - the long-run average
    p = p_sell(
        tenure_years=11,
        n_h=0.05,
        n_h_avg=0.05,
        i_current=0.04,
        i_avg=0.04,
        alpha=4.0,
        beta=5.0,
    )
    assert np.isclose(p, 1 / (12 * 11))


def test_p_sell_floors_at_zero():
    # extreme oversupply should never give a negative probability
    p = p_sell(
        tenure_years=11,
        n_h=100,
        n_h_avg=0.01,
        i_current=0.04,
        i_avg=0.04,
        alpha=4.0,
        beta=5.0,
    )
    assert p == 0.0


def test_p_sell_lock_in_effect_reduces_probability_when_mortgage_rate_is_low():
    # golden-handcuffs term: a mortgage well below current rates should make
    # selling less likely 
    base = p_sell(
        tenure_years=11,
        n_h=0.05,
        n_h_avg=0.05,
        i_current=0.07,
        i_avg=0.07,
        alpha=4.0,
        beta=5.0,
        i_mortgage=None,
    )
    locked_in = p_sell(
        tenure_years=11,
        n_h=0.05,
        n_h_avg=0.05,
        i_current=0.07,
        i_avg=0.07,
        alpha=4.0,
        beta=5.0,
        i_mortgage=0.03,
        gamma=1.0,
    )
    assert locked_in < base


#asking price EQ 7

def test_asking_price_matches_formula_with_zero_noise(zero_rng):
    p_bar, f_bar, alpha, beta, zeta = 250_000, 30, 0.04, 0.011, 1.0 / 31.0
    price = asking_price(p_bar, f_bar, alpha, beta, zeta, epsilon_std=0.5, rng=zero_rng)
    expected = np.exp(alpha + np.log(p_bar) - beta * np.log(zeta * (1 + f_bar)))
    assert np.isclose(price, expected)


def test_asking_price_is_always_positive(rng):
    # exp() by construction - regardless of how extreme the noise draw is
    for _ in range(500):
        price = asking_price(
            p_bar_tract=200_000,
            f_bar_tract=45,
            alpha=0.04,
            beta=0.011,
            zeta=1.0 / 31.0,
            epsilon_std=2.0,
            rng=rng,
        )
        assert price > 0


# price reduction eq 8


def test_price_reduction_never_goes_negative(rng):
    for _ in range(2000):
        price = rng.uniform(1, 500_000)
        result = price_reduction(price, reduction_prob=1.0, epsilon_std=2.0, rng=rng)
        assert result >= 0.0


def test_price_reduction_is_always_a_genuine_reduction_when_triggered(rng):
    # whenever a reduction fires, the new price must be <= the old price -
    # never an increase, never a sign flip
    for _ in range(500):
        price = rng.uniform(1000, 500_000)
        result = price_reduction(price, reduction_prob=1.0, epsilon_std=1.0, rng=rng)
        assert result <= price


def test_price_reduction_does_nothing_when_not_triggered(never_trigger_rng):
    price = 250_000
    result = price_reduction(
        price, reduction_prob=0.15, epsilon_std=0.08, rng=never_trigger_rng
    )
    assert result == price


def test_price_reduction_vacancy_multiplier_increases_trigger_probability(
    make_fixed_rng,
):
    # a rng that triggers only for prob >= 0.2 should trigger under the
    # investor multiplier (0.15*1.5=0.225) but not without it (0.15)
    borderline_rng = make_fixed_rng(random_value=0.2)
    without_multiplier = price_reduction(
        100_000,
        reduction_prob=0.15,
        epsilon_std=0.05,
        rng=borderline_rng,
        vacancy_tax_multiplier=1.0,
    )
    with_multiplier = price_reduction(
        100_000,
        reduction_prob=0.15,
        epsilon_std=0.05,
        rng=borderline_rng,
        vacancy_tax_multiplier=1.5,
    )
    assert without_multiplier == 100_000  # 0.2 >= 0.15, doesn't trigger
    assert with_multiplier < 100_000  # 0.2 < 0.225, triggers
