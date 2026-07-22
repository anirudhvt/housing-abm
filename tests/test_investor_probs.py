from housing_abm.equations.investor_probs import p_buy_investor, p_sell_investor


def test_p_buy_investor_policy_blocked_forces_zero():
    p = p_buy_investor(omega=10.0, beta=5.0, policy_blocked=True)
    assert p == 0.0


def test_p_buy_investor_increases_with_omega():
    low = p_buy_investor(omega=-1.0, beta=3.0)
    high = p_buy_investor(omega=1.0, beta=3.0)
    assert high > low


def test_p_buy_investor_bounded_between_zero_and_one():
    for omega in (-100, -1, 0, 1, 100):
        p = p_buy_investor(omega=omega, beta=3.0)
        assert 0.0 <= p <= 1.0


def test_p_sell_investor_forced_divestiture_forces_one():
    p = p_sell_investor(psi=10.0, beta=5.0, forced_divesture=True)
    assert p == 1.0


def test_p_sell_investor_decreases_with_psi():
    # higher effective yield from holding -> lower probability of selling
    low_yield = p_sell_investor(psi=-1.0, beta=3.0)
    high_yield = p_sell_investor(psi=1.0, beta=3.0)
    assert high_yield < low_yield
