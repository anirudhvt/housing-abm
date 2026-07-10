"""EQ 9 (expected yield, buy), EQ 12: (effective yield, sell)
delta: small_landlord delta = 0.3, institutional investor delta = 0.6 - investors value long term yield
Policy costs subtract directly from each yield 
"""



def expected_yield_buy(price: float, down_payment: float, delta:float,
                       g:float, kappa: float, r_bar: float,
                       monthly_mortgage: float, policy_cost: float = 0.0) -> float:
    '''EQ 9: omega = (p/d)*(delta*(g+kappa)+(1-delta)*r_bar) - m/d - policy_cost
    down_payment == price when purchase is all cash (m is zero)
    p/d: leverage (price/down_payment)
    delta: weight on capital yield
    g: estimation of monthly house price growth (eq 4)
    kappa: long-term average gross yield
    r_bar: current avg gross yield
    m/d: mortgage rate (mortgage/down_payment)'''

    leverage = price / down_payment if down_payment != 0 else 0
    omega = leverage * (delta * (g + kappa) + (1 - delta) * r_bar)
    omega -= monthly_mortgage / down_payment if down_payment != 0 else 0
    omega -= policy_cost
    return omega

def effective_yield_sell(price:float, equity: float, delta:float, 
                         g:float, kappa: float, r_bar: float,
                         monthly_mortgage: float, policy_cost: float = 0.0) -> float:
        '''EQ 12: same as eq 9 but with equity instead of down_payment'''
        equity = max(equity, 1e-6) #equity >0
        leverage = price/equity
        psi = leverage * (delta * (g + kappa) + (1 - delta) * r_bar)
        psi -= monthly_mortgage / equity 
        psi -= policy_cost
        return psi

