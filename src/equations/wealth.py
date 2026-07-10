"""EQ 1 (desired bank balance) and EQ 2 (consumption)"""

import numpy as np

def desired_bank_balance(income: float, alpha:float, beta:float, epsilon_std:float, rng: np.random.Generator) -> float:
    """Eq 1: ln(w) = alpha + beta*ln(y) + epsilon
    
    returns target bank balance w that the household should converge to given its income"""
    epsilon = rng.normal(0, epsilon_std)
    ln_w = alpha + beta * np.log(income) + epsilon
    return np.exp(ln_w)

def monthly_consumption(current_balance:float, desired_balance:float, alpha:float) -> float:
    """EQ 2: C = max(alpha*(b-w), 0)
    consumption occurs after rent/mortgage payment is made
    """
    return max(alpha * (current_balance - desired_balance), 0.0)

    