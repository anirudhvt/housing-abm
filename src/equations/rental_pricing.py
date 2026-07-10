"""EQ 11: rental pricing, differentiated by landlord type (adaption)"""

import numpy as np

def small_landlord_rent(r_bar_tract: float, f_bar_tract: float, alpha: float,
                        beta: float, zeta:float, epsilon_std: float,
                        reprice_prob:float, previous_rent:float | None, 
                        rng: np.random.Generator) -> float: 
    """ Small landlords: EQ 11, but chance of stickiness - reprice with reprice_prob
    otherwise relist at previous rent
    """
    if previous_rent is not None and rng.random() >= reprice_prob:
        return previous_rent
    # Calculate new rent based on the tract-level average and other parameters
    epsilon= rng.normal(0, epsilon_std)
    ln_q = alpha + np.log(r_bar_tract) - beta*np.log(zeta*(1+f_bar_tract))+epsilon
    return np.exp(ln_q)

def institutional_rent(market_rate: float, premium: float = 1.04) -> float:
    """Institutional: always market rate + premium (~4%)
    skip noise of EQ 11 - no stickiness, due to algorithmic tools (RealPage)"""
    return market_rate*premium