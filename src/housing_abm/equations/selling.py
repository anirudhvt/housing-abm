"""EQ 6: sell probability, EQ 7: asking price, EQ 8: price reduction"""
import numpy as np

def p_sell(tenure_years: float, n_h: float, n_h_avg: float,
           i_current: float, i_avg: float, alpha: float, beta: float,
           i_mortgage: float | None = None, gamma: float = 0.0) -> float:
    """EQ 6: P(sell) = 1/12 * max(1/tenure years (1 + alpha(n_h_avg - n_h)
    +beta(i_avg - i_current)),0)
    adapted term: 'golden handcuffs' lock-in effect: gamma(i_current - i_mortgage)
    NOT in original paper, pass i_mortgage = None to disable

    tenure_years: long term selling probability (once every _ years)
    n_h/n_h_avg: number of homes per capita on the market/moving average
    i_current/i_avg: current and average interest rates
    i_mortgage: current mortgage interest rate (if applicable)
    """
    base = 1 + alpha * (n_h_avg - n_h) + beta * (i_avg - i_current)
    if i_mortgage is not None:
        base += gamma * (i_current - i_mortgage)
    return 1/12 * max((1/tenure_years) * base, 0.0)

def asking_price(p_bar_tract:float, f_bar_tract:float, alpha:float, 
                 beta:float, zeta:float, epsilon_std:float,
                 rng: np.random.Generator) -> float:
    """EQ 7: ln(p_s) = alpha + ln(p_bar) - beta*ln(zeta*(1+f_bar))+epsilon
        p_bar_tract, f_bar_tract computed at CENSUS TRACT level, instead of area wide 
        
        p_bar_tract: average sold-price of homes of this quality
        f_bar_tract: average number of days on market for all house qualities
        """
    
    epsilon = rng.normal(0, epsilon_std)
    ln_ps = alpha + np.log(p_bar_tract) - beta * np.log(zeta * (1 + f_bar_tract)) + epsilon
    return np.exp(ln_ps)

def price_reduction(current_price: float, reduction_prob: float,epsilon_mean: float, 
                    epsilon_std:float, rng: np.random.Generator,
                    vacancy_tax_multiplier: float = 1.0) -> float:
    
    """EQ 8: chance of price reduction for each time step
    vacancy_tax_multiplier > 1: increases reduction probability for investor held units"""

    prob = min(reduction_prob * vacancy_tax_multiplier, 1.0)
    if rng.random() < prob:
        epsilon = rng.normal(epsilon_mean, epsilon_std)
        reduction_fraction = np.exp(epsilon)/100
        return current_price * (1 - reduction_fraction)
    return current_price