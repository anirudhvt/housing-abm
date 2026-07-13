"""EQ 3 (desired expenditure), EQ 4: (price appreciation expectatoins)
Adaptions from Baptista et al. (2016): 
- investors subtitute income for capital
- alpha and g are conmputed at census tract level
- alpha_institutional > alpha_household (more deterministic)"""

import numpy as np

def desired_expenditure(income_or_capital: float, g:float, alpha:float, 
                        beta:float, epsilon_std:float, 
                        rng: np.random.Generator,
                        mortgage_cap: float |None = None) -> float:
    """Eq 3: p_desired = alpha * y * exp(epsilon) / (1-beta*g)
    
    'income or capital' is household income for households and available
    fund capital for investors 
    'mortgage cap,' if provided, caps household's maximum loan + down payment (EQ 14)
    """
    epsilon = rng.normal(0, epsilon_std)
    p_desired = alpha * income_or_capital * np.exp(epsilon) / (1 - beta * g)
    if mortgage_cap is not None:
        p_desired = min(p_desired, mortgage_cap)
    return p_desired 


def price_appreciation_expectation(hpi_history: list[float], alpha:float) -> float:
    """EQ 4: g_t = alpha*[(h_t-1+h_t-2+h_t-3)/(h_t-13+h_t-14+h_t-15)-1]."""