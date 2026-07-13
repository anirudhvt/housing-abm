"""First-time Buyer (FTB): FHA Mortgage, no equity"""
from .base import HouseholdAgent

class FirstTimeBuyer(HouseholdAgent):
    
    LOAN_TYPE = 'fha'

    #inherited initialization
    
    def step(self):
        #EQ 3: desired expenditure capped by max loan
        #EQ 5: buy vs rent, EQ 17 down payment, place bid on ownership makret
        return