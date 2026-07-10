#Repeat/move-up buyer: conventional  mortgage, downpayment from equity
#Simultaneously lists current home - 'golden handcuffs'

from .base import HouseholdAgent


class RepeatBuyer(HouseholdAgent):

    LOAN_TYPE = "conventional"

    #inherited initialization
    
    def step(self):
        #EQ 6: sell decision with lock in term, EQ 7 asking price for current home
        #EQ 3/5 for new purchase, EQ 17 down payment from equity

        return