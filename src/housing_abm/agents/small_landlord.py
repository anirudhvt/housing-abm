#2-10 units, rental-yield focused (delta = 0.3)
#rent stickiness on lease renewal
#cash purchase if wealth > 2x price
from .base import HouseholdAgent

class SmallLandlord(HouseholdAgent):
    DELTA = 0.3

    def __init__(self, unique_id, model, income, age, tract_id):
        super().__init__(unique_id, model, income, age, tract_id)
        self.properties = [] #housingunit owned

    def step(self):
        #EQ 9: Expected yield, EQ 10 p_buy
        #EQ 11: rent with stickiness, EQ 12/13 sell decision
        #EQ 18: down_payment_investor(agent_type = 'small landlord')
        return