"""Institutional Investor agent: appreciation focused (delta = 0.6)
Capital constrained rather than income, prices at market value
subject to policy restrictoins"""
from .base import HouseholdAgent

class InstitutionalInvestor(HouseholdAgent):
    def __init__(self, unique_id, model, income, age, tract_id):
        super().__init__(unique_id, model, income=0, age=None, tract_id=tract_id)
        self.available_capital = available_capital #in place of income
        self.properties = []

    def step(self):
        #EQ 3: desired expenditure
        #policy cheks
        #EQ 9 Expected yield with policy costs, before EQ 10
        #EQ 11: no stickiness, EQ 12/13 sell decision

        return