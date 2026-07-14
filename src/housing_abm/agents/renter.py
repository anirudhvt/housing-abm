"""Renter agent, transitions to FTB when saved enough
exits neighborhood if rent burden too high (threshold) """
from .base import HouseholdAgent


class Renter(HouseholdAgent):
    def __init__(self, model, income, age, tract_id, rent_burden_exit_threshold: float = 0.30):
        super().__init__( model, income, age, tract_id)
        self.rent_burden_exit_threshold = rent_burden_exit_threshold
        self.status = "renting"


    def step(self):
        #pay rent, accumulate savings (EQ 1/2), rent burden
        #if lease end -> social housing -> buy/rent decision (EQ 5)

        return