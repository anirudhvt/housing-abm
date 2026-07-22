"""Renter agent, transitions to FTB when saved enough
exits neighborhood if rent burden too high (threshold)"""

from .base import HouseholdAgent


class Renter(HouseholdAgent):

    WEALTH_KEY = "renter"

    def __init__(
        self, model, income, age, tract_id, rent_burden_exit_threshold: float = 0.30
    ):
        super().__init__(model, income, age, tract_id)
        self.rent_burden_exit_threshold = rent_burden_exit_threshold
        # self.status = "renting" status starts at social housing, turns into renter when run_rental_market assigns house
        self.lease_months_remaining = 0

    def step(self):
        # pay rent, accumulate savings (EQ 1/2), rent burden
        # if lease end -> social housing -> buy/rent decision (EQ 5)

        self.refresh_desired_balance()  # reset using EQ 1

        if self.house is not None:  # not in social housing
            monthly_rent = self.house.rent
            self.apply_consumption(housing_cost=monthly_rent)

            # rent burden check using given threshold
            if monthly_rent / self.income > self.rent_burden_exit_threshold:
                self.model.exit_tract(self)  # exits if unaffordable
                return

            self.lease_months_remaining -= 1
            if self.lease_months_remaining > 0:
                return  # we don't model leaving leases early

            # otherwise, lease has ended, returning to social housing
            self.house.tenant = None
            self.house.on_rental_market = True  # relist house

            self.house = None
            self.status = "social_housing"
        else:  # not housed yet
            self.apply_consumption(housing_cost=0)  # no housing cost if not housed

        if self.status == "social_housing":
            # check if enough savings for downpayment on affordable house
            # actual bidding is handled in model.py
            # may promote agent to FTB or another rental bid
            self.model.queue_housing_decision(self)
