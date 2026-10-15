"""Renter: becomes a first-time buyer once it can afford to buy, and leaves the
tract if rent exceeds the burden threshold."""

from .base import HouseholdAgent


class Renter(HouseholdAgent):

    WEALTH_KEY = "renter"

    def __init__(
        self, model, income, age, tract_id, rent_burden_exit_threshold: float = 0.30
    ):
        super().__init__(model, income, age, tract_id)
        self.rent_burden_exit_threshold = rent_burden_exit_threshold
        self.lease_months_remaining = 0

    def step(self):
        self.refresh_desired_balance()

        if self.house is not None:
            monthly_rent = self.house.rent
            self.apply_consumption(housing_cost=monthly_rent)

            if monthly_rent / self.income > self.rent_burden_exit_threshold:
                self.model.exit_tract(self)
                return

            self.lease_months_remaining -= 1
            if self.lease_months_remaining > 0:
                return  # leases are never broken early

            self.model.end_tenancy(self.house)
            self.house = None
            self.status = "social_housing"
        else:
            self.apply_consumption(housing_cost=0)

        if self.status == "social_housing":
            # promotes to first-time buyer or queues a rental bid
            self.model.queue_housing_decision(self)
