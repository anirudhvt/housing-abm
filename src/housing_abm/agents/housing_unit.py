"""Passive housing unit agent; holds state only."""

from mesa import Agent


class HousingUnit(Agent):
    def __init__(self, model, tract_id: str, quality: float):
        super().__init__(model)
        self.tract_id = tract_id
        self.quality = quality
        self.owner = None
        self.tenant = None
        self.price = None
        self.rent = None
        # both counters advance once per step, so they count months
        self.days_on_market = 0
        self.day_vacant = 0
        # months before a vacated unit can be re-let
        self.void_months_remaining = 0
        self.on_sale_market = False
        self.on_rental_market = False
        self.mortgage_principal = 0.0
        self.mortgage_payment = 0.0
        self.mortgage_rate = None

    def step(self):
        pass
