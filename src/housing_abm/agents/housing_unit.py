"""Passive housing unit agent - holds state"""
from mesa import Agent


class HousingUnit(Agent):
    def __init__(self, model, tract_id: str, quality:float):
        super().__init__(model)
        self.tract_id = tract_id
        self.quality = quality
        self.owner = None    #owning agent
        self.tenant = None    #renting agent, if rented
        self.price = None      #current sale price, if listed
        self.rent = None       #current rent if listed
        self.days_on_market = 0
        self.day_vacant = 0
        self.on_sale_market = False
        self.on_rental_market = False
        self.mortgage_principal = 0.0
        self.mortgage_payment = 0.0
        self.mortgage_rate = None

    def step(self):
        pass


