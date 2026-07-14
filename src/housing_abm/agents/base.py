#base code for all household agents
from mesa import Agent

class HouseholdAgent(Agent):
    def __init__(self, model, income:float, age:int, tract_id: str):
        super().__init__(model)
        self.income = income
        self.age = age
        self.tract_id = tract_id
        self.bank_balance = 0.0
        self.desired_balance = 0.0 #eq 1, updated periodically
        self.status = "social housing" # or renting or owning
        self.house = None #referencce to HousingUnit, if existing

    