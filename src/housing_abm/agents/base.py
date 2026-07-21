#base code for all household agents
from mesa import Agent

from housing_abm.equations.wealth import desired_bank_balance, monthly_consumption

class HouseholdAgent(Agent):
    def __init__(self, model, income:float, age:int, tract_id: str):
        super().__init__(model)
        self.income = income
        self.age = age
        self.tract_id = tract_id
        self.bank_balance = 0.0
        self.desired_balance = desired_bank_balance(income=self.income, alpha=-30.0, beta= 4.0, epsilon_std=0.3, rng=self.model.random_gen) #eq 1, updated periodically
        self.status = "social_housing" # or renting or owning
        self.house = None #reference to HousingUnit, if existing
        self.owned_since_month = None #month they became an owner, for repeat buyers
        self.bridge_loan = 0.0 #down payment financed against unrealized equity, repaid when old house is sold


    def refresh_desired_balance(self):
        """EQ 1: re-drawn each step - noise varies over time, "
        income - wealth relationship stays fixed per agent type"""

        #grabbing relevant parameters from the passed file
        params = self.model.params["wealth_eq1"][self.WEALTH_KEY]
        self.desired_balance = desired_bank_balance(
            income=self.income, alpha = params["alpha"], 
            beta = params["beta"], epsilon_std = params["epsilon_std"],
            rng = self.model.random_gen
        )
    
    def apply_consumption(self, housing_cost: float):
        """EQ 2: rent/mortgage netted out before consumption"""
        alpha = self.model.params["consumption_eq2"]["alpha"]
        if housing_cost is None:
            print(
                self.__class__.__name__,
                self.status,
                self.house,
                housing_cost
            )
            raise ValueError(
                f"housing_cost is None for {self.__class__.__name__}, "
                f"house={self.house}"
                f"cost = {housing_cost}"
                f"status = {self.status}"
            )
        

        #available money
        net_inflow = self.bank_balance + self.income - housing_cost
        #use eq 2 from the equations folder
        consumption = monthly_consumption(net_inflow, self.desired_balance, alpha)
        self.bank_balance = net_inflow - consumption #update bank balance
        self.model.prevent_bankruptcy(self) #inject cash
        return consumption
    
    def essential_consumption(self) -> float:
        """fixed subsistence floor, netted out before computing disposable income
        config/baseline_params.yaml household_budget.essential_consumption_monthly
        for calibration and current plcaeholder"""
        return self.model.params.get("household_budget", {}).get("essential_consumption_monthly", 0.0)
    
    def step(self):
        raise NotImplementedError("implement in each subclass")