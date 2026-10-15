"""Base class for household agents: EQ 1 desired balance and EQ 2 consumption."""

from mesa import Agent

from housing_abm.equations.wealth import desired_bank_balance, monthly_consumption


class HouseholdAgent(Agent):
    def __init__(self, model, income: float, age: int, tract_id: str):
        super().__init__(model)
        self.income = income
        self.age = age
        self.tract_id = tract_id
        self.bank_balance = 0.0
        self.desired_balance = desired_bank_balance(
            income=self.income,
            alpha=-30.0,
            beta=4.0,
            epsilon_std=0.3,
            rng=self.model.random_gen,
        )  # placeholder until the first refresh_desired_balance()
        self.status = "social_housing"  # or "renting" / "owning"
        self.house = None
        self.owned_since_month = None
        self.bridge_loan = 0.0  # down payment financed against unsold equity

    def refresh_desired_balance(self):
        """EQ 1, re-drawn each step with fresh noise."""
        params = self.model.params["wealth_eq1"][self.WEALTH_KEY]
        self.desired_balance = desired_bank_balance(
            income=self.income,
            alpha=params["alpha"],
            beta=params["beta"],
            epsilon_std=params["epsilon_std"],
            rng=self.model.random_gen,
        )

    def apply_consumption(self, housing_cost: float):
        """EQ 2, applied after rent/mortgage is paid."""
        alpha = self.model.params["consumption_eq2"]["alpha"]
        if housing_cost is None:
            raise ValueError(
                f"housing_cost is None for {self.__class__.__name__}, "
                f"house={self.house}"
                f"cost = {housing_cost}"
                f"status = {self.status}"
            )

        net_inflow = self.bank_balance + self.income - housing_cost
        consumption = monthly_consumption(net_inflow, self.desired_balance, alpha)
        self.bank_balance = net_inflow - consumption
        self.model.prevent_bankruptcy(self)
        return consumption

    def essential_consumption(self) -> float:
        """Fixed monthly subsistence spend, subtracted to get disposable income."""
        return self.model.params.get("household_budget", {}).get(
            "essential_consumption_monthly", 0.0
        )

    def step(self):
        raise NotImplementedError("implement in each subclass")
