"""Repeat (move-up) buyer: conventional mortgage, down payment from the equity
in the home it lists for sale."""

from housing_abm.equations.expenditure import desired_expenditure
from housing_abm.equations.mortgage import down_payment_owner, max_loan_owner_occupier
from housing_abm.equations.selling import asking_price, p_sell

from .base import HouseholdAgent


class RepeatBuyer(HouseholdAgent):

    WEALTH_KEY = "repeat_buyer"
    LOAN_TYPE = "conventional"

    def __init__(self, model, income: float, age: int, tract_id: str):
        super().__init__(model, income, age, tract_id)
        self.house_to_sell = None

    def step(self):
        self.refresh_desired_balance()

        if self.house is not None and self.status == "renting":
            # renting after an unmatched bid, possibly still carrying the old home
            monthly_rent = self.house.rent
            still_carrying_old = (
                self.house_to_sell is not None and self.house_to_sell is not self.house
            )
            if still_carrying_old:
                monthly_rent += self.house_to_sell.mortgage_payment
            self.apply_consumption(housing_cost=monthly_rent)
            if self.lease_months_remaining > 0:
                return
            self.model.end_tenancy(self.house)
            self.house = None
            self.status = "social_housing"
            available_capital = self.bank_balance
            if self.house_to_sell is not None:
                available_capital = max(
                    0, self.house_to_sell.price - self.house_to_sell.mortgage_principal
                )
            self._bid_for_next_home(available_capital=available_capital)
            return

        if self.house is None:
            self.apply_consumption(housing_cost=0)
            self._bid_for_next_home(available_capital=self.bank_balance)
            return

        monthly_payment = self.house.mortgage_payment
        still_carrying_old = (
            self.house_to_sell is not None and self.house_to_sell is not self.house
        )
        if still_carrying_old:
            monthly_payment += self.house_to_sell.mortgage_payment
        self.apply_consumption(housing_cost=monthly_payment)

        if still_carrying_old:
            return  # wait for the old home to sell

        if self.house_to_sell is None:
            # EQ 6 sell decision, with the lock-in term
            sell_cfg = self.model.params["selling_eq6"]
            prob_sell = p_sell(
                tenure_years=sell_cfg["tenure_years"],
                n_h=self.model.houses_per_capita(self.tract_id),
                n_h_avg=self.model.houses_per_capita_avg(self.tract_id),
                i_current=self.model.mortgage_rate_annual,
                i_avg=self.model.mortgage_rate_avg,
                alpha=sell_cfg["alpha_stock"],
                beta=sell_cfg["beta_rate"],
                i_mortgage=self.house.mortgage_rate,
                gamma=sell_cfg["gamma_lockin"],
            )

            if self.model.random_gen.random() < prob_sell:
                tract = self.model.tracts[self.tract_id]
                asking_cfg = self.model.params["asking_price_eq7"]
                self.house.price = asking_price(  # EQ 7
                    p_bar_tract=tract.avg_sold_price(self.house.quality),
                    f_bar_tract=tract.avg_days_on_market(),
                    alpha=asking_cfg["alpha"],
                    beta=asking_cfg["beta"],
                    zeta=asking_cfg["zeta"],
                    epsilon_std=asking_cfg["epsilon_std"],
                    rng=self.model.random_gen,
                )
                # never list below the outstanding mortgage
                self.house.price = max(self.house.price, self.house.mortgage_principal)
                self.house.days_on_market = 0
                self.house_to_sell = self.house
                self.model.list_for_sale(self.house, seller=self)
            else:
                return

        # listed and unsold: bid each month against the expected equity
        equity = max(0, self.house.price - self.house.mortgage_principal)
        self._bid_for_next_home(available_capital=equity)

    def _bid_for_next_home(self, available_capital: float):
        """EQ 3 desired price and EQ 17 down payment for the next home.
        available_capital is unsold equity while listed, else the bank balance."""
        tract = self.model.tracts[self.tract_id]
        g = tract.appreciation_g(alpha=self.model.params["appreciation_eq4"]["alpha_household"])

        mort_cfg = self.model.mortgage_terms["conventional"]
        i_r_monthly = self.model.mortgage_rate_monthly
        loan_cap = max_loan_owner_occupier(
            bank_balance=available_capital,
            disposable_income=self.income - self.essential_consumption(),
            chi_max_ltv=mort_cfg["max_ltv"],
            dti_front=mort_cfg["front_end_dti_max"],
            i_r_monthly=i_r_monthly,
            term_months=mort_cfg["term_months"],
        )

        exp_params = self.model.params["expenditure_eq3"]
        price = desired_expenditure(
            income_or_capital=self.income * 12,
            g=g,
            alpha=exp_params["alpha_household"],
            beta=exp_params["beta"],
            epsilon_std=exp_params["epsilon_std"],
            rng=self.model.random_gen,
            mortgage_cap=loan_cap + available_capital,
        )

        down_cfg = self.model.params["downpayment_eq17"]["repeat_buyer"]
        down_payment = down_payment_owner(
            price=price,
            income_rank=self.income,
            income_cutoff=self.model.ftb_income_cutoff(down_cfg["floor_share_p_floor"]),
            d_minimum_pct=down_cfg["d_minimum_pct"],
            lognorm_m=down_cfg["lognorm_m"],
            lognorm_s=down_cfg["lognorm_s"],
            rng=self.model.random_gen,
        )

        if price < 0 or available_capital < mort_cfg["min_down_payment_pct"] * price:
            self.model.queue_rental_bid(self, fraction_of_income=0.33)
            return
        # cash actually paid is capped by the bank balance, not by unsold equity
        down_payment = min(down_payment, self.bank_balance, price)
        self.bridge_loan = max(0.0, down_payment - self.bank_balance)

        self.model.queue_ownership_bid(self, max_price=price, down_payment=down_payment)
