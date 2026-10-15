"""First-time buyer: FHA mortgage, no equity."""

from housing_abm.equations.buy_rent import p_buy
from housing_abm.equations.expenditure import desired_expenditure
from housing_abm.equations.mortgage import down_payment_owner, max_loan_owner_occupier

from .base import HouseholdAgent


class FirstTimeBuyer(HouseholdAgent):

    LOAN_TYPE = "fha"
    WEALTH_KEY = "first_time_buyer"

    def step(self):
        self.refresh_desired_balance()
        if self.house is not None and self.status == "owning":
            # pay the mortgage; promote to repeat buyer after the minimum tenure
            min_tenure = self.model.params["repeat_buyer_promotion"][
                "min_tenure_months"
            ]
            months_owned = self.model.current_month - self.owned_since_month
            monthly_payment = self.house.mortgage_payment
            self.apply_consumption(housing_cost=monthly_payment)
            if months_owned >= min_tenure:
                self.model._promote_to_repeat_buyer(self)
            return

        if self.house is not None and self.status == "renting":
            # renting while saving; re-decide buy vs rent when the lease ends
            monthly_rent = self.house.rent
            self.apply_consumption(housing_cost=monthly_rent)
            self.lease_months_remaining -= 1
            if self.lease_months_remaining > 0:
                return
            self.model.end_tenancy(self.house)
            self.house = None
            self.status = "social_housing"
            self._evaluate_buy_or_rent()
            return

        self.apply_consumption(housing_cost=0)
        self._evaluate_buy_or_rent()

    def _evaluate_buy_or_rent(self):
        """EQ 3 desired price, EQ 5 buy vs rent, EQ 17 down payment, then bid."""
        tract = self.model.tracts[self.tract_id]
        mort_cfg = self.model.mortgage_terms[self.LOAN_TYPE]
        i_r_monthly = self.model.mortgage_rate_monthly

        loan_cap = max_loan_owner_occupier(
            bank_balance=self.bank_balance,
            disposable_income=self.income - self.essential_consumption(),
            chi_max_ltv=mort_cfg["max_ltv"],
            dti_front=mort_cfg["front_end_dti_max"],
            i_r_monthly=i_r_monthly,
            term_months=mort_cfg["term_months"],
        )

        max_affordable_price = loan_cap + self.bank_balance
        g = tract.appreciation_g(alpha=self.model.params["appreciation_eq4"]["alpha_household"])

        exp_params = self.model.params["expenditure_eq3"]
        price = desired_expenditure(
            income_or_capital=self.income * 12,  # annual income
            g=g,
            alpha=exp_params["alpha_household"],
            beta=exp_params["beta"],
            epsilon_std=exp_params["epsilon_std"],
            rng=self.model.random_gen,
            mortgage_cap=max_affordable_price,
        )

        if self.bank_balance < mort_cfg["min_down_payment_pct"] * price:
            # can't cover the minimum down payment: rent instead
            self.model.queue_rental_bid(self, fraction_of_income=0.33)
            return

        quality = self.model.quality_affordable(price, self.tract_id)
        rent_q_monthly = self.model.market_rent_for_quality(quality, self.tract_id)
        estimated_down_payment = mort_cfg["min_down_payment_pct"] * price
        estimated_loan = price - estimated_down_payment
        monthly_mortgage = self.model.monthly_payment(
            estimated_loan, i_r_monthly, mort_cfg["term_months"]
        )

        buy_params = self.model.params["buy_rent_eq5"]
        prob = p_buy(
            rent_q=rent_q_monthly * 12,
            tau=buy_params["tau"],
            monthly_mortgage=monthly_mortgage,
            price=price,
            g=g,
            beta=buy_params["beta"],
            annual_income=self.income * 12,
        )

        if self.model.random_gen.random() < prob:
            down_cfg = self.model.params["downpayment_eq17"]["first_time_buyer"]
            income_cutoff = self.model.ftb_income_cutoff(
                down_cfg["floor_share_p_floor"]
            )
            down_payment = down_payment_owner(
                price=price,
                income_rank=self.income,
                income_cutoff=income_cutoff,
                d_minimum_pct=down_cfg["d_minimum_pct"],
                lognorm_m=down_cfg["lognorm_m"],
                lognorm_s=down_cfg["lognorm_s"],
                rng=self.model.random_gen,
            )

            down_payment = min(down_payment, self.bank_balance, price)
            self.model.queue_ownership_bid(
                self, max_price=price, down_payment=down_payment
            )
        else:
            self.model.queue_rental_bid(self, fraction_of_income=0.33)
