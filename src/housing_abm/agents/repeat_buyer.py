# Repeat/move-up buyer: conventional  mortgage, downpayment from equity
# Simultaneously lists current home - 'golden handcuffs'

from housing_abm.equations.expenditure import (
    desired_expenditure,
    price_appreciation_expectation,
)
from housing_abm.equations.mortgage import down_payment_owner, max_loan_owner_occupier
from housing_abm.equations.selling import asking_price, p_sell

from .base import HouseholdAgent


class RepeatBuyer(HouseholdAgent):

    WEALTH_KEY = "repeat_buyer"
    LOAN_TYPE = "conventional"

    # inherited initialization

    def __init__(self, model, income: float, age: int, tract_id: str):
        super().__init__(model, income, age, tract_id)
        self.house_to_sell = None  # house they are trying to sell

    def step(self):
        # EQ 6: sell decision with lock in term, EQ 7 asking price for current home
        # EQ 3/5 for new purchase, EQ 17 down payment from equity

        self.refresh_desired_balance()

        if self.house is not None and self.status == "renting":
            # sometimes falls back here via ownershrip market homeless-bidder
            monthly_rent = self.house.rent
            still_carrying_old = (
                self.house_to_sell is not None and self.house_to_sell is not self.house
            )
            if still_carrying_old:
                monthly_rent += (
                    self.house_to_sell.mortgage_payment
                )  # has to pay for old house as well
            self.apply_consumption(housing_cost=monthly_rent)
            if self.lease_months_remaining > 0:
                return
            # current lease is done, get rid of current rental and re-evaluate buy vs rent
            self.house.tenant = None
            self.house.on_rental_market = True
            self.house = None
            self.status = "social_housing"
            available_capital = self.bank_balance
            if (
                self.house_to_sell is not None
            ):  # if has a house to sell, use equity when selling
                available_capital = max(
                    0, self.house_to_sell.price - self.house_to_sell.mortgage_principal
                )  # equity from sale
                # should be add?
            self._bid_for_next_home(
                available_capital=available_capital
            )  # redecide what to do now
            return

        if self.house is None:
            self.apply_consumption(housing_cost=0)
            self._bid_for_next_home(
                available_capital=self.bank_balance
            )  # enter the market with available capital if not currently owning
            return

        monthly_payment = self.house.mortgage_payment
        still_carrying_old = (
            self.house_to_sell is not None and self.house_to_sell is not self.house
        )
        if (
            still_carrying_old
        ):  # if they are still carrying the old house, they pay for both
            monthly_payment += self.house_to_sell.mortgage_payment
        self.apply_consumption(housing_cost=monthly_payment)

        if still_carrying_old:
            return  # just waiting for old place to sell

        if self.house_to_sell is None:  # first time listing, deciding whether to sell
            sell_cfg = self.model.params["selling_eq6"]  # grab selling parameters
            prob_sell = p_sell(
                tenure_years=sell_cfg["tenure_years"],
                n_h=self.model.houses_per_capita(self.tract_id),  # TODO
                n_h_avg=self.model.houses_per_capita_avg(self.tract_id),
                i_current=self.model.mortgage_rate_annual,
                i_avg=self.model.mortgage_rate_avg,  # TODO
                alpha=sell_cfg["alpha_stock"],
                beta=sell_cfg["beta_rate"],
                i_mortgage=self.house.mortgage_rate,  # TODO
                gamma=sell_cfg["gamma_lockin"],
            )

            if self.model.random_gen.random() < prob_sell:  # chooses to sell
                tract = self.model.tracts[self.tract_id]
                asking_cfg = self.model.params["asking_price_eq7"]  # grab eq7 params
                self.house.price = asking_price(  # calculate asking price
                    p_bar_tract=tract.avg_sold_price(self.house.quality),
                    f_bar_tract=tract.avg_days_on_market(),
                    alpha=asking_cfg["alpha"],
                    beta=asking_cfg["beta"],
                    zeta=asking_cfg["zeta"],
                    epsilon_std=asking_cfg["epsilon_std"],
                    rng=self.model.random_gen,
                )
                # sometimes the noise from equation 6 prices a listing below the mortgage owed on it
                self.house.price = max(
                    self.house.price, self.house.mortgage_principal
                )  # can't sell for less than mortgage
                self.house.on_sale_market = True
                self.house.days_on_market = 0
                self.house_to_sell = self.house
                self.model.queue_listing(self.house, seller=self)  # TODO
            else:
                return  # decides not to sell, just keep paying mortgage and wait for next month

        # occurs when listed but still living at old house
        # bid for new house using anticipated equity from the sale
        # runs every month listing is live and unsold
        equity = max(
            0, self.house.price - self.house.mortgage_principal
        )  # prevent any negative equity from being used to buy a new house
        self._bid_for_next_home(available_capital=equity)  # or equity+bank_balance?

    def _bid_for_next_home(self, available_capital: float):
        """EQ 3/5, purchase decision for new home
        unrealized equity when listing old home, liquid bank once old home is osld"""
        tract = self.model.tracts[self.tract_id]
        g = tract.appreciation_g(alpha=self.model.params["appreciation_eq4"]["alpha_household"])

        mort_cfg = self.model.mortgage_terms[
            "conventional"
        ]  # loan terms for conventional loans
        i_r_monthly = self.model.mortgage_rate_monthly
        loan_cap = max_loan_owner_occupier(  # max conventional loan they can get
            bank_balance=available_capital,  # uses house equity for downpayment on next house
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
            alpha=exp_params["alpha_household"],  # income to yearly
            beta=exp_params["beta"],
            epsilon_std=exp_params["epsilon_std"],
            rng=self.model.random_gen,
            mortgage_cap=loan_cap + available_capital,  # equity helps afford more
        )

        down_cfg = self.model.params["downpayment_eq17"]["repeat_buyer"]
        down_payment = down_payment_owner(  # calculate downpayment from distribution clustered at minimum
            price=price,
            income_rank=self.income,
            income_cutoff=self.model.ftb_income_cutoff(down_cfg["floor_share_p_floor"]),
            d_minimum_pct=down_cfg["d_minimum_pct"],
            lognorm_m=down_cfg["lognorm_m"],
            lognorm_s=down_cfg["lognorm_s"],
            rng=self.model.random_gen,
        )

        # down_pamyne above may be based off unrealized equity
        # prevent bids from spending more real cash than bank_balance has

        if (
            price < 0 or available_capital < mort_cfg["min_down_payment_pct"] * price
        ):  # can't afford down payment, enter rental market
            self.model.queue_rental_bid(self, fraction_of_income=0.33)
            return
        down_payment = min(
            down_payment, self.bank_balance, price
        )  # can't pay more than you have or more than the price of the house
        self.bridge_loan = max(0.0, down_payment - self.bank_balance)

        self.model.queue_ownership_bid(self, max_price=price, down_payment=down_payment)
