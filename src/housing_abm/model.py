# top level Mesa model/initial skeleton, monthly cycle
import yaml
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from housing_abm.construction import run_construction
from housing_abm.demographics import process_aging_and_births, process_deaths
from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.renter import Renter
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.markets.ownership_market import (
    generate_placeholder_sale_stock,
    run_ownership_market,
)
from housing_abm.markets.rental_market import (
    generate_placeholder_rental_stock,
    run_rental_market,
)
from housing_abm.policy import load_policies
from housing_abm.tract import Tract
from housing_abm.interest_rate import update_mortgage_rate


class AtlantaHousingModel(Model):
    def __init__(
        self,
        config_path: str = "config/baseline_params.yaml",
        policy_paths: list[str] | None = None,
        seed: int | None = None,
        n_households: int | None = None,
    ):
        super().__init__(seed=seed)
        with open(config_path) as f:
            self.params = yaml.safe_load(f)  # take parameters from given config path

        # policies handled by policy.py

        self.random_gen = np.random.default_rng(seed=seed)
        self.current_month = 0
        # placeholder exogenous interest rate
        # TODO: swap for real series per month
        self.current_fed_rate_annual = 0.039
        self.current_fed_rate_monthly = self.current_fed_rate_annual / 12

        self.fed_rate_history = [
            self.current_fed_rate_annual
        ]  # placeholder, no history yet

        # EQ 16: mortgage rates = exogenous base rate from the bank + endogenous spread
        self.mortgage_rate_spread_annual = 0.0
        self.mortgage_rate_annual = self.current_fed_rate_annual
        self.mortgage_rate_monthly = self.mortgage_rate_annual / 12
        self.mortgage_rate_history = [self.mortgage_rate_annual]
        self.mortgage_rate_avg = self.mortgage_rate_annual
        self._monthly_new_lending = (
            0.0  # accumulated during run_ownership_market, drives EQ16
        )
        self.bankruptcy_injections_this_month = 0  # track to make sure not too many
        self.bankruptcy_injections_total = 0

        # different types of mortgages
        self.mortgage_terms = {"fha": {}, "conventional": {}, "investor_dscr": {}}
        with open("config/mortgage_terms.yaml") as f:
            self.mortgage_terms = yaml.safe_load(f)

        # load LTV/LTI policies, mutates mortgage terms
        load_policies(self, policy_paths)

        self.tracts = {"tract_001": Tract("tract_001")}  # placeholder storage of tracts

        # trailing history of houses_per_capita per tract
        self.houses_per_capita_history = {tract_id: [] for tract_id in self.tracts}

        # queue of renters in social housing looking to rent again
        # -queue housing decision(), drained by run_rental_market() in step()

        self._rental_bid_queue = []
        self._ownership_bid_queue = []
        self._resale_sellers = (
            {}
        )  # HouseholdUnit -> agent, keeps tracks of resale listings

        self.datacollector = DataCollector(
            # track all the relevant data
            model_reporters={
                "n_agents": lambda m: len(m.agents),
                "n_renting": lambda m: m._n_renting(),
                "n_owning": lambda m: sum(
                    1
                    for agent in m.agents
                    if getattr(agent, "status", None) == "owning"
                ),
                "n_social_housing": lambda m: sum(
                    1
                    for agent in m.agents
                    if getattr(agent, "status", None) == "social_housing"
                ),
                "n_first_time_buyers": lambda m: sum(
                    1 for a in m.agents if isinstance(a, FirstTimeBuyer)
                ),  # NEW
                "n_repeat_buyers": lambda m: sum(
                    1 for a in m.agents if isinstance(a, RepeatBuyer)
                ),  # NEW
                "n_small_landlords": lambda m: sum(
                    1 for a in m.agents if isinstance(a, SmallLandlord)
                ),
                "n_institutional_investors": lambda m: sum(
                    1 for a in m.agents if isinstance(a, InstitutionalInvestor)
                ),
                "n_renters": lambda m: sum(
                    1 for a in m.agents if isinstance(a, Renter)
                ),
                "n_investor_owned_units": lambda m: sum(
                    len(a.properties)
                    for a in m.agents
                    if isinstance(a, (SmallLandlord, InstitutionalInvestor))
                ),
                "mean_bank_balance": lambda m: m._mean_bank_balance(),
                "bankruptcy_injections_this_month": lambda m: m.bankruptcy_injections_this_month,
                "rental_vacancy_rate": lambda m: m._rental_vacancy_rate(),
                "homeownership_rate": lambda m: m._homeownership_rate(),
            }
        )
        # create initial renter population
        # TODO: replace placeholder income with real calibrated
        n_households = n_households or self.params.get("n_households", 100)
        for _ in range(n_households):
            income = float(
                self.random_gen.lognormal(mean=8.0, sigma=0.55)
            )  # realistic terms, generated by Claude - average ~$2,980 per month
            age = int(self.random_gen.integers(22, 65))
            Renter(
                model=self, income=income, age=age, tract_id="tract_001"
            )  # default initialization

        # create small landlord and institutional investor populations
        # TODO: replace placeholder counts/wealth draws with calibrated Atlanta investor shares
        n_small_landlords = self.params.get("simulation", {}).get(
            "n_small_landlords", 0
        )
        for _ in range(n_small_landlords):
            income = float(
                self.random_gen.lognormal(mean=9.8, sigma=0.5)
            )  # landlords skew higher-income than renters
            age = int(self.random_gen.integers(30, 70))
            landlord = SmallLandlord(
                model=self, income=income, age=age, tract_id="tract_001"
            )
            landlord.bank_balance = float(
                self.random_gen.lognormal(mean=11.5, sigma=0.6)
            )  # starting cash for down payments

        n_institutional_investors = self.params.get("simulation", {}).get(
            "n_institutional_investors", 0
        )
        for _ in range(n_institutional_investors):
            available_capital = float(
                self.random_gen.lognormal(mean=14.5, sigma=0.7)
            )  # much larger capital pools
            InstitutionalInvestor(
                model=self, available_capital=available_capital, tract_id="tract_001"
            )

        # placholder exogenous rental stock
        self.rental_units = generate_placeholder_rental_stock(self)
        self.for_sale_units = generate_placeholder_sale_stock(self)

    def step(self):
        self.current_month += 1
        self.bankruptcy_injections_this_month = 0

        # monthly cycle: demographics -> construction -> households decide
        # ownership market -> rental market -> interest rate update

        self.fed_rate_history.append(self.current_fed_rate_annual)
        self.fed_rate_avg = float(
            np.mean(self.fed_rate_history[-12:])
        )  # average of last 12 months

        for tract_id in self.tracts:
            history = self.houses_per_capita_history.setdefault(tract_id, [])
            history.append(self.houses_per_capita(tract_id))
            del history[:-24]  # 24 month trailing window

        # deal with demographic stuff
        process_aging_and_births(self)
        process_deaths(self)

        run_construction(self)

        self.agents.shuffle_do("step")
        # match queued renters against vacant rental stock

        run_ownership_market(self)  # buyers get first chance

        run_rental_market(self)

        # EQ 16 mortgage rate update
        update_mortgage_rate(self)

        self.datacollector.collect(self)

    def prevent_bankruptcy(self, agent):
        """If a household cannot afford mortgage/rent it goes bankrupt
        Our model doesn't include bankruptcy dynamics, so we artificially inject cash
        as much as necessary to bankrupt households
        Tracked via self.bankruptcy_injections_this_month"""
        if agent.bank_balance < 0:
            self.bankruptcy_injections_this_month += 1
            self.bankruptcy_injections_total += 1
            agent.bank_balance = 0

    # core indicator reporters
    def _n_renting(self):
        return sum(
            1 for agent in self.agents if getattr(agent, "status", None) == "renting"
        )  # checks status attritbute for renters

    def _mean_bank_balance(self):

        balances = [
            agent.bank_balance
            for agent in self.agents
            if hasattr(agent, "bank_balance")
        ]  # only applies to household agents
        return (
            float(np.mean(balances)) if balances else 0
        )  # if balances exist, return mean as float

    def _homeownership_rate(self):
        owners = sum(
            1 for agent in self.agents if getattr(agent, "status", None) == "owning"
        )
        total = len(
            [
                agent
                for agent in self.agents
                if hasattr(agent, "income")
                and getattr(agent, "properties", None) is None
            ]
        )
        return owners / total if total > 0 else 0.0

    def _investor_purchase_share(self):
        raise NotImplementedError

    def _rental_vacancy_rate(self):
        vacant = sum(
            1
            for unit in self.rental_units
            if unit.tenant is None and unit.on_rental_market
        )
        total = len(self.rental_units)
        return vacant / total if self.rental_units else 0.0

    def _median_rent(self):
        raise NotImplementedError

    def _transaction_volume(self):
        raise NotImplementedError

    # tract/quality getters
    def quality_affordable(
        self, price, tract_id
    ):  # given price, what quality can you afford
        tract = self.tracts[tract_id]
        return price / tract.price_per_quality

    def market_rent_for_quality(
        self, quality, tract_id
    ):  # given quality, what rent is it at a tract level
        tract = self.tracts[tract_id]
        return quality * tract.rent_per_quality

    def monthly_payment(
        self, principal, i_r_monthly, term_months
    ):  # monthly payment calculation
        if principal <= 0:
            return 0.0
        annuity_factor = (1 - (1 + i_r_monthly) ** (-term_months)) / i_r_monthly
        return principal / annuity_factor

    def ftb_income_cutoff(self, floor_share_p_floor):
        incomes = [
            agent.income
            for agent in self.agents
            if hasattr(agent, "income") and getattr(agent, "properties", None) is None
        ]  # filter out houses and investors
        return (
            float(np.quantile(incomes, floor_share_p_floor)) if incomes else 0.0
        )  # return the income you need to be above the cutoff

    # placeholder hooks
    def queue_housing_decision(self, agent):
        # mark and queue an agent in social housing for the housing-market process
        if isinstance(agent, Renter):  # seeing if can become first time buyer
            if self.for_sale_units:  # available houses
                min_price = min(u.price for u in self.for_sale_units)
                min_down = (
                    self.mortgage_terms["fha"]["min_down_payment_pct"] * min_price
                )  # loan regulation check
                if agent.bank_balance >= min_down:  # has enough money
                    self._promote_to_first_time_buyer(agent)
                    return
        if agent not in self._rental_bid_queue:  # everyone else
            self._rental_bid_queue.append(agent)

    def exit_tract(self, agent):  # remove agent from schedule
        # occurs if rent burden became high or they moved away
        # vacate unit if exists, then remove from market
        if agent.house is not None:
            agent.house.tenant = None
            agent.house.on_rental_market = True
            agent.house = None
        if agent in self._rental_bid_queue:
            self._rental_bid_queue.remove(agent)
        agent.remove()

    def queue_rental_bid(self, agent, fraction_of_income: float = 0.33):
        agent.rent_affordability_fraction = (
            fraction_of_income  # how much of their income do they bid for rent
        )
        self.queue_housing_decision(agent)

    def queue_ownership_bid(
        self, agent, max_price: float, down_payment: float
    ):  # creates a bid
        self._ownership_bid_queue.append(
            {"agent": agent, "max_price": max_price, "down_payment": down_payment}
        )

    def queue_listing(self, unit, seller):  # queues a house for sale
        """Register a resale listing, tracks who gets paid out once someone buys it"""
        self._resale_sellers[unit] = seller

    def _promote_to_first_time_buyer(
        self, renter
    ):  # promote when a renter has enough money for a down payment on a house
        # create new FTB agent, delete the old renter agent
        ftb = FirstTimeBuyer(
            model=self, income=renter.income, age=renter.age, tract_id=renter.tract_id
        )
        ftb.bank_balance = renter.bank_balance
        ftb.desired_balance = renter.desired_balance

        renter.remove()

    def _promote_to_repeat_buyer(
        self, ftb
    ):  # promote when FTB has been in their house for reqiured amount of time
        """carries over financial state, repoint reference to new agent"""
        rb = RepeatBuyer(
            model=self, income=ftb.income, age=ftb.age, tract_id=ftb.tract_id
        )
        rb.bank_balance = ftb.bank_balance
        rb.desired_balance = ftb.desired_balance
        rb.house = ftb.house
        rb.house.owner = rb  # point house at new agent
        rb.status = "owning"
        rb.owned_since_month = ftb.owned_since_month

        ftb.remove()

    def houses_per_capita(self, tract_id):
        """TODO: make per tract once multiple tracts exist"""
        return len(self.for_sale_units) / max(
            1, len([agent for agent in self.agents if hasattr(agent, "income")])
        )  # houses/non house agents

    def houses_per_capita_avg(self, tract_id):
        """placeholder, no rolling history tracked yet"""
        history = self.houses_per_capita_history.get(tract_id, [])
        if not history:
            return self.houses_per_capita(
                tract_id
            )  # if no history, return current value
        return float(np.mean(history))  # average of last 12 months
