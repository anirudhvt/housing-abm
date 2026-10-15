"""Top-level Mesa model: initial population, housing stock and the monthly cycle."""

import warnings

import yaml
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from housing_abm.construction import run_construction, run_investor_replenishment
from housing_abm.demographics import (
    process_aging_and_births,
    process_deaths,
    sample_stationary_ages,
)
from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.renter import Renter
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.external_data import load_g_series, load_monthly_growth_series
from housing_abm.equations.investor_propensity import landlord_selection_weight
from housing_abm.markets.ownership_market import (
    generate_placeholder_sale_stock,
    run_ownership_market,
)
from housing_abm.markets.rental_market import (
    generate_placeholder_rental_stock,
    run_rental_market,
)
from housing_abm.policies.investor_restrictions import (
    apply_forced_divestiture,
    update_ownership_cap_soft_state,
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
            self.params = yaml.safe_load(f)

        # Separate RNG streams so the rate path and demographics are identical
        # across the baseline and policy arms of a paired run.
        seed_seq = np.random.SeedSequence(seed)
        market_seq, macro_seq, demo_seq = seed_seq.spawn(3)
        self.random_gen = np.random.default_rng(market_seq)  # market/behavioural
        self.rng_macro = np.random.default_rng(macro_seq)  # exogenous rate path
        self.rng_demography = np.random.default_rng(demo_seq)  # births/deaths
        self.current_month = 0
        self.spinup_end_month = 0

        # exogenous fed rate, AR(1) around a long-run mean
        cfg = self.params.get("fed_rate_process", {})
        self._fed_rate_mean = cfg.get("mean_annual", 0.0453)
        self._fed_rate_phi = cfg.get("monthly_persistence", 0.98)
        self._fed_rate_sigma = cfg.get("monthly_noise_std", 0.0024)
        self.current_fed_rate_annual = self._fed_rate_mean
        self.current_fed_rate_monthly = self.current_fed_rate_annual / 12
        self.fed_rate_history = [self.current_fed_rate_annual]

        # EQ 16: mortgage rate = exogenous base rate + endogenous spread
        self.mortgage_rate_spread_annual = 0.0
        self.mortgage_rate_annual = self.current_fed_rate_annual
        self.mortgage_rate_monthly = self.mortgage_rate_annual / 12
        self.mortgage_rate_history = [self.mortgage_rate_annual]
        self.mortgage_rate_avg = self.mortgage_rate_annual
        self._monthly_new_lending = 0.0  # accumulated by the ownership market, drives EQ 16
        self.bankruptcy_injections_this_month = 0
        self.bankruptcy_injections_total = 0
        self.purchases_this_month = {"ftb": 0, "repeat": 0, "small_landlord": 0,
                                     "institutional": 0}
        # rent-burden exits this month, replaced one-for-one by in-migration
        self.households_displaced_pending = 0
        self.households_displaced_total = 0

        with open("config/mortgage_terms.yaml") as f:
            self.mortgage_terms = yaml.safe_load(f)

        load_policies(self, policy_paths)  # may override mortgage terms

        # Optional: drive appreciation/rent growth from real ZHVI/ZORI series
        # instead of the model's own transactions.
        external_g_series = None
        external_rent_growth_series = None
        calib_cfg = self.params.get("tract_calibration", {})
        sim_cfg = self.params.get("simulation", {})
        if sim_cfg.get("use_external_appreciation_data", False):
            zhvi_path = calib_cfg.get("zhvi_csv_path", "atlanta_zillow_zhvi.csv")
            zori_path = calib_cfg.get("zori_csv_path", "atlanta_zillow_zori.csv")
            try:
                external_g_series = load_g_series(zhvi_path)
                external_rent_growth_series = load_monthly_growth_series(zori_path)
            except (FileNotFoundError, KeyError) as e:
                warnings.warn(
                    f"use_external_appreciation_data is set but {zhvi_path}/"
                    f"{zori_path} could not be loaded ({e}); falling back to "
                    "endogenous appreciation."
                )

        # 2019 Atlanta price/rent anchors and the reference model's smoothing constants
        smoothing_cfg = self.params.get("market_smoothing", {})
        reference_price = calib_cfg.get("reference_price_per_quality", 250_000.0)
        reference_rent = calib_cfg.get("reference_rent_per_quality", 1400.0)
        cumulative_weight_beyond_year = smoothing_cfg.get(
            "cumulative_weight_beyond_year", 0.25
        )
        smoothing_factor = 1.0 - cumulative_weight_beyond_year ** (1.0 / 12.0)
        price_decay = smoothing_cfg.get("market_average_price_decay", 0.5)
        rent_decay = smoothing_cfg.get("market_average_rent_decay", 0.5)

        # single tract for now; multi-tract is a planned extension
        self.tracts = {
            "tract_001": Tract(
                "tract_001",
                price_per_quality=reference_price,
                rent_per_quality=reference_rent,
                reference_price_per_quality=reference_price,
                reference_rent_per_quality=reference_rent,
                smoothing_factor=smoothing_factor,
                price_decay=price_decay,
                rent_decay=rent_decay,
                external_g_series=external_g_series,
                external_rent_growth_series=external_rent_growth_series,
            )
        }

        self.houses_per_capita_history = {tract_id: [] for tract_id in self.tracts}

        self._rental_bid_queue = []
        self._ownership_bid_queue = []
        self._resale_sellers = {}  # HousingUnit -> seller, for resale payouts

        # set mirrors of the channel registries for O(1) membership tests
        self._for_sale_set = set()
        self._rental_set = set()
        # every unit that exists, in any tenure
        self.housing_units = set()

        # per-step memo slots (see _invalidate_step_caches)
        self._cache_incomes = None
        self._cache_houses_per_capita = None
        self._cache_n_households = None
        self._cache_active_for_sale = None

        self.datacollector = DataCollector(
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
                ),
                "n_repeat_buyers": lambda m: sum(
                    1 for a in m.agents if isinstance(a, RepeatBuyer)
                ),
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

        # Initial households. Small landlords are selected from the N households,
        # weighted by the SCF income-decile propensity curve, not added on top.
        n_households = n_households or sim_cfg.get("n_households", 100)
        demo_cfg = self.params["demographics"]
        entry_lo, entry_hi = demo_cfg["new_household_age_range"]
        initial_ages = sample_stationary_ages(
            self.random_gen, n_households, demo_cfg["mortality"], entry_lo, entry_hi
        )
        income_cfg = self.params.get("income_distribution", {})
        propensity_cfg = self.params.get("investor_propensity_eq", {})

        all_incomes = self.random_gen.lognormal(
            mean=income_cfg.get("household_lognormal_mean", 8.6),
            sigma=income_cfg.get("household_lognormal_sigma", 0.65),
            size=n_households,
        )

        n_small_landlords = round(
            n_households * sim_cfg.get("small_landlord_fraction", 0.0)
        )
        decile_probs = propensity_cfg.get(
            "small_landlord_income_decile_probs", [0.1] * 10
        )
        weights = np.array([
            landlord_selection_weight(
                inc, decile_probs,
                income_cfg.get("household_lognormal_mean", 8.6),
                income_cfg.get("household_lognormal_sigma", 0.65),
            )
            for inc in all_incomes
        ])
        landlord_indices = set()
        if n_small_landlords > 0 and weights.sum() > 0:
            weights_norm = weights / weights.sum()
            landlord_indices = set(self.random_gen.choice(
                n_households, size=n_small_landlords, replace=False, p=weights_norm,
            ))

        for i, (age, income) in enumerate(zip(initial_ages, all_incomes)):
            if i in landlord_indices:
                landlord = SmallLandlord(
                    model=self, income=float(income), age=int(age),
                    tract_id="tract_001",
                )
                landlord.bank_balance = float(
                    self.random_gen.lognormal(mean=11.5, sigma=0.6)
                )
            else:
                Renter(
                    model=self, income=float(income), age=int(age),
                    tract_id="tract_001",
                )

        # institutional investors are firms, outside the household population
        n_institutional_investors = round(
            n_households * sim_cfg.get("institutional_investor_fraction", 0.0)
        )
        for _ in range(n_institutional_investors):
            available_capital = float(
                self.random_gen.lognormal(mean=13.0, sigma=0.5)
            )
            InstitutionalInvestor(
                model=self, available_capital=available_capital, tract_id="tract_001",
            )

        rental_n_units = round(
            n_households
            * self.params.get("initial_rental_stock", {}).get(
                "units_per_household", 0.5
            )
        )
        self.rental_units = generate_placeholder_rental_stock(
            self, n_units=rental_n_units,
            base_rent=self.tracts["tract_001"].rent_per_quality,
        )
        self._rental_set = set(self.rental_units)
        self.housing_units.update(self.rental_units)
        self._assign_initial_rental_ownership()
        sale_n_units = round(
            n_households
            * self.params.get("initial_sale_stock", {}).get(
                "units_per_household", 0.667
            )
        )
        self.for_sale_units = generate_placeholder_sale_stock(
            self, n_units=sale_n_units
        )
        self._for_sale_set = set(self.for_sale_units)
        self.housing_units.update(self.for_sale_units)

    # ------------------------------------------------------------------
    # per-step caches: values read once per agent but fixed within a step
    # ------------------------------------------------------------------

    def _invalidate_step_caches(self):
        self._cache_incomes = None
        self._cache_houses_per_capita = None
        self._cache_n_households = None
        self._cache_active_for_sale = None

    def _household_incomes(self):
        """Incomes of owner-occupier-track agents (excludes investors/units)."""
        if self._cache_incomes is None:
            self._cache_incomes = np.fromiter(
                (
                    a.income
                    for a in self.agents
                    if getattr(a, "properties", None) is None
                    and getattr(a, "income", None) is not None
                ),
                dtype=float,
            )
        return self._cache_incomes

    def n_household_agents(self):
        """Count of decision-making household agents (excludes HousingUnits)."""
        if self._cache_n_households is None:
            self._cache_n_households = sum(
                1 for a in self.agents if getattr(a, "income", None) is not None
            )
        return self._cache_n_households

    def active_for_sale(self):
        """Units currently listed for sale."""
        if self._cache_active_for_sale is None:
            self._cache_active_for_sale = [
                u for u in self.for_sale_units if u.on_sale_market
            ]
        return self._cache_active_for_sale

    def _prune_registries(self):
        """Drop sold listings and units that have left the rental channel."""
        self.for_sale_units = [u for u in self.for_sale_units if u.on_sale_market]
        self._for_sale_set = set(self.for_sale_units)
        self.rental_units = [
            u
            for u in self.rental_units
            if u.tenant is not None
            or u.on_rental_market
            or getattr(u.owner, "properties", None) is not None
        ]
        self._rental_set = set(self.rental_units)

    def step(self):
        """Demographics -> construction -> agent decisions -> ownership market
        -> rental market -> interest rate update."""
        self.current_month += 1
        self.bankruptcy_injections_this_month = 0
        self.purchases_this_month = dict.fromkeys(self.purchases_this_month, 0)
        self._prune_registries()
        self._invalidate_step_caches()

        noise = self.rng_macro.normal(0.0, self._fed_rate_sigma)
        self.current_fed_rate_annual = self._fed_rate_mean + self._fed_rate_phi * (
            self.current_fed_rate_annual - self._fed_rate_mean
        ) + noise
        self.current_fed_rate_annual = max(self.current_fed_rate_annual, 0.0005)
        self.current_fed_rate_monthly = self.current_fed_rate_annual / 12
        self.fed_rate_history.append(self.current_fed_rate_annual)

        for tract_id in self.tracts:
            history = self.houses_per_capita_history.setdefault(tract_id, [])
            history.append(self.houses_per_capita(tract_id))
            del history[:-24]  # 24-month trailing window

        process_aging_and_births(self)
        process_deaths(self)

        run_construction(self)
        run_investor_replenishment(self)
        apply_forced_divestiture(self)  # hard ownership cap

        self.agents.shuffle_do("step")

        run_ownership_market(self)  # buyers get first pick
        for tract in self.tracts.values():
            tract.update_hpi_history()

        run_rental_market(self)
        for tract in self.tracts.values():
            tract.update_rent_history()

        update_ownership_cap_soft_state(self)
        update_mortgage_rate(self)

        self.datacollector.collect(self)

    def run_spinup(self, n_months: int):
        """Run n_months to settle endogenous dynamics, then discard collected data."""
        self.spinup_end_month = self.current_month + n_months
        for _ in range(n_months):
            self.step()
        for key in self.datacollector.model_vars:
            self.datacollector.model_vars[key] = []

    def prevent_bankruptcy(self, agent):
        """The model has no bankruptcy, so negative balances are topped up to
        zero and counted."""
        if agent.bank_balance < 0:
            self.bankruptcy_injections_this_month += 1
            self.bankruptcy_injections_total += 1
            agent.bank_balance = 0

    # core indicator reporters

    def _appreciation_g(self, alpha: float = 1.0):
        """EQ 4 averaged across tracts; None until a tract has 15 months of history."""
        values = [t.appreciation_g(alpha=alpha) for t in self.tracts.values()]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    def _n_renting(self):
        return sum(
            1 for agent in self.agents if getattr(agent, "status", None) == "renting"
        )

    def _mean_bank_balance(self):
        balances = [
            agent.bank_balance
            for agent in self.agents
            if hasattr(agent, "bank_balance")
        ]
        return float(np.mean(balances)) if balances else 0

    def _homeownership_rate(self):
        owners = sum(
            1 for agent in self.agents if getattr(agent, "status", None) == "owning"
        )
        total = self._household_incomes().size
        return owners / total if total > 0 else 0.0

    def _rental_vacancy_rate(self):
        """Vacant share of the rental stock (excluding units sold to owner-occupiers)."""
        stock = [
            u
            for u in self.rental_units
            if u.tenant is not None
            or u.on_rental_market
            or getattr(u.owner, "properties", None) is not None
        ]
        if not stock:
            return 0.0
        vacant = sum(1 for u in stock if u.tenant is None and u.on_rental_market)
        return vacant / len(stock)

    # tract/quality getters

    def quality_affordable(self, price, tract_id):
        """Quality of house a given price buys in this tract."""
        tract = self.tracts[tract_id]
        return price / tract.price_per_quality

    def market_rent_for_quality(self, quality, tract_id):
        tract = self.tracts[tract_id]
        return quality * tract.rent_per_quality

    def monthly_payment(self, principal, i_r_monthly, term_months):
        """Standard annuity payment."""
        if principal <= 0:
            return 0.0
        annuity_factor = (1 - (1 + i_r_monthly) ** (-term_months)) / i_r_monthly
        return principal / annuity_factor

    def ftb_income_cutoff(self, floor_share_p_floor):
        """Household income at the given quantile (EQ 17 floor cutoff)."""
        incomes = self._household_incomes()
        return float(np.quantile(incomes, floor_share_p_floor)) if incomes.size else 0.0

    # queues and transitions

    def queue_housing_decision(self, agent):
        """Promote a qualifying renter to first-time buyer, otherwise queue a rental bid."""
        if isinstance(agent, Renter):
            if self.renter_qualifies_as_buyer(agent):
                self._promote_to_first_time_buyer(agent)
                return
        if agent not in self._rental_bid_queue:
            self._rental_bid_queue.append(agent)

    def renter_qualifies_as_buyer(self, agent) -> bool:
        """True if the renter can fund the minimum down payment on the cheapest
        listing and also qualify (EQ 14) for the rest of the price."""
        listings = self.active_for_sale()
        if not listings:
            return False
        min_price = min(u.price for u in listings)

        terms = self.mortgage_terms["fha"]
        if agent.bank_balance < terms["min_down_payment_pct"] * min_price:
            return False

        from housing_abm.equations.mortgage import max_loan_owner_occupier

        loan_cap = max_loan_owner_occupier(
            bank_balance=agent.bank_balance,
            disposable_income=agent.income - agent.essential_consumption(),
            chi_max_ltv=terms["max_ltv"],
            dti_front=terms["front_end_dti_max"],
            i_r_monthly=self.mortgage_rate_monthly,
            term_months=terms["term_months"],
        )
        return loan_cap + agent.bank_balance >= min_price

    def exit_tract(self, agent):
        """Remove a household that leaves on rent burden, vacating its unit."""
        self.households_displaced_pending += 1
        self.households_displaced_total += 1
        if agent.house is not None:
            self.end_tenancy(agent.house)
            agent.house = None
        if agent in self._rental_bid_queue:
            self._rental_bid_queue.remove(agent)
        agent.remove()

    def queue_rental_bid(self, agent, fraction_of_income: float = 0.33):
        agent.rent_affordability_fraction = fraction_of_income
        self.queue_housing_decision(agent)

    def queue_ownership_bid(
        self, agent, max_price: float, down_payment: float, acquisition_tax: float = 0.0
    ):
        """Register a bid. acquisition_tax is paid on top of the down payment and
        buys no equity, so it is kept separate from down_payment."""
        self._ownership_bid_queue.append(
            {
                "agent": agent,
                "max_price": max_price,
                "down_payment": down_payment,
                "acquisition_tax": acquisition_tax,
            }
        )

    def queue_listing(self, unit, seller):
        """List a resale unit and record who gets paid when it sells."""
        self.list_for_sale(unit, seller=seller)

    def _assign_initial_rental_ownership(self):
        """Split the initial rental stock between institutional investors
        (config institutional_share) and small landlords."""
        cfg = self.params.get("initial_rental_stock", {})
        inst_share = cfg.get("institutional_share", 0.30)
        landlords = [a for a in self.agents if isinstance(a, SmallLandlord)]
        investors = [a for a in self.agents if isinstance(a, InstitutionalInvestor)]
        if not landlords and not investors:
            return

        units = list(self.rental_units)
        self.random_gen.shuffle(units)
        n_institutional = int(round(len(units) * inst_share)) if investors else 0

        def _hand_over(unit, owner):
            unit.owner = owner
            owner.properties.append(unit)

        for i, unit in enumerate(units):
            if i < n_institutional:
                owner = investors[i % len(investors)]
            elif landlords:
                owner = landlords[(i - n_institutional) % len(landlords)]
            else:
                owner = investors[i % len(investors)]
            _hand_over(unit, owner)

    def end_tenancy(self, unit):
        """Release a tenancy and start the unit's frictional void period."""
        unit.tenant = None
        unit.on_rental_market = True
        unit.void_months_remaining = self.params.get("rental_market", {}).get(
            "void_months", 0
        )

    def record_purchase(self, agent):
        """Tally a completed purchase against the buyer's class."""
        key = {
            "first_time_buyer": "ftb",
            "repeat_buyer": "repeat",
            "small_landlord": "small_landlord",
            "institutional_investor": "institutional",
        }.get(getattr(agent, "WEALTH_KEY", None))
        if key is not None:
            self.purchases_this_month[key] += 1

    def register_unit(self, unit):
        """Track a unit in the total housing stock (used for construction targets)."""
        self.housing_units.add(unit)

    def total_housing_stock(self) -> int:
        return len(self.housing_units)

    def list_for_sale(self, unit, seller=None):
        """Put a unit on the sale market and register it exactly once."""
        unit.on_sale_market = True
        if seller is not None:
            self._resale_sellers[unit] = seller
        if unit not in self._for_sale_set:
            self._for_sale_set.add(unit)
            self.for_sale_units.append(unit)
        self.register_unit(unit)
        self._cache_active_for_sale = None

    def add_rental_unit(self, unit):
        """Register a unit in the rental stock exactly once."""
        if unit not in self._rental_set:
            self._rental_set.add(unit)
            self.rental_units.append(unit)
        self.register_unit(unit)

    def drop_rental_unit(self, unit):
        """Remove a unit that has left the rental channel."""
        if unit in self._rental_set:
            self._rental_set.discard(unit)
            self.rental_units.remove(unit)

    def _promote_to_first_time_buyer(self, renter):
        """Replace a renter with a FirstTimeBuyer carrying the same finances."""
        ftb = FirstTimeBuyer(
            model=self, income=renter.income, age=renter.age, tract_id=renter.tract_id
        )
        ftb.bank_balance = renter.bank_balance
        ftb.desired_balance = renter.desired_balance

        renter.remove()

    def _promote_to_repeat_buyer(self, ftb):
        """Replace a FirstTimeBuyer with a RepeatBuyer, carrying over the house."""
        rb = RepeatBuyer(
            model=self, income=ftb.income, age=ftb.age, tract_id=ftb.tract_id
        )
        rb.bank_balance = ftb.bank_balance
        rb.desired_balance = ftb.desired_balance
        rb.house = ftb.house
        rb.house.owner = rb
        rb.status = "owning"
        rb.owned_since_month = ftb.owned_since_month

        ftb.remove()

    def houses_per_capita(self, tract_id):
        """Live sale listings per household (EQ 6 supply signal).
        TODO: make per tract once multiple tracts exist."""
        if self._cache_houses_per_capita is None:
            self._cache_houses_per_capita = len(self.active_for_sale()) / max(
                1, self.n_household_agents()
            )
        return self._cache_houses_per_capita

    def houses_per_capita_avg(self, tract_id):
        """Trailing 24-month mean of houses_per_capita (current value if no history)."""
        history = self.houses_per_capita_history.get(tract_id, [])
        if not history:
            return self.houses_per_capita(tract_id)
        return float(np.mean(history))
