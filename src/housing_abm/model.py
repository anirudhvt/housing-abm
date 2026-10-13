# top level Mesa model/initial skeleton, monthly cycle
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
            self.params = yaml.safe_load(f)  # take parameters from given config path

        # policies handled by policy.py

        # Independent RNG substreams.
        #
        # Common random numbers only reduce variance if the two arms of a
        # paired comparison actually consume the same randomness. With a
        # single stream, the first policy-induced decision shifts every
        # subsequent draw, so the baseline and policy arms of the same seed
        # get different interest-rate paths and different birth/death
        # sequences -- noise that has nothing to do with the policy but shows
        # up in full in the paired difference. Measured on the original
        # results, arm-to-arm correlation was ~0 (sometimes negative), so
        # pairing bought almost nothing.
        #
        # Splitting the exogenous processes into their own streams makes the
        # macro path and the demographic path bit-identical across arms for a
        # given seed, whatever the policy does, leaving the paired difference
        # to reflect the policy plus endogenous market noise only.
        seed_seq = np.random.SeedSequence(seed)
        market_seq, macro_seq, demo_seq = seed_seq.spawn(3)
        self.random_gen = np.random.default_rng(market_seq)  # market/behavioural
        self.rng_macro = np.random.default_rng(macro_seq)  # exogenous rate path
        self.rng_demography = np.random.default_rng(demo_seq)  # births/deaths
        self.current_month = 0
        self.spinup_end_month = 0

        #grab exogenous interest rates 
        cfg = self.params.get("fed_rate_process", {})
        self._fed_rate_mean = cfg.get("mean_annual", 0.0453)
        self._fed_rate_phi = cfg.get("monthly_persistence", 0.98)
        self._fed_rate_sigma = cfg.get("monthly_noise_std", 0.0024)
        self.current_fed_rate_annual = self._fed_rate_mean
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
        # purchases this month by buyer class, so the investor-vs-first-time-buyer
        # competition the policies target can be read directly rather than
        # inferred from the homeownership stock
        self.purchases_this_month = {"ftb": 0, "repeat": 0, "small_landlord": 0,
                                     "institutional": 0}
        # households that left the metro on rent burden this month, replaced
        # one-for-one by in-migration in process_aging_and_births
        self.households_displaced_pending = 0
        self.households_displaced_total = 0

        # different types of mortgages
        self.mortgage_terms = {"fha": {}, "conventional": {}, "investor_dscr": {}}
        with open("config/mortgage_terms.yaml") as f:
            self.mortgage_terms = yaml.safe_load(f)

        # population actually requested for this run, so scale-relative policy
        # thresholds resolve against it rather than against the config default
        self._configured_n_households = n_households or self.params.get(
            "simulation", {}
        ).get("n_households", 100)

        # load LTV/LTI policies, mutates mortgage terms
        load_policies(self, policy_paths)

        # Real ZHVI/ZORI series (month-over-month growth) can optionally drive
        # appreciation/rent-growth exogenously instead of computing them
        # endogenously from the model's own transactions -- off by default,
        # this is a separate design choice from the price-level calibration
        # below (which is always applied). Previously this flag was read from
        # config but never actually consulted here -- external_g_series and
        # external_rent_growth_series were hardcoded to None regardless of
        # its value, so the full 2015-2026 ZHVI/ZORI history already checked
        # into the repo (atlanta_zillow_zhvi.csv / atlanta_zillow_zori.csv)
        # was never used beyond its 2019 mean (tract_calibration above).
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

        # Calibrated 2019 Atlanta price/rent level (see config/baseline_params.yaml
        # tract_calibration for provenance) and the reference model's smoothing
        # constants (market_smoothing) -- both ported from the Java reference
        # implementation. smoothing_factor is derived from the reference's
        # CUMULATIVE_WEIGHT_BEYOND_YEAR the same way its own Config.java does.
        smoothing_cfg = self.params.get("market_smoothing", {})
        reference_price = calib_cfg.get("reference_price_per_quality", 250_000.0)
        reference_rent = calib_cfg.get("reference_rent_per_quality", 1400.0)
        cumulative_weight_beyond_year = smoothing_cfg.get(
            "cumulative_weight_beyond_year", 0.25
        )
        smoothing_factor = 1.0 - cumulative_weight_beyond_year ** (1.0 / 12.0)
        price_decay = smoothing_cfg.get("market_average_price_decay", 0.5)
        rent_decay = smoothing_cfg.get("market_average_rent_decay", 0.5)

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
        }  # placeholder storage of tracts, multi-tract is a planned extension

        # trailing history of houses_per_capita per tract
        self.houses_per_capita_history = {tract_id: [] for tract_id in self.tracts}

        # queue of renters in social housing looking to rent again
        # -queue housing decision(), drained by run_rental_market() in step()

        self._rental_bid_queue = []
        self._ownership_bid_queue = []
        self._resale_sellers = (
            {}
        )  # HouseholdUnit -> agent, keeps tracks of resale listings

        # identity sets mirroring the two channel registries, so membership
        # tests are O(1) instead of a linear scan over the whole housing stock
        self._for_sale_set = set()
        self._rental_set = set()
        # authoritative stock: every unit that exists, in any tenure
        self.housing_units = set()

        # per-step memo slots (see _invalidate_step_caches)
        self._cache_incomes = None
        self._cache_houses_per_capita = None
        self._cache_n_households = None
        self._cache_active_for_sale = None

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
        # Create initial household population.
        # SmallLandlords are selected FROM the N households (weighted by the
        # SCF income-decile propensity curve), not appended on top. This
        # matches the reference Java model's BTL gene mechanism and keeps
        # the household count at exactly N for construction targeting.
        n_households = n_households or self.params.get("simulation", {}).get(
            "n_households", 100
        )
        demo_cfg = self.params["demographics"]
        entry_lo, entry_hi = demo_cfg["new_household_age_range"]
        initial_ages = sample_stationary_ages(
            self.random_gen, n_households, demo_cfg["mortality"], entry_lo, entry_hi
        )
        income_cfg = self.params.get("income_distribution", {})
        propensity_cfg = self.params.get("investor_propensity_eq", {})
        sim_cfg = self.params.get("simulation", {})

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

        # InstitutionalInvestors are corporate entities, separate from the
        # household population and excluded from construction targeting.
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

        

        # placholder exogenous rental stock
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
    # per-step caches
    #
    # ftb_income_cutoff() and houses_per_capita() are each called once per
    # agent per step but depend only on model-level state that is fixed
    # within a step. Recomputing them per agent made the step O(N^2), which
    # capped the population at a size where sampling noise swamped every
    # policy effect. They are memoised here and invalidated once per step.
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
        """Units actually listed for sale right now.

        model.for_sale_units is an append-only registry; sold units keep
        their entry with on_sale_market False. Anything reading "current
        supply" has to filter, or it reads cumulative construction instead.
        """
        if self._cache_active_for_sale is None:
            self._cache_active_for_sale = [
                u for u in self.for_sale_units if u.on_sale_market
            ]
        return self._cache_active_for_sale

    def _prune_registries(self):
        """Drop stale entries so the registries track live stock.

        for_sale_units previously grew monotonically because _settle_purchase
        never removed a sold unit, so len(for_sale_units) measured every house
        ever built. rental_units likewise retained units that had been sold to
        owner-occupiers, inflating the denominator of the rental vacancy rate.
        """
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
        self.current_month += 1
        self.bankruptcy_injections_this_month = 0
        self.purchases_this_month = dict.fromkeys(self.purchases_this_month, 0)
        self._prune_registries()
        self._invalidate_step_caches()

        # monthly cycle: demographics -> construction -> households decide
        # ownership market -> rental market -> interest rate update


        #exogenous fed rate
        noise = self.rng_macro.normal(0.0, self._fed_rate_sigma)
        self.current_fed_rate_annual = self._fed_rate_mean + self._fed_rate_phi * (
            self.current_fed_rate_annual - self._fed_rate_mean
        ) + noise
        self.current_fed_rate_annual = max(self.current_fed_rate_annual, 0.0005)
        self.current_fed_rate_monthly = self.current_fed_rate_annual / 12

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
        run_investor_replenishment(self)
        apply_forced_divestiture(self) #hard owernship cap, force selling

        self.agents.shuffle_do("step")
        # match queued renters against vacant rental stock

        run_ownership_market(self)  # buyers get first chance

        for tract in self.tracts.values():
            tract.update_hpi_history()

        run_rental_market(self)

        # rent-side mirror of update_hpi_history above: form this month's
        # rent signal from the lettings that just cleared
        for tract in self.tracts.values():
            tract.update_rent_history()

        update_ownership_cap_soft_state(self) #roll soft-cap counters

        # EQ 16 mortgage rate update
        update_mortgage_rate(self)

        self.datacollector.collect(self)

    def run_spinup(self, n_months: int):
        """Run model for n_months, discard data to settle endogenous dynamics"""
        self.spinup_end_month = self.current_month + n_months
        for _ in range(n_months):
            self.step()
        for key in self.datacollector.model_vars:
            self.datacollector.model_vars[key] = []

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

    def _appreciation_g(self, alpha: float = 1.0):
        """EQ 4 aggregated across tracts - mean trailing appreication.
          Returns none until every tract has 15 months of hpi history"""
        values = [t.appreciation_g(alpha = alpha) for t in self.tracts.values()]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None 
        


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
        total = self._household_incomes().size
        return owners / total if total > 0 else 0.0

    def _investor_purchase_share(self):
        raise NotImplementedError

    def _rental_vacancy_rate(self):
        """Vacant share of the *rental* stock.

        The denominator must exclude units that have left the rental channel
        (sold to an owner-occupier); _prune_registries keeps that invariant.
        """
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
        incomes = self._household_incomes()  # cached per step
        return (
            float(np.quantile(incomes, floor_share_p_floor)) if incomes.size else 0.0
        )  # return the income you need to be above the cutoff

    # placeholder hooks
    def queue_housing_decision(self, agent):
        # mark and queue an agent in social housing for the housing-market process
        if isinstance(agent, Renter):  # seeing if can become first time buyer
            if self.renter_qualifies_as_buyer(agent):
                self._promote_to_first_time_buyer(agent)
                return
        if agent not in self._rental_bid_queue:  # everyone else
            self._rental_bid_queue.append(agent)

    def renter_qualifies_as_buyer(self, agent) -> bool:
        """Can this renter both fund a down payment and carry a mortgage?

        The monthly cycle (Section 2.C, step 4) specifies two gates on the
        renter -> first-time-buyer transition: accumulated savings above the
        minimum down payment, *and* income sufficient to qualify for a mortgage
        under the EQ 14 affordability and loan-to-income constraints. Only the
        savings gate was implemented.

        With the income gate missing, any renter holding the down payment on
        the cheapest listing was promoted, including households whose income
        could never support the loan. They then sat in the first-time-buyer
        pool indefinitely without transacting: at the default configuration
        the median first-time buyer's income was half the population median,
        and 151 first-time buyers between them produced about one purchase a
        month. That pool dilutes every first-time-buyer statistic with
        households who were never going to buy, which inflates the variance of
        the study's headline outcome without carrying any signal.
        """
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

    def exit_tract(self, agent):  # remove agent from schedule
        # occurs if rent burden became high or they moved away
        self.households_displaced_pending += 1
        self.households_displaced_total += 1
        # vacate unit if exists, then remove from market
        if agent.house is not None:
            self.end_tenancy(agent.house)
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
        self, agent, max_price: float, down_payment: float, acquisition_tax: float = 0.0
    ):  # creates a bid
        """Register a bid.

        acquisition_tax is cash the buyer must hand over on top of the down
        payment (a purchase tax). It is tracked separately because it is not
        equity in the house: folding it into down_payment would shrink the
        mortgage principal, so the tax would silently reduce the buyer's
        borrowing instead of costing them anything.
        """
        self._ownership_bid_queue.append(
            {
                "agent": agent,
                "max_price": max_price,
                "down_payment": down_payment,
                "acquisition_tax": acquisition_tax,
            }
        )

    def queue_listing(self, unit, seller):  # queues a house for sale
        """Register a resale listing, tracks who gets paid out once someone buys it.

        This previously only recorded the seller, without adding the unit to
        model.for_sale_units. run_ownership_market only ever iterates that
        registry, so every resale listing -- repeat buyers trading up, landlord
        and investor disposals, and hard-cap forced divestiture -- was flagged
        for sale but never actually offered to buyers. That severed the supply
        -release channel every financial-penalty policy depends on, and left
        repeat buyers permanently stuck carrying an unsellable home.
        """
        self.list_for_sale(unit, seller=seller)

    def _assign_initial_rental_ownership(self):
        """Distribute the initial rental stock across landlords and investors.

        generate_placeholder_rental_stock leaves every unit with owner=None.
        Nothing ever gives those units an owner, so they sit permanently
        tenanted and outside every decision rule: no agent can ever sell them,
        they never reach the sale market, and no investor policy can touch
        them. At the default configuration that is ~35% of the housing stock
        held inert, which both caps the attainable homeownership rate and
        dilutes every measured policy effect by the same factor.

        Shares come from config so the realized institutional share of the
        rental stock is a calibration target rather than an accident.
        """
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
        """Release a tenancy and start the unit's frictional void period.

        Without a void, a unit vacated during the agent step is re-let by the
        rental market in the same month, so turnover produces no vacancy at
        all and the measured rental vacancy rate sits at ~0 in every run. That
        both mis-states the level against the ACS figure and censors one of
        the three reported outcomes at a floor, where a policy can move it in
        only one direction.
        """
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
        """Track a unit in the authoritative housing stock.

        for_sale_units and rental_units are channel registries and are pruned,
        so neither can stand in for total stock -- an owner-occupied home
        belongs to neither. Construction sizes its build against this.
        """
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
        """Listings currently on the sale market per household agent.

        EQ 6 reads this as the market-supply signal, so it has to be live
        listings, not the cumulative count of units ever constructed.
        TODO: make per tract once multiple tracts exist
        """
        if self._cache_houses_per_capita is None:
            self._cache_houses_per_capita = len(self.active_for_sale()) / max(
                1, self.n_household_agents()
            )
        return self._cache_houses_per_capita

    def houses_per_capita_avg(self, tract_id):
        """placeholder, no rolling history tracked yet"""
        history = self.houses_per_capita_history.get(tract_id, [])
        if not history:
            return self.houses_per_capita(
                tract_id
            )  # if no history, return current value
        return float(np.mean(history))  # average of last 12 months
