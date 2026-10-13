"""demographics: birth, mortality, aging, inheritance"""

import numpy as np

from housing_abm.agents.renter import Renter
from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord
from housing_abm.equations.investor_propensity import landlord_birth_probability

# don't include investor, modeleded as a fund/firm, not a mortal household - doesn't die
HOUSEHOLD_TYPES = (Renter, FirstTimeBuyer, RepeatBuyer, SmallLandlord)


def annual_death_probability(age, cfg: dict):
    """Logistic age hazard, scaled by the calibration constant."""
    midpoint = cfg["age_midpoint"]
    scale = cfg["age_scale"]
    base = cfg["base_annual_rate"]
    annual_rate = base + (1 - base) / (1 + np.exp(-(age - midpoint) / scale))
    return np.minimum(annual_rate * cfg["mortality_scale"], 0.99)


def stationary_age_distribution(cfg: dict, entry_lo: int, entry_hi: int,
                                max_age: int = 140):
    """Age distribution of a population in demographic steady state.

    Drawing initial ages uniformly over [22, 65) -- as the model previously
    did -- seeds a population with no one near the mortality midpoint, so
    deaths stay near zero for several decades while births continue. The
    population then grows, hits the hazard wall together as one cohort, and
    collapses. That transient is longer than the whole simulation, so no
    spin-up length reaches a steady state and every reported outcome is
    measured mid-drift.

    Sampling from the stationary distribution instead starts the model where
    the spin-up was supposed to end. Returns (ages, probabilities).
    """
    ages = np.arange(0, max_age + 1)
    hazard = annual_death_probability(ages, cfg)
    survival = np.concatenate([[1.0], np.cumprod(1.0 - hazard)])
    weights = np.zeros(max_age + 1)
    for a0 in range(entry_lo, entry_hi):
        weights[a0:] += survival[a0:max_age + 1] / survival[a0]
    total = weights.sum()
    return ages, weights / total


def sample_stationary_ages(rng, n: int, cfg: dict, entry_lo: int, entry_hi: int):
    ages, probs = stationary_age_distribution(cfg, entry_lo, entry_hi)
    return rng.choice(ages, size=n, p=probs)


def implied_death_rate(cfg: dict, entry_lo: int, entry_hi: int) -> float:
    """Annual death rate implied by the hazard, i.e. 1 / mean household life.

    A stable population needs this to equal the birth rate; Baptista et al.
    scale the mortality pdf by a constant to enforce exactly that.
    """
    _, probs = stationary_age_distribution(cfg, entry_lo, entry_hi)
    ages = np.arange(0, 141)
    survival_mass = 0.0
    hazard = annual_death_probability(ages, cfg)
    survival = np.concatenate([[1.0], np.cumprod(1.0 - hazard)])
    for a0 in range(entry_lo, entry_hi):
        survival_mass += (survival[a0:141] / survival[a0]).sum()
    mean_lifetime = survival_mass / (entry_hi - entry_lo)
    return 1.0 / mean_lifetime


def monthly_death_probability(age: float, cfg: dict) -> float:
    """Logistic age-dependent function, scaled by a calibration constant."""
    midpoint = cfg["age_midpoint"]
    scale = cfg["age_scale"]
    base = cfg["base_annual_rate"]
    annual_rate = base + (1 - base) / (1 + np.exp(-(age - midpoint) / scale))
    annual_rate = min(annual_rate * cfg["mortality_scale"], 0.99)  # can't be too high
    return 1 - (1 - annual_rate) ** (1.0 / 12.0)  # convert to monthly


def process_aging_and_births(model):
    """Age every household by a year every 12 steps
    draw ner births as poisson process on annual birth rate"""
    if model.current_month % 12 == 0:  # year has passed
        for agent in list(model.agents):
            if isinstance(agent, HOUSEHOLD_TYPES):
                agent.age += 1

    cfg = model.params["demographics"]
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))
    expected_births = n_households * cfg["birth_rate_annual"] / 12.0
    n_births = model.rng_demography.poisson(expected_births)

    # Households pushed out by rent burden leave the metro permanently. Left
    # uncompensated that is a pure population sink of ~0.4%/yr on top of
    # mortality, which tips the birth/death balance negative and makes the
    # population -- and every rate computed against it -- drift downward for
    # the whole run. Real metros replace out-migrants with in-migrants, so
    # they are replaced one-for-one here.
    n_replacements = model.households_displaced_pending
    model.households_displaced_pending = 0

    age_lo, age_hi = cfg["new_household_age_range"]
    income_cfg = model.params.get("income_distribution", {})
    propensity_cfg = model.params.get("investor_propensity_eq", {})
    sim_cfg = model.params.get("simulation", {})
    target_fraction = sim_cfg.get("small_landlord_fraction", 0.0)
    decile_probs = propensity_cfg.get(
        "small_landlord_income_decile_probs", [0.1] * 10
    )
    mu = income_cfg.get("household_lognormal_mean", 8.6)
    sigma = income_cfg.get("household_lognormal_sigma", 0.65)

    for _ in range(n_births + n_replacements):
        age = int(model.rng_demography.integers(age_lo, age_hi))
        income = float(model.rng_demography.lognormal(mean=mu, sigma=sigma))
        p_landlord = landlord_birth_probability(
            income, target_fraction, decile_probs, mu, sigma,
        )
        if model.rng_demography.random() < p_landlord:
            landlord = SmallLandlord(
                model=model, income=income, age=age, tract_id="tract_001",
            )
            landlord.bank_balance = float(
                model.rng_demography.lognormal(mean=11.5, sigma=0.6)
            )
        else:
            Renter(model=model, income=income, age=age, tract_id="tract_001")


def process_deaths(model):
    """kill households based on age, transfer wealth to random living heir"""
    cfg = model.params["demographics"]["mortality"]
    households = [a for a in model.agents if isinstance(a, HOUSEHOLD_TYPES)]
    if len(households) < 2:
        return  # need at least one potential heir

    deceased = [
        a
        for a in households
        if model.rng_demography.random() < monthly_death_probability(a.age, cfg)
    ]  #
    # randomly select deceased people

    for agent in deceased:
        heirs = [h for h in households if h is not agent and h not in deceased]
        if not heirs:  # no one to take money
            _liquidate_estate(model, agent)
            agent.remove()
            continue
        # transfer to new heir
        heir = heirs[model.rng_demography.integers(0, len(heirs))]
        _transfer_estate(model, agent, heir)
        agent.remove()


def _vacate_and_delist(model, agent):
    """Pull deceased out of bid queues to prevent posthumous matching"""
    # remove from rental bid and ownership bid queues
    if agent in model._rental_bid_queue:
        model._rental_bid_queue.remove(agent)
    model._ownership_bid_queue[:] = [
        b for b in model._ownership_bid_queue if b["agent"] is not agent
    ]


def _transfer_estate(model, deceased, heir):
    """heir gets financial wealth and housing
    renting tenancy terminated, mortgages written off"""
    _vacate_and_delist(model, deceased)
    heir.bank_balance += deceased.bank_balance

    # owner-occupied home
    if deceased.house is not None and deceased.status == "owning":
        unit = deceased.house
        unit.mortgage_principal = 0.0  # written off
        unit.mortgage_payment = 0.0
        unit.owner = heir  # transfer to new heir
        if getattr(heir, "properties", None) is not None:
            # heir is a landlord: treat the inherited home as another rental property
            heir.properties.append(unit)
            unit.tenant = None
            unit.on_rental_market = True
            if unit.rent is None:
                unit.rent = model.tracts[unit.tract_id].rent_per_quality * unit.quality
            model.add_rental_unit(unit)
        elif heir.house is None and isinstance(heir, (FirstTimeBuyer, RepeatBuyer)):
            # heir is currently unhoused: they move in directly
            heir.house = unit
            heir.status = "owning"
            heir.owned_since_month = model.current_month
        else:
            # heir already has a home of their own: list the inherited house for
            # sale rather than letting it vanish from the housing stock
            model.list_for_sale(unit, seller=heir)

    # renting tenancy ends
    if deceased.house is not None and deceased.status == "renting":
        model.end_tenancy(deceased.house)

    # landlord's rental portfolio
    if getattr(deceased, "properties", None):  # deceased is a landlord
        for unit in list(deceased.properties):  # transfer all properties to heir
            unit.owner = heir
            if getattr(heir, "properties", None) is not None:
                heir.properties.append(unit)
            else:
                # heir isn't set up to be a landlord: liquidate via the normal
                # resale channel instead of just handing them a rental unit
                if unit.tenant is not None:
                    # evict existing tenant before selling unit
                    _evict_tenant(model, unit)
                unit.on_rental_market = False
                model.drop_rental_unit(unit)
                model.list_for_sale(unit, seller=heir)
        deceased.properties.clear()

    # any pending listing (repeat buyer mid-sale) transfers its payout too
    house_to_sell = getattr(deceased, "house_to_sell", None)
    if house_to_sell is not None and house_to_sell in model._resale_sellers:
        model._resale_sellers[house_to_sell] = heir


def _evict_tenant(model, unit):
    """terminate existing tenancy on a unit about to change hands"""
    tenant = unit.tenant
    unit.tenant = None
    tenant.house = None
    tenant.status = "social_housing"
    model.queue_housing_decision(tenant)  # back into the market next step


def _liquidate_estate(model, agent):
    """No heir available, pull estate out of circulation.

    These units leave the housing stock entirely, so they must also leave the
    stock registry -- otherwise construction keeps counting demolished homes
    against its house-to-household target and under-builds forever.
    """
    _vacate_and_delist(model, agent)
    if agent.house is not None:
        if agent.status == "owning":
            _withdraw_unit(model, agent.house)
        else:
            # the deceased was a tenant: the unit belongs to someone else, so
            # release the tenancy rather than demolishing another agent's
            # property (which also left stale tenant pointers behind)
            model.end_tenancy(agent.house)
    for unit in getattr(agent, "properties", None) or []:
        if unit.tenant is not None:
            _evict_tenant(model, unit)
        _withdraw_unit(model, unit)


def _withdraw_unit(model, unit):
    """Remove a unit from every registry and from the housing stock."""
    unit.owner = None
    unit.on_sale_market = False
    unit.on_rental_market = False
    model.drop_rental_unit(unit)
    model.housing_units.discard(unit)
