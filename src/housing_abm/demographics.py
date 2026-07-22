"""demographics: birth, mortality, aging, inheritance"""

import numpy as np

from housing_abm.agents.renter import Renter
from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord

# don't include investor, modeleded as a fund/firm, not a mortal household - doesn't die
HOUSEHOLD_TYPES = (Renter, FirstTimeBuyer, RepeatBuyer, SmallLandlord)


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
    n_births = model.random_gen.poisson(expected_births)
    age_lo, age_hi = cfg["new_household_age_range"]
    for _ in range(n_births):  # create new households within age range, start as renter
        age = int(model.random_gen.integers(age_lo, age_hi))
        income = float(model.random_gen.lognormal(mean=9.4, sigma=0.55))
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
        if model.random_gen.random() < monthly_death_probability(a.age, cfg)
    ]  #
    # randomly select deceased people

    for agent in deceased:
        heirs = [h for h in households if h is not agent and h not in deceased]
        if not heirs:  # no one to take money
            _liquidate_estate(model, agent)
            agent.remove()
            continue
        # transfer to new heir
        heir = heirs[model.random_gen.integers(0, len(heirs))]
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
            if unit not in model.rental_units:
                model.rental_units.append(unit)
        elif heir.house is None and isinstance(heir, (FirstTimeBuyer, RepeatBuyer)):
            # heir is currently unhoused: they move in directly
            heir.house = unit
            heir.status = "owning"
            heir.owned_since_month = model.current_month
        else:
            # heir already has a home of their own: list the inherited house for
            # sale rather than letting it vanish from the housing stock
            unit.on_sale_market = True
            model._resale_sellers[unit] = heir
            if unit not in model.for_sale_units:
                model.for_sale_units.append(unit)

    # renting tenancy ends
    if deceased.house is not None and deceased.status == "renting":
        deceased.house.tenant = None
        deceased.house.on_rental_market = True

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
                unit.on_sale_market = True
                unit.on_rental_market = False
                model._resale_sellers[unit] = heir
                if unit not in model.for_sale_units:
                    model.for_sale_units.append(unit)
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
    """No heir available, pull estate out of circulation"""
    _vacate_and_delist(model, agent)
    if agent.house is not None:
        agent.house.owner = None
        agent.house.on_sale_market = False
        agent.house.on_rental_market = False
    for unit in getattr(agent, "properties", None) or []:
        if unit.tenant is not None:
            _evict_tenant(model, unit)
        unit.owner = None
        unit.on_sale_market = False
        unit.on_rental_market = False
