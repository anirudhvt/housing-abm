"""Section 3.1 step 2 + Appendix A1
Adds new for sale stock whenever house to household ratio falls below target.
Investor replenishment now CONVERTS existing renters into landlords (BTL gene
approach from the reference) instead of appending extra agents."""

from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.agents.renter import Renter
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord
from housing_abm.equations.investor_propensity import select_future_landlords

HOUSEHOLD_TYPES = (Renter, FirstTimeBuyer, RepeatBuyer, SmallLandlord)


def run_construction(model):
    cfg = model.params["construction"]
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))
    total_units = len(set(model.for_sale_units) | set(model.rental_units))
    target_units = cfg["target_house_to_household_ratio"] * n_households
    deficit = int(round(target_units - total_units))
    if deficit <= 0:
        return

    tract = model.tracts["tract_001"]
    for _ in range(deficit):
        unit = HousingUnit(model=model, tract_id="tract_001", quality=1.0)
        unit.price = tract.price_per_quality
        unit.on_sale_market = True
        model.for_sale_units.append(unit)


def _convert_renter_to_landlord(model, renter):
    """Convert an existing Renter into a SmallLandlord, preserving financial state."""
    landlord = SmallLandlord(
        model=model, income=renter.income, age=renter.age, tract_id=renter.tract_id
    )
    landlord.bank_balance = renter.bank_balance
    landlord.desired_balance = renter.desired_balance

    if renter.house is not None:
        renter.house.tenant = None
        renter.house.on_rental_market = True
        renter.house = None

    if renter in model._rental_bid_queue:
        model._rental_bid_queue.remove(renter)
    model._ownership_bid_queue[:] = [
        b for b in model._ownership_bid_queue if b["agent"] is not renter
    ]

    renter.remove()
    return landlord


def run_investor_replenishment(model):
    """Keeps investor population proportional to household base.
    SmallLandlords are CONVERTED from existing renters (BTL gene),
    not appended as extra agents."""
    sim_cfg = model.params.get("simulation", {})
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))

    n_small_landlords = sum(1 for a in model.agents if isinstance(a, SmallLandlord))
    target_small_landlords = round(
        n_households * sim_cfg.get("small_landlord_fraction", 0.0)
    )
    deficit = target_small_landlords - n_small_landlords
    if deficit > 0:
        eligible = [
            a for a in model.agents
            if isinstance(a, Renter) and a.status == "social_housing"
        ]
        if not eligible:
            eligible = [a for a in model.agents if isinstance(a, Renter)]
        selected = select_future_landlords(eligible, deficit, model.random_gen)
        for renter in selected:
            _convert_renter_to_landlord(model, renter)

    n_institutional = sum(
        1 for a in model.agents if isinstance(a, InstitutionalInvestor)
    )
    target_institutional = round(
        n_households * sim_cfg.get("institutional_investor_fraction", 0.0)
    )
    for _ in range(target_institutional - n_institutional):
        available_capital = float(model.random_gen.lognormal(mean=14.5, sigma=0.7))
        InstitutionalInvestor(
            model=model, available_capital=available_capital, tract_id="tract_001"
        )
