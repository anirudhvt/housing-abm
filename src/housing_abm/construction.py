"""Section 3.1 step 2 + Appendix A1
Adds new for sale stock whenever house to household ratio falls below target

Placeholder: new units priced at tract current pricing, rather than house price datasets
"""

from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.agents.renter import Renter
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord

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
    for _ in range(deficit):  # create new houses to fill the gap, put on sale market
        unit = HousingUnit(model=model, tract_id="tract_001", quality=1.0)
        unit.price = tract.price_per_quality
        unit.on_sale_market = True
        model.for_sale_units.append(unit)

def run_investor_replenishment(model):
    """Keeps investor population proportional to household base"""
    sim_cfg = model.params.get("simulation", {})
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))
 
    n_small_landlords = sum(1 for a in model.agents if isinstance(a, SmallLandlord))
    target_small_landlords = round(n_households * sim_cfg.get("small_landlord_fraction", 0.0))
    for _ in range(target_small_landlords - n_small_landlords): #fill in gap with normal landlords
        income = float(model.random_gen.lognormal(mean=9.8, sigma=0.5))
        age = int(model.random_gen.integers(30, 70))
        landlord = SmallLandlord(model=model, income=income, age=age, tract_id="tract_001")
        landlord.bank_balance = float(model.random_gen.lognormal(mean=11.5, sigma=0.6))
 
    n_institutional_investors = sum(
        1 for a in model.agents if isinstance(a, InstitutionalInvestor)
    )
    target_institutional_investors = round(
        n_households * sim_cfg.get("institutional_investor_fraction", 0.0)
    )
    for _ in range(target_institutional_investors - n_institutional_investors): #fill in gap with normal investors
        available_capital = float(model.random_gen.lognormal(mean=14.5, sigma=0.7))
        InstitutionalInvestor(
            model=model, available_capital=available_capital, tract_id="tract_001"
        )
