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
    total_units = model.total_housing_stock()
    target_units = cfg["target_house_to_household_ratio"] * n_households
    deficit = int(round(target_units - total_units))
    if deficit <= 0:
        return

    tract = model.tracts["tract_001"]
    for _ in range(deficit):
        unit = HousingUnit(model=model, tract_id="tract_001", quality=1.0)
        unit.price = tract.price_per_quality
        model.list_for_sale(unit)


def run_investor_replenishment(model):
    """Keeps institutional investor population proportional to household base.

    SmallLandlord replenishment is handled by the demographic birth process
    (each newborn household is probabilistically assigned the landlord gene
    weighted by the SCF income-decile curve), matching the reference Java
    model's BTL mechanism. InstitutionalInvestors are corporate entities
    outside the household population and need explicit replenishment.
    """
    sim_cfg = model.params.get("simulation", {})
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))

    n_institutional_investors = sum(
        1 for a in model.agents if isinstance(a, InstitutionalInvestor)
    )
    target_institutional_investors = round(
        n_households * sim_cfg.get("institutional_investor_fraction", 0.0)
    )
    for _ in range(target_institutional_investors - n_institutional_investors):
        available_capital = float(model.random_gen.lognormal(mean=13.0, sigma=0.5))
        InstitutionalInvestor(
            model=model, available_capital=available_capital, tract_id="tract_001",
        )
