"""Section 3.1 step 2 + Appendix A1
Adds new for sale stock whenever house to household ratio falls below target

Placeholder: new units priced at tract current pricing, rather than house price datasets
"""

from housing_abm.agents.first_time_buyer import FirstTimeBuyer
from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.agents.renter import Renter
from housing_abm.agents.repeat_buyer import RepeatBuyer
from housing_abm.agents.small_landlord import SmallLandlord

HOUSEHOLD_TYPES = (Renter, FirstTimeBuyer, RepeatBuyer, SmallLandlord)


def run_construction(model):
    cfg = model.params["construction"]
    n_households = sum(1 for a in model.agents if isinstance(a, HOUSEHOLD_TYPES))
    total_units = len(model.for_sale_units) + len(model.rental_units)
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
