"""Construction (Section 3.1 step 2, Appendix A1) and institutional investor
replenishment. New units are priced at the tract's current price per quality."""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.agents.institutional_investor import InstitutionalInvestor
from housing_abm.demographics import HOUSEHOLD_TYPES


def run_construction(model):
    """Build new for-sale units whenever stock falls below the target per household."""
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
    """Keep the institutional investor count proportional to households
    (small landlords are replenished through births)."""
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
