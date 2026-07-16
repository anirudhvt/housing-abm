""" EQ 11-based rental market: placeholder stock"""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.rental_pricing import small_landlord_rent


def generate_placeholder_rental_stock(model, n_units: int = 150, base_rent: float = 1400.0): #if not given, provides default values
    "Creates fixed rental stock for skeleton market"
    #TODO: replace with tract-based generation 

    units = []
    for _ in range(n_units):
        unit = HousingUnit(model = model, tract_id = "tract_001", quality= 1.0)
        #placeholder rent, small noise around base rent
        unit.rent = small_landlord_rent(
            r_bar_tract  = base_rent, f_bar_tract = 0.0, alpha = 0.0,
            beta = 0.0, zeta = 1.0, epsilon_std = 0.05,
            reprice_prob = 1.0, previous_rent = None, rng=model.random_gen,
        )
        unit.on_rental_market = True
        units.append(unit)
    return units

def run_rental_market(model):
    """Matches queued renters against vacant units
    no bidding implemented yet"""
    vacant_units = [unit for unit in model.rental_units if unit.on_rental_market and unit.tenant is None]
    model.random_gen.shuffle(vacant_units)

    matched = []
    for agent in model._rental_bid_queue:
        if not vacant_units: 
            break #no more units this month, unmatched agents stay queued
        unit = vacant_units.pop()
        unit.tenant = agent
        unit.on_rental_market = False
        agent.house = unit
        agent.status = "renting"
        agent.lease_months_remaining = 12 #placeholder fixed lease length
        matched.append(agent)
    
    for agent in matched:
        model._rental_bid_queue.remove(agent)
        


        