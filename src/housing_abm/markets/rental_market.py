""" EQ 11-based rental market: placeholder stock"""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.rental_pricing import sample_lease_length, small_landlord_rent


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
    single-round bidding, price stays static
    TODO: multi-round bid convergence"""
    vacant_units = [unit for unit in model.rental_units if unit.on_rental_market and unit.tenant is None]
    if not vacant_units or not model._rental_bid_queue: #no houses or no renters
        return

    #each queued agent's affordable rent, based on affordable fraction
    #33% default 

    bids = {}
    for agent in model._rental_bid_queue:
        fraction = getattr(agent, "rent_affordability_fraction", 0.33)
        bids[agent] = fraction*agent.income #raw amount of money bid

    model.random_gen.shuffle(vacant_units)
    matched_agents = []

    for unit in vacant_units: #loop through houses and find highest bidder
        eligible = [(agent, bid) for agent, bid in bids.items() #all bidders who can afford and want this house
                     if bid >= unit.rent and agent not in matched_agents]
        if not eligible:
            continue #no one eligible, unit stays unmatched

        winner, winning_bid = max(eligible, key = lambda pair: pair[1]) 

        #assign house to winner
        unit.tenant = winner
        unit.on_rental_market = False
        winner.house = unit
        winner.status = "renting"
        lease_length = sample_lease_length(model.random_gen)
        #to avoid leases lining up, on the first step of the model we give agents a varied head start
        if getattr(winner, "_ever_leased", False): #randomly start somewhere in the lease
            lease_length = int(model.random_gen.integers(1, lease_length+1))
        winner._ever_leased = True #flag so we don't do this again
        winner.lease_months_remaining = lease_length

        matched_agents.append(winner) #done looking for a house


    for agent in matched_agents:
        model._rental_bid_queue.remove(agent)
        


        