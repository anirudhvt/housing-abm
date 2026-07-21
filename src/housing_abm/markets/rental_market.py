""" EQ 11-based rental market: placeholder stock"""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.rental_pricing import sample_lease_length, small_landlord_rent
from housing_abm.equations.market_matching import sample_bid_up_multiplier, max_rounds


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


def _settle_lease(model, unit, winner, final_rent):
    """Assign house to winner and start a lease"""
    unit.rent = final_rent
    unit.tenant = winner
    unit.on_rental_market = False
    winner.house = unit
    winner.status = "renting"
    lease_length = sample_lease_length(model.random_gen)
    #to avoid leases lining up, on the first step of the model we give agents a varied head start
    if getattr(winner, "_ever_leased", False): #randomly start somewhere in the lease
        lease_length = int(model.random_gen.integers(1, lease_length + 1))
    winner._ever_leased = True #flag so we don't do this again
    winner.lease_months_remaining = lease_length

def run_rental_market(model):
    """Multi round double auction clearing of queued renters"""
    vacant_units = [unit for unit in model.rental_units if unit.on_rental_market and unit.tenant is None]
    if not vacant_units or not model._rental_bid_queue: #no houses or no renters
        return

    #each queued agent's affordable rent, based on affordable fraction
    #33% default 

    bids = {}
    for agent in model._rental_bid_queue:
        fraction = getattr(agent, "rent_affordability_fraction", 0.33)
        bids[agent] = fraction*agent.income #raw amount of money bid

    auction_cfg = model.params["market_clearing_a4"]
    #see the agents and houses on the market
    remaining_agents = list(model._rental_bid_queue)
    remaining_units = list(vacant_units)

    n_rounds = max_rounds(
        n_bids = len(remaining_agents), n_offers = len(remaining_units),
        n_households = len(model.agents), round_floor = auction_cfg["round_floor"]
    )

    matched_agents = []

    for _round in range(n_rounds):
        if not remaining_agents or not remaining_units: #bidders or houses ran out
            break

        #phase 1: remaining renters claim best quality unit they can afford
        claims = {} #unit -> list of agents
        for agent in remaining_agents: 
            affordable = [u for u in remaining_units if bids[agent] >= u.rent]
            if not affordable: #nothing on the market is cheap enough
                continue
            best_unit = max(affordable, key=lambda u: u.quality) 
            claims.setdefault(best_unit, []).append(agent)

        if not claims:
            break #no one can afford anythnig

        #Phase 2: resolve each claimed unit
        leased_units = []
        for unit, claimants in claims.items():
            if len(claimants) == 1: #only 1 person wants that house
                winner = claimants[0]
                final_rent = unit.rent
            else:  #bid up to settle ties
                multiplier = sample_bid_up_multiplier(
                    model.random_gen, n_bids = len(claimants),
                    bid_up_pct = auction_cfg["bid_up_pct"],
                    arrival_window_days = auction_cfg["arrival_window_days"],
                    month_days = auction_cfg["month_days"],
                    max_multiplier = auction_cfg["max_multiplier"],
                )
                bid_up_rent = unit.rent * multiplier 
                still_afford = [a for a in claimants if bids[a] >= bid_up_rent] #agents who can still afford the house
                if not still_afford:
                    continue #priced everyone out, try again later
                winner = model.random_gen.choice(still_afford) #choose a random person to get the house
                final_rent = bid_up_rent
            #assign winner their house
            _settle_lease(model, unit, winner, final_rent) 
            matched_agents.append(winner)
            leased_units.append(unit)

        remaining_agents = [a for a in remaining_agents if a not in matched_agents]
        remaining_units = [u for u in remaining_units if u not in leased_units]

        #unmatched bidders resubmit next month
        model._rental_bid_queue = [] #clear queue to prevent carryover issues



    
        


        