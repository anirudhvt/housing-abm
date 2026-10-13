"""EQ 11-based rental market: placeholder stock"""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.rental_pricing import (
    sample_lease_length,
    small_landlord_rent,
)
from housing_abm.equations.market_matching import sample_bid_up_multiplier, max_rounds
from housing_abm.markets.offer_book import OfferBook


def generate_placeholder_rental_stock(
    model, n_units: int = 150, base_rent: float = 1400.0
):  # if not given, provides default values
    "Creates fixed rental stock for skeleton market"
    # TODO: replace with tract-based generation

    # same mean-preserving lognormal quality spread as the sale stock, so
    # rental and owner-occupied units are drawn from one housing distribution
    cfg = model.params.get("initial_sale_stock", {})
    quality_sigma = cfg.get("quality_sigma", 0.45)
    mu = -(quality_sigma ** 2) / 2  # E[quality] == 1
    tract = model.tracts["tract_001"]

    units = []
    for _ in range(n_units):
        quality = float(model.random_gen.lognormal(mean=mu, sigma=quality_sigma))
        unit = HousingUnit(model=model, tract_id="tract_001", quality=quality)
        # rental units need a sale price too: once they have an owner, that
        # owner can decide to sell them, and the sale market needs a price
        unit.price = tract.price_per_quality * quality
        # placeholder rent, small noise around base rent
        unit.rent = small_landlord_rent(
            r_bar_tract=base_rent * quality,
            f_bar_tract=0.0,
            alpha=0.0,
            beta=0.0,
            zeta=1.0,
            epsilon_std=0.05,
            reprice_prob=1.0,
            previous_rent=None,
            rng=model.random_gen,
        )
        unit.on_rental_market = True
        units.append(unit)
    return units


def _settle_lease(model, unit, winner, final_rent):
    """Assign house to winner and start a lease"""
    # the rent-side mirror of _settle_purchase's record_sale: this is the only
    # place a letting actually completes, so it is where the rental market's
    # price signal has to be formed (see Tract.update_rent_history)
    model.tracts[unit.tract_id].record_letting(
        rent=final_rent, quality=unit.quality, days_vacant=unit.day_vacant
    )
    unit.rent = final_rent
    unit.tenant = winner
    unit.on_rental_market = False
    unit.day_vacant = 0
    winner.house = unit
    winner.status = "renting"
    lease_length = sample_lease_length(model.random_gen)
    # to avoid leases lining up, on the first step of the model we give agents a varied head start
    if getattr(winner, "_ever_leased", False):  # randomly start somewhere in the lease
        lease_length = int(model.random_gen.integers(1, lease_length + 1))
    winner._ever_leased = True  # flag so we don't do this again
    winner.lease_months_remaining = lease_length


def run_rental_market(model):
    """Multi round double auction clearing of queued renters"""
    reduction = model.params.get("rental_market", {}).get("vacant_rent_reduction", 0.0)
    for unit in model.rental_units:
        if unit.tenant is None and unit.on_rental_market:
            # a unit that already sat unfilled last month cuts its asking rent
            # (reference 3.4.4, the rental analogue of EQ8). Applied before the
            # counter increments so a freshly-listed unit gets one clear month
            # at its EQ11 asking rent first.
            if reduction and unit.day_vacant > 0 and unit.rent:
                unit.rent *= 1.0 - reduction
            unit.day_vacant += 1

    # a unit still inside its void period counts as vacant stock but is not
    # yet available to let. Availability is read before the counter is
    # decremented, so a unit vacated during this month's agent step actually
    # sits out a month rather than being re-let immediately.
    vacant_units = [
        unit
        for unit in model.rental_units
        if unit.on_rental_market
        and unit.tenant is None
        and unit.void_months_remaining <= 0
    ]
    for unit in model.rental_units:
        if unit.void_months_remaining > 0:
            unit.void_months_remaining -= 1
    if not vacant_units or not model._rental_bid_queue:  # no houses or no renters
        return

    # each queued agent's affordable rent, based on affordable fraction
    # 33% default

    bids = {}
    for agent in model._rental_bid_queue:
        fraction = getattr(agent, "rent_affordability_fraction", 0.33)
        bids[agent] = fraction * agent.income  # raw amount of money bid

    auction_cfg = model.params["market_clearing_a4"]
    # see the agents and houses on the market
    remaining_agents = list(model._rental_bid_queue)
    remaining_units = list(vacant_units)

    n_rounds = max_rounds(
        n_bids=len(remaining_agents),
        n_offers=len(remaining_units),
        n_households=len(model.agents),
        round_floor=auction_cfg["round_floor"],
    )

    matched_agents = []
    rejected = {} #agent -> set of units they've lost a bid on this month

    for _round in range(n_rounds):
        if not remaining_agents or not remaining_units:  # bidders or houses ran out
            break

        # phase 1: remaining renters claim best quality unit they can afford.
        # Rent-sorted book with a prefix argmax over quality, so each renter
        # costs O(log n) instead of a scan over every vacant unit -- see
        # housing_abm.markets.offer_book.
        book = OfferBook(
            remaining_units, lambda u: u.quality, model.random_gen, price_attr="rent"
        )
        claims = {}  # unit -> list of agents
        for agent in remaining_agents:
            best_unit = book.best_affordable(
                bids[agent], excluded=rejected.get(agent)
            )
            if best_unit is None:  # nothing on the market is cheap enough
                continue
            claims.setdefault(best_unit, []).append(agent)

        if not claims:
            break  # no one can afford anythnig

        # Phase 2: resolve each claimed unit
        leased_units = []
        for unit, claimants in claims.items():
            if len(claimants) == 1:  # only 1 person wants that house
                winner = claimants[0]
                final_rent = unit.rent
                losers = []
            else:  # bid up to settle ties
                multiplier = sample_bid_up_multiplier(
                    model.random_gen,
                    n_bids=len(claimants),
                    bid_up_pct=auction_cfg["bid_up_pct"],
                    arrival_window_days=auction_cfg["arrival_window_days"],
                    month_days=auction_cfg["month_days"],
                    max_multiplier=auction_cfg["max_multiplier"],
                )
                bid_up_rent = unit.rent * multiplier
                still_afford = [
                    a for a in claimants if bids[a] >= bid_up_rent
                ]  # agents who can still afford the house
                if not still_afford:
                    for a in claimants: 
                        rejected.setdefault(a, set()).add(unit)
                    continue  # priced everyone out, try again later
                winner = model.random_gen.choice(
                    still_afford
                )  # choose a random person to get the house
                final_rent = bid_up_rent
                losers = [a for a in claimants if a != winner]
            #losers of contested unit try something else next time
            for a in losers:
                rejected.setdefault(a, set()).add(unit)
            # assign winner their house
            _settle_lease(model, unit, winner, final_rent)
            matched_agents.append(winner)
            leased_units.append(unit)

        matched_ids = {id(a) for a in matched_agents}
        leased_ids = {id(u) for u in leased_units}
        remaining_agents = [a for a in remaining_agents if id(a) not in matched_ids]
        remaining_units = [u for u in remaining_units if id(u) not in leased_ids]

        # unmatched bidders resubmit next month
        model._rental_bid_queue = []  # clear queue to prevent carryover issues
