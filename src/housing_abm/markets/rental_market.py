"""Rental market: initial rental stock, vacancy repricing and multi-round
double-auction clearing of queued renters."""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.rental_pricing import (
    sample_lease_length,
    small_landlord_rent,
)
from housing_abm.equations.market_matching import sample_bid_up_multiplier, max_rounds
from housing_abm.markets.offer_book import OfferBook


def generate_placeholder_rental_stock(
    model, n_units: int = 150, base_rent: float = 1400.0
):
    """Initial rental stock, same quality distribution as the sale stock.
    TODO: replace with tract-based generation."""
    cfg = model.params.get("initial_sale_stock", {})
    quality_sigma = cfg.get("quality_sigma", 0.45)
    mu = -(quality_sigma ** 2) / 2  # E[quality] == 1
    tract = model.tracts["tract_001"]

    units = []
    for _ in range(n_units):
        quality = float(model.random_gen.lognormal(mean=mu, sigma=quality_sigma))
        unit = HousingUnit(model=model, tract_id="tract_001", quality=quality)
        unit.price = tract.price_per_quality * quality  # owners may sell later
        # placeholder rent: small noise around base rent
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
    """Let the unit to the winner, record the letting, and start a lease."""
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
    # agents who have leased before get a random length so expiries stagger
    if getattr(winner, "_ever_leased", False):
        lease_length = int(model.random_gen.integers(1, lease_length + 1))
    winner._ever_leased = True
    winner.lease_months_remaining = lease_length


def run_rental_market(model):
    """Cut rents on units vacant since last month, then clear queued renters."""
    reduction = model.params.get("rental_market", {}).get("vacant_rent_reduction", 0.0)
    for unit in model.rental_units:
        if unit.tenant is None and unit.on_rental_market:
            # rental analogue of EQ 8; new listings get one month at full rent
            if reduction and unit.day_vacant > 0 and unit.rent:
                unit.rent *= 1.0 - reduction
            unit.day_vacant += 1

    # units in their void period count as vacant but can't be let yet
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
    if not vacant_units or not model._rental_bid_queue:
        return

    # each renter bids a fixed share of income (default 33%)
    bids = {}
    for agent in model._rental_bid_queue:
        fraction = getattr(agent, "rent_affordability_fraction", 0.33)
        bids[agent] = fraction * agent.income

    auction_cfg = model.params["market_clearing_a4"]
    remaining_agents = list(model._rental_bid_queue)
    remaining_units = list(vacant_units)

    n_rounds = max_rounds(
        n_bids=len(remaining_agents),
        n_offers=len(remaining_units),
        n_households=len(model.agents),
        round_floor=auction_cfg["round_floor"],
    )

    matched_agents = []
    rejected = {}  # agent -> units they were outbid on this month

    for _round in range(n_rounds):
        if not remaining_agents or not remaining_units:
            break

        # phase 1: each renter claims the best-quality unit it can afford
        book = OfferBook(
            remaining_units, lambda u: u.quality, model.random_gen, price_attr="rent"
        )
        claims = {}  # unit -> list of agents
        for agent in remaining_agents:
            best_unit = book.best_affordable(
                bids[agent], excluded=rejected.get(agent)
            )
            if best_unit is None:
                continue
            claims.setdefault(best_unit, []).append(agent)

        if not claims:
            break

        # phase 2: as in the ownership market, contested units are bid up
        leased_units = []
        for unit, claimants in claims.items():
            if len(claimants) == 1:
                winner = claimants[0]
                final_rent = unit.rent
                losers = []
            else:
                multiplier = sample_bid_up_multiplier(
                    model.random_gen,
                    n_bids=len(claimants),
                    bid_up_pct=auction_cfg["bid_up_pct"],
                    arrival_window_days=auction_cfg["arrival_window_days"],
                    month_days=auction_cfg["month_days"],
                    max_multiplier=auction_cfg["max_multiplier"],
                )
                bid_up_rent = unit.rent * multiplier
                still_afford = [a for a in claimants if bids[a] >= bid_up_rent]
                if not still_afford:
                    for a in claimants:
                        rejected.setdefault(a, set()).add(unit)
                    continue
                winner = model.random_gen.choice(still_afford)
                final_rent = bid_up_rent
                losers = [a for a in claimants if a != winner]
            # losers try a different unit next round
            for a in losers:
                rejected.setdefault(a, set()).add(unit)
            _settle_lease(model, unit, winner, final_rent)
            matched_agents.append(winner)
            leased_units.append(unit)

        matched_ids = {id(a) for a in matched_agents}
        leased_ids = {id(u) for u in leased_units}
        remaining_agents = [a for a in remaining_agents if id(a) not in matched_ids]
        remaining_units = [u for u in remaining_units if id(u) not in leased_ids]

        # unmatched renters re-queue next month
        model._rental_bid_queue = []
