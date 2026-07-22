"""Placeholder ownership market, one round bidding
TODO: replace with multi-round matching"""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.market_matching import (
    sample_bid_up_multiplier,
    max_rounds,
    expected_gross_rental_yield,
)
from housing_abm.equations.selling import price_reduction
from housing_abm.policy import enforce_lti_policies


import numpy as np


def generate_placeholder_sale_stock(
    model,
    n_units: int | None = None,
    quality_mean: float | None = None,
    quality_sigma: float | None = None,
):
    """Generate a for sale stock with a realistic price spread"""
    cfg = model.params.get("initial_sale_stock", {})
    n_units = n_units if n_units is not None else cfg.get("n_units", 200)
    quality_mean = (
        quality_mean if quality_mean is not None else cfg.get("quality_mean", 1.0)
    )
    quality_sigma = (
        quality_sigma if quality_sigma is not None else cfg.get("quality_sigma", 0.45)
    )

    # mean-preserving lognormal draw:
    mu = np.log(quality_mean) - (quality_sigma**2) / 2
    tract = model.tracts["tract_001"]

    units = []
    for _ in range(n_units):
        quality = float(model.random_gen.lognormal(mean=mu, sigma=quality_sigma))
        unit = HousingUnit(model=model, tract_id="tract_001", quality=quality)
        unit.price = tract.price_per_quality * quality
        unit.on_sale_market = True
        units.append(unit)
    return units


def _is_investor(agent) -> bool:
    return getattr(agent, "properties", None) is not None


def _preferred_offer(bid, affordable_offers, model):
    """Phase 1 matching: investors want expected rental yield,
    everyone else wants highest quality"""
    if _is_investor(bid["agent"]):

        def yield_key(unit):
            tract = model.tracts[unit.tract_id]
            rent_estimate = (
                unit.rent
                if unit.rent is not None
                else tract.rent_per_quality * unit.quality
            )
            return expected_gross_rental_yield(
                rent_estimate, unit.price, tract.avg_days_on_market()
            )

        return max(
            affordable_offers, key=yield_key
        )  # search through affordable households for best yield
    return max(
        affordable_offers, key=lambda unit: unit.quality
    )  # everyone else looks for quality


def _settle_purchase(model, unit, agent, down_payment, final_price):
    """finish one winnig bid, assign all characteristics"""
    unit.price = final_price  # bid up may have raised price

    # if resale, settle previous owner.
    previous_owner = model._resale_sellers.pop(unit, None)
    if previous_owner is not None:
        payoff = unit.mortgage_principal
        proceeds = unit.price - payoff
        previous_owner.bank_balance += proceeds
        previous_owner.house_to_sell = None
        if (
            getattr(previous_owner, "properties", None) is not None
            and unit in previous_owner.properties
        ):
            # investor/landlord: just drop this one property, they may hold others
            previous_owner.properties.remove(unit)
        elif previous_owner.house is unit:
            # owner-occupier hasn't secured replacement
            previous_owner.house = None
            previous_owner.status = "social_housing"  # they look for a house

    # record sale
    model.tracts[unit.tract_id].record_sale(
        price=unit.price, quality=unit.quality, days_on_market=unit.days_on_market
    )

    # assign characteristics of the bought house according to the bid/loan terms
    principal = max(
        unit.price - down_payment, 0.0
    )  # never negative, regardless of  down_payment sizing
    i_r_monthly = model.mortgage_rate_monthly  # EQ16 spread-adjusted rate
    term_months = model.mortgage_terms[agent.LOAN_TYPE]["term_months"]
    unit.mortgage_principal = principal
    unit.mortgage_payment = model.monthly_payment(principal, i_r_monthly, term_months)
    unit.mortgage_rate = i_r_monthly
    model._monthly_new_lending += (
        principal  # feeds EQ16's spread update at end of month
    )

    unit.owner = agent
    unit.on_sale_market = False
    unit.days_on_market = 0
    agent.bank_balance -= down_payment
    if _is_investor(agent):
        # investor/landlord: accumulate, don't overwrite prior purchases
        agent.properties.append(unit)
        # bought from for_sale stock -> now needs to sell on the rental market
        unit.tenant = None
        unit.on_rental_market = True
        if unit.rent is None:
            # initial listing rent at tract market rate; landlord/investor repricing logic
            # (EQ11) takes over from here in subsequent months
            unit.rent = model.tracts[unit.tract_id].rent_per_quality * unit.quality
        if unit not in model.rental_units:
            model.rental_units.append(unit)
    else:  # otherwise, assign it to the agent
        unit.on_rental_market = False
        unit.tenant = None
        agent.house = unit
        agent.status = "owning"
        agent.owned_since_month = model.current_month


def run_ownership_market(model):
    """Multi round double auction clearing of queued buyers against for-sale houses"""
    for_sale = [
        u for u in model.for_sale_units if u.on_sale_market
    ]  # and u.owner is None

    # EQ 8 repricing if stale listing, regardless of if there are agents bidding
    price_cfg = model.params["price_reduction_eq8"]
    for unit in for_sale:
        unit.days_on_market += 1
        is_investor_listing = (
            getattr(model._resale_sellers.get(unit), "properties", None) is not None
        )
        multiplier = (
            price_cfg["vacancy_tax_multiplier_investor"] if is_investor_listing else 1.0
        )
        unit.price = price_reduction(
            current_price=unit.price,
            reduction_prob=price_cfg["reduction_prob"],
            epsilon_std=price_cfg["epsilon_std"],
            rng=model.random_gen,
            vacancy_tax_multiplier=multiplier,
        )
        # cant have repeated cuts lower a listing below payoff
        unit.price = max(unit.price, unit.mortgage_principal)

    if not model._ownership_bid_queue:  # no houses or no prospective buyers
        return

    enforce_lti_policies(model)  # cap over-limit bids before matching

    auction_cfg = model.params["market_clearing_a4"]
    remaining_bids = list(model._ownership_bid_queue)
    remaining_offers = [u for u in for_sale]  # exclude nothing

    n_rounds = max_rounds(
        n_bids=len(remaining_bids),
        n_offers=len(remaining_offers),
        n_households=len(model.agents),
        round_floor=auction_cfg["round_floor"],
    )

    matched_bids = []

    for _round in range(n_rounds):
        if not remaining_bids or not remaining_offers:  # no one bidding or no houses
            break

        # Phase 1: every remaining bid chooses preferred affordable offer
        claims = {}  # offer -> list of bids
        for bid in remaining_bids:
            affordable = [
                u
                for u in remaining_offers
                if bid["max_price"] >= u.price and bid["agent"] is not u.owner
            ]
            if not affordable:
                continue  # can't afford anything left this round
            best_offer = _preferred_offer(bid, affordable, model)
            claims.setdefault(best_offer, []).append(bid)

        if not claims:
            break  # no bid can afford any remaining offer- more rounds doesn't help

        # Phase 2: resolve ties/multple bids on a house
        sold_offers = []
        for unit, claimants in claims.items():
            if len(claimants) == 1:  # only one bid, just give to that one person
                winning_bid = claimants[0]
                final_price = unit.price  # no bid up
            else:
                # EQ21: seller bids the price up, then a random still-affording bidder wins
                multiplier = sample_bid_up_multiplier(  # same as rental market
                    model.random_gen,
                    n_bids=len(claimants),
                    bid_up_pct=auction_cfg["bid_up_pct"],
                    arrival_window_days=auction_cfg["arrival_window_days"],
                    month_days=auction_cfg["month_days"],
                    max_multiplier=auction_cfg["max_multiplier"],
                )
                bid_up_price = unit.price * multiplier
                still_afford = [b for b in claimants if b["max_price"] >= bid_up_price]
                if not still_afford:
                    # bid-up priced everyone out this round, retry next round
                    continue
                winning_bid = model.random_gen.choice(still_afford)
                final_price = bid_up_price

            agent, down_payment = winning_bid["agent"], winning_bid["down_payment"]
            _settle_purchase(model, unit, agent, down_payment, final_price)
            matched_bids.append(winning_bid)
            sold_offers.append(unit)

        remaining_bids = [b for b in remaining_bids if b not in matched_bids]
        remaining_offers = [u for u in remaining_offers if u not in sold_offers]

    # bids left over after all rounds go back to the rental market
    unmatched = [bid for bid in model._ownership_bid_queue if bid not in matched_bids]
    for bid in unmatched:
        agent = bid["agent"]
        if (
            agent.house is None
        ):  # only if they don't have a house, they go to the rental market
            model.queue_rental_bid(agent)
            # repeat buyer whose house didn't sell keeps current home, retries next step

    model._ownership_bid_queue = []
