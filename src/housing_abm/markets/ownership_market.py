"""Placeholder ownership market, one round bidding"""

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.market_matching import (
    pick_preferred,
    sample_bid_up_multiplier,
    max_rounds,
    expected_gross_rental_yield,
)
from housing_abm.equations.selling import price_reduction
from housing_abm.policy import enforce_lti_policies
from housing_abm.markets.offer_book import OfferBook
from housing_abm.policies.investor_restrictions import (
    agent_policy_tag,
    eligible_units_for_tag,
    filter_ownership_cap_bids,
    has_unit_level_policies,
    passes_unit_level_policies,
)


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


def _quality_key(unit):
    """Owner-occupiers want the best house they can afford."""
    return unit.quality


def _make_yield_key(model):
    """Investors want the best expected gross rental yield (EQ 19/20).

    avg_days_on_market is a tract-level quantity, so it is read once per round
    rather than once per (bid, unit) pair.
    """
    dom = {tid: tract.avg_days_on_market() for tid, tract in model.tracts.items()}
    rent_per_quality = {
        tid: tract.rent_per_quality for tid, tract in model.tracts.items()
    }

    def yield_key(unit):
        rent_estimate = (
            unit.rent
            if unit.rent is not None
            else rent_per_quality[unit.tract_id] * unit.quality
        )
        return expected_gross_rental_yield(rent_estimate, unit.price, dom[unit.tract_id])

    return yield_key


def _build_offer_books(model, offers):
    """One price-sorted book per (policy class, preference) combination.

    Unit-level policies gate which listings a class may bid on, so each class
    that faces a different gate needs its own book. With no such policy active
    -- the baseline, and the tax scenarios -- households and investors share
    the two books built from the full offer list.
    """
    rng = model.random_gen
    yield_key = _make_yield_key(model)
    if not has_unit_level_policies(model):
        household_book = OfferBook(offers, _quality_key, rng)
        investor_book = OfferBook(offers, yield_key, rng)
        return {
            None: household_book,
            "small_landlord": investor_book,
            "institutional": investor_book,
        }

    books = {None: OfferBook(offers, _quality_key, rng)}
    for tag in ("small_landlord", "institutional"):
        books[tag] = OfferBook(
            eligible_units_for_tag(model, tag, offers), yield_key, rng
        )
    return books


def _settle_purchase(model, unit, agent, down_payment, final_price, acquisition_tax=0.0):
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
    # the purchase tax leaves the buyer's balance but buys no equity
    agent.bank_balance -= down_payment + acquisition_tax
    model.record_purchase(agent)
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
        model.add_rental_unit(unit)
    else:  # otherwise, assign it to the agent
        unit.on_rental_market = False
        unit.tenant = None
        # the unit has left the rental channel; leaving it in model.rental_units
        # inflates the rental-vacancy denominator with owner-occupied homes
        model.drop_rental_unit(unit)
        agent.house = unit
        agent.status = "owning"
        agent.owned_since_month = model.current_month


def _withdraw_stale_listings(model, for_sale):
    """Let sellers give up on a listing that will not sell.

    Nothing ever took a listing off the sale market, so a seller whose home
    did not clear stayed listed forever -- and a repeat buyer carrying an
    unsold home returns early from step(), so they could neither buy nor
    re-price nor withdraw. At the default configuration that deadlocked about
    a fifth of repeat buyers and left the median listing sitting on the market
    for roughly 87 months, against the three to six months of inventory a real
    market runs.

    Real sellers withdraw and try again later. A withdrawn home reverts to
    ordinary owner-occupation, and its owner rolls the EQ 6 selling
    probability again from the next month, so the listing can return when
    conditions improve. Units with no owner (unsold new construction) stay
    listed, since there is no one to withdraw them.
    """
    limit = model.params.get("sale_market", {}).get("withdraw_after_months", 0)
    if not limit:
        return
    for unit in list(for_sale):
        if unit.days_on_market < limit or unit.owner is None:
            continue
        seller = model._resale_sellers.pop(unit, None)
        unit.on_sale_market = False
        unit.days_on_market = 0
        model._cache_active_for_sale = None
        if seller is not None and getattr(seller, "house_to_sell", None) is unit:
            seller.house_to_sell = None
        if getattr(unit.owner, "properties", None) is not None:
            # investor-held: back onto the rental market rather than idle
            unit.on_rental_market = True
            model.add_rental_unit(unit)


def run_ownership_market(model):
    """Multi round double auction clearing of queued buyers against for-sale houses"""
    for_sale = model.active_for_sale()

    # EQ 8 repricing if stale listing, regardless of if there are agents bidding
    price_cfg = model.params["price_reduction_eq8"]
    vacancy_tax_active = any(p.get("type") == "vacancy_tax" for p in model.policies)
    for unit in for_sale:
        unit.days_on_market += 1
        is_investor_listing = (
            getattr(model._resale_sellers.get(unit), "properties", None) is not None
        )
        multiplier = (
            price_cfg["vacancy_tax_multiplier_investor"] if (is_investor_listing and vacancy_tax_active) else 1.0
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

    _withdraw_stale_listings(model, for_sale)

    if not model._ownership_bid_queue:  # no houses or no prospective buyers
        return

    enforce_lti_policies(model)  # cap over-limit bids before matching
    filter_ownership_cap_bids(model) #drop bids over ownership cap

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
    household_bids = [b for b in remaining_bids if not _is_investor(b["agent"])] #give household buyers first position
    investor_bids  = [b for b in remaining_bids if _is_investor(b["agent"])]
    if household_bids:
        remaining_bids = household_bids + investor_bids

    for _round in range(n_rounds):
        if not remaining_bids or not remaining_offers:  # no one bidding or no houses
            break

        # Phase 1: every remaining bid chooses preferred affordable offer
        books = _build_offer_books(model, remaining_offers)
        claims = {}  # offer -> list of bids
        for bid in remaining_bids:
            agent = bid["agent"]
            tag = agent_policy_tag(agent) if _is_investor(agent) else None
            best_offer = books[tag].best_affordable(
                bid["max_price"], exclude_owner=agent
            )
            if best_offer is None:
                continue  # can't afford anything left this round
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
            _settle_purchase(
                model,
                unit,
                agent,
                down_payment,
                final_price,
                acquisition_tax=winning_bid.get("acquisition_tax", 0.0),
            )
            matched_bids.append(winning_bid)
            sold_offers.append(unit)

        sold_ids = {id(u) for u in sold_offers}
        matched_ids = {id(b) for b in matched_bids}
        remaining_bids = [b for b in remaining_bids if id(b) not in matched_ids]
        remaining_offers = [u for u in remaining_offers if id(u) not in sold_ids]

    # bids left over after all rounds go back to the rental market
    matched_ids = {id(b) for b in matched_bids}
    unmatched = [bid for bid in model._ownership_bid_queue if id(bid) not in matched_ids]
    for bid in unmatched:
        agent = bid["agent"]
        if (
            agent.house is None
        ):  # only if they don't have a house, they go to the rental market
            model.queue_rental_bid(agent)
            # repeat buyer whose house didn't sell keeps current home, retries next step

    model._ownership_bid_queue = []
