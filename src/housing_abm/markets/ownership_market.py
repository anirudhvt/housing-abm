"""Ownership market: EQ 8 repricing, stale-listing withdrawal, and multi-round
double-auction clearing of queued bids against for-sale units."""

import numpy as np

from housing_abm.agents.housing_unit import HousingUnit
from housing_abm.equations.market_matching import (
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
)


def generate_placeholder_sale_stock(
    model,
    n_units: int | None = None,
    quality_mean: float | None = None,
    quality_sigma: float | None = None,
):
    """Initial for-sale stock with lognormal quality (mean-preserving)."""
    cfg = model.params.get("initial_sale_stock", {})
    n_units = n_units if n_units is not None else cfg.get("n_units", 200)
    quality_mean = (
        quality_mean if quality_mean is not None else cfg.get("quality_mean", 1.0)
    )
    quality_sigma = (
        quality_sigma if quality_sigma is not None else cfg.get("quality_sigma", 0.45)
    )

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
    """Investors want the best expected gross rental yield (EQ 19/20)."""
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
    """One offer book per bidder class. Investor classes get their own filtered
    books only when a unit-level policy (waiting period, geographic) is active."""
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
    """Settle a winning bid: pay the seller, record the sale, set up the
    mortgage, and hand the unit to the buyer."""
    unit.price = final_price

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
            previous_owner.properties.remove(unit)
        elif previous_owner.house is unit:
            # owner-occupier sold before securing a replacement
            previous_owner.house = None
            previous_owner.status = "social_housing"

    model.tracts[unit.tract_id].record_sale(
        price=unit.price, quality=unit.quality, days_on_market=unit.days_on_market
    )

    principal = max(unit.price - down_payment, 0.0)
    i_r_monthly = model.mortgage_rate_monthly  # EQ 16 spread-adjusted rate
    term_months = model.mortgage_terms[agent.LOAN_TYPE]["term_months"]
    unit.mortgage_principal = principal
    unit.mortgage_payment = model.monthly_payment(principal, i_r_monthly, term_months)
    unit.mortgage_rate = i_r_monthly
    model._monthly_new_lending += principal  # feeds the EQ 16 spread update

    unit.owner = agent
    unit.on_sale_market = False
    unit.days_on_market = 0
    # the purchase tax leaves the buyer's balance but buys no equity
    agent.bank_balance -= down_payment + acquisition_tax
    model.record_purchase(agent)
    if _is_investor(agent):
        # goes onto the rental market; EQ 11 repricing takes over next month
        agent.properties.append(unit)
        unit.tenant = None
        unit.on_rental_market = True
        if unit.rent is None:
            unit.rent = model.tracts[unit.tract_id].rent_per_quality * unit.quality
        model.add_rental_unit(unit)
    else:
        unit.on_rental_market = False
        unit.tenant = None
        model.drop_rental_unit(unit)
        agent.house = unit
        agent.status = "owning"
        agent.owned_since_month = model.current_month


def _withdraw_stale_listings(model, for_sale):
    """Withdraw owned listings unsold after withdraw_after_months. Owners can
    relist later via EQ 6; investor units go back on the rental market."""
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
            unit.on_rental_market = True
            model.add_rental_unit(unit)


def run_ownership_market(model):
    """Reprice and prune listings, then clear queued bids by double auction."""
    for_sale = model.active_for_sale()

    # EQ 8 repricing of every listing, whether or not anyone is bidding
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
        # never cut below the outstanding mortgage
        unit.price = max(unit.price, unit.mortgage_principal)

    _withdraw_stale_listings(model, for_sale)

    if not model._ownership_bid_queue:
        return

    enforce_lti_policies(model)
    filter_ownership_cap_bids(model)

    auction_cfg = model.params["market_clearing_a4"]
    remaining_bids = list(model._ownership_bid_queue)
    remaining_offers = list(for_sale)

    n_rounds = max_rounds(
        n_bids=len(remaining_bids),
        n_offers=len(remaining_offers),
        n_households=len(model.agents),
        round_floor=auction_cfg["round_floor"],
    )

    matched_bids = []
    # household bids go first in the queue
    household_bids = [b for b in remaining_bids if not _is_investor(b["agent"])]
    investor_bids = [b for b in remaining_bids if _is_investor(b["agent"])]
    if household_bids:
        remaining_bids = household_bids + investor_bids

    for _round in range(n_rounds):
        if not remaining_bids or not remaining_offers:
            break

        # phase 1: each bid claims its preferred affordable offer
        books = _build_offer_books(model, remaining_offers)
        claims = {}  # offer -> list of bids
        for bid in remaining_bids:
            agent = bid["agent"]
            tag = agent_policy_tag(agent) if _is_investor(agent) else None
            best_offer = books[tag].best_affordable(
                bid["max_price"], exclude_owner=agent
            )
            if best_offer is None:
                continue
            claims.setdefault(best_offer, []).append(bid)

        if not claims:
            break  # nothing left is affordable to anyone

        # phase 2: single claims sell at asking; contested ones are bid up (EQ 21)
        # and go to a random bidder who can still afford them
        sold_offers = []
        for unit, claimants in claims.items():
            if len(claimants) == 1:
                winning_bid = claimants[0]
                final_price = unit.price
            else:
                multiplier = sample_bid_up_multiplier(
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
                    continue  # priced everyone out; retry next round
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

    # unmatched bidders without a home go to the rental market; owners retry next month
    matched_ids = {id(b) for b in matched_bids}
    unmatched = [bid for bid in model._ownership_bid_queue if id(bid) not in matched_ids]
    for bid in unmatched:
        agent = bid["agent"]
        if agent.house is None:
            model.queue_rental_bid(agent)

    model._ownership_bid_queue = []
