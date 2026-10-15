"""Investor-restriction policies. Config shapes:

  - type: waiting_period
    days: 60
    applies_to: [institutional]  

  - type: ownership_cap
    max_units_absolute: 100
    enforcement: hard | soft
    soft_target_exceedance_fraction: 0.10   # only used if enforcement: soft
    soft_window_months: 12                   # only used if enforcement: soft
    applies_to: [institutional]

  - type: geographic_restriction
    restricted_tracts: [tract_001]
    applies_to: [institutional]

  - type: purchase_tax
    rate: 0.05
    applies_to: [institutional]

  - type: vacancy_tax
    annual_rate: 0.03
    vacancy_threshold_days: 90
    applies_to: [small_landlord, institutional]

  - type: portfolio_tax
    brackets:
      - {min_units: 0,   max_units: 10,  rate: 0.0}
      - {min_units: 10,  max_units: 50,  rate: 0.01}
      - {min_units: 50,  max_units: 200, rate: 0.03}
      - {min_units: 200, max_units: null, rate: 0.05}
    applies_to: [institutional]
"""
from collections import deque

# one model step is one calendar month; policy configs are written in days
DAYS_PER_MODEL_STEP = 30

_WEALTH_KEY_TO_TAG = {
    "small_landlord": "small_landlord",
    "institutional_investor": "institutional",
}


def _agent_tag(agent) -> str | None:
    return _WEALTH_KEY_TO_TAG.get(getattr(agent, "WEALTH_KEY", None))


def _applies(policy: dict, agent) -> bool:
    tag = _agent_tag(agent)
    if tag is None:
        return False
    applies_to = policy.get("applies_to", ["small_landlord", "institutional"])
    return tag in applies_to


def _policies_of_type(model, policy_type: str) -> list[dict]:
    """Policies of one type, from an index built once per model (hot path)."""
    index = getattr(model, "_policy_type_index", None)
    if index is None:
        index = {}
        for policy in model.policies:
            index.setdefault(policy.get("type"), []).append(policy)
        model._policy_type_index = index
    return index.get(policy_type, ())


# ---------------------------------------------------------------------------
# waiting_period + geographic_restriction, checked during run_ownership_market
# ---------------------------------------------------------------------------

def has_unit_level_policies(model) -> bool:
    """True if any policy restricts which units an agent may bid on."""
    return bool(
        _policies_of_type(model, "waiting_period")
        or _policies_of_type(model, "geographic_restriction")
    )


def eligible_units_for_tag(model, tag, units):
    """Units an agent class may bid on under waiting-period and geographic
    restrictions (resolved once per class per round)."""
    DAYS = DAYS_PER_MODEL_STEP
    thresholds = [
        p["days"] / DAYS
        for p in _policies_of_type(model, "waiting_period")
        if tag is not None and tag in p.get("applies_to", ["small_landlord", "institutional"])
    ]
    restricted = set()
    for p in _policies_of_type(model, "geographic_restriction"):
        if tag is not None and tag in p.get(
            "applies_to", ["small_landlord", "institutional"]
        ):
            restricted.update(p.get("restricted_tracts", []))

    if not thresholds and not restricted:
        return units
    min_days = max(thresholds) if thresholds else None
    return [
        u
        for u in units
        if (min_days is None or u.days_on_market >= min_days)
        and u.tract_id not in restricted
    ]


def agent_policy_tag(agent):
    """Public accessor for an agent's policy class tag."""
    return _agent_tag(agent)


# ---------------------------------------------------------------------------
# ownership_cap
# ---------------------------------------------------------------------------

def _get_or_init_soft_cap_state(model, policy: dict) -> dict:

    if not hasattr(model, "_ownership_cap_soft_state"):
        model._ownership_cap_soft_state = {}
    key = id(policy)
    if key not in model._ownership_cap_soft_state:
        window = policy.get("soft_window_months", 12)
        model._ownership_cap_soft_state[key] = {
            "n_over_cap_history": deque(maxlen=window),
            "n_total_history": deque(maxlen=window),
            "n_over_cap_this_month": 0.0,
            "n_total_this_month": 0.0,
            "allowed_prob": policy.get("soft_target_exceedance_fraction", 0.10),
        }
    return model._ownership_cap_soft_state[key]


def filter_ownership_cap_bids(model):
    """Before matching, drop bids from investors at the cap (hard), or keep
    them with the adaptive allowed probability (soft)."""
    for policy in _policies_of_type(model, "ownership_cap"):
        max_units = policy["max_units_absolute"]
        enforcement = policy.get("enforcement", "hard")
        surviving_bids = []
        for bid in model._ownership_bid_queue:
            agent = bid["agent"]
            if not _applies(policy, agent):
                surviving_bids.append(bid)
                continue

            would_exceed = len(agent.properties) >= max_units
            if not would_exceed:
                surviving_bids.append(bid)
                continue

            if enforcement == "hard":
                continue

            state = _get_or_init_soft_cap_state(model, policy)
            state["n_total_this_month"] += 1
            if model.random_gen.random() < state["allowed_prob"]:
                state["n_over_cap_this_month"] += 1
                surviving_bids.append(bid)

        model._ownership_bid_queue = surviving_bids


def update_ownership_cap_soft_state(model):
    """End of month: nudge the soft cap's allowed probability toward the
    target exceedance share over the trailing window."""
    for policy in _policies_of_type(model, "ownership_cap"):
        if policy.get("enforcement") != "soft":
            continue
        state = _get_or_init_soft_cap_state(model, policy)
        state["n_over_cap_history"].append(state["n_over_cap_this_month"])
        state["n_total_history"].append(state["n_total_this_month"])

        total_over = sum(state["n_over_cap_history"])
        total_all = sum(state["n_total_history"])
        if total_all > 0:
            realized = total_over / total_all
            target = policy.get("soft_target_exceedance_fraction", 0.10)
            error = target - realized
            state["allowed_prob"] = min(max(state["allowed_prob"] + 0.1 * error, 0.0), 1.0)

        state["n_over_cap_this_month"] = 0.0
        state["n_total_this_month"] = 0.0


def apply_forced_divestiture(model):
    """Hard ownership cap: list excess vacant units for sale. Occupied units
    wait for their lease to end."""
    for policy in _policies_of_type(model, "ownership_cap"):
        if policy.get("enforcement", "hard") != "hard":
            continue
        max_units = policy["max_units_absolute"]
        for agent in model.agents:
            if not _applies(policy, agent):
                continue
            excess = len(agent.properties) - max_units
            if excess <= 0:
                continue
            # vacant units first
            candidates = sorted(
                (u for u in agent.properties if not u.on_sale_market),
                key=lambda u: (u.tenant is not None, getattr(u, "days_on_market", 0)),
            )
            for unit in candidates[:excess]:
                if unit.tenant is not None:
                    continue  # never evict
                tract = model.tracts[unit.tract_id]
                unit.price = tract.avg_sold_price(unit.quality)
                unit.price = max(unit.price, unit.mortgage_principal)
                unit.on_rental_market = False
                model.list_for_sale(unit, seller=agent)


# ---------------------------------------------------------------------------
# taxes: vacancy and portfolio taxes are holding costs subtracted from EQ 9/12;
# the purchase tax raises the cash outlay at acquisition
# ---------------------------------------------------------------------------

def _portfolio_tax_rate(brackets: list[dict], n_units: int) -> float:
    for bracket in brackets:
        lo, hi = bracket["min_units"], bracket["max_units"]
        if n_units >= lo and (hi is None or n_units < hi):
            return bracket["rate"]
    return 0.0


def compute_policy_cost(model, agent) -> float:
    """Annual holding cost (vacancy + portfolio tax) on this agent's yield.
    The purchase tax is excluded; see acquisition_price_multiplier."""
    total = 0.0

    for policy in _policies_of_type(model, "vacancy_tax"):
        if not _applies(policy, agent):
            continue
        # day_vacant counts months, so convert the threshold from days
        threshold_months = policy.get("vacancy_threshold_days", 90) / DAYS_PER_MODEL_STEP
        n_vacant_over_threshold = sum(
            1 for u in agent.properties
            if u.tenant is None and getattr(u, "day_vacant", 0) >= threshold_months
        )
        if n_vacant_over_threshold > 0 and agent.properties:
            # scaled by the share of the portfolio vacant past the threshold
            total += policy["annual_rate"] * (n_vacant_over_threshold / len(agent.properties))

    for policy in _policies_of_type(model, "portfolio_tax"):
        if _applies(policy, agent):
            total += _portfolio_tax_rate(policy["brackets"], len(agent.properties))

    return total


def acquisition_price_multiplier(model, agent) -> float:
    """1 + purchase tax rate. The tax raises the investor's cash outlay in EQ 9,
    while the return still accrues on market value."""
    multiplier = 1.0
    for policy in _policies_of_type(model, "purchase_tax"):
        if _applies(policy, agent):
            multiplier += policy["rate"]
    return multiplier