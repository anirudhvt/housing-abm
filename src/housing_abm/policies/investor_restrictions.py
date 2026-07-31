"""six investor-restriction policies 


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
    return [p for p in model.policies if p.get("type") == policy_type]


# ---------------------------------------------------------------------------
# waiting_period + geographic_restriction, checked during run_ownership_market
# ---------------------------------------------------------------------------

def passes_unit_level_policies(model, agent, unit) -> bool:
    """call this alongside the existing max_price/ownership check when
    building each bid's `affordable` list in run_ownership_market.
    Returns True if `agent` is allowed to buy `unit` under all active
    waiting_period and geographic_restriction policies.

    """
    DAYS_PER_MODEL_STEP = 30  # one model step = one month, approximated as 30 days
    for policy in _policies_of_type(model, "waiting_period"):
        if not _applies(policy, agent):
            continue
        threshold_in_model_steps = policy["days"] / DAYS_PER_MODEL_STEP
        if unit.days_on_market < threshold_in_model_steps:
            return False
    for policy in _policies_of_type(model, "geographic_restriction"):
        if _applies(policy, agent) and unit.tract_id in policy.get("restricted_tracts", []):
            return False
    return True


#ownership cap

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
    """Call once per month, before ownership-market matching - removes bids from investors who would
    exceed an active ownership_cap policy's max_units_absolute.
    """
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
                continue  # drop the bid entirely

            # soft mode: keep the bid with the currently adapted probability
            state = _get_or_init_soft_cap_state(model, policy)
            state["n_total_this_month"] += 1
            if model.random_gen.random() < state["allowed_prob"]:
                state["n_over_cap_this_month"] += 1
                surviving_bids.append(bid)
            # else: dropped this round, agent can try again next month

        model._ownership_bid_queue = surviving_bids


def update_ownership_cap_soft_state(model):
    """Call once per month (end of step, after ownership market clears
    Updates proportions 
    """
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
    """Hard-mode ownership_cap: force list excess units for sale. =
    """
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
            # force-list the excess units with the shortest remaining tenancy
            # first (least disruptive to sitting tenants) - vacant units first,
            # then occupied ones if still over the cap
            candidates = sorted(
                (u for u in agent.properties if not u.on_sale_market),
                key=lambda u: (u.tenant is not None, getattr(u, "days_on_market", 0)),
            )
            for unit in candidates[:excess]:
                if unit.tenant is not None:
                    continue  # don't evict; wait for lease to end naturally
                tract = model.tracts[unit.tract_id]
                unit.price = tract.avg_sold_price(unit.quality)
                unit.price = max(unit.price, unit.mortgage_principal)
                unit.on_sale_market = True
                unit.on_rental_market = False
                model.queue_listing(unit, seller=agent)


# ---------------------------------------------------------------------------
# purchase_tax, vacancy_tax, portfolio_tax: all three feed into a single
# policy_cost figure subtracted directly from Omega/Psi (Eq 9/12) 
# ---------------------------------------------------------------------------

def _portfolio_tax_rate(brackets: list[dict], n_units: int) -> float:
    for bracket in brackets:
        lo, hi = bracket["min_units"], bracket["max_units"]
        if n_units >= lo and (hi is None or n_units < hi):
            return bracket["rate"]
    return 0.0


def compute_policy_cost(model, agent) -> float:
    """Total policy-driven cost  on this agent's expected/effective yield
    this month
    """
    total = 0.0

    for policy in _policies_of_type(model, "purchase_tax"):
        if _applies(policy, agent):
            total += policy["rate"]

    for policy in _policies_of_type(model, "vacancy_tax"):
        if not _applies(policy, agent):
            continue
        threshold = policy.get("vacancy_threshold_days", 90)
        n_vacant_over_threshold = sum(
            1 for u in agent.properties
            if u.tenant is None and getattr(u, "day_vacant", 0) >= threshold
        )
        if n_vacant_over_threshold > 0 and agent.properties:
            # scale the annual rate by the share of the portfolio sitting
            # vacant past the threshold, rather than applying it flat
            # regardless of how much of the portfolio is actually vacant
            total += policy["annual_rate"] * (n_vacant_over_threshold / len(agent.properties))

    for policy in _policies_of_type(model, "portfolio_tax"):
        if _applies(policy, agent):
            total += _portfolio_tax_rate(policy["brackets"], len(agent.properties))

    return total