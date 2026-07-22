"""EQ 16: morgage rate = exogenous bank rate + endogenous spread
Spread adjusts monthly based on new mortgage lending relative to target supply
lending above target pushes spread and rate up (lower demand), vice versa"""


def update_mortgage_rate(model):
    cfg = model.params["interest_rate_eq16"]
    M_t = model._monthly_new_lending
    T = cfg["target_monthly_supply"]
    model.mortgage_rate_spread_annual += cfg["alpha"] * (M_t - T)
    model.mortgage_rate_spread_annual = min(
        max(model.mortgage_rate_spread_annual, cfg["spread_floor"]),
        cfg["spread_ceiling"],
    )
    model.mortgage_rate_annual = (
        model.current_fed_rate_annual + model.mortgage_rate_spread_annual
    )
    model.mortgage_rate_monthly = model.mortgage_rate_annual / 12.0

    model.mortgage_rate_history.append(model.mortgage_rate_annual)
    model.mortgage_rate_avg = sum(model.mortgage_rate_history[-12:]) / len(
        model.mortgage_rate_history[-12:]
    )

    model._monthly_new_lending = 0.0  # reset for next month
