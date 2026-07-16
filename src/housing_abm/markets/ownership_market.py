"""Placeholder ownership market, first come first serve matching
TODO: replace with multi-round matching"""


from housing_abm.agents.housing_unit import HousingUnit


def generate_placeholder_sale_stock(model, n_units: int = 100, base_price: float = 150_000.0):
    "generate fixed for sale stock for skeleton model"
    units = []
    for _ in range(n_units):
        unit = HousingUnit(model= model, tract_id = "tract_001", quality = 1.0)
        unit.price = base_price
        unit.on_sale_market = True
        units.append(unit)
    return units

def run_ownership_market(model):
    """Matches queued buyers to houses whose prices they can afford (max_price)"""
    for_sale = [u for u in model.for_sale_units if u.on_sale_market and u.owner is None]
    model.random_gen.shuffle(for_sale)

    matched = []
    for bid in model._ownership_bid_queue:
        agent, max_price, down_payment = bid["agent"], bid["max_price"], bid["down_payment"]
        affordable = [u for u in for_sale if u.price <= max_price]
        if not affordable: #asking price is too high for each house
            model.queue_rental_bid(bid["agent"])
            matched.append(bid)
            continue
        unit = affordable[0] #otherwise, select the first affordable unit
        for_sale.remove(unit)

        
        #assign characteristics of the bought house according to the bid/loan terms
        principal = unit.price - down_payment
        i_r_monthly = model.current_fed_rate_monthly
        term_months = model.mortgage_terms[agent.LOAN_TYPE]["term_months"]
        unit.mortgage_principal = principal
        unit.mortgage_payment = model.monthly_payment(principal, i_r_monthly, term_months)
        #agent buys the house

        unit.owner = agent
        unit.on_sale_market = False
        agent.bank_balance -= down_payment
        agent.house = unit
        agent.status = "owning"
        matched.append(bid)

    for bid in matched:
        model._ownership_bid_queue.remove(bid)
