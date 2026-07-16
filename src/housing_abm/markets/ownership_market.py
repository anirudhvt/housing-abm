"""Placeholder ownership market, one round bidding
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
    if not for_sale or not model._ownership_bid_queue: #no houses or no prospective buyers
        return
    
    model.random_gen.shuffle(for_sale)

    matched_bids = []


    for unit in for_sale:
        eligibile = [bid for bid in model._ownership_bid_queue 
                     if bid["max_price"] >= unit.price and bid not in matched_bids]
        if not eligibile:
            continue #no one can afford

        winning_bid = max(eligibile, key = lambda b: b["max_price"])
        agent, down_payment = winning_bid["agent"], winning_bid["down_payment"]

        
        #assign characteristics of the bought house according to the bid/loan terms
        principal = unit.price - down_payment
        i_r_monthly = model.current_fed_rate_monthly
        term_months = model.mortgage_terms[agent.LOAN_TYPE]["term_months"]
        unit.mortgage_principal = principal
        unit.mortgage_payment = model.monthly_payment(principal, i_r_monthly, term_months)
        #winning agent buys the house

        unit.owner = agent
        unit.on_sale_market = False
        agent.bank_balance -= down_payment
        agent.house = unit
        agent.status = "owning"
        matched_bids.append(winning_bid)

    #unmatched bidders go back to rental market
    unmatched = [bid for bid in model._ownership_bid_queue if bid not in matched_bids]
    for bid in unmatched:
        model.queue_rental_bid(bid["agent"])

    for bid in matched_bids + unmatched:
        model._ownership_bid_queue.remove(bid)
