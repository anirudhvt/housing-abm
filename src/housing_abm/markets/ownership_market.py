"""Placeholder ownership market, one round bidding
TODO: replace with multi-round matching"""


from housing_abm.agents.housing_unit import HousingUnit


def generate_placeholder_sale_stock(model, n_units: int = 200, base_price: float = 150_000.0):
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
    for_sale = [u for u in model.for_sale_units if u.on_sale_market] #and u.owner is None
    if not model._ownership_bid_queue: #no houses or no prospective buyers
        return
    
    model.random_gen.shuffle(for_sale)

    matched_bids = []


    for unit in for_sale:
        eligibile = [bid for bid in model._ownership_bid_queue 
                     if bid["max_price"] >= unit.price
                       and bid not in matched_bids
                       and bid["agent"] is not unit.owner] #exclude current owner from bidding on their own house
        if not eligibile:
            continue #no one can afford

        winning_bid = max(eligibile, key = lambda b: b["max_price"])
        agent, down_payment = winning_bid["agent"], winning_bid["down_payment"]

        #if resale, settle previous owner
        previous_owner = model._resale_sellers.pop(unit, None)
        if previous_owner is not None:
            payoff = unit.mortgage_principal
            proceeds = unit.price - payoff
            bridge = getattr(previous_owner, "bridge_loan", 0.0)
            proceeds -= bridge #repay any bridge loan
            previous_owner.bridge_loan = 0.0 #reset bridge loan
            previous_owner.bank_balance += proceeds
            previous_owner.house_to_sell = None
            if previous_owner.house is unit:
                #previous ownerr hasn't secured replacement
                #fix pointers
                previous_owner.house = None
                previous_owner.status = "social_housing"
        
        #record sale
        model.tracts[unit.tract_id].record_sale(
            price = unit.price, 
            quality = unit.quality, 
            days_on_market = unit.days_on_market) 

        
        #assign characteristics of the bought house according to the bid/loan terms
        principal = unit.price - down_payment
        i_r_monthly = model.current_fed_rate_monthly
        term_months = model.mortgage_terms[agent.LOAN_TYPE]["term_months"]
        unit.mortgage_principal = principal
        unit.mortgage_payment = model.monthly_payment(principal, i_r_monthly, term_months)
        unit.mortgage_rate = i_r_monthly
        #winning agent buys the house

        unit.owner = agent
        unit.on_sale_market = False
        

        shortfall = max(0.0, down_payment - agent.bank_balance)
        agent.bank_balance -= (down_payment - shortfall) #pay down payment from bank balance,
        if shortfall > 0.0:
            agent.bridge_loan += shortfall  #finance the rest with a bridge loan, to be repaid when old house is sold


        agent.house = unit
        agent.status = "owning"
        agent.owned_since_month = model.current_month
        matched_bids.append(winning_bid)

    #unmatched bidders go back to rental market
    unmatched = [bid for bid in model._ownership_bid_queue if bid not in matched_bids]
    for bid in unmatched:
        agent = bid["agent"]
        if agent.house is None: #only if they don't have a house, they go to the rental market
            model.queue_rental_bid(agent)
            #repeat buyer whose house didn't sell keeps current home, retries next step

    for bid in matched_bids + unmatched:
        model._ownership_bid_queue.remove(bid)
