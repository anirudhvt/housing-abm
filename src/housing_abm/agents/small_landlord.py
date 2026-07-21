#2-10 units, rental-yield focused (delta = 0.3)
#rent stickiness on lease renewal
#cash purchase if wealth > 2x price
from housing_abm.equations.expenditure import price_appreciation_expectation
from housing_abm.equations.investor_probs import p_buy_investor, p_sell_investor
from housing_abm.equations.investor_yield import effective_yield_sell, expected_yield_buy
from housing_abm.equations.mortgage import down_payment_investor, passes_investor_dscr
from housing_abm.equations.rental_pricing import small_landlord_rent
from housing_abm.equations.selling import asking_price

from .base import HouseholdAgent

class SmallLandlord(HouseholdAgent):
    
    WEALTH_KEY = "small_landlord"
    LOAN_TYPE = "investor_dscr" #new loan terms
    DELTA = 0.3 #more risk averse than institutional investors


    def __init__(self, model, income, age, tract_id):
        super().__init__(model, income, age, tract_id)
        self.properties = [] #housingunit owned
        self.house_to_sell = None #unused, kept for ownership market 

    def step(self):
        #EQ 9: Expected yield, EQ 10 p_buy
        #EQ 11: rent with stickiness, EQ 12/13 sell decision
        #EQ 18: down_payment_investor(agent_type = 'small landlord')
        
        self.refresh_desired_balance() #reset using EQ 1

        net_rental_cash_flow = self._collect_rent_and_pay_mortgages() 
        #negative housing cost folds into EQ 2 similar to a rent/mortgage payment
        self.apply_consumption(housing_cost = -net_rental_cash_flow) #negative cash flow is a housing cost, positive cash flow is a housing benefit

        self._reprice_vacant_units() #price reduction mechanic
        #buy or sell any units
        self._evaluate_sell_decision()
        self._evaluate_buy_decision()


    def _collect_rent_and_pay_mortgages(self) -> float:
        """sum rent across tenant properties minus mortgage payments
        vacant units still carry mortgage"""
        total_rent = sum(u.rent or 0.0 for u in self.properties if u.tenant is not None) #non vacant units generate rent
        total_mortgage = sum(u.mortgage_payment or 0.0 for u in self.properties) #pay for all properties, vacant or not
        return total_rent - total_mortgage
    
    def _reprice_vacant_units(self):
        """EQ 11: sticky repricing for vacant listed units"""
        cfg = self.model.params["rental_pricing_eq11"]["small_landlord"]
        for unit in self.properties:
            if unit.on_rental_market and unit.tenant is None:
                tract = self.model.tracts[unit.tract_id]
                unit.rent = small_landlord_rent(
                    r_bar_tract=tract.rent_per_quality * unit.quality,
                    f_bar_tract=tract.avg_days_on_market(),
                    alpha=cfg["alpha"], beta=cfg["beta"], zeta=cfg["zeta"],
                    epsilon_std=cfg["epsilon_std"], reprice_prob=cfg["reprice_prob"],
                    previous_rent=unit.rent, rng=self.model.random_gen,
                )
        
    def _evaluate_sell_decision(self):
        """EQ 12/13, sell probability based on effective yield"""
        yield_cfg = self.model.params["investor_yield_eq9_eq12"]
        prob_cfg = self.model.params["investor_probs_eq10_eq13"]
        for unit in list(self.properties): #check for each property
            if unit.on_sale_market: #already listed, don't relist
                continue
            tract = self.model.tracts[unit.tract_id]
            current_value = tract.avg_sold_price(unit.quality) #current market value of the unit
            equity = current_value - unit.mortgage_principal 
            g = price_appreciation_expectation(
                tract.hpi_history, alpha=self.model.params["appreciation_eq4"]["alpha_household"]
            )
            psi = effective_yield_sell(
                price = current_value, equity =equity, delta = self.DELTA, 
                g = g,kappa=yield_cfg["kappa"], r_bar=tract.gross_rental_yield(),
                monthly_mortgage=unit.mortgage_payment, policy_cost=yield_cfg["policy_cost"]
            )
            prob_sell = p_sell_investor(psi, beta = prob_cfg["beta_small_landlord"])
            if self.model.random_gen.random() < prob_sell: #chooses to sell
                asking_cfg = self.model.params["asking_price_eq7"]
                unit.price = asking_price( #generates price from metrics on the tract level
                    p_bar_tract=current_value, f_bar_tract=tract.avg_days_on_market(),
                    alpha=asking_cfg["alpha"], beta=asking_cfg["beta"], zeta=asking_cfg["zeta"],
                    epsilon_std=asking_cfg["epsilon_std"], rng=self.model.random_gen,
                )
                unit.price = max(unit.price, unit.mortgage_principal) #see repeat_buyer.py comment
                unit.on_sale_market = True
                self.model.queue_listing(unit, seller=self)
            
    
    def _evaluate_buy_decision(self):
        """EQ 9/10; expected yield -> purchase probability -> bid on ownership market"""
        yield_cfg = self.model.params["investor_yield_eq9_eq12"]
        prob_cfg = self.model.params["investor_probs_eq10_eq13"]
        down_cfg = self.model.params["downpayment_eq18"]["small_landlord"]
        dscr_cfg = self.model.params["investor_dscr_eq15"]
        mort_cfg = self.model.mortgage_terms[self.LOAN_TYPE]

        tract = self.model.tracts[self.tract_id]

        target_price = tract.price_per_quality 
        g = price_appreciation_expectation(
            tract.hpi_history, alpha = self.model.params["appreciation_eq4"]["alpha_household"]
        )

        down_payment, is_cash = down_payment_investor( #follows cash rule for buying if has enough money
            price=target_price, wealth=self.bank_balance, agent_type="small_landlord",
            mu=down_cfg["mu"], sigma=down_cfg["sigma"], p_cash=0.0,
            d_minimum_pct=down_cfg["d_minimum_pct"], rng=self.model.random_gen,
        )

        if self.bank_balance < down_payment:
            return #can't afford to purchase this month
        down_payment = min(down_payment, target_price) #can't pay more than the price

        #evaluate if they can get the loan
        proposed_loan = 0.0 if is_cash else target_price - down_payment 
        i_r_monthly = self.model.mortgage_rate_monthly
        monthly_mortgage = 0.0 if is_cash else self.model.monthly_payment(
            proposed_loan, dscr_cfg["i_btl_monthly"], mort_cfg["term_months"]
        )

        if not is_cash:
            passes = passes_investor_dscr( #bank conditions on loans
                bank_balance=self.bank_balance,
                expected_annual_rent_yield=tract.gross_rental_yield(),
                xi_icr=dscr_cfg["xi_icr"], i_btl_monthly=dscr_cfg["i_btl_monthly"],
                proposed_loan=proposed_loan, chi_max_ltv=mort_cfg["max_ltv"],
            )
            if not passes: #bank doesn't allow loan
                return
        
        omega = expected_yield_buy(
            price=target_price, down_payment=down_payment, delta=self.DELTA,
            g=g, kappa=yield_cfg["kappa"], r_bar=tract.gross_rental_yield(),
            monthly_mortgage=monthly_mortgage, policy_cost=yield_cfg["policy_cost"],
        )

        prob_buy = p_buy_investor(omega, beta=prob_cfg["beta_small_landlord"])

        if self.model.random_gen.random() < prob_buy: #chooses to buy, bid on ownership market
            self.model.queue_ownership_bid(self, max_price = target_price, down_payment = down_payment)
                
            
