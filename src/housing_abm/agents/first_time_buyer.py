"""First-time Buyer (FTB): FHA Mortgage, no equity"""
from housing_abm.equations.buy_rent import p_buy
from housing_abm.equations.mortgage import down_payment_owner, max_loan_owner_occupier

from .base import HouseholdAgent
import numpy as np
from housing_abm.equations.expenditure import desired_expenditure, price_appreciation_expectation

class FirstTimeBuyer(HouseholdAgent):
    
    LOAN_TYPE = 'fha'
    WEALTH_KEY = 'first_time_buyer'

    #inherited initialization
    
    def step(self):
        #EQ 3: desired expenditure capped by max loan
        #EQ 5: buy vs rent, EQ 17 down payment, place bid on ownership makret

        self.refresh_desired_balance()
        if self.house is not None: #same as renter
            monthly_payment = self.house.mortgage_payment
            self.apply_consumption(housing_cost = monthly_payment)
            return #already owns, selling handled by repeatbuyer logic
                   #newly owning ftbs don't relist immediately
        
        self.apply_consumption(housing_cost = 0)

        #grab information
        tract = self.model.tracts[self.tract_id]
        mort_cfg = self.model.mortgage_terms[self.LOAN_TYPE] #fha loan terms
        i_r_monthly = self.model.current_fed_rate_monthly

        loan_cap = max_loan_owner_occupier(
            bank_balance = self.bank_balance,
            #essential consumption is a flat constant, so we subtract to get disposable income
            disposable_income = self.income - self.essential_consumption(),
            #fha loan regulations
            chi_max_ltv = mort_cfg["max_ltv"], dti_front = mort_cfg["front_end_dti_max"],
            i_r_monthly = i_r_monthly, term_months = mort_cfg["term_months"])
                    
        #if they spent whole bank balance + maximum loan from bank                
        max_affordable_price = loan_cap + self.bank_balance

        #account for appreciation of the house
        g = price_appreciation_expectation(tract.hpi_history, 
                                           alpha = self.model.params["appreciation_eq4"]["alpha_household"])
    
        exp_params = self.model.params["expenditure_eq3"]
        price = desired_expenditure(
            income_or_capital = self.income, g=g, alpha=exp_params["alpha_household"], 
            beta=exp_params["beta"], epsilon_std=exp_params["epsilon_std"], 
            rng=self.model.random_gen, mortgage_cap = max_affordable_price)
        
        if self.bank_balance < mort_cfg["min_down_payment_pct"] * price:
            #not enough money for minimum down payment
            #enter rental market again
            self.model.queue_rental_bid(self, fraction_of_income = 0.33)
            return 
        
        #what quality house can they afford
        quality = self.model.quality_affordable(price, self.tract_id)
        rent_q = self.model.market_rent_for_quality(quality, self.tract_id)
        monthly_mortgage = self.model.monthly_payment(loan_cap, i_r_monthly, mort_cfg["term_months"])

        #now, weigh owning vs renting via equation 5

        buy_params = self.model.params["buy_rent_eq5"]
        prob = p_buy(rent_q = rent_q, tau = buy_params["tau"], beta = buy_params["beta"], rent_income_fraction = buy_params["rent_income_fraction"],
                     monthly_mortgage = monthly_mortgage, price=price, g=g,
                     beta = buy_params["beta"])

        if self.model.random_gen.random() < prob: #chooses buying with logistic probability
            down_cfg = self.model.params["downpayment_eq17"]["first_time_buyer"] 
            income_cutoff = self.model.ftb_income_cutoff(down_cfg["floor_share_p_floor"])
            down_payment = down_payment_owner(
                price = price, income_rank = self.income, income_cutoff=income_cutoff,
                d_minimum_pct = down_cfg["d_minimum_pct"], lognormal_m = down_cfg["lognormal_m"],
                lognormal_s = down_cfg["lognormal_s"], rng=self.model.random_gen)
            self.model.queue_ownership_bid(self, max_price = price, down_payment = down_payment) #enters housing market
        else: #chooses to rent
            self.model.queue_rental_bid(self, fraction_of_income = 0.33) 



    
            