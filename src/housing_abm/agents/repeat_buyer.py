#Repeat/move-up buyer: conventional  mortgage, downpayment from equity
#Simultaneously lists current home - 'golden handcuffs'

from housing_abm import tract
from housing_abm.equations.expenditure import desired_expenditure, price_appreciation_expectation
from housing_abm.equations.mortgage import down_payment_owner, max_loan_owner_occupier
from housing_abm.equations.selling import asking_price, p_sell

from .base import HouseholdAgent


class RepeatBuyer(HouseholdAgent):


    WEALTH_KEY = 'repeat_buyer'
    LOAN_TYPE = "conventional"

    #inherited initialization


    def __init__(self, model, income: float, age: int, tract_id: str):
        super().__init__(model, income, age, tract_id)
        self.house_to_sell = None #house they are trying to sell
    
    def step(self):
        #EQ 6: sell decision with lock in term, EQ 7 asking price for current home
        #EQ 3/5 for new purchase, EQ 17 down payment from equity

        self.refresh_desired_balance()

        if self.house is not None and self.status == "renting":
            #sometimes falls back here via ownershrip market homeless-bidder
            monthly_rent = self.house.rent
            self.apply_consumption(housing_cost = monthly_rent)
            if self.lease_months_remaining > 0:
                return
            self.house.tenant = None
            self.house.on_rental_market = True
            self.house = None
            self.status = "social_housing"
            self._bid_for_next_home(available_capital = self.bank_balance) #redecide what to do now
            return 




        if self.house is None: 
            self.apply_consumption(housing_cost = 0)
            self._bid_for_next_home(available_capital = self.bank_balance) 
            #TODO; enters the market again
            return
        
        monthly_payment = self.house.mortgage_payment
        still_carrying_old = self.house_to_sell is not None and self.house_to_sell is not self.house
        if still_carrying_old and not self.house_to_sell.on_sale_market: #old house got removed from the market
            self.house_to_sell = None #check this
            still_carrying_old = False  
        if still_carrying_old: #if they are still carrying the old house, they pay for both
            monthly_payment += self.house_to_sell.mortgage_payment
        self.apply_consumption(housing_cost = monthly_payment)

        if still_carrying_old:
            return #just waiting for old place to sell
        
        if self.house_to_sell is None: #first time listing, deciding whether to sell
            sell_cfg = self.model.params["selling_eq6"]#grab selling parameters
            prob_sell = p_sell(
                tenure_years = sell_cfg["tenure_years"],
                n_h = self.model.houses_per_capita(self.tract_id), #TODO
                n_h_avg = self.model.houses_per_capita_avg(self.tract_id),
                i_current = self.model.current_fed_rate_annual,
                i_avg = self.model.fed_rate_avg, #TODO
                alpha = sell_cfg["alpha_stock"], beta = sell_cfg["beta_rate"],
                i_mortgage = self.house.mortgage_rate, #TODO
                gamma=sell_cfg["gamma_lockin"])

            if self.model.random_gen.random() < prob_sell: #chooses to sell
                tract = self.model.tracts[self.tract_id]
                asking_cfg = self.model.params["asking_price_eq7"]#grab eq7 params
                self.house.price = asking_price( #calculate asking price
                    p_bar_tract = tract.avg_sold_price(self.house.quality), 
                    f_bar_tract = tract.avg_days_on_market(),
                    alpha = asking_cfg["alpha"],
                    beta = asking_cfg["beta"],
                    zeta = asking_cfg["zeta"],
                    epsilon_std = asking_cfg["epsilon_std"],
                    rng = self.model.random_gen
                )
                self.house.on_sale_market = True
                self.house_to_sell = self.house
                self.model.queue_listing(self.house, seller = self) #TODO



                #listed and still living at old house
                #bid for next home using anticipated equity from sale

                equity = self.house.price - self.house.mortgage_principal
                self._bid_for_next_home(available_capital = equity) #or equity+bank_balance?
                

    def _bid_for_next_home(self, available_capital: float):
        """EQ 3/5, purchase decision for new home 
        unrealized equity when listing old home, liquid bank once old home is osld"""
        tract = self.model.tracts[self.tract_id]
        g = price_appreciation_expectation(
                    tract.hpi_history, alpha=self.model.params["appreciation_eq4"]["alpha_household"]
                )
        mort_cfg = self.model.mortgage_terms["conventional"]#loan terms for conventional loans
        i_r_monthly = self.model.current_fed_rate_monthly
        loan_cap = max_loan_owner_occupier( #max conventional loan they can get
            bank_balance = available_capital, #uses house equity for downpayment on next house
            disposable_income = self.income - self.essential_consumption(),
            chi_max_ltv = mort_cfg["max_ltv"], dti_front = mort_cfg["front_end_dti_max"],
            i_r_monthly = i_r_monthly, term_months = mort_cfg["term_months"])
        
        exp_params = self.model.params["expenditure_eq3"]
        price = desired_expenditure(
            income_or_capital = self.income*12, g=g, alpha=exp_params["alpha_household"], #income to yearly
            beta=exp_params["beta"], epsilon_std = exp_params["epsilon_std"], 
            rng = self.model.random_gen, mortgage_cap = loan_cap +available_capital #equity helps afford more
        )

        down_cfg = self.model.params["downpayment_eq17"]["repeat_buyer"]
        down_payment = down_payment_owner( #calculate downpayment from distribution clustered at minimum
            price = price, income_rank = self.income,
            income_cutoff = self.model.ftb_income_cutoff(down_cfg["floor_share_p_floor"]), 
            d_minimum_pct = down_cfg["d_minimum_pct"], 
            lognorm_m = down_cfg["lognorm_m"],
            lognorm_s = down_cfg["lognorm_s"], rng = self.model.random_gen)


        if down_payment > available_capital+ self.bank_balance: #can't afford down payment, enter rental market
            self.model.queue_rental_bid(self, fraction_of_income = 0.33)
            return
        
        self.model.queue_ownership_bid(self, max_price = price, down_payment = down_payment)






