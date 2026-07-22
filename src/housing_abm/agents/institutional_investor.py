"""Institutional Investor agent: appreciation focused (delta = 0.6)
Capital constrained rather than income, prices at market value
subject to policy restrictions"""

from housing_abm.equations.expenditure import price_appreciation_expectation
from housing_abm.equations.investor_probs import p_buy_investor, p_sell_investor
from housing_abm.equations.investor_yield import (
    effective_yield_sell,
    expected_yield_buy,
)
from housing_abm.equations.mortgage import down_payment_investor, passes_investor_dscr
from housing_abm.equations.rental_pricing import institutional_rent
from housing_abm.equations.selling import asking_price

from .base import HouseholdAgent


class InstitutionalInvestor(HouseholdAgent):

    WEALTH_KEY = "institutional_investor"
    LOAN_TYPE = "investor_dscr"
    DELTA = 0.6

    def __init__(self, model, available_capital: float, tract_id: str):
        # income/age don't apply, we override EQ 1/2
        super().__init__(model, income=1.0, age=None, tract_id=tract_id)
        self.available_capital = available_capital  # in place of income
        self.properties = []
        self.house_to_sell = None  # unused, kept for ownership market

    @property
    def bank_balance(self):
        """rename bank_balance so that ownership_market functions"""
        return self.available_capital

    @bank_balance.setter
    def bank_balance(self, value):
        self.available_capital = value

    def refresh_desired_balance(self):
        """EQ 1 doesn't apply: investors are capital constrained, not income"""

    def step(self):
        net_rental_cash_flow = (
            self._collect_rent_and_pay_mortgages()
        )  # analogous to landlords, collect rent and pay mortgage
        self.available_capital += net_rental_cash_flow  # add profit to balance
        self.model.prevent_bankruptcy(self)  # inject cash to prevent bankruptcy

        self._reprice_vacant_units()  # no stickiness
        # see if they should buy or sell any houses
        self._evaluate_sell_decision()
        self._evaluate_buy_decision()

    def _collect_rent_and_pay_mortgages(self) -> float:
        total_rent = sum(
            u.rent or 0.0 for u in self.properties if u.tenant is not None
        )  # only collect rent if someone lives there
        total_mortgage = sum(
            u.mortgage_payment or 0.0 for u in self.properties
        )  # pay all mortgages
        return total_rent - total_mortgage  # could be negative

    def _reprice_vacant_units(self):
        # instead of stickiness, due to programs used by investors they reprice at market rate
        premium = self.model.params["rental_pricing_eq11"]["institutional"]["premium"]
        for unit in self.properties:
            if unit.on_rental_market and unit.tenant is None:  # currently selling
                tract = self.model.tracts[unit.tract_id]
                market_rate = tract.rent_per_quality * unit.quality
                unit.rent = institutional_rent(
                    market_rate, premium=premium
                )  # reprice at market rate + premium

    def _evaluate_sell_decision(self):
        """copy as before/small landlord"""
        yield_cfg = self.model.params["investor_yield_eq9_eq12"]
        prob_cfg = self.model.params["investor_probs_eq10_eq13"]
        for unit in list(self.properties):  # see if they should sell any unit
            if unit.on_sale_market:  # already on sale, ignore
                continue
            if unit.tenant is not None:
                continue  # wait until vacant
            tract = self.model.tracts[unit.tract_id]
            current_value = tract.avg_sold_price(unit.quality)
            equity = current_value - unit.mortgage_principal
            g = price_appreciation_expectation(
                tract.hpi_history,
                alpha=self.model.params["appreciation_eq4"]["alpha_institutional"],
            )
            psi = effective_yield_sell(
                price=current_value,
                equity=equity,
                delta=self.DELTA,
                g=g,
                kappa=yield_cfg["kappa"],
                r_bar=tract.gross_rental_yield(),
                monthly_mortgage=unit.mortgage_payment,
                policy_cost=yield_cfg["policy_cost"],
            )
            prob_sell = p_sell_investor(psi, beta=prob_cfg["beta_institutional"])
            if self.model.random_gen.random() < prob_sell:
                asking_cfg = self.model.params["asking_price_eq7"]
                unit.price = asking_price(
                    p_bar_tract=current_value,
                    f_bar_tract=tract.avg_days_on_market(),
                    alpha=asking_cfg["alpha"],
                    beta=asking_cfg["beta"],
                    zeta=asking_cfg["zeta"],
                    epsilon_std=asking_cfg["epsilon_std"],
                    rng=self.model.random_gen,
                )
                unit.price = max(
                    unit.price, unit.mortgage_principal
                )  # see repeat_buyer.py comment
                unit.on_sale_market = True
                unit.on_rental_market = False
                self.model.queue_listing(unit, seller=self)

    def _evaluate_buy_decision(self):
        # see if they should buy, subject to policy restrictions
        # uses investor parameters
        yield_cfg = self.model.params["investor_yield_eq9_eq12"]
        prob_cfg = self.model.params["investor_probs_eq10_eq13"]
        down_cfg = self.model.params["downpayment_eq18"]["institutional"]
        dscr_cfg = self.model.params["investor_dscr_eq15"]
        mort_cfg = self.model.mortgage_terms[self.LOAN_TYPE]

        tract = self.model.tracts[self.tract_id]
        target_price = tract.price_per_quality
        g = price_appreciation_expectation(
            tract.hpi_history,
            alpha=self.model.params["appreciation_eq4"]["alpha_institutional"],
        )

        down_payment, is_cash = down_payment_investor(
            price=target_price,
            wealth=self.available_capital,
            agent_type="institutional",
            mu=down_cfg["mu"],
            sigma=down_cfg["sigma"],
            p_cash=down_cfg["p_cash"],
            d_minimum_pct=down_cfg["d_minimum_pct"],
            rng=self.model.random_gen,
        )
        if self.available_capital < down_payment:
            return
        down_payment = min(down_payment, target_price)

        proposed_loan = 0.0 if is_cash else target_price - down_payment
        monthly_mortgage = (
            0.0
            if is_cash
            else self.model.monthly_payment(
                proposed_loan, dscr_cfg["i_btl_monthly"], mort_cfg["term_months"]
            )
        )

        if not is_cash:
            passes = passes_investor_dscr(
                bank_balance=self.available_capital,
                expected_annual_rent_yield=tract.gross_rental_yield(),
                xi_icr=dscr_cfg["xi_icr"],
                i_btl_monthly=dscr_cfg["i_btl_monthly"],
                proposed_loan=proposed_loan,
                chi_max_ltv=mort_cfg["max_ltv"],
            )
            if not passes:
                return

        omega = expected_yield_buy(
            price=target_price,
            down_payment=down_payment,
            delta=self.DELTA,
            g=g,
            kappa=yield_cfg["kappa"],
            r_bar=tract.gross_rental_yield(),
            monthly_mortgage=monthly_mortgage,
            policy_cost=yield_cfg["policy_cost"],
        )
        prob_buy = p_buy_investor(omega, beta=prob_cfg["beta_institutional"])

        if self.model.random_gen.random() < prob_buy:
            self.model.queue_ownership_bid(
                self, max_price=target_price, down_payment=down_payment
            )
