"""Institutional investor: appreciation focused (delta = 0.6), capital- rather
than income-constrained, and prices rents at market plus a premium."""

from housing_abm.equations.investor_probs import p_buy_investor, p_sell_investor
from housing_abm.equations.investor_yield import (
    effective_yield_sell,
    expected_yield_buy,
)
from housing_abm.equations.mortgage import down_payment_investor, passes_investor_dscr
from housing_abm.equations.rental_pricing import institutional_rent
from housing_abm.equations.selling import asking_price
from housing_abm.policies.investor_restrictions import (
    acquisition_price_multiplier,
    compute_policy_cost,
)

from .base import HouseholdAgent


class InstitutionalInvestor(HouseholdAgent):

    WEALTH_KEY = "institutional_investor"
    LOAN_TYPE = "investor_dscr"
    DELTA = 0.6

    @property
    def delta(self) -> float:
        """EQ 9/12 weight on capital gain; config delta_institutional overrides DELTA."""
        return self.model.params["investor_yield_eq9_eq12"].get("delta_institutional", self.DELTA)

    def __init__(self, model, available_capital: float, tract_id: str):
        # income/age don't apply; EQ 1/2 are overridden
        super().__init__(model, income=1.0, age=None, tract_id=tract_id)
        self.available_capital = available_capital
        self.properties = []
        self.house_to_sell = None  # unused; the ownership market expects it

    @property
    def bank_balance(self):
        """Alias for available_capital, so shared market code works unchanged."""
        return self.available_capital

    @bank_balance.setter
    def bank_balance(self, value):
        self.available_capital = value

    def refresh_desired_balance(self):
        """EQ 1 doesn't apply: investors are capital constrained, not income"""

    def step(self):
        net_rental_cash_flow = self._collect_rent_and_pay_mortgages()
        self.available_capital += net_rental_cash_flow
        self._distribute_to_shareholders(net_rental_cash_flow)
        self.model.prevent_bankruptcy(self)

        self._reprice_vacant_units()
        self._evaluate_sell_decision()
        self._evaluate_buy_decision()

    def _distribute_to_shareholders(self, net_cash_flow: float) -> None:
        """Pay out a share of positive cash flow (REIT-style), so capital
        doesn't compound without limit."""
        if net_cash_flow <= 0:
            return
        payout_ratio = self.model.params["investor_yield_eq9_eq12"].get(
            "institutional_payout_ratio", 0.0
        )
        self.available_capital -= payout_ratio * net_cash_flow

    def _collect_rent_and_pay_mortgages(self) -> float:
        total_rent = sum(
            u.rent or 0.0 for u in self.properties if u.tenant is not None
        )
        total_mortgage = sum(u.mortgage_payment or 0.0 for u in self.properties)
        return total_rent - total_mortgage

    def _reprice_vacant_units(self):
        """Algorithmic pricing: vacant units always reset to market rent + premium."""
        premium = self.model.params["rental_pricing_eq11"]["institutional"]["premium"]
        for unit in self.properties:
            if unit.on_rental_market and unit.tenant is None:
                tract = self.model.tracts[unit.tract_id]
                market_rate = tract.rent_per_quality * unit.quality
                unit.rent = institutional_rent(market_rate, premium=premium)

    def _evaluate_sell_decision(self):
        """EQ 12/13: sell vacant units based on effective yield."""
        yield_cfg = self.model.params["investor_yield_eq9_eq12"]
        prob_cfg = self.model.params["investor_probs_eq10_eq13"]
        for unit in list(self.properties):
            if unit.on_sale_market:
                continue
            if unit.tenant is not None:
                continue  # only sell once the lease ends
            tract = self.model.tracts[unit.tract_id]
            current_value = tract.avg_sold_price(unit.quality)
            equity = current_value - unit.mortgage_principal
            g = tract.appreciation_g(alpha=self.model.params["appreciation_eq4"]["alpha_institutional"])

            psi = effective_yield_sell(
                price=current_value,
                equity=equity,
                delta=self.delta,
                g=g,
                kappa=yield_cfg["kappa"],
                r_bar=tract.gross_rental_yield(),
                monthly_mortgage=unit.mortgage_payment,
                policy_cost=compute_policy_cost(self.model, self),
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
                unit.price = max(unit.price, unit.mortgage_principal)
                unit.on_rental_market = False
                self.model.list_for_sale(unit, seller=self)

    def _evaluate_buy_decision(self):
        """EQ 9/10: expected yield -> purchase probability -> ownership bid."""
        yield_cfg = self.model.params["investor_yield_eq9_eq12"]
        prob_cfg = self.model.params["investor_probs_eq10_eq13"]
        down_cfg = self.model.params["downpayment_eq18"]["institutional"]
        dscr_cfg = self.model.params["investor_dscr_eq15"]
        mort_cfg = self.model.mortgage_terms[self.LOAN_TYPE]

        tract = self.model.tracts[self.tract_id]
        target_price = tract.price_per_quality
        g = tract.appreciation_g(alpha=self.model.params["appreciation_eq4"]["alpha_institutional"])

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
        down_payment = min(down_payment, target_price)
        # purchase tax raises the cash outlay; returns still accrue on market value
        tax_multiplier = acquisition_price_multiplier(self.model, self)
        cash_outlay = down_payment + (tax_multiplier - 1.0) * target_price
        if self.available_capital < cash_outlay:
            return

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
            down_payment=cash_outlay,
            delta=self.delta,
            g=g,
            kappa=yield_cfg["kappa"],
            r_bar=tract.gross_rental_yield(),
            monthly_mortgage=monthly_mortgage,
            policy_cost=compute_policy_cost(self.model, self),
        )
        prob_buy = p_buy_investor(omega, beta=prob_cfg["beta_institutional"])

        if self.model.random_gen.random() < prob_buy:
            self.model.queue_ownership_bid(
                self,
                max_price=target_price,
                down_payment=down_payment,
                acquisition_tax=cash_outlay - down_payment,
            )
