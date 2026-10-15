"""Per-tract market state: smoothed price and rent levels, ported from the
reference Java model's HousingMarketStats (Baptista et al. 2016)."""


class Tract:
    def __init__(
        self,
        tract_id: str,
        price_per_quality: float = 250_000.0,
        rent_per_quality: float = 1400.0,
        hpi_history: list[float] | None = None,
        external_g_series: list[float] | None = None,
        external_rent_growth_series: list[float] | None = None,
        reference_price_per_quality: float | None = None,
        reference_rent_per_quality: float | None = None,
        smoothing_factor: float = 0.1091,
        price_decay: float = 0.5,
        rent_decay: float = 0.5,
    ):
        self.tract_id = tract_id
        self.price_per_quality = price_per_quality
        self.rent_per_quality = rent_per_quality
        # default: 15 flat months, i.e. zero initial appreciation
        self.hpi_history = (
            hpi_history if hpi_history is not None else [price_per_quality] * 15
        )

        # fixed calibrated anchor that price_per_quality reverts toward
        self.reference_price_per_quality = (
            reference_price_per_quality
            if reference_price_per_quality is not None
            else price_per_quality
        )
        # smoothed ratio of transacted prices to the reference-implied level
        self.house_price_index = 1.0
        # EMA weight on new transactions, and share retained before reversion
        self.smoothing_factor = smoothing_factor
        self.price_decay = price_decay
        self.rent_decay = rent_decay

        # rent-side mirrors of the price fields above
        self.reference_rent_per_quality = (
            reference_rent_per_quality
            if reference_rent_per_quality is not None
            else rent_per_quality
        )
        self.rent_index = 1.0
        self.rent_history = [rent_per_quality] * 15

        # this month's completed sales (price, quality), reset by update_hpi_history
        self._monthly_sales: list[tuple[float, float]] = []
        self.recent_days_on_market = []  # trailing window, days on market for recent sales
        # rent-side equivalents, consumed by update_rent_history
        self._monthly_lettings: list[tuple[float, float]] = []
        self.recent_days_vacant = []  # trailing window, days-to-let for recent lettings

        # optional real ZHVI/ZORI series
        self.external_g_series = external_g_series
        self.external_rent_growth_series = external_rent_growth_series
        self._g_index = 0
        self._rent_growth_index = 0

    def record_sale(
        self, price: float, quality: float, days_on_market: float, window: int = 60
    ):
        """Record a completed sale for the price EMA and the days-on-market window."""
        self._monthly_sales.append((price, quality))
        self.recent_days_on_market.append(days_on_market)
        self.recent_days_on_market = self.recent_days_on_market[-window:]

    def avg_sold_price(self, quality: float) -> float:
        """Smoothed market price for a house of the given quality."""
        return self.price_per_quality * quality

    def avg_days_on_market(self) -> float:
        if not self.recent_days_on_market:
            return 30.0  # default placeholder
        return sum(self.recent_days_on_market) / len(self.recent_days_on_market)

    def record_letting(
        self, rent: float, quality: float, days_vacant: float, window: int = 60
    ):
        """Record a completed letting for the rent EMA and the days-to-let
        window (EQ 11's f_bar)."""
        self._monthly_lettings.append((rent, quality))
        self.recent_days_vacant.append(days_vacant)
        self.recent_days_vacant = self.recent_days_vacant[-window:]

    def market_rent(self, quality: float) -> float:
        """Smoothed market rent for a house of the given quality."""
        return self.rent_per_quality * quality

    def avg_days_vacant(self) -> float:
        if not self.recent_days_vacant:
            return 30.0  # default placeholder
        return sum(self.recent_days_vacant) / len(self.recent_days_vacant)

    def update_hpi_history(self, window: int = 24):
        """Update the smoothed price after the ownership market clears.

        Stage 1 (only if there were sales): EMA this month's prices into
        price_per_quality and house_price_index. Stage 2 (every month): revert
        toward reference_price_per_quality * house_price_index.
        """
        if self._monthly_sales:
            month_avg = sum(p / q for p, q in self._monthly_sales if q > 0) / len(
                self._monthly_sales
            )
            self.price_per_quality = (
                self.smoothing_factor * month_avg
                + (1.0 - self.smoothing_factor) * self.price_per_quality
            )
            sum_sold_price = sum(p for p, q in self._monthly_sales if q > 0)
            sum_reference_price = sum(
                self.reference_price_per_quality * q
                for p, q in self._monthly_sales
                if q > 0
            )
            if sum_reference_price > 0:
                # smoothed rather than overwritten: with one tract-wide series a
                # single outlier month would otherwise drive the reversion target
                monthly_hpi = sum_sold_price / sum_reference_price
                self.house_price_index = (
                    self.smoothing_factor * monthly_hpi
                    + (1.0 - self.smoothing_factor) * self.house_price_index
                )
            self._monthly_sales = []

        self.price_per_quality = (
            self.price_decay * self.price_per_quality
            + (1.0 - self.price_decay)
            * (self.house_price_index * self.reference_price_per_quality)
        )

        self.hpi_history.append(self.price_per_quality)
        self.hpi_history = self.hpi_history[-window:]

        if self.external_g_series:
            self._g_index += 1

    def update_rent_history(self, window: int = 24):
        """Rent-side mirror of update_hpi_history, run after the rental market
        clears. An external rent-growth series, if set, replaces the update."""
        if self.external_rent_growth_series:
            idx = self._rent_growth_index % len(
                self.external_rent_growth_series
            )  # cycle the series once exhausted
            self.rent_per_quality *= 1.0 + self.external_rent_growth_series[idx]
            self._rent_growth_index += 1
            self._monthly_lettings = []
            self.rent_history.append(self.rent_per_quality)
            self.rent_history = self.rent_history[-window:]
            return

        if self._monthly_lettings:
            month_avg = sum(r / q for r, q in self._monthly_lettings if q > 0) / len(
                self._monthly_lettings
            )
            self.rent_per_quality = (
                self.smoothing_factor * month_avg
                + (1.0 - self.smoothing_factor) * self.rent_per_quality
            )
            sum_let_rent = sum(r for r, q in self._monthly_lettings if q > 0)
            sum_reference_rent = sum(
                self.reference_rent_per_quality * q
                for r, q in self._monthly_lettings
                if q > 0
            )
            if sum_reference_rent > 0:
                monthly_index = sum_let_rent / sum_reference_rent
                self.rent_index = (
                    self.smoothing_factor * monthly_index
                    + (1.0 - self.smoothing_factor) * self.rent_index
                )
            self._monthly_lettings = []

        self.rent_per_quality = (
            self.rent_decay * self.rent_per_quality
            + (1.0 - self.rent_decay)
            * (self.rent_index * self.reference_rent_per_quality)
        )

        self.rent_history.append(self.rent_per_quality)
        self.rent_history = self.rent_history[-window:]

    def appreciation_g(self, alpha: float = 1.0) -> float | None:
        """EQ 4 trailing appreciation; None until hpi_history has 15 months."""

        if self.external_g_series:
            idx = self._g_index % len(self.external_g_series)
            raw_g = self.external_g_series[idx]
            return max(min(alpha * raw_g, 0.25), -0.10)

        from housing_abm.equations.expenditure import price_appreciation_expectation

        if len(self.hpi_history) < 15:
            return None
        return price_appreciation_expectation(self.hpi_history, alpha=alpha)

    def gross_rental_yield(self) -> float:
        """r_bar for EQ 9/12: annual gross rent per quality"""
        if self.price_per_quality <= 0:
            return 0.0
        return (self.rent_per_quality * 12) / self.price_per_quality
