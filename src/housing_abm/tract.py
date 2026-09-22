"""per-tract state container"""


class Tract:
    def __init__(
        self,
        tract_id: str,
        price_per_quality: float = 250_000.0,
        rent_per_quality: float = 1400.0,
        hpi_history: list[float] | None = None,
        external_g_series: list[float] | None = None,
        external_rent_growth_series: list[float] | None = None,
        rent_decay: float = 0.5,
    ):
        self.tract_id = tract_id
        self.price_per_quality = price_per_quality
        self.rent_per_quality = rent_per_quality
        self.reference_rent_per_quality = rent_per_quality
        self.hpi_history = (
            hpi_history if hpi_history is not None else [price_per_quality] * 15
        )
        self.recent_sales = []
        self.recent_days_on_market = []

        self._monthly_lettings = []
        self.recent_days_vacant = []
        self.rent_history = [rent_per_quality] * 15
        self.rent_decay = rent_decay

        self.external_g_series = external_g_series
        self.external_rent_growth_series = external_rent_growth_series
        self._g_index = 0
        self._rent_growth_index = 0

    def record_sale(
        self, price: float, quality: float, days_on_market: float, window: int = 60
    ):
        self.recent_sales.append((price, quality))
        self.recent_days_on_market.append(days_on_market)
        self.recent_sales = self.recent_sales[-window:]
        self.recent_days_on_market = self.recent_days_on_market[-window:]

    def record_letting(
        self, rent: float, quality: float, days_vacant: float, window: int = 60
    ):
        self._monthly_lettings.append((rent, quality))
        self.recent_days_vacant.append(days_vacant)
        self._monthly_lettings = self._monthly_lettings[-window:]
        self.recent_days_vacant = self.recent_days_vacant[-window:]

    def avg_sold_price(self, quality: float) -> float:
        per_quality = [p / q for p, q in self.recent_sales if q > 0]
        if not per_quality:
            return self.price_per_quality * quality
        per_quality.sort()
        n = len(per_quality)
        mid = n // 2
        median = (
            per_quality[mid]
            if n % 2 == 1
            else (per_quality[mid - 1] + per_quality[mid]) / 2.0
        )
        return median * quality

    def avg_days_on_market(self) -> float:
        if not self.recent_days_on_market:
            return 30.0
        return sum(self.recent_days_on_market) / len(self.recent_days_on_market)

    def avg_days_vacant(self) -> float:
        if not self.recent_days_vacant:
            return 30.0
        return sum(self.recent_days_vacant) / len(self.recent_days_vacant)

    def market_rent(self, quality: float = 1.0) -> float:
        per_quality = [r / q for r, q in self._monthly_lettings if q > 0]
        if not per_quality:
            return self.rent_per_quality * quality
        per_quality.sort()
        n = len(per_quality)
        mid = n // 2
        median = (
            per_quality[mid]
            if n % 2 == 1
            else (per_quality[mid - 1] + per_quality[mid]) / 2.0
        )
        return median * quality

    def gross_rental_yield(self) -> float:
        if self.price_per_quality <= 0:
            return 0.0
        return (self.rent_per_quality * 12) / self.price_per_quality

    def update_hpi_history(self, window: int = 24):
        current_index = self.avg_sold_price(quality=1.0)
        self.hpi_history.append(current_index)
        self.hpi_history = self.hpi_history[-window:]
        if len(self.recent_sales) >= 5:
            self.price_per_quality = current_index

        if self.external_rent_growth_series:
            idx = self._rent_growth_index % len(self.external_rent_growth_series)
            self.rent_per_quality *= 1.0 + self.external_rent_growth_series[idx]
            self._rent_growth_index += 1

        if self.external_g_series:
            self._g_index += 1

    def update_rent_history(self, window: int = 24):
        realized = self.market_rent(quality=1.0)
        alpha = self.rent_decay
        smoothed = alpha * realized + (1 - alpha) * self.rent_per_quality
        ref = self.reference_rent_per_quality
        new_rent = smoothed + 0.25 * (ref - smoothed)
        self.rent_per_quality = new_rent
        self.rent_history.append(new_rent)
        self.rent_history = self.rent_history[-window:]
        self._monthly_lettings.clear()

        if self.external_rent_growth_series:
            idx = self._rent_growth_index % len(self.external_rent_growth_series)
            self.rent_per_quality *= 1.0 + self.external_rent_growth_series[idx]

    def appreciation_g(self, alpha: float = 1.0) -> float | None:
        """EQ 4: trailing appreciation estimate."""
        if self.external_g_series:
            idx = self._g_index % len(self.external_g_series)
            raw_g = self.external_g_series[idx]
            return max(min(alpha * raw_g, 0.25), -0.10)

        from housing_abm.equations.expenditure import price_appreciation_expectation

        if len(self.hpi_history) < 15:
            return None
        return price_appreciation_expectation(self.hpi_history, alpha=alpha)
