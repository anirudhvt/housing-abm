"""per-tract state container
TODO: replace with real hpi_history during calibration"""


class Tract:
    def __init__(
        self,
        tract_id: str,
        price_per_quality: float = 250_000.0,
        rent_per_quality: float = 1400.0,
        hpi_history: list[float] | None = None,
        external_g_series: list[float] | None = None,
        external_rent_growth_series: list[float] | None = None
    ):
        self.tract_id = tract_id
        self.price_per_quality = price_per_quality
        self.rent_per_quality = rent_per_quality
        # 15 flat months, zero appreciation trend for placeholder
        self.hpi_history = (
            hpi_history if hpi_history is not None else [price_per_quality] * 15
        )  # placeholder value if not given
        self.recent_sales = []  # list of (price, quality) tuples
        self.recent_days_on_market = []  # list of days on market for recent sales

        #real world ZHVI/ZORI series
        self.external_g_series = external_g_series
        self.external_rent_growth_series = external_rent_growth_series
        self._g_index = 0
        self._rent_growth_index = 0

    def record_sale(
        self, price: float, quality: float, days_on_market: float, window: int = 60
    ):
        """trailing window of recent transactions"""
        self.recent_sales.append((price, quality))
        self.recent_days_on_market.append(days_on_market)
        self.recent_sales = self.recent_sales[-window:]
        self.recent_days_on_market = self.recent_days_on_market[-window:]

    def avg_sold_price(self, quality: float) -> float:
        """median price of recent sales for a given quality
        falls back to price_per_quality if no sales recorded"""
        per_quality = [p / q for p, q in self.recent_sales if q > 0]
        if not per_quality:  # no recent purchases
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
            return 30.0  # default placeholder
        return sum(self.recent_days_on_market) / len(
            self.recent_days_on_market
        )  # average

    def gross_rental_yield(self) -> float:
        """r_bar for EQ 9/12: annualized gross rent/price, per quality"""
        if self.price_per_quality <= 0:
            return 0.0
        return (self.rent_per_quality * 12) / self.price_per_quality

    def update_hpi_history(self, window: int = 24):
        """Append this month's tract level price index to history
        Call once ownership market has run"""
        self.hpi_history.append(self.avg_sold_price(quality=1.0))
        self.hpi_history = self.hpi_history[-window:]

        if self.external_rent_growth_series:
            idx = self._rent_growth_index % len(self.external_rent_growth_series) #repeat/cycle data when finished
            self.rent_per_quality *= 1.0 + self.external_rent_growth_series[idx]
            self._rent_growth_index += 1

        if self.external_g_series:
            self._g_index += 1

    def appreciation_g(self, alpha: float = 1.0) -> float | None: 
        """EQ 4: trailing appreciation estimate.
 
            g = alpha * ( (h[-1]+h[-2]+h[-3]) / (h[-13]+h[-14]+h[-15]) - 1 )
 
        Returns None if hpi_history doesn't yet have 15 months of data."""

        if self.external_g_series:
            idx = self._g_index % len(self.external_g_series)
            raw_g = self.external_g_series[idx]
            return max(min(alpha*raw_g,0.25), -0.10) #clamp down the appreciatoin

        from housing_abm.equations.expenditure import price_appreciation_expectation

        if len(self.hpi_history) < 15:
            return None
        return price_appreciation_expectation(self.hpi_history, alpha=alpha)

    def gross_rental_yield(self) -> float:
        """r_bar for EQ 9/12: annual gross rent per quality"""
        if self.price_per_quality <= 0:
            return 0.0
        return (self.rent_per_quality * 12)/self.price_per_quality

