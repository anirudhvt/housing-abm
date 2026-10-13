"""per-tract state container

Price formation follows the reference Java model (Baptista et al. 2016,
HousingMarketStats.java): a two-stage smoothed price series rather than a raw
overwrite from recent transactions. See Tract.update_hpi_history for why.

Rent formation mirrors it exactly -- see Tract.update_rent_history. Before
that existed, rent_per_quality was set once from config and never updated by
the rental market: across 240 simulated months it took exactly one value,
while actual unit rents in the same run ranged 740-7,846. Since that frozen
number is the market signal read by EQ5's buy-vs-rent decision, EQ9/12's
investor yield, EQ11's rent setting and the investor yield ranking, the
reference's stabilising loop (investors buy -> rental supply up -> rents
fall -> renting more attractive) could not operate at all. See
docs/reference_gap_analysis.md Problem 1.
"""


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
        # 15 flat months, zero appreciation trend for placeholder
        self.hpi_history = (
            hpi_history if hpi_history is not None else [price_per_quality] * 15
        )  # placeholder value if not given

        # Fixed calibrated "fundamental" price level (reference Java model's
        # referencePricePerQuality, from real HPI data). price_per_quality
        # reverts toward this * house_price_index every month -- see
        # update_hpi_history. Immutable after construction; defaults to the
        # tract's starting price if no independent calibration is given.
        self.reference_price_per_quality = (
            reference_price_per_quality
            if reference_price_per_quality is not None
            else price_per_quality
        )
        # House price index: ratio of actual transacted prices to the
        # reference-implied level, updated from each month's completed sales
        # (reference Java model's housePriceIndex). Starts at 1.0 (transacting
        # at the calibrated reference level).
        self.house_price_index = 1.0
        # Monthly smoothing weight on new transactions (EMA) and the fraction
        # of the smoothed price retained each month before blending in the
        # reversion term. See config/baseline_params.yaml market_smoothing for
        # where these values come from.
        self.smoothing_factor = smoothing_factor
        self.price_decay = price_decay
        self.rent_decay = rent_decay

        # Rent-side mirrors of the three price fields above: a fixed
        # calibrated anchor, an index of actual-to-reference transacted rent,
        # and the smoothed series agents actually read.
        self.reference_rent_per_quality = (
            reference_rent_per_quality
            if reference_rent_per_quality is not None
            else rent_per_quality
        )
        self.rent_index = 1.0
        self.rent_history = [rent_per_quality] * 15

        # This month's completed sales (price, quality), consumed and reset
        # each time update_hpi_history runs. Distinct from
        # recent_days_on_market below, which tracks a longer trailing window.
        self._monthly_sales: list[tuple[float, float]] = []
        self.recent_days_on_market = []  # trailing window, days on market for recent sales
        # rent-side equivalents, consumed by update_rent_history
        self._monthly_lettings: list[tuple[float, float]] = []
        self.recent_days_vacant = []  # trailing window, days-to-let for recent lettings

        #real world ZHVI/ZORI series
        self.external_g_series = external_g_series
        self.external_rent_growth_series = external_rent_growth_series
        self._g_index = 0
        self._rent_growth_index = 0

    def record_sale(
        self, price: float, quality: float, days_on_market: float, window: int = 60
    ):
        """Record a completed sale: this month's realized-price sample (fed
        into update_hpi_history's EMA) and the trailing days-on-market window
        (fed into EQ7's f_bar_tract and EQ19/20's occupancy estimate)."""
        self._monthly_sales.append((price, quality))
        self.recent_days_on_market.append(days_on_market)
        self.recent_days_on_market = self.recent_days_on_market[-window:]

    def avg_sold_price(self, quality: float) -> float:
        """Current smoothed market price for a house of the given quality.

        Matches the reference Java model's getExpAvSalePriceForQuality: reads
        directly off the smoothed price series (see update_hpi_history) rather
        than a raw median of recent transactions, so asking prices, investor
        yield calculations, and forced-divestiture pricing all see the same
        stabilized price level appreciation itself is computed from.
        """
        return self.price_per_quality * quality

    def avg_days_on_market(self) -> float:
        if not self.recent_days_on_market:
            return 30.0  # default placeholder
        return sum(self.recent_days_on_market) / len(
            self.recent_days_on_market
        )  # average

    def record_letting(
        self, rent: float, quality: float, days_vacant: float, window: int = 60
    ):
        """Record a completed letting -- the rent-side mirror of record_sale.

        Feeds update_rent_history's EMA and the trailing days-to-let window
        that EQ11 uses as its f_bar (the reference is explicit that EQ11's
        f_bar is days on the *rental* market, not the ownership market).
        """
        self._monthly_lettings.append((rent, quality))
        self.recent_days_vacant.append(days_vacant)
        self.recent_days_vacant = self.recent_days_vacant[-window:]

    def market_rent(self, quality: float) -> float:
        """Current smoothed market rent for a house of the given quality --
        the rent-side mirror of avg_sold_price."""
        return self.rent_per_quality * quality

    def avg_days_vacant(self) -> float:
        if not self.recent_days_vacant:
            return 30.0  # default placeholder, matches avg_days_on_market
        return sum(self.recent_days_vacant) / len(self.recent_days_vacant)

    def update_hpi_history(self, window: int = 24):
        """Update the smoothed price level and appreciation history.

        Two-stage update, ported from the reference Java model's
        HousingMarketStats.postClearingRecord() (see module docstring and
        config/baseline_params.yaml market_smoothing for the full rationale):

        Stage 1 (EMA): if any sales completed this month, blend their average
        price into the smoothed price at a fixed monthly weight
        (smoothing_factor), and update the house price index from the ratio
        of actual to reference-implied transacted value. Skipped in months
        with no sales -- the smoothed price and HPI simply hold their
        previous values, they are not undefined or reset to zero.

        Stage 2 (reversion): every month, regardless of whether stage 1 ran,
        pull the smoothed price back toward reference_price_per_quality *
        house_price_index. This is what stage 1 alone lacks: nothing in an
        EMA prevents it drifting arbitrarily far from fundamentals over a
        long sales drought, since it only ever updates toward whatever
        (possibly noisy, possibly sparse) transactions happen to occur.

        Call once ownership market has run.
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
                # EMA'd, not a raw overwrite. The reference Java model pools a
                # month's sales across many quality bins before forming this
                # ratio, which averages out a lot of single-transaction noise
                # even in an otherwise thin market. This model has one price
                # series for the whole tract (no quality bins to pool across),
                # so its monthly HPI sample is exactly as sparse as its
                # monthly price sample -- and stage 2 below applies it at 50%
                # weight. Left unsmoothed, an outlier month's ratio would feed
                # straight back into the very reversion term meant to damp
                # that same outlier, roughly doubling its effective weight
                # instead of damping it.
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
        """Update the smoothed rent level from this month's completed lettings.

        Exact mirror of update_hpi_history, for the same reasons: stage 1
        blends this month's realized rents into the smoothed series at
        smoothing_factor and updates a rent index (actual-to-reference
        transacted rent, itself EMA'd so one thin month can't set it
        outright); stage 2 reverts toward reference_rent_per_quality *
        rent_index every month so the series can't drift arbitrarily far
        from the calibrated level during a letting drought.

        Call once the rental market has run.

        When an external rent-growth series is configured, that drives the
        rent level instead and the endogenous update is skipped -- the same
        override relationship external_g_series has with appreciation_g.
        """
        if self.external_rent_growth_series:
            idx = self._rent_growth_index % len(
                self.external_rent_growth_series
            )  # repeat/cycle data when finished
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
