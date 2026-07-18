"""per-tract state container
TODO: replace with real hpi_history during calibration"""

class Tract:
    def __init__(self, tract_id: str, price_per_quality: float = 250_000.0, 
                 rent_per_quality: float = 1400.0, hpi_history: list[float] | None = None):
        self.tract_id = tract_id
        self.price_per_quality = price_per_quality
        self.rent_per_quality = rent_per_quality
        #15 flat months, zero appreciation trend for placeholder
        self.hpi_history = hpi_history if hpi_history is not None else [100.0]*15 #placeholder value if not given
        self.recent_sales = [] #list of (price, quality) tuples
        self.recent_days_on_market = [] #list of days on market for recent sales

    def record_sale(self, price: float, quality: float, days_on_market: float, window: int = 12):
        """trailing window of recent transactions"""
        self.recent_sales.append((price, quality))
        self.recent_days_on_market.append(days_on_market)
        self.recent_sales = self.recent_sales[-window:]
        self.recent_days_on_market = self.recent_days_on_market[-window:]

    def avg_sold_price(self, quality: float) -> float:
        """average price of recent sales for a given quality
        falls back to price_per_quality if no sales recorded"""
        per_quality = [p / q for p, q in self.recent_sales if q > 0]
        if not per_quality: #no recent purchases
            return self.price_per_quality * quality
        return (sum(per_quality) / len(per_quality)) * quality

    def avg_days_on_market(self) -> float:
        if not self.recent_days_on_market:
            return 30.0 #default placeholder
        return sum(self.recent_days_on_market) / len(self.recent_days_on_market) #average
    
    