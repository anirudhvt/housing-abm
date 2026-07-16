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