"""Price-sorted offer book with a running argmax of the preference key, so each
bid finds its best affordable offer in O(log n). Ties are broken by random jitter."""

from bisect import bisect_right


class OfferBook:
    def __init__(self, offers, key_fn, rng, price_attr="price"):
        decorated = []
        for unit in offers:
            price = getattr(unit, price_attr)
            if price is None:
                continue
            # random jitter breaks ties in the preference key uniformly
            decorated.append((price, key_fn(unit), rng.random(), unit))
        decorated.sort(key=lambda row: row[0])

        self.prices = [row[0] for row in decorated]
        self.keys = [(row[1], row[2]) for row in decorated]
        self.units = [row[3] for row in decorated]

        self.argmax = []
        best_index = -1
        best_key = None
        for i, key in enumerate(self.keys):
            if best_key is None or key > best_key:
                best_index, best_key = i, key
            self.argmax.append(best_index)

    def __len__(self):
        return len(self.units)

    def best_affordable(self, max_price, exclude_owner=None, excluded=None):
        """Highest-preference offer priced at or below max_price, skipping the
        bidder's own listings and any units in `excluded`."""
        hi = bisect_right(self.prices, max_price) - 1
        if hi < 0:
            return None

        def blocked(unit):
            if exclude_owner is not None and unit.owner is exclude_owner:
                return True
            return bool(excluded) and unit in excluded

        candidate = self.units[self.argmax[hi]]
        if not blocked(candidate):
            return candidate
        # top pick is blocked for this bidder (rare): scan the affordable prefix
        best_unit = None
        best_key = None
        for i in range(hi + 1):
            if blocked(self.units[i]):
                continue
            if best_key is None or self.keys[i] > best_key:
                best_key, best_unit = self.keys[i], self.units[i]
        return best_unit
