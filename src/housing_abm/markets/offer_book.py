"""Price-sorted offer book with prefix-argmax lookup.

Phase 1 of the double auction has every remaining bid pick its preferred
affordable offer. Written directly that is a scan over all offers for every
bid, so the clearing step costs O(bids x offers) per round -- quadratic in
population, and quadratic with a large constant, since the scan also ran the
per-unit policy predicate for every (bid, offer) pair.

That cost is what pinned the study at 300 households, where the binomial noise
on a rate computed over a few hundred agents is the same order as the policy
effects being estimated. Population is the most direct lever on that noise, so
the matching step needs to not be quadratic.

Offers are sorted by price once per round with a running argmax of the
preference key. A bid's preferred offer is then a binary search for its budget
followed by one array lookup: O(log n) per bid instead of O(n).

Ties in the preference key -- common, because the construction sector emits
units at exactly quality 1.0 -- are broken by a per-round random jitter drawn
from the model RNG, which reproduces the uniform random tie-break the previous
implementation did explicitly.
"""

from bisect import bisect_right


class OfferBook:
    def __init__(self, offers, key_fn, rng, price_attr="price"):
        decorated = []
        for unit in offers:
            price = getattr(unit, price_attr)
            if price is None:
                continue
            # (key, jitter) makes the ordering total, so the argmax is unique
            # and uniformly distributed among genuine ties
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
        """Highest-preference offer priced at or below max_price.

        exclude_owner skips listings the bidder owns; excluded skips a set of
        units the bidder has already been outbid on this month.
        """
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
        # the top pick is blocked for this bidder specifically; rare, so a scan
        # of the affordable prefix is cheap enough
        best_unit = None
        best_key = None
        for i in range(hi + 1):
            if blocked(self.units[i]):
                continue
            if best_key is None or self.keys[i] > best_key:
                best_key, best_unit = self.keys[i], self.units[i]
        return best_unit
