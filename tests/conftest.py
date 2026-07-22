"""Shared pytest fixtures for the equation test suite."""

import numpy as np
import pytest


class FixedRNG:
    """Fake random generator that outputs determined values"""

    def __init__(
        self,
        normal_value: float = 0.0,
        random_value: float = 0.5,
        geometric_value: int = 1,
    ):
        self._normal_value = normal_value
        self._random_value = random_value
        self._geometric_value = geometric_value

    def normal(self, loc=0.0, scale=1.0):
        return loc + self._normal_value * scale

    def random(self):
        return self._random_value

    def geometric(self, p):
        return self._geometric_value

    def integers(self, low, high=None):
        return low if high is None else low

    def poisson(self, lam):
        return int(round(lam))

    def choice(self, seq):
        return seq[0]

    def shuffle(self, seq):
        pass


@pytest.fixture
def rng():
    """A real RNG for statistical/property-based checks."""
    return np.random.default_rng(12345)


@pytest.fixture
def zero_rng():
    """A fake RNG with epsilon/noise pinned to 0,"""
    return FixedRNG(normal_value=0.0)


@pytest.fixture
def never_trigger_rng():
    """random() always returns 1.0 - fails any `rng.random() < prob` check,
    """
    return FixedRNG(random_value=1.0)


@pytest.fixture
def always_trigger_rng():
    """random() always returns 0.0 - passes any `rng.random() < prob` check
    (as long as prob > 0)."""
    return FixedRNG(random_value=0.0)


@pytest.fixture
def make_fixed_rng():
    """Factory fixture: make_fixed_rng(random_value=0.2) -> FixedRNG instance."""
    return FixedRNG
