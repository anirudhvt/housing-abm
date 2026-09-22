"""Tests for investor_propensity: BTL gene selection."""

import numpy as np
import pytest
from housing_abm.equations.investor_propensity import (
    landlord_selection_weight,
    select_future_landlords,
    SCF_DECILE_PROBS,
)


def test_weight_bottom_decile():
    incomes = list(range(100, 1100, 100))
    w = landlord_selection_weight(100, incomes)
    assert np.isclose(w, SCF_DECILE_PROBS[1])


def test_weight_top_decile():
    incomes = list(range(100, 1100, 100))
    w = landlord_selection_weight(1000, incomes)
    assert np.isclose(w, SCF_DECILE_PROBS[9])


def test_weight_increases_with_income():
    incomes = list(range(100, 1100, 100))
    w_low = landlord_selection_weight(200, incomes)
    w_high = landlord_selection_weight(900, incomes)
    assert w_high > w_low


def test_weight_empty_incomes_returns_middle():
    w = landlord_selection_weight(50000, [])
    assert np.isclose(w, SCF_DECILE_PROBS[4])


class FakeAgent:
    def __init__(self, income):
        self.income = income


def test_select_returns_correct_count():
    rng = np.random.default_rng(42)
    agents = [FakeAgent(inc) for inc in range(1000, 11000, 1000)]
    selected = select_future_landlords(agents, 3, rng)
    assert len(selected) == 3


def test_select_favors_higher_income():
    rng = np.random.default_rng(42)
    agents = [FakeAgent(inc) for inc in range(1000, 11000, 1000)]
    selections = {id(a): 0 for a in agents}
    for seed in range(200):
        rng2 = np.random.default_rng(seed)
        sel = select_future_landlords(agents, 1, rng2)
        selections[id(sel[0])] += 1
    top_half = sum(selections[id(a)] for a in agents[5:])
    bottom_half = sum(selections[id(a)] for a in agents[:5])
    assert top_half > bottom_half


def test_select_no_duplicates():
    rng = np.random.default_rng(42)
    agents = [FakeAgent(inc) for inc in range(1000, 6000, 1000)]
    selected = select_future_landlords(agents, 5, rng)
    assert len(set(id(a) for a in selected)) == 5


def test_select_empty_returns_empty():
    rng = np.random.default_rng(42)
    assert select_future_landlords([], 5, rng) == []


def test_select_zero_target():
    rng = np.random.default_rng(42)
    agents = [FakeAgent(1000)]
    assert select_future_landlords(agents, 0, rng) == []
