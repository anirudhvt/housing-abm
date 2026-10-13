"""Paired policy experiment harness.

Design notes -- why this differs from a naive two-arm Monte Carlo.

1. Shared spin-up, then fork.
   The original harness built two independent models per seed and ran each
   through its own spin-up with the policy active from month 1. Passing the
   same seed to both does not make them comparable: the policy changes agent
   decisions in the first month, every later random draw shifts, and after a
   long spin-up the two arms are simply two unrelated realisations that happen
   to share a seed. Measured on the original results, the correlation between
   arms was ~0 and sometimes negative, so differencing them *added* variance
   rather than removing it (sd of the paired difference ~= sqrt(2) x sd of one
   arm, the signature of independent draws).

   Here each seed is spun up once, with no policy, and the resulting model is
   forked. Both arms therefore begin from a bit-identical microstate -- same
   agents, same balances, same houses, same prices -- and differ only by the
   intervention. This is the textbook common-random-numbers setup, and it also
   halves the compute, since the spin-up runs once per seed rather than twice.

2. Time-averaged outcomes.
   Outcomes are averaged over the measurement window rather than read from the
   final month (see housing_abm.metrics).

3. Split RNG streams.
   The exogenous interest-rate path and the demographic process draw from
   dedicated substreams (see AtlantaHousingModel.__init__), so those are
   identical across the two arms of a seed no matter what the policy does.
"""

from __future__ import annotations

import copy

import numpy as np

from housing_abm.metrics import ALL_METRICS, run_window, window_means
from housing_abm.model import AtlantaHousingModel
from housing_abm.policy import load_policies


def build_and_spinup(seed, n_households, spinup, config_path):
    """One no-policy model carried through spin-up, ready to fork."""
    model = AtlantaHousingModel(
        config_path=config_path, n_households=n_households, seed=seed
    )
    if spinup:
        model.run_spinup(spinup)
    return model


def fork_with_policy(model, policy_paths):
    """Deep-copy a spun-up model and switch the policy on in the copy."""
    treated = copy.deepcopy(model)
    if policy_paths:
        load_policies(treated, policy_paths)
    return treated


def run_paired_seed(seed, n_households, spinup, months, policy_paths, config_path):
    """Run one seed's baseline and policy arm from a shared spun-up state.

    Returns (baseline_means, policy_means, baseline_series, policy_series).
    """
    spun_up = build_and_spinup(seed, n_households, spinup, config_path)

    baseline_model = copy.deepcopy(spun_up)
    policy_model = fork_with_policy(spun_up, policy_paths)

    baseline_series = run_window(baseline_model, months)
    policy_series = run_window(policy_model, months)

    return (
        window_means(baseline_series),
        window_means(policy_series),
        baseline_series,
        policy_series,
    )


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def paired_summary(baseline_values, policy_values, n_boot=20_000, rng=None):
    """Paired comparison of one metric across seeds.

    Reports the normal-theory paired t interval and a bootstrap interval over
    the same paired differences; when they disagree materially the normal
    approximation is not safe for that metric.
    """
    b = np.asarray(baseline_values, dtype=float)
    p = np.asarray(policy_values, dtype=float)
    keep = ~(np.isnan(b) | np.isnan(p))
    b, p = b[keep], p[keep]
    n = b.size
    if n < 2:
        return None

    diff = p - b
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)

    from scipy import stats

    t_crit = float(stats.t.ppf(0.975, n - 1))
    t_stat = mean / se if se > 0 else np.inf * np.sign(mean)
    p_value = float(2 * stats.t.sf(abs(t_stat), n - 1)) if se > 0 else 0.0

    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    boot_lo, boot_hi = np.percentile(boot, [2.5, 97.5])

    # correlation between arms: the diagnostic for whether pairing is working
    corr = float(np.corrcoef(b, p)[0, 1]) if n > 2 and b.std() > 0 and p.std() > 0 else np.nan
    # variance reduction actually achieved relative to unpaired sampling
    var_unpaired = b.var(ddof=1) + p.var(ddof=1)
    var_paired = diff.var(ddof=1)
    vrf = float(var_unpaired / var_paired) if var_paired > 0 else np.inf

    return {
        "n": int(n),
        "baseline_mean": float(b.mean()),
        "policy_mean": float(p.mean()),
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": float(se),
        "ci_lo": mean - t_crit * se,
        "ci_hi": mean + t_crit * se,
        "boot_ci_lo": float(boot_lo),
        "boot_ci_hi": float(boot_hi),
        "t_stat": float(t_stat),
        "p_value": p_value,
        "arm_correlation": corr,
        "variance_reduction_factor": vrf,
        "seeds_same_direction": int(max((diff > 0).sum(), (diff < 0).sum())),
    }


def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjustment.

    Six policies are tested against the same baseline on three outcomes each,
    so the family-wise error rate needs controlling before any single result
    is called significant. Holm is uniformly more powerful than Bonferroni and
    makes no independence assumption, which matters here because the arms
    share a spun-up state.
    """
    p_values = list(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * p_values[i]
        running = max(running, value)
        adjusted[i] = min(running, 1.0)
    return adjusted


def minimum_detectable_effect(sd_diff, n_seeds, power=0.80, alpha=0.05):
    """Smallest paired difference detectable at the given power.

    Reported alongside every null result so a null can be read as "the effect
    is smaller than X" rather than as "there is no effect".
    """
    from scipy import stats

    if n_seeds < 2:
        return np.nan
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sd_diff / np.sqrt(n_seeds))


def seeds_needed_for(effect, sd_diff, power=0.80, alpha=0.05):
    """Seeds required to detect `effect` at the given power."""
    from scipy import stats

    if effect == 0:
        return np.inf
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float(np.ceil(((z_a + z_b) * sd_diff / abs(effect)) ** 2))
