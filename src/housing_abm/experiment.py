"""Paired policy experiments: each seed is spun up once with no policy, then
forked into baseline and policy arms that start from an identical state.
Also holds the paired inference helpers."""

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
    """Paired t and bootstrap intervals for one metric across seeds."""
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

    # arm-to-arm correlation shows whether pairing is reducing variance
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
    """Holm-Bonferroni step-down adjustment across the policy x outcome grid."""
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
    """Smallest paired difference detectable at the given power."""
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
