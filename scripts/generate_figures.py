"""
Generate Figure 4 (policy comparison bar chart) and
Figure 5 (trade-off scatter plot) from policy comparison CSVs.

Usage — from your repo root:
    python scripts/make_figures.py --results results/ --output figures/

Or with the test data:
    python make_figures.py --results /tmp/test_results/ --output /tmp/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_abm.experiment import paired_summary  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────

# arm name in all_policies_raw.csv -> display label
POLICY_ARMS = {
    "waiting_period":     "Waiting Period",
    "ownership_cap_soft": "Ownership Cap (Soft)",
    "ownership_cap_hard": "Ownership Cap (Hard)",
    "purchase_tax":       "Purchase Tax",
    "vacancy_tax":        "Vacancy Tax",
    "portfolio_tax":      "Portfolio Tax",
}

# legacy one-file-per-policy layout, still read as a fallback
POLICY_FILES = {label: f"{arm}.csv" for arm, label in POLICY_ARMS.items()}

# Colors: access restrictions vs financial penalties
POLICY_COLORS = {
    "Waiting Period":       "#2E75B6",   # blue  — access
    "Ownership Cap (Soft)": "#2E75B6",
    "Ownership Cap (Hard)": "#2E75B6",
    "Purchase Tax":         "#C55A11",   # orange — financial
    "Vacancy Tax":          "#C55A11",
    "Portfolio Tax":        "#C55A11",
}

METRICS = {
    "homeownership_rate":    "Homeownership Rate",
    "rental_vacancy_rate":   "Rental Vacancy Rate",
    "annual_appreciation_g": "Annual Price Appreciation",
    "ftb_purchase_share":    "First-Time-Buyer Purchase Share",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _summarise(policy_label, metric, baseline_values, policy_values):
    """One row of the figure summary, using the shared paired estimator.

    This defers to housing_abm.experiment.paired_summary so the intervals
    plotted are exactly the ones the runner reports -- t critical values with
    the right degrees of freedom, and sd with ddof=1. The previous version
    used 1.96 with a population standard deviation, which understates the
    interval at these seed counts.
    """
    summary = paired_summary(baseline_values, policy_values, n_boot=5000)
    if summary is None:
        return None
    return {
        "policy": policy_label,
        "metric": metric,
        "mean_diff": summary["mean_diff"],
        "ci_lower": summary["ci_lo"],
        "ci_upper": summary["ci_hi"],
        "boot_lower": summary["boot_ci_lo"],
        "boot_upper": summary["boot_ci_hi"],
        "p_value": summary["p_value"],
        "arm_correlation": summary["arm_correlation"],
        "n_seeds": summary["n"],
        "pct_same_dir": summary["seeds_same_direction"] / summary["n"],
    }


def load_policy_results(results_dir: str) -> pd.DataFrame:
    """Paired differences per policy and metric.

    Prefers all_policies_raw.csv, where every policy shares one baseline arm
    from the same spun-up seeds. Falls back to the older layout of one CSV per
    policy, each with its own baseline arm.
    """
    combined = os.path.join(results_dir, "all_policies_raw.csv")
    rows = []

    if os.path.exists(combined):
        df = pd.read_csv(combined)
        baseline = df[df["arm"] == "baseline"].set_index("seed")
        for arm, policy_label in POLICY_ARMS.items():
            treated = df[df["arm"] == arm].set_index("seed")
            if treated.empty:
                print(f"  WARNING: no rows for arm {arm} — skipping")
                continue
            seeds = baseline.index.intersection(treated.index)
            for metric in METRICS:
                if metric not in baseline.columns:
                    continue
                row = _summarise(
                    policy_label,
                    metric,
                    baseline.loc[seeds, metric].tolist(),
                    treated.loc[seeds, metric].tolist(),
                )
                if row:
                    rows.append(row)
        return pd.DataFrame(rows)

    for policy_label, filename in POLICY_FILES.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {policy_label}")
            continue
        df = pd.read_csv(path)
        baseline = df[df["arm"] == "baseline"].set_index("seed")
        policy = df[df["arm"] == "policy"].set_index("seed")
        seeds = baseline.index.intersection(policy.index)
        for metric in METRICS:
            if metric not in baseline.columns:
                continue
            row = _summarise(
                policy_label,
                metric,
                baseline.loc[seeds, metric].tolist(),
                policy.loc[seeds, metric].tolist(),
            )
            if row:
                rows.append(row)

    return pd.DataFrame(rows)


# ── Figure 4 — Bar chart ─────────────────────────────────────────────────────

def make_figure4(summary: pd.DataFrame, output_dir: str):
    """Horizontal bar chart of homeownership rate diff per policy."""
    hr = summary[summary["metric"] == "homeownership_rate"].copy()
    hr = hr.sort_values("mean_diff", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [POLICY_COLORS.get(p, "#888888") for p in hr["policy"]]
    y_pos  = np.arange(len(hr))

    # Bars
    bars = ax.barh(y_pos, hr["mean_diff"], color=colors, alpha=0.85,
                   height=0.6, zorder=3)

    # Error bars (95% CI)
    xerr_lower = hr["mean_diff"] - hr["ci_lower"]
    xerr_upper = hr["ci_upper"] - hr["mean_diff"]
    ax.errorbar(
        hr["mean_diff"], y_pos,
        xerr=[xerr_lower, xerr_upper],
        fmt="none", color="#333333",
        capsize=4, linewidth=1.2, zorder=4
    )

    # Zero line
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=2)

    # Value labels, placed clear of the error bar rather than at the bar end,
    # and scaled to the data range so they do not collide as effect sizes change
    pad = 0.02 * (hr["ci_upper"].max() - hr["ci_lower"].min())
    for i, row in hr.iterrows():
        val = row["mean_diff"]
        anchor = row["ci_upper"] if val >= 0 else row["ci_lower"]
        ax.text(
            anchor + (pad if val >= 0 else -pad),
            i,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5,
            color="#222222",
        )

    # Axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hr["policy"], fontsize=10)
    ax.set_xlabel(
        "Mean paired change in homeownership rate vs. baseline (95% CI)",
        fontsize=10,
    )
    ax.set_title("Policy Effects on the Homeownership Rate",
                 fontsize=11, fontweight="bold", pad=12)

    # Legend
    blue_patch  = mpatches.Patch(color="#2E75B6", alpha=0.85, label="Access restriction")
    orange_patch = mpatches.Patch(color="#C55A11", alpha=0.85, label="Financial penalty")
    ax.legend(
        handles=[blue_patch, orange_patch],
        fontsize=9,
        loc="lower right",
        framealpha=0.9,
    )

    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure4_policy_comparison.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    out_path_png = os.path.join(output_dir, "figure4_policy_comparison.png")
    plt.savefig(out_path_png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path_png}")
    plt.close()


# ── Figure 5 — Trade-off scatter ─────────────────────────────────────────────

def make_figure5(summary: pd.DataFrame, output_dir: str):
    """Scatter: homeownership rate gain vs rental vacancy change."""
    hr = summary[summary["metric"] == "homeownership_rate"].set_index("policy")["mean_diff"]
    rv = summary[summary["metric"] == "rental_vacancy_rate"].set_index("policy")["mean_diff"]

    # Only plot policies present in both
    policies = hr.index.intersection(rv.index)
    x = hr.loc[policies].values   # homeownership diff
    y = rv.loc[policies].values   # vacancy diff

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [POLICY_COLORS.get(p, "#888888") for p in policies]
    ax.scatter(x, y, s=120, c=colors, alpha=0.9, zorder=5, edgecolors="#333333",
               linewidths=0.6)

    # Label each point — offset to avoid overlap
    offsets = {
        "Waiting Period":       ( 0.0008,  0.0010),
        "Ownership Cap (Soft)": ( 0.0008, -0.0015),
        "Ownership Cap (Hard)": ( 0.0008,  0.0010),
        "Purchase Tax":         ( 0.0008,  0.0010),
        "Vacancy Tax":          ( 0.0008,  0.0010),
        "Portfolio Tax":        ( 0.0008,  0.0010),
    }
    for policy, xi, yi in zip(policies, x, y):
        dx, dy = offsets.get(policy, (0.0008, 0.0010))
        ax.annotate(
            policy,
            (xi, yi),
            xytext=(xi + dx, yi + dy),
            fontsize=8.5,
            color="#222222",
        )

    # Quadrant reference lines
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=2)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Legend
    blue_patch   = mpatches.Patch(color="#2E75B6", alpha=0.85, label="Access restriction")
    orange_patch = mpatches.Patch(color="#C55A11", alpha=0.85, label="Financial penalty")
    #ax.legend(handles=[blue_patch, orange_patch], fontsize=9,
    #          loc="lower left", framealpha=0.9)

    ax.set_xlabel("Change in First-Time Homeownership Rate vs. Baseline",
                  fontsize=10)
    ax.set_ylabel("Change in Rental Vacancy Rate vs. Baseline",
                  fontsize=10)
    ax.set_title("Figure 4: Policy Trade-offs: Homeownership vs. Rental Supply",
                 fontsize=11, fontweight="bold", pad=12)

    ax.grid(linewidth=0.3, alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure5_tradeoff_scatter.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    out_path_png = os.path.join(output_dir, "figure5_tradeoff_scatter.png")
    plt.savefig(out_path_png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path_png}")
    plt.close()


# ── Figure 4b — Rental vacancy bar chart ────────────────────────────────────

def make_figure4b(summary: pd.DataFrame, output_dir: str):
    """Horizontal bar chart of rental vacancy rate diff per policy.
    Same format as Figure 4 so the two sit side by side in the paper."""
    rv = summary[summary["metric"] == "rental_vacancy_rate"].copy()
    rv = rv.sort_values("mean_diff", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [POLICY_COLORS.get(p, "#888888") for p in rv["policy"]]
    y_pos  = np.arange(len(rv))

    # Bars — use slightly different alpha to distinguish from Figure 4
    ax.barh(y_pos, rv["mean_diff"], color=colors, alpha=0.75,
            height=0.6, zorder=3)

    # Error bars (95% CI)
    xerr_lower = rv["mean_diff"] - rv["ci_lower"]
    xerr_upper = rv["ci_upper"] - rv["mean_diff"]
    ax.errorbar(
        rv["mean_diff"], y_pos,
        xerr=[xerr_lower, xerr_upper],
        fmt="none", color="#333333",
        capsize=4, linewidth=1.2, zorder=4
    )

    # Zero line
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=2)

    # Value labels
    for i, row in rv.iterrows():
        x_offset = 0.0003 if row["mean_diff"] >= 0 else -0.0003
        ha = "left" if row["mean_diff"] >= 0 else "right"
        ax.text(
            row["mean_diff"] + x_offset, i,
            f"{row['mean_diff']:+.4f}",
            va="center", ha=ha, fontsize=8.5, color="#222222"
        )
    # Axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rv["policy"], fontsize=10)
    ax.set_xlabel(
        "Mean Change in Rental Vacancy Rate vs. Baseline",
        fontsize=10
    )
    ax.set_title(
        "Figure 3: Policy Effects on Rental Vacancy Rate",
        fontsize=11, fontweight="bold", pad=12
    )

    blue_patch   = mpatches.Patch(color="#2E75B6", alpha=0.75, label="Access restriction")
    orange_patch = mpatches.Patch(color="#C55A11", alpha=0.75, label="Financial penalty")
    

    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure4b_rental_vacancy.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    out_path_png = os.path.join(output_dir, "figure4b_rental_vacancy.png")
    plt.savefig(out_path_png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path_png}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def make_figure_ftb_share(summary: pd.DataFrame, output_dir: str):
    """Policy effect on the first-time-buyer share of purchases.

    This is the flow the policies act on directly -- who wins the bidding on a
    given listing -- rather than the homeownership stock, which only moves as
    fast as that flow accumulates. It is also the better-powered outcome: it is
    close to white noise month to month, so averaging the measurement window
    buys a much larger reduction in standard error than it does for the stocks.
    """
    data = summary[summary["metric"] == "ftb_purchase_share"].copy()
    if data.empty:
        print("  no ftb_purchase_share rows — skipping")
        return
    data = data.sort_values("mean_diff").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [POLICY_COLORS.get(p, "#888888") for p in data["policy"]]
    y_pos = np.arange(len(data))

    ax.barh(y_pos, data["mean_diff"], color=colors, alpha=0.85, height=0.6, zorder=3)
    ax.errorbar(
        data["mean_diff"],
        y_pos,
        xerr=[
            data["mean_diff"] - data["ci_lower"],
            data["ci_upper"] - data["mean_diff"],
        ],
        fmt="none",
        color="#333333",
        capsize=4,
        linewidth=1.2,
        zorder=4,
    )
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=2)

    pad = 0.02 * (data["ci_upper"].max() - data["ci_lower"].min())
    for i, row in data.iterrows():
        value = row["mean_diff"]
        anchor = row["ci_upper"] if value >= 0 else row["ci_lower"]
        ax.text(
            anchor + (pad if value >= 0 else -pad),
            i,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color="#222222",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["policy"], fontsize=10)
    ax.set_xlabel(
        "Mean change in first-time-buyer share of purchases vs. baseline",
        fontsize=10,
    )
    ax.set_title(
        "Policy Effects on the First-Time-Buyer Share of Purchases",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    blue = mpatches.Patch(color="#2E75B6", alpha=0.85, label="Access restriction")
    orange = mpatches.Patch(color="#C55A11", alpha=0.85, label="Financial penalty")
    ax.legend(handles=[blue, orange], fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.12)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"figure_ftb_purchase_share.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Directory containing policy comparison CSVs")
    parser.add_argument("--output", required=True,
                        help="Directory to write figures into")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading policy comparison results...")
    summary = load_policy_results(args.results)

    if summary.empty:
        print("No results found — check that your policy comparison CSVs are in the results directory.")
        return

    print(f"Loaded results for: {summary['policy'].unique().tolist()}")
    print()

    print("Generating Figure 4...")
    make_figure4(summary, args.output)

    print("Generating Figure 4b (rental vacancy)...")
    make_figure4b(summary, args.output)

    print("Generating Figure 5...")
    make_figure5(summary, args.output)

    print("Generating first-time-buyer purchase share figure...")
    make_figure_ftb_share(summary, args.output)

    print("\nDone. Check your output directory for:")
    print("  figure4_policy_comparison.pdf/png")
    print("  figure4b_rental_vacancy.pdf/png")
    print("  figure5_tradeoff_scatter.pdf/png")
    print("  figure_ftb_purchase_share.pdf/png")


if __name__ == "__main__":
    main()