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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────

POLICY_FILES = {
    "Waiting Period":       "waiting_period.csv",
    "Ownership Cap (Soft)": "ownership_cap_soft.csv",
    "Ownership Cap (Hard)": "ownership_cap_hard.csv",
    "Purchase Tax":         "purchase_tax.csv",
    "Vacancy Tax":          "vacancy_tax.csv",
    "Portfolio Tax":        "portfolio_tax.csv",
}

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
    "homeownership_rate":    "First-Time Homeownership Rate",
    "rental_vacancy_rate":   "Rental Vacancy Rate",
    "annual_appreciation_g": "Annual Price Appreciation",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_policy_results(results_dir: str) -> pd.DataFrame:
    """Load all policy CSVs and compute paired differences per seed."""
    rows = []
    for policy_label, filename in POLICY_FILES.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {policy_label}")
            continue

        df = pd.read_csv(path)
        baseline = df[df["arm"] == "baseline"].set_index("seed")
        policy   = df[df["arm"] == "policy"].set_index("seed")

        common_seeds = baseline.index.intersection(policy.index)
        for metric in METRICS:
            if metric not in baseline.columns:
                continue
            diffs = policy.loc[common_seeds, metric].astype(float).values \
                  - baseline.loc[common_seeds, metric].astype(float).values
            n = len(diffs)
            mean_diff = diffs.mean()
            se = diffs.std() / np.sqrt(n)
            rows.append({
                "policy":    policy_label,
                "metric":    metric,
                "mean_diff": mean_diff,
                "ci_lower":  mean_diff - 1.96 * se,
                "ci_upper":  mean_diff + 1.96 * se,
                "n_seeds":   n,
                "pct_same_dir": max((diffs > 0).mean(), (diffs < 0).mean()),
            })

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

    # Value labels on each bar
    for i, row in hr.iterrows():
        val = row['mean_diff']
        x_offset = 0.001 if val >= 0 else -0.001
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + x_offset, i,
            f"{val:+.3f}",
            va="center", ha=ha, fontsize=8.5, color="#222222"
    )

    # Axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hr["policy"], fontsize=10)
    ax.set_xlabel("Mean Change in Homeownership Rate vs. Baseline",
                  fontsize=10)
    ax.set_title("Figure 2: Policy Effects on First-Time Homeownership Rate",
                 fontsize=11, fontweight="bold", pad=12)

    # Legend
    blue_patch  = mpatches.Patch(color="#2E75B6", alpha=0.85, label="Access restriction")
    orange_patch = mpatches.Patch(color="#C55A11", alpha=0.85, label="Financial penalty")
    #ax.legend(handles=[blue_patch, orange_patch], fontsize=9,
    #         loc="lower right", framealpha=0.9)

    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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

    print("\nDone. Check your output directory for:")
    print("  figure4_policy_comparison.pdf/png")
    print("  figure4b_rental_vacancy.pdf/png")
    print("  figure5_tradeoff_scatter.pdf/png")


if __name__ == "__main__":
    main()