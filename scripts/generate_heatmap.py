"""Sensitivity heatmap: policy effect by swept parameter value.

Reads the CSV written by run_sensitivity.py, which now carries the full paired
summary per cell (effect, interval, p-value), so the heatmap can mark which
cells are actually distinguishable from zero rather than colouring point
estimates that may be noise. Cells whose 95% interval excludes zero are
outlined; the rest are drawn flat.

Built on matplotlib alone so it does not depend on seaborn being installed,
and it honours --input/--output rather than hard-coding paths as the previous
version did while the README passed them.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

LABELS = {
    "waiting_period": "Waiting period",
    "ownership_cap_soft": "Ownership cap (soft)",
    "ownership_cap_hard": "Ownership cap (hard)",
    "purchase_tax": "Purchase tax",
    "vacancy_tax": "Vacancy tax",
    "portfolio_tax": "Portfolio tax",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/sensitivity_beta_institutional.csv")
    parser.add_argument("--output", default="figures/figure6_sensitivity_heatmap.png")
    parser.add_argument("--outcome", default="homeownership_rate")
    args = parser.parse_args()

    sens = pd.read_csv(args.input)
    data = sens[sens["outcome"] == args.outcome]
    if data.empty:
        raise SystemExit(f"no rows for outcome {args.outcome} in {args.input}")

    param = data["parameter"].iloc[0]
    effects = data.pivot_table(index="policy", columns="value", values="mean_diff")
    lo = data.pivot_table(index="policy", columns="value", values="ci_lo")
    hi = data.pivot_table(index="policy", columns="value", values="ci_hi")

    order = [p for p in LABELS if p in effects.index]
    effects, lo, hi = effects.loc[order], lo.loc[order], hi.loc[order]

    limit = float(np.nanmax(np.abs(effects.values)))
    fig, ax = plt.subplots(figsize=(1.5 * effects.shape[1] + 4, 0.62 * len(order) + 2.4))
    image = ax.imshow(
        effects.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
    )

    for row in range(effects.shape[0]):
        for col in range(effects.shape[1]):
            value = effects.values[row, col]
            if np.isnan(value):
                continue
            significant = not (lo.values[row, col] <= 0 <= hi.values[row, col])
            # white text on the saturated ends, dark in the pale middle
            shade = "white" if abs(value) > 0.6 * limit else "#16202B"
            ax.text(
                col,
                row,
                f"{value:+.3f}" + ("\n*" if significant else ""),
                ha="center",
                va="center",
                fontsize=8.5,
                color=shade,
                fontweight="bold" if significant else "normal",
            )
            if significant:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5), 1, 1,
                        fill=False, edgecolor="#16202B", linewidth=1.6,
                    )
                )

    ax.set_xticks(range(effects.shape[1]))
    ax.set_xticklabels([f"{v:g}" for v in effects.columns])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[p] for p in order])
    ax.set_xlabel(f"{param} value", fontsize=10)
    ax.set_title(
        f"Paired effect on {args.outcome.replace('_', ' ')} across {param}\n"
        "outlined cells: 95% interval excludes zero",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xticks(np.arange(-0.5, effects.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    bar = fig.colorbar(image, ax=ax)
    bar.set_label("Mean paired difference from baseline", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for path in (args.output, os.path.splitext(args.output)[0] + ".pdf"):
        plt.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
