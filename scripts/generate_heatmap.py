import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sens = pd.read_csv("results/sensitivity_beta_institutional.csv")

# Drop geographic and portfolio tax, focus on homeownership
hr = sens[
    (sens["metric"] == "homeownership_rate") &
    (sens["policy"] != "baseline") &
    (~sens["policy"].isin(["geographic", "portfolio_tax"]))
]

pivot = (
    hr.groupby(["policy", "param_value"])["diff_from_baseline"]
    .mean()
    .unstack("param_value")
)

# Clean up policy labels
pivot.index = [p.replace("_", " ").title() for p in pivot.index]

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(
    pivot,
    annot=True, fmt=".3f",
    cmap="RdYlGn", center=0,
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Mean diff from baseline"}
)
ax.set_title(
    label="Figure 6: Homeownership Rate Gain by Policy and β_institutional",
    fontsize=10, fontweight="bold"  
)
ax.set_xlabel("β_institutional value", fontsize=10)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("figures/figure6_sensitivity_heatmap.pdf", bbox_inches="tight", dpi=150)
plt.savefig("figures/figure6_sensitivity_heatmap.png", bbox_inches="tight", dpi=150)