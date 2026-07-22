"""Run model for N months and plot all core indicators tracked by the
model's DataCollector."""

import matplotlib.pyplot as plt

from housing_abm.model import AtlantaHousingModel

N_MONTHS = 500
N_HOUSEHOLDS = 300
SEED = 42

model = AtlantaHousingModel(n_households=N_HOUSEHOLDS, seed=SEED)
for _ in range(N_MONTHS):
    model.step()

df = model.datacollector.get_model_vars_dataframe()
df.index.name = "Month"

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
fig.suptitle(
    f"Baseline run — n={N_HOUSEHOLDS} households, seed={SEED}, {N_MONTHS} months",
    fontsize=14,
)

# 1) household population by housing status
ax = axes[0, 0]
ax.plot(df.index, df["n_renting"], label="Renting", color="#2b6cb0")
ax.plot(df.index, df["n_owning"], label="Owning", color="#2f855a")
ax.plot(df.index, df["n_social_housing"], label="Social housing", color="#c05621")
ax.plot(
    df.index,
    df["n_renting"] + df["n_owning"] + df["n_social_housing"],
    label="Total households",
    color="#4a5568",
    linestyle="--",
)
ax.set_ylabel("# households")
ax.set_title("Household population by status")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 2) buyer/investor agent type counts
ax = axes[0, 1]
ax.plot(df.index, df["n_first_time_buyers"], label="First-time buyers", color="#2b6cb0")
ax.plot(df.index, df["n_repeat_buyers"], label="Repeat buyers", color="#2f855a")
ax.plot(df.index, df["n_small_landlords"], label="Small landlords", color="#c05621")
ax.plot(
    df.index,
    df["n_institutional_investors"],
    label="Institutional investors",
    color="#805ad5",
)
ax.plot(df.index, df["n_renters"], label="Renters", color="#FC94A6")

ax.set_ylabel("# agents")
ax.set_title("Agent population by type")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 3) investor-owned rental units
ax = axes[1, 0]
ax.plot(df.index, df["n_investor_owned_units"], color="#805ad5")
ax.set_ylabel("# units")
ax.set_title("Investor-owned units (landlords + institutions)")
ax.grid(alpha=0.3)

# 4) rental vacancy rate
ax = axes[1, 1]
ax.plot(df.index, df["rental_vacancy_rate"] * 100, color="#c05621")
ax.set_ylabel("Rate (%)")
ax.set_title("Rental vacancy rate")
ax.grid(alpha=0.3)

# 5) homeownership rate
ax = axes[2, 0]
ax.plot(df.index, df["homeownership_rate"] * 100, color="#2f855a")
ax.set_ylabel("Rate (%)")
ax.set_title("Homeownership rate")
ax.grid(alpha=0.3)

# 6) mean household bank balance
ax = axes[2, 1]
ax.plot(df.index, df["mean_bank_balance"], color="#2b6cb0")
ax.set_ylabel("$")
ax.set_title("Mean household bank balance")
ax.grid(alpha=0.3)

# 7) bankruptcy injections
ax = axes[3, 0]
ax.plot(df.index, df["bankruptcy_injections_this_month"], color="#c53030")
ax.set_ylabel("# injections")
ax.set_xlabel("Month")
ax.set_title("Bankruptcy injections per month")
ax.grid(alpha=0.3)

# unused panel - keep axes grid even, hide it
axes[3, 1].axis("off")
axes[3, 0].set_xlabel("Month")
axes[2, 1].set_xlabel("Month")

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = "baseline_run.png"
fig.savefig(out_path, dpi=150)
print(f"Results saved to {out_path}")
