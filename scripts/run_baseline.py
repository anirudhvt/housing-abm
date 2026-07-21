"""Run model for N months and plot core indicators"""

import matplotlib.pyplot as plt

from housing_abm.model import AtlantaHousingModel

N_MONTHS = 100
N_HOUSEHOLDS = 300
SEED = 42

model = AtlantaHousingModel(n_households=N_HOUSEHOLDS, seed=SEED)
for _ in range(N_MONTHS):
    model.step()

df = model.datacollector.get_model_vars_dataframe()
df.index.name = "Month"

fig, axes = plt.subplots(4, 1, figsize=(8,6), sharex = True)

axes[0].plot(df.index, df["n_renting"], color="#2b6cb0")
axes[0].set_ylabel("# households renting")
axes[0].set_title(f"Renter population over time (n={N_HOUSEHOLDS} households, seed={SEED})")
axes[0].grid(alpha=0.3)
 
axes[1].plot(df.index, df["rental_vacancy_rate"] * 100, color="#c05621")
axes[1].set_ylabel("Rental vacancy rate (%)")
axes[1].set_xlabel("Month")
axes[1].set_title("Rental market vacancy rate")
axes[1].grid(alpha=0.3)

axes[2].plot(df.index, df["n_owning"], color="#2b6cb0")
axes[2].set_ylabel("# households owning")
axes[2].set_title(f"Owner population over time (n={N_HOUSEHOLDS} households, seed={SEED})")
axes[2].grid(alpha=0.3)

axes[3].plot(df.index, df["n_social_housing"], color="#2b6cb0")
axes[3].grid(alpha=0.3)




fig.tight_layout()
out_path = "baseline_run.png"
fig.savefig(out_path, dpi = 150)
print(f"Results saved to {out_path}")