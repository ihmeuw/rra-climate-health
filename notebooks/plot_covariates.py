"""
Plot scattered changes in main model covariates over time for a select number of
countries:
- Namibia
- Angola
- Rwanda
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from db_queries import get_location_metadata

# Variables ####################################################################
DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_09_02.01/data.parquet"
PLOT_OUTPUT = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/"

OUTPUT_PATH = (
    Path(PLOT_OUTPUT) / "covariates_by_country.pdf"
)  # TODO: set final destination
locs = get_location_metadata(location_set_id=39, release_id=16)
countries = ["Namibia", "Angola", "Rwanda"]
vars_of_interest = ["consumption_pd_cumul", "days_over_30C_monthly_cumul"]
VAR_LABELS = {
    "consumption_pd_cumul": "Consumption per day",
    "days_over_30C_monthly_cumul": "Days over 30C (monthly)",
}
POINT_COLOR = "#9dc0d8"
BOX_COLOR = "#1f4257"


# Load data and format #########################################################
df = pd.read_parquet(DATA_PATH)
df = df.merge(locs[["ihme_loc_id", "location_name"]], on="ihme_loc_id", how="left")
df = df[df["location_name"].isin(countries)]
df["ihme_loc_id"].unique()

# select max age_month per indv_id
df = df.loc[df.groupby("indv_id")["age_month"].idxmax()]

# Plot variables of interest ###################################################
# Small multiples: one row per country, one column per variable. Every panel gets
# its own x and y scale, and a per-birth-year box plot is drawn over the points to
# summarize the distribution.
fig, axes = plt.subplots(
    nrows=len(countries),
    ncols=len(vars_of_interest),
    figsize=(11, 11),
)

for row, country in enumerate(countries):
    subset = df[df["location_name"] == country]
    for col, var in enumerate(vars_of_interest):
        ax = axes[row, col]
        ax.scatter(
            subset["birth_year"],
            subset[var],
            color=POINT_COLOR,
            s=8,
            alpha=0.3,
            linewidths=0,
            zorder=1,
        )

        # One box per birth year, positioned on the numeric x axis.
        grouped = subset.dropna(subset=[var]).groupby("birth_year")[var]
        positions = [year for year, _ in grouped]
        distributions = [values.to_numpy() for _, values in grouped]
        if distributions:
            ax.boxplot(
                distributions,
                positions=positions,
                widths=0.6,
                manage_ticks=False,
                showfliers=False,
                zorder=2,
                boxprops={"color": BOX_COLOR, "linewidth": 0.8},
                whiskerprops={"color": BOX_COLOR, "linewidth": 0.8},
                capprops={"color": BOX_COLOR, "linewidth": 0.8},
                medianprops={"color": BOX_COLOR, "linewidth": 1.4},
            )

        ax.grid(True, color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{country} — {VAR_LABELS.get(var, var)}", fontsize=10)
        ax.set_xlabel("Birth year")
        ax.set_ylabel(VAR_LABELS.get(var, var))

fig.suptitle("Model covariates by birth year", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
