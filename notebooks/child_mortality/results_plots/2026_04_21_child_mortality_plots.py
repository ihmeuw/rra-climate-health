"""
Plot code corresponding to child mortality run:

date launched: 4/8/2026
spec:
model <- scam(child_mortality ~
                age_1_m+
                age_3_m+
                age_6_m+
                age_12_m+
                age_24_m+
                age_36_m+
                age_48_m+
                age_60_m+
                sex_id +
                s(consumption_pd, bs="mpd") +
                s(days_over_30C, bs="mpi")+
                total_precipitation+
                birth_year+
                s(ihme_loc_id, bs = "re"),
              family = binomial(link = "logit"),
              data = df_model)
call:
sbatch -J cm_full_no_re_splines --mem=800G -c 6 -A proj_integrated_analytics -t 5-24 -p long.q -o /ihme/temp/slurmoutput/elyeb/output/%x.o%j -e /ihme/temp/slurmoutput/elyeb/errors/%x.e%j /ihme/singularity-images/rstudio/shells/execR.sh -s /ihme/homes/elyeb/repos/rra-climate-health/notebooks/child_mortality/child_mortality_logistic_splines_full_no_re.R
- 22332528


"""

import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from lifelines import CoxPHFitter  # for Cox survival models
from pymer4.models import Lmer
import os
import geopandas as gpd
from rra_climate_health.data import ClimateMalnutritionData, DEFAULT_ROOT
from pathlib import Path


DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/data_within_bin.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_04_13.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2026_04_21.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

## READ IN DATA ################################################################

# Raw data
df_raw = pd.read_parquet(DATA_PATH)
df_raw = df_raw[df_raw["int_birth_year_diff_months"] <= 120]


# Modeled data
df_model = pd.read_parquet(
    RESULTS_PATH
    + "cm_splines_full_no_re_custom_knots_v2_input_predictions_both_re_fe.parquet"
)
# df_model["age_month_old"] = df_model["age_month"]
# df_model = df_model.rename(columns={"days_over_30C_monthly": "days_over_30C"})


# df_model_me = pd.read_parquet(
#     RESULTS_PATH
#     + "cm_splines_full_no_re_custom_knots_v2_input_predictions_both_re_fe.parquet"
# )
df_model_me = pd.read_csv(
    RESULTS_PATH + "cm_splines_full_no_re_custom_knots_v2_predictions_cumulative_me.csv"
)


df_model_fe = pd.read_csv(
    RESULTS_PATH
    + "cm_splines_full_no_re_custom_knots_v2_predictions_with_both_splines_ranged.csv"
)

# Preliminary formatting
df_model = df_model.rename(columns={"days_over_30C_monthly": "days_over_30C"})
df_model.dropna(subset=["consumption_pd", "days_over_30C"], inplace=True)
df_model["age_month_old"] = df_model["age_month"]
df_model["age_month"] = 0
for v in [
    "age_1_m",
    "age_3_m",
    "age_6_m",
    "age_12_m",
    "age_24_m",
    "age_36_m",
    "age_48_m",
    "age_60_m",
]:
    months = int(v.split("_")[1])
    df_model.loc[df_model[v] == 1, "age_month"] = months


df_model_me = df_model_me.rename(columns={"days_over_30C_monthly": "days_over_30C"})

df_model_fe = df_model_fe.rename(columns={"days_over_30C_monthly": "days_over_30C"})
df_model_fe["age_month"] = 60

# Get max age obs for me
df_max_age = df_model.copy()
df_max_age = (
    df_model.sort_values("age_month")
    .groupby("indv_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

# EXPLORE DATA #################################################################

# Scatter days_over_30C and days_over_30C_monthly
df_raw_scatter = df_raw[["days_over_30C", "days_over_30C_monthly"]].drop_duplicates()
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df_raw,
    x="days_over_30C",
    y="days_over_30C_monthly",
    alpha=0.1,
)
plt.xlabel("Days over 30°C (avg annual weighted by time bin)", fontsize=13)
plt.ylabel("Days over 30°C (avg monthly weighted by time bin)", fontsize=13)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, f"days_over_30_annual_vs_monthly_scatter.png"),
    bbox_inches="tight",
)
# Not very informative.

# Try grouping by location
df_raw_scatter_loc = (
    df_raw.groupby("ihme_loc_id")[["days_over_30C", "days_over_30C_monthly"]]
    .mean()
    .reset_index()
)
print(len(df_raw_scatter_loc))

plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df_raw_scatter_loc,
    x="days_over_30C",
    y="days_over_30C_monthly",
    alpha=0.7,
)
plt.xlabel("Days over 30°C (avg annual weighted by time bin)", fontsize=13)
plt.ylabel("Days over 30°C (avg monthly weighted by time bin)", fontsize=13)
plt.savefig(
    os.path.join(PLOT_PATH, f"days_over_30_annual_vs_monthly_scatter_loc_grouped.png"),
    bbox_inches="tight",
)

# perfectly correlated

# Plot avg mortality by location against days_over_30C and days_over_30C_monthly
# in two side-by-side scatter plots
df_raw_scatter_loc_mort = (
    df_raw.groupby("ihme_loc_id")[
        ["days_over_30C", "days_over_30C_monthly", "child_mortality"]
    ]
    .mean()
    .reset_index()
)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
sns.scatterplot(
    data=df_raw_scatter_loc_mort, x="days_over_30C", y="child_mortality", ax=axes[0]
)
sns.scatterplot(
    data=df_raw_scatter_loc_mort,
    x="days_over_30C_monthly",
    y="child_mortality",
    ax=axes[1],
)
plt.show()
# Both look the same

# Try additionally grouping by int_year and int_month
df_raw_scatter_loc_time = (
    (
        df_raw.groupby(["ihme_loc_id", "int_year", "int_month"])[
            ["days_over_30C", "days_over_30C_monthly", "child_mortality"]
        ]
    )
    .mean()
    .reset_index()
)

plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df_raw_scatter_loc_time,
    x="days_over_30C",
    y="days_over_30C_monthly",
    alpha=0.7,
)
plt.xlabel("Days over 30°C (avg annual weighted by time bin)", fontsize=13)
plt.ylabel("Days over 30°C (avg monthly weighted by time bin)", fontsize=13)
plt.savefig(
    os.path.join(
        PLOT_PATH, f"days_over_30_annual_vs_monthly_scatter_loc_time_grouped.png"
    ),
    bbox_inches="tight",
)

# There are many more points where monthly is low but annual is high
# than vice versa, which makes sense given that the monthly variable is more likely to be 0

# What happens when I group by invd ID and age_month?
df_raw_scatter_indv = (
    df_raw.groupby(["indv_id", "age_month"])[
        ["days_over_30C", "days_over_30C_monthly", "child_mortality"]
    ]
    .mean()
    .reset_index()
)
print(len(df_raw_scatter_indv))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
sns.scatterplot(
    data=df_raw_scatter_indv, x="days_over_30C", y="child_mortality", ax=axes[0]
)
sns.scatterplot(
    data=df_raw_scatter_indv,
    x="days_over_30C_monthly",
    y="child_mortality",
    ax=axes[1],
)
plt.show()

# not informative

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
sns.scatterplot(
    data=df_raw_scatter_loc_time,
    x="days_over_30C",
    y="child_mortality",
    ax=axes[0],
    alpha=0.1,
)
sns.scatterplot(
    data=df_raw_scatter_loc_time,
    x="days_over_30C_monthly",
    y="child_mortality",
    ax=axes[1],
    alpha=0.1,
)
plt.show()
# Some gap in days_over_30C data concentration

# Make two histograms for data density for each
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_raw.days_over_30C, bins=30)
plt.xlabel("Days over 30°C (avg annual weighted by time bin)", fontsize=13)
plt.subplot(1, 2, 2)
sns.histplot(df_raw.days_over_30C_monthly, bins=30)
plt.xlabel("Days over 30°C (avg monthly weighted by time bin)", fontsize=13)
plt.tight_layout()
# Not informative

# Group by days_over_30C and days_over_30C_monthly and plot avg mortality in each bin
# create 30 bins for each variable
df_raw["days_over_30C_bin"] = pd.cut(df_raw["days_over_30C"], bins=30)
df_raw["days_over_30C_monthly_bin"] = pd.cut(df_raw["days_over_30C_monthly"], bins=30)

df_raw_annual_grouped = (
    df_raw.groupby("days_over_30C_bin")["child_mortality"].mean().reset_index()
)
df_raw_monthly_grouped = (
    df_raw.groupby("days_over_30C_monthly_bin")["child_mortality"].mean().reset_index()
)
df_raw_annual_grouped["days_over_30C_bin_numeric"] = df_raw_annual_grouped[
    "days_over_30C_bin"
].cat.codes
df_raw_monthly_grouped["days_over_30C_monthly_bin_numeric"] = df_raw_monthly_grouped[
    "days_over_30C_monthly_bin"
].cat.codes

df_raw_annual_grouped["days_over_30C_bin_str"] = df_raw_annual_grouped[
    "days_over_30C_bin"
].astype(str)
df_raw_monthly_grouped["days_over_30C_monthly_bin_str"] = df_raw_monthly_grouped[
    "days_over_30C_monthly_bin"
].astype(str)


# plot side-by-side line plots of avg mortality by days_over_30C and days_over_30C_monthly
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 18), constrained_layout=True)

# Plot for annual data
sns.lineplot(
    data=df_raw_annual_grouped,
    x="days_over_30C_bin_str",
    y="child_mortality",
    ax=axes[0],
)
axes[0].set_xlabel("Days over 30°C (avg annual weighted by time bin)", fontsize=13)
axes[0].tick_params(axis="x", labelrotation=90, labelsize=8)  # Rotate and set font size

# Plot for monthly data
sns.lineplot(
    data=df_raw_monthly_grouped,
    x="days_over_30C_monthly_bin_str",
    y="child_mortality",
    ax=axes[1],
)
axes[1].set_xlabel("Days over 30°C (avg monthly weighted by time bin)", fontsize=13)
axes[1].tick_params(axis="x", labelrotation=90, labelsize=8)  # Rotate and set font size
# plt.show()
plt.savefig(
    os.path.join(PLOT_PATH, f"avg_mortality_per_do30_bin.png"),
    bbox_inches="tight",
)


# Use the actual bins in custom_x_bins instead of equal-width bins
spline_knot_bins = [0, 1.5, 5.25, 9.3, 15.5, 31]
 
df_raw["days_over_30C_bin"] = pd.cut(df_raw["days_over_30C"], bins=spline_knot_bins, include_lowest=True, right=False)
df_raw["days_over_30C_monthly_bin"] = pd.cut(df_raw["days_over_30C_monthly"], bins=spline_knot_bins, include_lowest=True, right=False)

df_raw_annual_grouped = (
    df_raw.groupby("days_over_30C_bin")["child_mortality"].mean().reset_index()
)
df_raw_monthly_grouped = (
    df_raw.groupby("days_over_30C_monthly_bin")["child_mortality"].mean().reset_index()
)

df_raw_annual_grouped["days_over_30C_bin_str"] = df_raw_annual_grouped[
    "days_over_30C_bin"
].astype(str)
df_raw_monthly_grouped["days_over_30C_monthly_bin_str"] = df_raw_monthly_grouped[
    "days_over_30C_monthly_bin"
].astype(str)


# plot avg mortality by binned days_over_30C_monthly
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

# Plot for monthly data
sns.lineplot(
    data=df_raw_monthly_grouped,
    x="days_over_30C_monthly_bin_str",
    y="child_mortality",
    ax=ax,
)
ax.set_xlabel("Days over 30°C (avg monthly weighted by time bin)", fontsize=13)
ax.tick_params(axis="x", labelrotation=90, labelsize=8)
plt.savefig(
    os.path.join(PLOT_PATH, f"avg_mortality_per_do30_knot_bin.png"),
    bbox_inches="tight",
)

# Disaggregate by age 
df_raw["age_month_old"] = df_raw["age_month"]
df_raw["age_month"] = 0
for v in [
    "age_1_m",
    "age_3_m",
    "age_6_m",
    "age_12_m",
    "age_24_m",
    "age_36_m",
    "age_48_m",
    "age_60_m",
]:
    months = int(v.split("_")[1])
    df_raw.loc[df_raw[v] == 1, "age_month"] = months


ages = [1, 3, 6, 12, 24, 36, 48, 60]
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 18), constrained_layout=True)

for ax, age in zip(axes.flatten(), ages):
    if age == 1:
        prev_age = 0
    else:
        prev_age = ages[ages.index(age) - 1]+1
    df_age = df_raw[df_raw["age_month"] == age]
    df_age_grouped = (
        df_age.groupby("days_over_30C_monthly_bin")["child_mortality"].mean().reset_index()
    )
    df_age_grouped["days_over_30C_monthly_bin_str"] = df_age_grouped[
        "days_over_30C_monthly_bin"
    ].astype(str)

    sns.lineplot(
        data=df_age_grouped,
        x="days_over_30C_monthly_bin_str",
        y="child_mortality",
        ax=ax,
    )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Days over 30°C (avg monthly weighted by time bin)", fontsize=11)
    ax.set_title(f"Avg Mortality - Ages {prev_age} to {age} Months", fontsize=13)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.annotate(
        f"{len(df_age):,} obs",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
    )

pdf_path = os.path.join(PLOT_PATH, "avg_mortality_per_do30_knot_bin_by_age.pdf")
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)
plt.close(fig)


# Custom plot: x axis has age group bins. Two bars in each: one of alive and one 
# for dead, with the bars showing the average days_over_30C_monthly in each group. 
# Also create a box-and-whiskers plot of avg days_over_30C_monthly by age group, disaggregated by alive vs dead.
bins = [0, 1, 3, 6, 12, 24, 36, 48, 60]
labels = ["0-1m", "2-3m", "4-6m", "7-12m", "13-24m", "25-36m", "37-48m", "49-60m"]
df_raw["age_group"] = pd.Categorical(df_raw["age_group"], categories=labels, ordered=True)
df_raw["age_group"] = pd.cut(
    df_raw["age_month"],
    bins=bins,
    labels=labels,
    include_lowest=True  # Include the lowest value in the first bin
)

# df_raw["age_group"] = pd.Categorical(
#     df_raw["age_month"].map(
#         {a: f"{0 if a == 1 else ages[ages.index(a)-1]+1}-{a}m" for a in ages}
#     ),
#     categories=[f"{0 if a == 1 else ages[ages.index(a)-1]+1}-{a}m" for a in ages],
#     ordered=True,
# )
df_raw["status"] = df_raw["child_mortality"].map({0: "Alive", 1: "Dead"})
df_raw["age_group"].value_counts()
df_raw["status"].value_counts()

# Grouped bar chart: avg days_over_30C_monthly by age group & alive/dead
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
sns.barplot(
    data=df_raw,
    x="age_group",
    y="days_over_30C_monthly_within_bin",
    hue="status",
    estimator="mean",
    errorbar="ci",
    ax=ax,
)
ax.set_xlabel("Age Group", fontsize=13)
ax.set_ylabel("Avg Days over 30°C (monthly)", fontsize=13)
ax.set_title("Avg Days over 30°C by Age Group: Alive vs Dead", fontsize=16)
ax.legend(title="Status")
plt.savefig(
    os.path.join(PLOT_PATH, "avg_do30_monthly_by_age_alive_vs_dead_bar.png"),
    bbox_inches="tight",
)
plt.close(fig)

# Box-and-whiskers plot
# fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
# sns.boxplot(
#     data=df_raw,
#     x="age_group",
#     y="days_over_30C_monthly",
#     hue="status",
#     showfliers=False,
#     ax=ax,
# )
# ax.set_xlabel("Age Group", fontsize=13)
# ax.set_ylabel("Days over 30°C (monthly)", fontsize=13)
# ax.set_title("Days over 30°C by Age Group: Alive vs Dead", fontsize=16)
# ax.legend(title="Status")
# plt.savefig(
#     os.path.join(PLOT_PATH, "do30_monthly_by_age_alive_vs_dead_boxplot.png"),
#     bbox_inches="tight",
# )
# plt.close(fig)

### Create version not disaggregated:

fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
sns.barplot(
    data=df_raw,
    x="age_group",
    y="days_over_30C_monthly_within_bin",
    # hue="status",
    estimator="mean",
    errorbar="ci",
    ax=ax,
)
ax.set_xlabel("Age Group", fontsize=13)
ax.set_ylabel("Avg Days over 30°C (monthly)", fontsize=13)
ax.set_title("Avg Days over 30°C by Age Group", fontsize=16)
plt.savefig(
    os.path.join(PLOT_PATH, "avg_do30_monthly_by_age_bar.png"),
    bbox_inches="tight",
)
plt.close(fig)

# Box-and-whiskers plot
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
sns.boxplot(
    data=df_raw,
    x="age_group",
    y="days_over_30C_monthly",
    hue="status",
    showfliers=False,
    ax=ax,
)
ax.set_xlabel("Age Group", fontsize=13)
ax.set_ylabel("Days over 30°C (monthly)", fontsize=13)
ax.set_title("Days over 30°C by Age Group: Alive vs Dead", fontsize=16)
ax.legend(title="Status")
plt.savefig(
    os.path.join(PLOT_PATH, "do30_monthly_by_age_alive_vs_dead_boxplot.png"),
    bbox_inches="tight",
)
plt.close(fig)


# spot-check outliers


[c for c in df_raw.columns]
df_explain = df_raw[
    (df_raw["days_over_30C_monthly"] >= 30) & (df_raw["days_over_30C"] <= 50)
]


## FUNCTIONS ###################################################################
def plot_heat_map_person_time(
    data: pd.DataFrame,
    outfile: str,
    title: str,
    bin_cols: list,
    format: str = ".2f",
    multiply_by: int = 1,
    vmin: float = None,
    vmax: float = None,
    show_colorbar: bool = True,
    x_bins: list = None,
    y_bins: list = None,
):
    heatmap_df = data.copy()
    for col in bin_cols:
        new_bin_col = f"{col}_bin"
        # heatmap_df[new_bin_col] = pd.qcut(
        #     heatmap_df[col], 10, retbins=False, duplicates="drop"
        # )
        if x_bins is not None:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=x_bins, include_lowest=True, right=False
            )
        else:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=10, include_lowest=True
            )
    if y_bins is None:
        heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
            heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
        )
    else:
        heatmap_df["consumption_pd_bin"] = pd.cut(
            heatmap_df.consumption_pd,
            bins=y_bins,
            include_lowest=True,
            right=False,
        )

    for col in bin_cols:
        figsize = (6, 6) if not show_colorbar else (7, 6)
        plt.figure(figsize=figsize)

        heatmap_data = (
            heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[
                "model_predictions"
            ]
            .sum()
            .unstack()
            / heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["age_month"]
            .sum()
            .unstack()
        )
        heatmap_data *= multiply_by

        if vmin is None:
            vmin = heatmap_data.min().min()
        if vmax is None:
            vmax = heatmap_data.max().max()
        colorbin_interval = (vmax - vmin) / 10
        boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
        cmap = plt.get_cmap("RdYlBu_r", len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

        x_tick_vals = heatmap_df.groupby([new_bin_col])[col].min().values.tolist() + [
            int(heatmap_df[col].max())
        ]
        y_tick_vals = heatmap_df.groupby(
            ["consumption_pd_bin"]
        ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]
        x_ticks = range(len(x_tick_vals) + 1)
        y_ticks = range(len(y_tick_vals) + 1)
        x_labs = [f"{x_tick_vals[i]:.1f}" for i in range(len(x_tick_vals))]
        y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]

        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=format,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 10, "weight": "regular"},
            cbar=show_colorbar,
        )
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        # ax.set_xticks(x_ticks)
        n_rows, n_cols = heatmap_data.shape
        ax.set_xticks(np.arange(n_cols + 1))
        ax.set_yticks(np.arange(n_rows + 1))
        ax.set_xticklabels(x_labs, rotation=45, fontsize=10)
        # ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=10)
        ax.set_xlabel("Days over 30°C", fontsize=13)
        ax.set_ylabel("Daily consumption", fontsize=13)
        ax.set_title(title, fontsize=18)

        plt.tight_layout()
        # plt.savefig(
        #     os.path.join(PLOT_PATH, f"{outfile}_{col}.png"), bbox_inches="tight"
        # )
        # plt.close()
        plt.show()


def plot_heat_map_person_time_grid(
    data: pd.DataFrame,
    bin_cols: list,
    ax=None,
    title: str = "",
    format: str = ".2f",
    multiply_by: int = 1,
    vmin: float = None,
    vmax: float = None,
    show_colorbar: bool = True,
    x_bins: list = None,
    y_bins: list = None,
):
    heatmap_df = data.copy()
    for col in bin_cols:
        new_bin_col = f"{col}_bin"

        if x_bins is not None:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=x_bins, include_lowest=True, right=False
            )
        else:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=10, include_lowest=True
            )
    if y_bins is None:
        heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
            heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
        )
    else:
        heatmap_df["consumption_pd_bin"] = pd.cut(
            heatmap_df.consumption_pd,
            bins=y_bins,
            include_lowest=True,
            right=False,
        )

    for col in bin_cols:

        heatmap_data = (
            heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[
                "model_predictions"
            ]
            .sum()
            .unstack()
            / heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["age_month"]
            .sum()
            .unstack()
        )
        heatmap_data *= multiply_by

        if vmin is None:
            vmin = heatmap_data.min().min()
        if vmax is None:
            vmax = heatmap_data.max().max()

        # Create a new figure if no Axes object is provided
        if ax is None:
            figsize = (6, 6) if not show_colorbar else (7, 6)
            fig, ax = plt.subplots(figsize=figsize)

        colorbin_interval = (vmax - vmin) / 10
        boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
        cmap = plt.get_cmap("RdYlBu_r", len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

        x_tick_vals = heatmap_df.groupby([new_bin_col])[col].min().values.tolist() + [
            int(heatmap_df[col].max())
        ]
        y_tick_vals = heatmap_df.groupby(
            ["consumption_pd_bin"]
        ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]
        x_ticks = range(len(x_tick_vals) + 1)
        y_ticks = range(len(y_tick_vals) + 1)
        x_labs = [f"{x_tick_vals[i]:.1f}" for i in range(len(x_tick_vals))]
        y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]

        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=format,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            cbar=show_colorbar,
            ax=ax,
            annot_kws={"size": 8, "weight": "regular"},
        )
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        # ax.set_xticks(x_ticks)
        n_rows, n_cols = heatmap_data.shape
        ax.set_xticks(np.arange(n_cols + 1))
        ax.set_yticks(np.arange(n_rows + 1))
        ax.set_xticklabels(x_labs, rotation=45, fontsize=10)
        # ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=10)
        ax.set_xlabel("Days over 30°C", fontsize=13)
        ax.set_ylabel("Daily consumption", fontsize=13)
        ax.set_title(title, fontsize=18)


def plot_data_pts_per_cell(
    data: pd.DataFrame,
    outfile: str,
    title: str,
    bin_cols: list,
    vmin: float = None,
    vmax: float = None,
    # format: str = ".2f",
    show_colorbar: bool = True,
    x_bins: list = None,
    y_bins: list = None,
):
    """
    data = df_model.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    )
    bin_cols=columns_to_bin
    title=""
    outfile=""
    show_colorbar = True
    x_bins=custom_x_bins
    y_bins=custom_y_bins

    """
    heatmap_df = data.copy()
    for col in bin_cols:
        new_bin_col = f"{col}_bin"

        if x_bins is not None:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=x_bins, include_lowest=True, right=False
            )
        else:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=10, include_lowest=True
            )
    if y_bins is None:
        heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
            heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
        )
    else:
        heatmap_df["consumption_pd_bin"] = pd.cut(
            heatmap_df.consumption_pd,
            bins=y_bins,
            include_lowest=True,
            right=False,
        )
    """
    col = bin_cols[0]
    """
    for col in bin_cols:
        figsize = (6, 6) if not show_colorbar else (7, 6)
        plt.figure(figsize=figsize)

        heatmap_data = (
            heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[
                "model_predictions"
            ]
            .size()
            .unstack()
        )

        # vmin = heatmap_data.min().min()
        # vmax = heatmap_data.max().max()
        colorbin_interval = (vmax - vmin) / 10
        boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
        cmap = plt.get_cmap("RdYlBu_r", len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

        x_tick_vals = heatmap_df.groupby(
            ["days_over_30C_bin"]
        ).days_over_30C.min().astype(int).values.tolist() + [
            int(heatmap_df.days_over_30C.max())
        ]
        y_tick_vals = heatmap_df.groupby(
            ["consumption_pd_bin"]
        ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]

        x_labs = [f"{x_tick_vals[i]}" for i in range(len(x_tick_vals))]
        y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]

        formatted_heatmap_data = heatmap_data.applymap(
            lambda x: f"{int(x):,}" if not pd.isnull(x) else ""
        )

        ax = sns.heatmap(
            heatmap_data,
            # annot=True,
            annot=formatted_heatmap_data,
            fmt="",
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 10, "weight": "regular"},
            cbar=show_colorbar,
        )
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

        n_rows, n_cols = heatmap_data.shape
        ax.set_xticks(np.arange(n_cols + 1))
        ax.set_yticks(np.arange(n_rows + 1))
        ax.set_xticklabels(x_labs, rotation=45, fontsize=10)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=10)
        ax.set_xlabel("Days over 30°C", fontsize=13)
        ax.set_ylabel("Daily consumption", fontsize=13)
        ax.set_title(title, fontsize=18)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"{outfile}_{col}.png"))
        plt.close()
        # plt.show()


# MAKE HEATMAPS ################################################################

# Heat maps of variables
columns_to_bin = [
    # "mean_temperature",
    # "total_precipitation",
    # "relative_humidity",
    # "mean_high_temperature",
    # "mean_low_temperature",
    # "precipitation_days",
    "days_over_30C",
    # "days_over_26C",
]

custom_x_bins = [0, 0.1, 2, 4, 9, 31]

custom_y_bins = [
    0,
    0.784781,
    1.180789,
    1.541445,
    1.950251,
    2.465952,
    3.103463,
    4.003564,
    5.541124,
    9.413681,
    112.879922,
]

# testing with actual knots
# custom_x_bins = [0, 1.5, 5.25, 9.3, 15.5, 31]
# custom_y_bins = [0, 2.0, 5.0, 10.0, 20.0, 30.0, 112.879922]

## MAKE SIDE-BY-SIDE HEATMAPS TOGETHER #########################################


multiply_by_val = 1000

versions = [
    "child_mortality",
    "pred_prob_me",
    # "pred_prob_re",
]

# Compute shared color scale across observed and predicted
heatmap_df = df_model_me.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
    heatmap_df.consumption_pd, 10, retbins=True
)

all_values = []
for col in columns_to_bin:
    for version in versions:
        heatmap_data = (
            heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[version]
            .sum()
            .unstack()
            / heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["age_month"]
            .sum()
            .unstack()
        )
        all_values.append(heatmap_data.values.flatten())

all_values = np.concatenate(all_values)
vmin = np.nanmin(all_values) * multiply_by_val
vmax = np.nanmax(all_values) * multiply_by_val

# Side-by-side heatmap: observed vs predicted (with RE)
versions_labeled = [
    ("child_mortality", "Observed Mortality"),
    ("pred_prob_re", "Predicted (with RE)"),
]

model_name = "Child Mortality"


# pdf_path = os.path.join(PLOT_PATH, "child_mortality_version_2026_04_13.pdf")
# with PdfPages(pdf_path) as pdf:
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)

#     for col_idx, (version, version_label) in enumerate(versions_labeled):
#         data = df_model.rename(columns={version: "model_predictions"})
#         plot_heat_map_person_time_grid(
#             data=data,
#             bin_cols=columns_to_bin,
#             ax=axes[col_idx],
#             title=f"{model_name} - {version_label}",
#             multiply_by=multiply_by_val,
#             vmin=vmin,
#             vmax=vmax,
#             show_colorbar=(col_idx == 1),
#             # x_bins=custom_x_bins,
#             # y_bins=custom_y_bins,
#         )

#     pdf.savefig(fig)
#     plt.close(fig)

# print(f"PDF saved to {pdf_path}")


# Individual plots
vmin = 0.05
vmax = 3.3
plot_heat_map_person_time(
    data=df_model_me.rename(columns={"child_mortality": "model_predictions"}),
    outfile="raw_heatmap_child_mortality_2026_04_16",
    title="Observed Mortality (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


# plot_heat_map_person_time(
#     data=df_model_me.rename(columns={"pred_prob_re": "model_predictions"}),
#     outfile="predicted_heatmap_child_mortality_2026_04_16_me",
#     title="Predicted Mortality with RE (per 1000 person-months)",
#     bin_cols=columns_to_bin,
#     format=".2f",
#     multiply_by=multiply_by_val,
#     # vmin=vmin,
#     # vmax=vmax,
#     show_colorbar=False,
#     x_bins=custom_x_bins,
#     y_bins=custom_y_bins,
# )

plot_heat_map_person_time(
    data=df_model_me.rename(columns={"pred_prob_me": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_16_me",
    title="Predicted Mortality with RE (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

plot_heat_map_person_time(
    data=df_model_fe.rename(columns={"pred_prob": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_16_fe",
    title="Predicted Mortality with FE (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


# Test out other heat maps
# plot_heat_map_person_time(
#     data=df_max_age.rename(columns={"pred_prob_re": "model_predictions"}),
#     outfile="test",
#     title="Predicted Mortality with RE (per 1000 person-months)",
#     bin_cols=columns_to_bin,
#     format=".2f",
#     multiply_by=multiply_by_val,
#     # vmin=vmin,
#     # vmax=vmax,
#     show_colorbar=False,
#     x_bins=custom_x_bins,
#     y_bins=custom_y_bins,
# )


# Trying cumulative hazard equivalent
vmin = 0.05
vmax = 3.0
plot_heat_map_person_time(
    data=df_max_age.rename(columns={"child_mortality": "model_predictions"}),
    outfile="raw_heatmap_child_mortality_2026_04_16",
    title="Observed Mortality (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


plot_heat_map_person_time(
    data=df_max_age.rename(columns={"mortality_me": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_16_me",
    title="Predicted Mortality with RE (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

plot_heat_map_person_time(
    data=df_max_age.rename(columns={"mortality_fe": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_16_fe",
    title="Predicted Mortality with FE (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


# Trying cumhaz
vmin = 0.05
vmax = 3.0
plot_heat_map_person_time(
    data=df_model.rename(columns={"child_mortality": "model_predictions"}),
    outfile="raw_heatmap_child_mortality_2026_04_16",
    title="Observed Mortality (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


plot_heat_map_person_time(
    data=df_model.rename(columns={"cumhaz_me": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_20_me",
    title="Predicted Mortality with RE (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

plot_heat_map_person_time(
    data=df_model.rename(columns={"cumhaz_fe": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_20_fe",
    title="Predicted Mortality with FE (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


# Try cumhaz for fe
plot_heat_map_person_time(
    data=df_model_fe.rename(columns={"cumhaz_fe": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_16_fe",
    title="Predicted Mortality with FE (per 1000 person-months)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)
# similar to cum_mortality_prob but slightly higher


## Try heatmaps but for only one age-group at a time

[c for c in df_model.columns]
df_model["age_month"].value_counts()

plot_heat_map_person_time(
    data=df_model[df_model["age_month"] == 1].rename(
        columns={"child_mortality": "model_predictions"}
    ),
    outfile="raw_heatmap_child_mortality_2026_04_16",
    title="Observed Mortality",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


plot_heat_map_person_time(
    data=df_model[df_model["age_month"] == 1].rename(
        columns={"cumhaz_me": "model_predictions"}
    ),
    outfile="predicted_heatmap_child_mortality_2026_04_20_me",
    title="Predicted Mortality PPT with RE",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

plot_heat_map_person_time(
    data=df_model[df_model["age_month"] == 1].rename(
        columns={"cumhaz_fe": "model_predictions"}
    ),
    outfile="predicted_heatmap_child_mortality_2026_04_20_fe",
    title="Predicted Mortality PPT without RE",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


multiply_by_val = 1000

# Build a single PDF with one row per age_month and three columns:
# observed child_mortality, cumhaz_me, cumhaz_fe.
# Use a global color scale across all grouped values in all 3 x N panels.

df_model["age_month_fixed"] = df_model["age_month"]
df_model["age_month"] = df_model["age_month_old"]
df_model["age_month"] = df_model["age_month_fixed"]

age_month_values = np.sort(df_model["age_month"].dropna().unique())

versions_labeled_age = [
    ("child_mortality", "Observed Mortality"),
    ("cumhaz_me", "Predicted with RE"),
    ("cumhaz_fe", "Predicted without RE"),
]

all_grouped_values = []
row_scales = {}
for age_month_val in age_month_values:
    age_subset = df_model[df_model["age_month"] <= age_month_val].copy()
    # Apply the same deduplication as in the plot loop
    age_subset = (
        age_subset.sort_values("age_month")
        .groupby("indv_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    row_grouped_values = []
    for version, _ in versions_labeled_age:
        heatmap_df = age_subset.rename(columns={version: "model_predictions"}).copy()

        for col in columns_to_bin:
            if custom_x_bins is not None:
                heatmap_df[f"{col}_bin"] = pd.cut(
                    heatmap_df[col],
                    bins=custom_x_bins,
                    include_lowest=True,
                    right=False,
                )
            else:
                heatmap_df[f"{col}_bin"] = pd.cut(
                    heatmap_df[col], bins=10, include_lowest=True
                )

        if custom_y_bins is None:
            heatmap_df["consumption_pd_bin"] = pd.qcut(
                heatmap_df.consumption_pd, 10, duplicates="drop"
            )
        else:
            heatmap_df["consumption_pd_bin"] = pd.cut(
                heatmap_df.consumption_pd,
                bins=custom_y_bins,
                include_lowest=True,
                right=False,
            )

        for col in columns_to_bin:
            grouped_vals = (
                heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[
                    "model_predictions"
                ]
                .sum()
                .unstack()
                / heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["age_month"]
                .sum()
                .unstack()
            )
            vals = grouped_vals.values.flatten()
            all_grouped_values.append(vals)
            row_grouped_values.append(vals)

    row_grouped_values = np.concatenate(row_grouped_values)
    row_scales[age_month_val] = (
        np.nanmin(row_grouped_values) * multiply_by_val,
        np.nanmax(row_grouped_values) * multiply_by_val,
    )


from tqdm import tqdm
pdf_path_age_panels = os.path.join(PLOT_PATH, "child_mortality_8x3_cumulative_unique_indv.pdf")

with PdfPages(pdf_path_age_panels) as pdf:
    fig, axes = plt.subplots(
        nrows=len(age_month_values),
        ncols=3,
        figsize=(16, 4.2 * len(age_month_values)),
        constrained_layout=True,
    )

    if len(age_month_values) == 1:
        axes = np.array([axes])

    for row_idx, age_month_val in tqdm(enumerate(age_month_values)):
        # Try making up until rather than within age bin
        age_subset = df_model[df_model["age_month"] <= age_month_val].copy()
        # Try only including 1 individual at a time
        df_max_age_subset = (
            age_subset.sort_values("age_month")
            .groupby("indv_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        row_vmin, row_vmax = row_scales[age_month_val]

        for col_idx, (version, version_label) in enumerate(versions_labeled_age):
            plot_heat_map_person_time_grid(
                data=df_max_age_subset.rename(columns={version: "model_predictions"}),
                bin_cols=columns_to_bin,
                ax=axes[row_idx, col_idx],
                title=f"{version_label}\nage_month<={int(age_month_val)}",
                multiply_by=multiply_by_val,
                vmin=row_vmin,
                vmax=row_vmax,
                show_colorbar=(col_idx == 2),
                x_bins=custom_x_bins,
                y_bins=custom_y_bins,
            )

    pdf.savefig(fig)
    plt.close(fig)

print(
    f"Saved age-specific 8x3 heatmap PDF to {pdf_path_age_panels} "
    f"with row-specific scales by age_month."
)


# Make plot of synthic data
vmin = 0.1
vmax = 3.3
plot_heat_map_person_time(
    data=df_model_fe.rename(columns={"cumhaz_fe": "model_predictions"}),
    outfile="predicted_heatmap_child_mortality_2026_04_16_fe",
    title="Predicted without RE (age_month==60)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

# Can I make fixed effects plot monotonic with fine-grained y bins?
df_model["age_month"] = df_model["age_month_old"]
df_model["age_month"] = df_model["age_month_fixed"]
custom_y_bins = [0,0.05,0.1,0.15,0.2,0.25,0.5,0.75, 1, 1.25,1.3,1.4,1.45,1.5, 1.75,2,2.25,2.5,3,4,5.25, 9.3, 15.5,17,18,20,22.5,23.5,25,27.5,31,40]

custom_y_bins = [0, 1.5, 5.25, 9.3, 15.5, 31]
plot_heat_map_person_time(
    data=df_model[df_model["age_month"]==60].rename(
        columns={"cumhaz_fe": "model_predictions"}
    ),
    outfile="predicted_heatmap_child_mortality_2026_04_21_fe_fine_grained",
    title="Predicted Mortality PPT without RE",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)



# Check underlying predictions for non-monotonicity:
# Use the discretized age periods (not age_month_old)
df_check = df_model.copy()
df_check["age_month_period"] = df_model["age_month_fixed"]  # 1,3,6,12,24,36,48,60

df_check["cumhaz_fe_rounded"] = df_check["cumhaz_fe"].round(8)

# Sort once, then use groupby + shift — no Python loops
df_check = df_check.sort_values(["age_month_period", "days_over_30C", "consumption_pd"])

df_check["cumhaz_fe_lag"] = df_check.groupby(
    ["age_month_period", "days_over_30C"], observed=True
)["cumhaz_fe_rounded"].shift(1)

# Drop NaN lags and rows where value didn't change
df_check = df_check.dropna(subset=["cumhaz_fe_lag"])
df_check = df_check[df_check["cumhaz_fe_rounded"] != df_check["cumhaz_fe_lag"]]

# Non-monotonic: cumhaz_fe went UP as consumption_pd increased
examine_all = df_check[df_check["cumhaz_fe_rounded"] > df_check["cumhaz_fe_lag"]].copy()

print(f"Non-monotonic rows: {len(examine_all):,}")
print(f"Unique age periods affected: {sorted(examine_all['age_month_period'].unique())}")
print(f"Unique days_over_30C affected: {examine_all['days_over_30C'].nunique()}")
examine_all[["age_month_period", "days_over_30C", "consumption_pd", "cumhaz_fe_rounded", "cumhaz_fe_lag"]].head(20)



# Investigate remaining non-montonicities between top left cells
def get_rows_for_heatmap_cell(
    df,
    age_month_val,
    version_col,  # e.g. "child_mortality", "cumhaz_me", "cumhaz_fe"
    x_col="days_over_30C",
    y_col="consumption_pd",
    x_bins=None,
    y_bins=None,
    x_bin_idx=0,  # 0-based column index in heatmap
    y_bin_idx=0,  # 0-based row index in heatmap
):
    d = df[df["age_month"] == age_month_val].copy()
    d = d.rename(columns={version_col: "model_predictions"})

    d[f"{x_col}_bin"] = pd.cut(d[x_col], bins=x_bins, include_lowest=True, right=False)
    d[f"{y_col}_bin"] = pd.cut(d[y_col], bins=y_bins, include_lowest=True, right=False)

    x_cats = d[f"{x_col}_bin"].cat.categories
    y_cats = d[f"{y_col}_bin"].cat.categories

    x_interval = x_cats[x_bin_idx]
    y_interval = y_cats[y_bin_idx]

    cell_rows = d[
        (d[f"{x_col}_bin"] == x_interval) & (d[f"{y_col}_bin"] == y_interval)
    ].copy()

    # optional metadata columns so the returned object is still self-describing
    cell_rows["x_interval"] = str(x_interval)
    cell_rows["y_interval"] = str(y_interval)
    cell_rows["n_rows_in_cell"] = len(cell_rows)
    cell_rows["n_individuals_in_cell"] = (
        cell_rows["indv_id"].nunique() if "indv_id" in cell_rows.columns else np.nan
    )

    return cell_rows


l2 = get_rows_for_heatmap_cell(
    df=df_model,
    age_month_val=1,
    version_col="cumhaz_fe",
    x_col="days_over_30C",
    y_col="consumption_pd",
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
    x_bin_idx=0,
    y_bin_idx=1,
)


l3 = get_rows_for_heatmap_cell(
    df=df_model,
    age_month_val=1,
    version_col="cumhaz_fe",
    x_col="days_over_30C",
    y_col="consumption_pd",
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
    x_bin_idx=1,
    y_bin_idx=1,
)


## try manual
# heatmap_data = (
#     heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[
#         "model_predictions"
#     ]
#     .sum()
#     .unstack()
#     / heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["age_month"]
#     .sum()
#     .unstack()
# )
l2[]
l2_amount = 1000 * l2["model_predictions"].sum() / l2["age_month"].sum()
l3_amount = 1000 * l3["model_predictions"].sum() / l3["age_month"].sum()

# Cell totals
f"l2_amount: {l2_amount:.3f} | l3_amount: {l3_amount:.3f}"

# numerators
f"l2 numerator: {l2['model_predictions'].sum():,} | l3 numerator: {l3['model_predictions'].sum():,}"

# the max l2 prediction should be less than the min l3
f"l2 max pred: {l2['model_predictions'].max():,} | l3 min pred: {l3['model_predictions'].min():,}"

# filter each
l2_explain = l2[l2["model_predictions"] > l3["model_predictions"].min()]
l3_explain = l3[l3["model_predictions"] < l2["model_predictions"].max()]

# could it be that the consumption is dominating the days over 30? 
l2_explain["consumption_pd"].describe() # similar range
l3_explain["consumption_pd"].describe()
l2_explain["days_over_30C"].describe() # discrete
l3_explain["days_over_30C"].describe()

# Compare against
# originial 
