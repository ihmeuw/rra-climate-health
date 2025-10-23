import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from lifelines import CoxPHFitter  # for Cox survival models
from pymer4.models import Lmer
import os

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_22.01/data.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_22.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_22.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)


## FUNCTIONS:
def plot_indvs_by_age_group(data):
    # Get individuals per age_month
    if "indv_id" not in data.columns:
        data = data.copy()
        data["indv_id"] = (
            data[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
        )
    agg_age = (
        data.groupby(["age_month"])["indv_id"]
        .nunique()
        .reset_index()
        .rename(columns={"indv_id": "unique_individuals"})
    )
    plt.figure(figsize=(30, 5))
    ax = agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False)
    plt.title("Unique Individuals per Age Month in Raw Data")
    months = agg_age["age_month"].values
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.tight_layout()


def plot_heat_map(
    data: pd.DataFrame,
    outfile: str,
    title: str,
    bin_cols: list,
    format: str = ".2f",
    vmin: float = None,
    vmax: float = None,
):

    heatmap_df = data.copy()
    for col in bin_cols:
        heatmap_df[f"{col}_bin"] = pd.qcut(
            heatmap_df[col], 10, retbins=False, duplicates="drop"
        )
    heatmap_df["consumption"], ldi_bins = pd.qcut(
        heatmap_df.consumption, 10, retbins=True, duplicates="drop"
    )

    # Create a discrete colormap with steps matching your rounded values
    bounds = np.round(np.arange(0, round(vmax + 0.002, 4), 0.001), 4)  # steps of 0.001
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    for col in columns_to_bin:
        plt.figure(figsize=(10, 8))
        heatmap_data = (
            heatmap_df.groupby(["consumption", f"{col}_bin"])["model_predictions"]
            .mean()
            .unstack()
        )

        if vmin and vmax:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
                vmin=vmin,
                vmax=vmax,
                norm=norm,
            )
        else:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
            )
        plt.title(
            f"{title}\nby Consumption and {col.replace('_', ' ').title()}",
            fontsize=20,
        )

        # Set rounded axis labels
        ax1.set_xticklabels(
            [f"{int(x.left)}–{int(x.right)}" for x in heatmap_data.columns],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        ax1.set_yticklabels(
            [f"{int(y.left)}–{int(y.right)}" for y in heatmap_data.index],
            rotation=0,
            fontsize=10,
        )
        ax1.set_xlabel("Binned " + col.replace("_", " ").title(), fontsize=16)
        ax1.set_ylabel("Consumption Bin", fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"{outfile}_{col}.png"))
        plt.close()


## READ IN DATA

# Raw data
df = pd.read_parquet(DATA_PATH)

# Modeled data

df_model = pd.read_parquet(
    RESULTS_PATH + "predictions_subset_100pct_model_interaction.parquet"
)

# Neonatal predictions
neonatal = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_16.01/neonatal.parquet"
)

# df_neo = pd.read_parquet(RESULTS_PATH + "neonatal/neonatal_mortality_1_mo.parquet")
df_neo = pd.read_parquet(
    RESULTS_PATH + "neonatal/neonatal_mortality_subset_100pct_model_do30.parquet"
)


## CONSTANTS

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


## MAKE SIDE-BY-SIDE HEATMAPS TOGETHER

# Raw data only
df["child_mortality_scaled"] = df["child_mortality"] / df["age_month"]
plot_heat_map(
    data=df.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_22_ppt",
    title="Raw Data Child Mortality (per Person-only)",
    bin_cols=columns_to_bin,
    format=".3f",
    # vmin=vmin,
    # vmax=vmax,
)

# just use df_model
df_model["child_mortality_scaled"] = df_model["child_mortality"] / df_model["age_month"]
df_model["mortality_fe_cum_scaled"] = df_model["mortality_fe"] / df_model["age_month"]
df_model["mortality_me_cum_scaled"] = df_model["mortality_me"] / df_model["age_month"]
df_model["mortality_point_fe_scaled"] = (
    df_model["mortality_point_fe"] / df_model["age_month"]
)
df_model["mortality_point_me_scaled"] = (
    df_model["mortality_point_me"] / df_model["age_month"]
)

# Make plots for point estimates ###############################################
heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    "child_mortality_scaled",
    "mortality_point_fe_scaled",
    "mortality_point_me_scaled",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

# plot raw data heatmaps with consistent color scale
# without scaled time
plot_heat_map(
    data=df_model.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt_original",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled fixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_fe_scaled": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_23_ppt_original",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled mixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_me_scaled": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_23_ppt_original",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

# Make plots for cumulative estimates ##########################################
heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    "child_mortality_scaled",
    "mortality_fe_cum_scaled",
    "mortality_me_cum_scaled",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

# plot raw data heatmaps with consistent color scale
# without scaled time
plot_heat_map(
    data=df_model.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot scaled fixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_fe_cum_scaled": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot scaled mixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_me_cum_scaled": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    # vmin=vmin,
    # vmax=vmax,
)

# Make plots with unscaled data ################################################
heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    "child_mortality_scaled",
    "mortality_fe",
    "mortality_me",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=df_model.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled fixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled mixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

# Make plots with unscaled point data ##########################################
heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    "child_mortality_scaled",
    "mortality_point_fe",
    "mortality_point_me",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=df_model.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled fixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled mixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    # vmin=vmin,
    # vmax=vmax,
)

# Make plots with 60 age_months removed ########################################
heatmap_df = df_model[df_model.age_month < 60].copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    "child_mortality_scaled",
    "mortality_fe_cum_scaled",
    "mortality_me_cum_scaled",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=df_model[df_model.age_month < 60].rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled fixed effects predictions
plot_heat_map(
    data=df_model[df_model.age_month < 60].rename(
        columns={
            "mortality_fe_cum_scaled": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled mixed effects predictions
plot_heat_map(
    data=df_model[df_model.age_month < 60].rename(
        columns={
            "mortality_me_cum_scaled": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

# Data exploration #############################################################

# get avg point probablity by age month
df_age_group = df_model.groupby("age_month").agg(
    {
        "child_mortality": "mean",
        "child_mortality_scaled": "mean",
        "mortality_fe": "mean",
        "mortality_me": "mean",
        "mortality_point_fe": "mean",
        "mortality_point_me": "mean",
        "mortality_fe_cum_scaled": "mean",
        "mortality_me_cum_scaled": "mean",
        "mortality_point_fe_scaled": "mean",
        "mortality_point_me_scaled": "mean",
    }
)

# plot avg mortality and mortality_scaled probability by age month
plt.figure(figsize=(12, 8))
plt.plot(
    df_age_group.index, df_age_group["child_mortality"], label="Avg Child Mortality"
)
plt.plot(
    df_age_group.index,
    df_age_group["child_mortality_scaled"],
    label="Avg Child Mortality (Scaled)",
)
plt.xlabel("Age (Months)")
plt.ylabel("Probability")
plt.title("Average Child Mortality by Age Month")
plt.legend()
# plt.show()
plt.savefig(os.path.join(PLOT_PATH, f"avg_child_mortality_by_age_month_unexploded.png"))
plt.close()


# plot the modeled averages by age_month
plt.figure(figsize=(12, 8))
plt.plot(
    df_age_group.index,
    df_age_group["mortality_fe"],
    label="Avg Fixed Effects Cumulative",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_me"],
    label="Avg Mixed Effects Cumulative",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_fe"],
    label="Avg Fixed Effects Point",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_me"],
    label="Avg Mixed Effects Point",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_fe_cum_scaled"],
    label="Avg Fixed Effects Cumulative (Scaled)",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_me_cum_scaled"],
    label="Avg Mixed Effects Cumulative (Scaled)",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_fe_scaled"],
    label="Avg Fixed Effects Point (Scaled)",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_me_scaled"],
    label="Avg Mixed Effects Point (Scaled)",
)
plt.xlabel("Age (Months)")
plt.ylabel("Probability")
plt.title("Average Modeled Child Mortality by Age Month")
plt.legend()
# plt.show()
plt.savefig(os.path.join(PLOT_PATH, f"avg_child_mortality_modeled_by_age_month.png"))
plt.close()

# Try exploding data to compare raw
df_exploded = df_model.copy()
df_exploded["months_to_expand"] = df_exploded.apply(
    lambda x: [y for y in range(1, x["age_month"] + 1)],
    axis=1,
)
df_exploded = df_exploded.explode("months_to_expand")

# adjust mortality s.t. children who died at age_month have previous months as 0
df_exploded["months_to_expand"] = df_exploded["months_to_expand"].astype(int)
df_exploded.loc[
    df_exploded["months_to_expand"] < df_exploded["age_month"], "child_mortality"
] = 0

# get avg point probablity by age month
df_exploded["child_mortality_scaled"] = (
    df_exploded["child_mortality"] / df_exploded["months_to_expand"]
)
df_exploded_age_group = df_exploded.groupby("months_to_expand").agg(
    {
        "child_mortality": "mean",
        "child_mortality_scaled": "mean",
    }
)

# plot avg mortality and mortality_scaled probability by age month
plt.figure(figsize=(12, 8))
plt.plot(
    df_exploded_age_group.index,
    df_exploded_age_group["child_mortality"],
    label="Avg Child Mortality",
)
plt.plot(
    df_exploded_age_group.index,
    df_exploded_age_group["child_mortality_scaled"],
    label="Avg Child Mortality (Scaled)",
)
plt.xlabel("Age (Months)")
plt.ylabel("Probability")
plt.title("Average Child Mortality by Age Month")
plt.legend()
# plt.show()
plt.savefig(os.path.join(PLOT_PATH, f"avg_child_mortality_by_age_month_exploded.png"))
plt.close()

# compare to plot of models without cumulative to get scales similar on y axis
plt.figure(figsize=(12, 8))
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_fe"],
    label="Avg Fixed Effects Point",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_me"],
    label="Avg Mixed Effects Point",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_fe_scaled"],
    label="Avg Fixed Effects Point (Scaled)",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_me_scaled"],
    label="Avg Mixed Effects Point (Scaled)",
)
plt.xlabel("Age (Months)")
plt.ylabel("Probability")
plt.title("Average Modeled Child Mortality by Age Month")
plt.legend()
# plt.show()
plt.savefig(
    os.path.join(
        PLOT_PATH, f"avg_child_mortality_modeled_by_age_month_non_cumulative.png"
    )
)
plt.close()

# compare to plot of models without cumulative to df_exploded["child_mortality_scaled"]
plt.figure(figsize=(12, 8))
plt.plot(
    df_exploded_age_group.index,
    df_exploded_age_group["child_mortality_scaled"],
    label="Avg Child Mortality (Scaled)",
    linestyle="dashed",
    color="black",
    linewidth=3,
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_fe_scaled"],
    label="Avg Fixed Effects Point (Scaled)",
)
plt.plot(
    df_age_group.index,
    df_age_group["mortality_point_me_scaled"],
    label="Avg Mixed Effects Point (Scaled)",
)
plt.xlabel("Age (Months)")
plt.ylabel("Probability")
plt.title(
    "Average Modeled Child Mortality by Age Month and Exploded Scaled Child Mortality"
)
plt.legend()
# plt.show()
plt.savefig(
    os.path.join(PLOT_PATH, f"child_mortality_scaled_vs_modeled_points_scaled.png")
)
plt.close()

# Compare scatters against consumption and days over 30
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# 1. Exploded Child Mortality (Scaled) vs Consumption
axs[0, 0].scatter(
    df_exploded["consumption"],
    df_exploded["child_mortality_scaled"],
    label="Exploded Child Mortality (Scaled)",
    alpha=0.5,
)
axs[0, 0].set_xlabel("Consumption")
axs[0, 0].set_ylabel("Probability")
axs[0, 0].set_title("Exploded Child Mortality (Scaled) vs Consumption")
axs[0, 0].legend()

# 2. Unexploded Child Mortality (Scaled) vs Consumption
axs[0, 1].scatter(
    df_model["consumption"],
    df_model["child_mortality_scaled"],
    label="Unexploded Child Mortality (Scaled)",
    alpha=0.5,
)
axs[0, 1].set_xlabel("Consumption")
axs[0, 1].set_ylabel("Probability")
axs[0, 1].set_title("Unexploded Child Mortality (Scaled) vs Consumption")
axs[0, 1].legend()

# 3. Avg Fixed Effects Point (Scaled) vs Consumption
axs[1, 0].scatter(
    df_model["consumption"],
    df_model["mortality_point_fe_scaled"],
    label="Avg Fixed Effects Point (Scaled)",
    alpha=0.5,
)
axs[1, 0].set_xlabel("Consumption")
axs[1, 0].set_ylabel("Probability")
axs[1, 0].set_title("Fixed Effects Point Mortality (Scaled) vs Consumption")
axs[1, 0].legend()

# 4. Avg Mixed Effects Point (Scaled) vs Consumption
axs[1, 1].scatter(
    df_model["consumption"],
    df_model["mortality_point_me_scaled"],
    label="Avg Mixed Effects Point (Scaled)",
    alpha=0.5,
)
axs[1, 1].set_xlabel("Consumption")
axs[1, 1].set_ylabel("Probability")
axs[1, 1].set_title("Mixed Effects Point Mortality (Scaled) vs Consumption")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "child_mortality_vs_modeled_points_by_consumption_2x2.png")
)
plt.close()

# same but for days over 30
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# 1. Exploded Child Mortality (Scaled) vs Days Over 30
axs[0, 0].scatter(
    df_exploded["days_over_30C"],
    df_exploded["child_mortality_scaled"],
    label="Exploded Child Mortality (Scaled)",
    alpha=0.5,
)
axs[0, 0].set_xlabel("Days Over 30")
axs[0, 0].set_ylabel("Probability")
axs[0, 0].set_title("Exploded Child Mortality (Scaled) vs Days Over 30")
axs[0, 0].legend()

# 2. Unexploded Child Mortality (Scaled) vs Days Over 30
axs[0, 1].scatter(
    df_model["days_over_30C"],
    df_model["child_mortality_scaled"],
    label="Unexploded Child Mortality (Scaled)",
    alpha=0.5,
)
axs[0, 1].set_xlabel("Days Over 30")
axs[0, 1].set_ylabel("Probability")
axs[0, 1].set_title("Unexploded Child Mortality (Scaled) vs Days Over 30")
axs[0, 1].legend()

# 3. Avg Fixed Effects Point (Scaled) vs Days Over 30
axs[1, 0].scatter(
    df_model["days_over_30C"],
    df_model["mortality_point_fe_scaled"],
    label="Avg Fixed Effects Point (Scaled)",
    alpha=0.5,
)
axs[1, 0].set_xlabel("Days Over 30")
axs[1, 0].set_ylabel("Probability")
axs[1, 0].set_title("Fixed Effects Point Mortality (Scaled) vs Days Over 30")
axs[1, 0].legend()

# 4. Avg Mixed Effects Point (Scaled) vs Days Over 30
axs[1, 1].scatter(
    df_model["days_over_30C"],
    df_model["mortality_point_me_scaled"],
    label="Avg Mixed Effects Point (Scaled)",
    alpha=0.5,
)
axs[1, 1].set_xlabel("Days Over 30")
axs[1, 1].set_ylabel("Probability")
axs[1, 1].set_title("Mixed Effects Point Mortality (Scaled) vs Days Over 30")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "child_mortality_vs_modeled_points_by_days_over_30_2x2.png")
)
plt.close()

# Compare average mortality and modeled against consumption and days over 30 ###


df_exploded_consumption_bins = df_exploded.copy()
for col in columns_to_bin:
    df_exploded_consumption_bins[f"{col}_bin"] = pd.qcut(
        df_exploded_consumption_bins[col], 10, retbins=False, duplicates="drop"
    )
df_exploded_consumption_bins["consumption"], ldi_bins = pd.qcut(
    df_exploded_consumption_bins.consumption, 10, retbins=True
)

df_unexploded_consumption_bins = df_model.copy()
for col in columns_to_bin:
    df_unexploded_consumption_bins[f"{col}_bin"] = pd.qcut(
        df_unexploded_consumption_bins[col], 10, retbins=False, duplicates="drop"
    )
df_unexploded_consumption_bins["consumption"], ldi_bins = pd.qcut(
    df_unexploded_consumption_bins.consumption, 10, retbins=True
)

# plot the differences between exploded and unexploded data by quantiles
for col in columns_to_bin:
    plt.figure(figsize=(10, 8))
    exploded_means = (
        df_exploded_consumption_bins.groupby(["consumption", f"{col}_bin"])[
            "child_mortality_scaled"
        ]
        .mean()
        .unstack()
    )
    exploded_means_unscaled = (
        df_exploded_consumption_bins.groupby(["consumption", f"{col}_bin"])[
            "child_mortality"
        ]
        .mean()
        .unstack()
    )
    unexploded_fe_means = (
        df_unexploded_consumption_bins.groupby(["consumption", f"{col}_bin"])[
            "mortality_point_fe"
        ]
        .mean()
        .unstack()
    )
    unexploded_me_means = (
        df_unexploded_consumption_bins.groupby(["consumption", f"{col}_bin"])[
            "mortality_point_me"
        ]
        .mean()
        .unstack()
    )
    unexploded_mortality = (
        df_unexploded_consumption_bins.groupby(["consumption", f"{col}_bin"])[
            "child_mortality_scaled"
        ]
        .mean()
        .unstack()
    )
    unexploded_mortality_unscaled = (
        df_unexploded_consumption_bins.groupby(["consumption", f"{col}_bin"])[
            "child_mortality"
        ]
        .mean()
        .unstack()
    )
    # Convert interval columns to midpoints for plotting
    bin_midpoints = [interval.mid for interval in exploded_means.columns]

    plt.plot(
        bin_midpoints,
        exploded_means.mean(),
        label="Exploded Child Mortality (Scaled)",
    )
    plt.plot(
        bin_midpoints,
        exploded_means_unscaled.mean(),
        label="Exploded Child Mortality (Unscaled)",
    )
    plt.plot(
        bin_midpoints,
        unexploded_fe_means.mean(),
        label="Unexploded Fixed Effects Point Mortality",
    )
    plt.plot(
        bin_midpoints,
        unexploded_me_means.mean(),
        label="Unexploded Mixed Effects Point Mortality",
    )
    plt.plot(
        bin_midpoints,
        unexploded_mortality.mean(),
        label="Unexploded Child Mortality (Scaled)",
    )
    plt.plot(
        bin_midpoints,
        unexploded_mortality_unscaled.mean(),
        label="Unexploded Child Mortality (Unscaled)",
    )
    plt.xlabel(f"Binned {col.replace('_', ' ').title()} (bin midpoint)")
    plt.ylabel("Average Probability")
    plt.title(
        f"Average Child Mortality vs Modeled by Binned Consumption and {col.replace('_', ' ').title()}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_PATH,
            f"avg_child_mortality_vs_modeled_by_consumption_and_{col}_bins.png",
        )
    )
    plt.close()

# Plot mean mortality unexploded and exploded by days over 30
df_exploded_binned = df_exploded.copy()
df_exploded_binned["days_over_30C_bin"] = pd.qcut(
    df_exploded_binned["days_over_30C"], 10, retbins=False, duplicates="drop"
)

df_unexploded_binned = df_model.copy()
df_unexploded_binned["days_over_30C_bin"] = pd.qcut(
    df_unexploded_binned["days_over_30C"], 10, retbins=False, duplicates="drop"
)


exploded_means = df_exploded_binned.groupby("days_over_30C_bin")[
    "child_mortality_scaled"
].mean()
unexploded_means = df_unexploded_binned.groupby("days_over_30C_bin")[
    "child_mortality_scaled"
].mean()

fig, axs = plt.subplots(1, 2, figsize=(16, 12))
# Convert interval columns to midpoints for plotting
bin_midpoints_exploded = [interval.mid for interval in exploded_means.index]
bin_midpoints_unexploded = [interval.mid for interval in unexploded_means.index]

axs[0].plot(
    bin_midpoints_exploded,
    exploded_means,
    label="Exploded Child Mortality (Scaled)",
)
axs[0].set_xlabel("Binned Days Over 30C (bin midpoint)")
axs[0].set_ylabel("Average Probability")
axs[0].set_title("Average Exploded Child Mortality by Binned Days Over 30C")
axs[0].legend()

axs[1].plot(
    bin_midpoints_unexploded,
    unexploded_means,
    label="Unexploded Child Mortality (Scaled)",
)
axs[1].set_xlabel("Binned Days Over 30C (bin midpoint)")
axs[1].set_ylabel("Average Probability")
axs[1].set_title("Average Unexploded Child Mortality by Binned Days Over 30C")
axs[1].legend()
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "child_mortality_vs_days_30_exploded_unexploded.png")
)
plt.close()

# Plot mean mortality unexploded and exploded by consumption
df_exploded_binned = df_exploded.copy()
df_exploded_binned["consumption_bin"] = pd.qcut(
    df_exploded_binned["consumption"], 10, retbins=False, duplicates="drop"
)

df_unexploded_binned = df_model.copy()
df_unexploded_binned["consumption_bin"] = pd.qcut(
    df_unexploded_binned["consumption"], 10, retbins=False, duplicates="drop"
)


exploded_means = df_exploded_binned.groupby("consumption_bin")[
    "child_mortality_scaled"
].mean()

unexploded_means = df_unexploded_binned.groupby("consumption_bin")[
    "child_mortality_scaled"
].mean()

fig, axs = plt.subplots(1, 2, figsize=(16, 12))
# Convert interval columns to midpoints for plotting
bin_midpoints_exploded = [interval.mid for interval in exploded_means.index]
bin_midpoints_unexploded = [interval.mid for interval in unexploded_means.index]

axs[0].plot(
    bin_midpoints_exploded,
    exploded_means,
    label="Exploded Child Mortality (Scaled)",
)
axs[0].set_xlabel("Binned Consumption (bin midpoint)")
axs[0].set_ylabel("Average Probability")
axs[0].set_title("Average Exploded Child Mortality by Binned Consumption")
axs[0].legend()

axs[1].plot(
    bin_midpoints_unexploded,
    unexploded_means,
    label="Unexploded Child Mortality (Scaled)",
)
axs[1].set_xlabel("Binned Consumption (bin midpoint)")
axs[1].set_ylabel("Average Probability")
axs[1].set_title("Average Unexploded Child Mortality by Binned Consumption")
axs[1].legend()
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "child_mortality_vs_consumption_exploded_unexploded.png")
)
plt.close()

# Reconcile these single-variable trends with what happens in the heatmaps
plot_heat_map(
    data=df_exploded.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    # vmin=vmin,
    # vmax=vmax,
)

# Replot heat maps comparing exploded data against unexploded modeled data #####

heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    "mortality_point_fe",
    "mortality_point_me",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

heatmap_df = df_exploded.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

for col in columns_to_bin:
    vals = (
        heatmap_df.groupby(["consumption", f"{col}_bin"])["child_mortality_scaled"]
        .mean()
        .values
    )
    all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=df_exploded.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled fixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled mixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

# Replot heat maps comparing exploded data against unexploded modeled data, scaled #####

heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    "mortality_point_fe_scaled",
    "mortality_point_me_scaled",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

heatmap_exploded_df = df_exploded.copy()
for col in columns_to_bin:
    heatmap_exploded_df[f"{col}_bin"] = pd.qcut(
        heatmap_exploded_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_exploded_df["consumption"], ldi_bins = pd.qcut(
    heatmap_exploded_df.consumption, 10, retbins=True
)

for col in columns_to_bin:
    vals = (
        heatmap_exploded_df.groupby(["consumption", f"{col}_bin"])[
            "child_mortality_scaled"
        ]
        .mean()
        .values
    )
    all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=df_exploded.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled fixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_fe_scaled": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled mixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_me_scaled": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_23_ppt",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)


# Plot heatmap on raw exploded data before collapsing by avg climate vars ######
# save temp files
df_climate = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_with_climate.parquet"
)

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df_climate["child_mortality"] = 1 - df_climate["child_alive"]
df_climate["consumption"] = df_climate["ldipc_weighted_no_match"]
df_climate["child_mortality_scaled"] = (
    df_climate["child_mortality"] / df_climate["months_to_expand"]
)

plot_heat_map(
    data=df_climate.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_23_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

## Neonatal
neonatal.rename(columns={"ldipc_weighted_no_match": "consumption"}, inplace=True)

# get min and max values for color scale consistency across plots
# df_non_neo = df[df["age_month"] > 0]
raw_heatmap_df = neonatal.copy()
# raw_heatmap_df = df_exploded[df_exploded["age_month"] == 1]
for col in columns_to_bin:
    raw_heatmap_df[f"{col}_bin"] = pd.qcut(
        raw_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
raw_heatmap_df["consumption"], ldi_bins = pd.qcut(
    raw_heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)

me_heatmap_df = df_neo_me.copy()
for col in columns_to_bin:
    me_heatmap_df[f"{col}_bin"] = pd.qcut(
        me_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
me_heatmap_df["consumption"], ldi_bins = pd.qcut(
    me_heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)

fe_heatmap_df = df_neo_fe.copy()
for col in columns_to_bin:
    fe_heatmap_df[f"{col}_bin"] = pd.qcut(
        fe_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
fe_heatmap_df["consumption"], ldi_bins = pd.qcut(
    fe_heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)

all_values = []
for data in [raw_heatmap_df, me_heatmap_df, fe_heatmap_df]:
    for col in columns_to_bin:
        if "model_predictions" in data.columns:
            append_val = (
                data.groupby(["consumption", f"{col}_bin"])["model_predictions"]
                .mean()
                .values
            )
            print(append_val)
            all_values.append(append_val)
        elif "child_mortality" in data.columns:
            append_val = (
                data.groupby(["consumption", f"{col}_bin"])["child_mortality"]
                .mean()
                .values
            )
            print(append_val)
            all_values.append(append_val)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=neonatal.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_neonatal_child_mortality_10_17",
    title="Raw Data Neonatal Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=df_neo_fe.copy(),
    outfile="neo_heatmap_child_mortality_100pc_do30_fe_10_17",
    title="Neonatal Modeled Child Mortality (without random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=df_neo_me.copy(),
    outfile="neo_heatmap_child_mortality_100pc_do30_me_10_17",
    title="Neonatal Modeled Child Mortality (with random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

## Explore raw data
df_raw = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_matched_10_06_2025.parquet"
)
key_vars = [
    "nid",
    "psu",
    "birth_year",
    "birth_month",
    "int_year",
    "int_month",
    "age_month",
    "hh_id",
    "geospatial_id",
    "line_id",
    "lat",
    "long",
    "child_alive",
]
df_raw.dropna(subset=key_vars, inplace=True)
df_raw["age_month"] = df_raw["age_month"].astype(int)
df_raw = df_raw[df_raw["age_month"] <= 60]
df_raw["indv_id"] = (
    df_raw[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
)

# Get individuals per age_month
agg_age = (
    df_raw.groupby(["age_month"])["indv_id"]
    .nunique()
    .reset_index()
    .rename(columns={"indv_id": "unique_individuals"})
)
plt.figure(figsize=(30, 5))
ax = agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False)
plt.title("Unique Individuals per Age Month in Raw Data")
months = agg_age["age_month"].values
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ",")))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, f"unique_indv_children_per_age_month_raw_data.png"))
plt.close()

## Plot predictions by country, me and fe, in bar charts
# Calculate mean predictions by country
country_preds = (
    df_model.groupby("ihme_loc_id")
    .agg({"mortality_fe_manual": "mean", "mortality_me_manual": "mean"})
    .reset_index()
)

# Sort by mixed effects prediction for better visualization
country_preds = country_preds.sort_values("mortality_me_manual", ascending=False)
# country_preds[['mortality_fe_manual', 'mortality_me_manual']]
# Create figure
fig, ax = plt.subplots(figsize=(20, 8))

# Set up bar positions
x = np.arange(len(country_preds))
width = 0.35

# Create bars
bars1 = ax.bar(
    x - width / 2,
    country_preds["mortality_fe_manual"],
    width,
    label="Fixed Effects",
    color="skyblue",
    alpha=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    country_preds["mortality_me_manual"],
    width,
    label="Mixed Effects (with frailty)",
    color="coral",
    alpha=0.8,
)

# Customize plot
ax.set_xlabel("Country (ihme_loc_id)", fontsize=14)
ax.set_ylabel("Average Mortality Probability", fontsize=14)
ax.set_title(
    "Comparison of Fixed Effects vs Mixed Effects Predictions by Country", fontsize=16
)
ax.set_xticks(x)
ax.set_xticklabels(country_preds["ihme_loc_id"], rotation=90, ha="right", fontsize=8)
ax.legend(fontsize=12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# Add grid for easier reading
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "fe_vs_me_predictions_by_country.png"), dpi=300)
plt.close()

## Plot scatterplot between two predictions
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=country_preds,
    x="mortality_fe_manual",
    y="mortality_me_manual",
    hue="ihme_loc_id",
    s=100,
    palette="tab20",
)
plt.plot([0, 0.2], [0, 0.2], color="gray", linestyle="--")  # 45-degree line
plt.xlabel("Fixed Effects Prediction", fontsize=14)
plt.ylabel("Mixed Effects Prediction", fontsize=14)
plt.title("Scatterplot of Fixed vs Mixed Effects Predictions by Country", fontsize=16)
plt.xlim(
    0, country_preds[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1
)
plt.ylim(
    0, country_preds[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1
)
plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.legend(
    title="Country (ihme_loc_id)",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8,
)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "fe_vs_me_predictions_scatter_by_country.png"), dpi=300
)
plt.close()

## Plot scatterplot between two predictions for full dataset
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_model,
    x="mortality_fe_manual",
    y="mortality_me_manual",
    hue="ihme_loc_id",
    s=20,
    alpha=0.5,
    palette="tab20",
)
plt.plot([0, 0.2], [0, 0.2], color="gray", linestyle="--")  # 45-degree line
plt.xlabel("Fixed Effects Prediction", fontsize=14)
plt.ylabel("Mixed Effects Prediction", fontsize=14)
plt.title(
    "Scatterplot of Fixed vs Mixed Effects Predictions for Individuals", fontsize=16
)
plt.xlim(0, df_model[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1)
plt.ylim(0, df_model[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1)
plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "fe_vs_me_predictions_scatter_for_individuals.png"), dpi=300
)
plt.close()

## Get correlation between country effects and days over 30
cntry_effects = pd.read_csv(
    RESULTS_PATH + "model_summaries/frailty_estimates_subset_05pct_model_do30.csv"
)
df_do30 = df_model.groupby("ihme_loc_id")["days_over_30C"].mean().reset_index()
cntry_effects = cntry_effects.merge(df_do30, on="ihme_loc_id", how="left")
cntry_effects.rename(columns={"frailty": "country_frailty"}, inplace=True)
cntry_effects = pd.DataFrame(cntry_effects)
# scatter
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=cntry_effects,
    x="days_over_30C",
    y="country_frailty",
    hue="ihme_loc_id",
    s=100,
    palette="tab20",
)
plt.xlabel("Average Days Over 30C", fontsize=14)
plt.ylabel("Country Frailty Estimate", fontsize=14)
plt.title(
    "Scatterplot of Country Frailty vs Average Days Over 30C by Country", fontsize=16
)
# plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ",")))
plt.legend(
    title="Country (ihme_loc_id)",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8,
)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "country_frailty_vs_days_over_30C_scatter.png"), dpi=300
)
plt.close()


# scatter predicted mortality against explanatory vars

px.scatter(df_model_me, y="model_predictions", x="consumption")

px.scatter(df_model_me, y="model_predictions", x="days_over_30C")


df_loc_group = (
    df_model_me.groupby(["ihme_loc_id"])
    .agg(
        {
            "model_predictions": "mean",
            # "mortality_me_manual": "mean",
            # "mortality_fe_manual": "mean",
            "days_over_30C": "mean",
            "mean_temperature": "mean",
            "total_precipitation": "mean",
            "relative_humidity": "mean",
            "mean_high_temperature": "mean",
            "mean_low_temperature": "mean",
            "precipitation_days": "mean",
            "days_over_26C": "mean",
            "consumption": "mean",
        }
    )
    .reset_index()
)
df_loc_group = df_loc_group.sort_values("model_predictions", ascending=True)
px.scatter(df_loc_group, y="model_predictions", x="ihme_loc_id")
