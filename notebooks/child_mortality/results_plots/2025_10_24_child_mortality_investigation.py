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
    heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
    )

    # Create a discrete colormap with steps matching your rounded values
    bounds = np.round(
        np.arange(round(vmin, 4), round(vmax + 0.001, 4), 0.001), 4
    )  # steps of 0.001
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    for col in columns_to_bin:
        plt.figure(figsize=(10, 8))
        heatmap_data = (
            heatmap_df.groupby(["consumption_pd", f"{col}_bin"])["model_predictions"]
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
            f"{title}\nby Consumption per Day and {col.replace('_', ' ').title()}",
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


def plot_heat_map_person_time(
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
    bounds = np.round(np.arange(0, round(vmax + 0.001, 4), 0.001), 4)  # steps of 0.001
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    for col in columns_to_bin:
        plt.figure(figsize=(10, 8))
        heatmap_data = (
            heatmap_df.groupby(["consumption", f"{col}_bin"])["model_predictions"]
            .sum()
            .unstack()
            / heatmap_df.groupby(["consumption", f"{col}_bin"])["age_month"]
            .sum()
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


def plot_heat_map_neo(
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
    bounds = np.round(
        np.arange(round(vmin, 3), round(vmax + 0.001, 3), 0.01), 4
    )  # steps of 0.01
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
# df_model = pd.read_parquet(
#     RESULTS_PATH + "predictions_subset_100pct_model_no_interaction.parquet"
# )
# df_model = pd.read_parquet(
#     RESULTS_PATH + "predictions_subset_100pct_model_10yr_cutoff.parquet"
# )

# Neonatal predictions
neonatal = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/neonatal/predictions_neonatal_logistic_interaction.parquet"
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
# df_model["mortality_fe_cum_scaled"] = df_model["mortality_fe"] / df_model["age_month"]
# df_model["mortality_me_cum_scaled"] = df_model["mortality_me"] / df_model["age_month"]
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
    # "child_mortality",
    "mortality_point_fe",
    "mortality_point_me",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

# Raw child_mortality has to be treated differently to get vmax
version = "child_mortality"
heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(
    heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)
for col in columns_to_bin:
    heatmap_data = (
        heatmap_df.groupby(["consumption", f"{col}_bin"])[version].sum().unstack()
        / heatmap_df.groupby(["consumption", f"{col}_bin"])["age_month"].sum().unstack()
    )
    vals = heatmap_data.values
    all_values.append(vals.flatten())


all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

# plot raw data heatmaps with consistent color scale
# without scaled time
plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_24_ppt_interaction_scen2",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

# plot unscaled fixed effects predictions
plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "mortality_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_24_ppt_interaction_cumsum",
    title="Modeled Child Mortality without Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot unscaled mixed effects predictions
plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "mortality_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_24_ppt_interaction_cumsum",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

####
# plot unscaled fixed effects predictions
plot_heat_map(
    data=df_model.rename(
        columns={
            "mortality_point_fe_scaled": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_24_ppt_interaction_scen2",
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
    outfile="me_heatmap_child_mortality_10_24_ppt_interaction_scen2",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

# Make plots for point estimates unscaled ######################################
heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

all_values = []
versions = [
    # "child_mortality",
    "mortality_point_fe",
    "mortality_point_me",
]
for col in columns_to_bin:
    for version in versions:
        vals = heatmap_df.groupby(["consumption", f"{col}_bin"])[version].mean().values
        all_values.append(vals)

# Raw child_mortality has to be treated differently to get vmax
version = "child_mortality"
heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(
    heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)
for col in columns_to_bin:
    heatmap_data = (
        heatmap_df.groupby(["consumption", f"{col}_bin"])[version].sum().unstack()
        / heatmap_df.groupby(["consumption", f"{col}_bin"])["age_month"].sum().unstack()
    )
    vals = heatmap_data.values
    all_values.append(vals.flatten())


all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

# plot raw data heatmaps with consistent color scale
# without scaled time
plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_24_ppt_interaction_unscaled",
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
    outfile="fe_heatmap_child_mortality_10_24_ppt_interaction_unscaled",
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
    outfile="me_heatmap_child_mortality_10_24_ppt_interaction_unscaled",
    title="Modeled Child Mortality with Random Effects (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

## Neonatal

# get min and max values for color scale consistency across plots
heatmap_df = neonatal.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
    heatmap_df.consumption_pd, 10, retbins=True
)

all_values = []
versions = [
    "child_mortality",
    "pred_fe",
    "pred_me",
]
for col in columns_to_bin:
    for version in versions:
        vals = (
            heatmap_df.groupby(["consumption_pd", f"{col}_bin"])[version].mean().values
        )
        all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=neonatal.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_neonatal_consumption_pd",
    title="Raw Data Neonatal Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=neonatal.rename(
        columns={
            "pred_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_neonatal_consumption_pd",
    title="Neonatal Modeled Child Mortality (without random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=neonatal.rename(
        columns={
            "pred_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_neonatal_consumption_pd",
    title="Neonatal Modeled Child Mortality (with random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
