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

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_24.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)


## FUNCTIONS:


def plot_heat_map(
    data: pd.DataFrame,
    outfile: str,
    title: str,
    bin_cols: list,
    format: str = ".2f",
    multiply_by: int = 1,
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
    # bounds = np.round(
    #     np.arange(round(vmin, 4), round(vmax + 0.001, 4), 0.001), 4
    # )  # steps of 0.001
    # norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    for col in columns_to_bin:
        plt.figure(figsize=(10, 8))
        heatmap_data = (
            heatmap_df.groupby(["consumption_pd", f"{col}_bin"])["model_predictions"]
            .mean()
            .unstack()
        )

        heatmap_data *= multiply_by_val

        if vmin and vmax:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
                vmin=vmin,
                vmax=vmax,
                # norm=norm,
            )
        else:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
            )
        if multiply_by > 1:
            plt.title(
                f"{title}\nby Consumption per Day and {col.replace('_', ' ').title()}, (x{multiply_by:,})",
                fontsize=20,
            )
        else:
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
            [f"{y.left:.1f}–{y.right:.1f}" for y in heatmap_data.index],
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
    multiply_by: int = 1,
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
    # bounds = np.round(np.arange(0, round(vmax + 0.001, 4), 0.001), 4)  # steps of 0.001
    # norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    for col in columns_to_bin:
        plt.figure(figsize=(10, 8))
        heatmap_data = (
            heatmap_df.groupby(["consumption_pd", f"{col}_bin"])["model_predictions"]
            .sum()
            .unstack()
            / heatmap_df.groupby(["consumption_pd", f"{col}_bin"])["age_month"]
            .sum()
            .unstack()
        )

        heatmap_data *= multiply_by

        if vmin and vmax:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
                vmin=vmin,
                vmax=vmax,
                # norm=norm,
            )
        else:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
            )
        if multiply_by > 1:
            plt.title(
                f"{title}\nby Consumption per Day and {col.replace('_', ' ').title()}, (x{multiply_by:,})",
                fontsize=20,
            )
        else:
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
            [f"{y.left:.1f}–{y.right:.1f}" for y in heatmap_data.index],
            rotation=0,
            fontsize=10,
        )
        ax1.set_xlabel("Binned " + col.replace("_", " ").title(), fontsize=16)
        ax1.set_ylabel("Consumption per Day Bin", fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"{outfile}_{col}.png"))
        plt.close()


## READ IN DATA

# Raw data
df = pd.read_parquet(DATA_PATH)

# Modeled data

# df_model = pd.read_parquet(RESULTS_PATH + "predictions_cm_v7_filtered.parquet")
# df_model = pd.read_parquet(RESULTS_PATH + "predictions_cm_v7_subset.parquet")
df_model = pd.read_parquet(RESULTS_PATH + "predictions_cm_v3_subset.parquet")


# Neonatal predictions
neonatal = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/neonatal/predictions_nm_v3.parquet"
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


## MAKE SIDE-BY-SIDE HEATMAPS TOGETHER #########################################

# Make plots for cumulative estimates ###############################################

multiply_by_val = 1000  # for easier to read heatmaps

heatmap_df = df_model.copy()
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
    "cumhaz_fe",
    "cumhaz_me",
]
for col in columns_to_bin:
    for version in versions:
        heatmap_data = (
            heatmap_df.groupby(["consumption_pd", f"{col}_bin"])[version]
            .sum()
            .unstack()
            / heatmap_df.groupby(["consumption_pd", f"{col}_bin"])["age_month"]
            .sum()
            .unstack()
        )
        vals = heatmap_data.values
        all_values.append(vals.flatten())

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

vmin *= multiply_by_val
vmax *= multiply_by_val

# plot raw data heatmaps with consistent color scale
# without scaled time
plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_28_ppt",
    title="Raw Data Child Mortality (per Person-Time)",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=1000,
    vmin=vmin,  # 0.5
    vmax=vmax,  # 4
)


plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "cumhaz_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_10_29_ppt_v3",
    title="Modeled Child Mortality with Random Effects",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=1000,
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "cumhaz_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_10_29_ppt_v3",
    title="Modeled Child Mortality without Random Effects",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=1000,
    vmin=vmin,
    vmax=vmax,
)


## Neonatal

# get min and max values for color scale consistency across plots

multiply_by_val = 1000  # for easier to read heatmaps


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

vmin *= multiply_by_val
vmax *= multiply_by_val


plot_heat_map(
    data=neonatal.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_neonatal_consumption_pd_10_29",
    title="Raw Data Neonatal Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=neonatal.rename(
        columns={
            "pred_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_neonatal_consumption_pd_10_29_v3",
    title="Neonatal Modeled Child Mortality (without random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=neonatal.rename(
        columns={
            "pred_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_neonatal_consumption_pd_10_29_v3",
    title="Neonatal Modeled Child Mortality (with random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
)
