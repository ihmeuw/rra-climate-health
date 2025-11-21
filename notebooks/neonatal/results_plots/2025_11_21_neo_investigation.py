import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

import os

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2025_11_20.01/neonatal/neonatal_data.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_11_20.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/plots/2025_11_20.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

## READ IN DATA

# Raw data
data = pd.read_parquet(DATA_PATH)

# Modeled data

# Neonatal predictions
model = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_015_3_mo_sensititivy_model_summary.parquet"
)


## FUNCTIONS


def plot_heat_map(
    data: pd.DataFrame,
    outfile: str,
    title: str,
    bin_cols: list,
    format: str = ".2f",
    multiply_by: int = 1,
    vmin: float = None,
    vmax: float = None,
    show_colorbar: bool = True,
):
    """
    data = model.copy()
    data.rename(columns={
        "child_mortality": "model_predictions",
    }, inplace=True)
    outfile="raw_heatmap_neonatal_11_21"
    title=""
    bin_cols=columns_to_bin
    format=".2f"
    multiply_by=multiply_by_val
    vmin=vmin
    vmax=vmax
    show_colorbar=False
    """
    heatmap_df = data.copy()
    for col in bin_cols:
        new_bin_col = f"{col}_bin"
        # heatmap_df[new_bin_col] = pd.qcut(
        #     heatmap_df[col], 10, retbins=False, duplicates="drop"
        # )
        heatmap_df[f"{col}_bin"] = pd.cut(heatmap_df[col], bins=10, include_lowest=True)
    heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
    )

    for col in bin_cols:
        figsize = (6, 6) if not show_colorbar else (7, 6)
        plt.figure(figsize=figsize)

        heatmap_data = (
            heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[
                "model_predictions"
            ]
            .mean()
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

        x_tick_vals = heatmap_df.groupby([new_bin_col])[col].min().astype(
            int
        ).values.tolist() + [int(heatmap_df.groupby([new_bin_col])[col].max().max())]
        # y_tick_vals = heatmap_df.groupby(
        #     ["consumption_pd_bin"]
        # ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]

        # Generate tick values for y-axis (both edges of the bins)
        y_tick_vals = heatmap_df.groupby(
            ["consumption_pd_bin"]
        ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]
        x_ticks = range(len(x_tick_vals) + 1)
        # y_ticks = range(len(y_tick_vals) + 1)
        y_ticks = range(len(y_tick_vals))
        x_labs = [f"{x_tick_vals[i]}" for i in range(len(x_tick_vals))]
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
        ax.set_xticks(x_ticks)
        n_rows, n_cols = heatmap_data.shape
        ax.set_xticks(np.arange(n_cols + 1))
        ax.set_yticks(np.arange(n_rows + 1))
        ax.set_xticklabels(x_labs, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=10)
        ax.set_xlabel(new_bin_col, fontsize=13)
        ax.set_ylabel("Daily consumption", fontsize=13)
        ax.set_title(title, fontsize=18)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"{outfile}_{col}.png"))
        plt.close()


## CONSTANTS

# Heat maps of variables
columns_to_bin = [
    # "mean_temperature",
    # "total_precipitation",
    # "relative_humidity",
    # "mean_high_temperature",
    # "mean_low_temperature",
    # "precipitation_days",
    # "days_over_30C",
    # "days_over_26C",
    "days_over_30C_prev_3_mo_avg"
]


## MAKE SIDE-BY-SIDE HEATMAPS TOGETHER #########################################

# Make plots for cumulative estimates ###############################################

## Neonatal

# get min and max values for color scale consistency across plots

multiply_by_val = 1000  # for easier to read heatmaps


heatmap_df = model.copy()
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
    data=model.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_neonatal_11_21",
    title="",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
)


plot_heat_map(
    data=model.rename(
        columns={
            "pred_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_neonatal_11_21",
    title="",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
)

plot_heat_map(
    data=model.rename(
        columns={
            "pred_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_neonatal_11_21",
    title="",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
)
###############################################################################
