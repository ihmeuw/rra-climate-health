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


DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_13.01/data_binned.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_04_13.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2026_04_17.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

## READ IN DATA

# Raw data
df_raw = pd.read_parquet(DATA_PATH)
df_raw = df_raw[df_raw["int_birth_year_diff_months"] <= 120]

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


## FUNCTIONS:
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

        x_tick_vals = heatmap_df.groupby(
            ["days_over_30C_bin"]
        ).days_over_30C.min().astype(int).values.tolist() + [
            int(heatmap_df.days_over_30C.max())
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
        ax.set_xticklabels(x_labs, rotation=45, fontsize=11)
        # ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=11)
        ax.set_xlabel("Days over 30°C", fontsize=13)
        ax.set_ylabel("Daily consumption", fontsize=13)
        ax.set_title(title, fontsize=15, pad=20)


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

# custom_x_bins = [0, 1, 14, 46, 98, 283]
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
