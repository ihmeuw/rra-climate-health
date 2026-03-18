"""
Plot code corresponding to child mortality run:

date launched: 3/3/26
spec:
model <- emfrail(Surv(age_month, child_mortality) ~ consumption_pd +
                   days_over_30C +
                   total_precipitation +
                   sex_id +
                   birth_year +
                   survival::cluster(ihme_loc_id),
                 data = df_model,
                 verbose = TRUE)
call:
sbatch -J birth_year_child_mortality --mem=900G -c 6 -A proj_rapidresponse -t 5-24 -p long.q -o /ihme/temp/slurmoutput/elyeb/output/%x.o%j -e /ihme/temp/slurmoutput/elyeb/errors/%x.e%j /ihme/singularity-images/rstudio/shells/execR.sh -s /ihme/homes/elyeb/repos/rra-climate-health/notebooks/child_mortality/child_mortality_v7_factored_birthyear.R
data: /mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_02_27.01/data.parquet

Updates to plotting code:
1. Add scaling feature to be consistent with neonatal mortality heatmaps. > Done
2. Make plot of data points per cell for raw data.
3. Make color bar same across heatmaps.

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


DATA_PATH_OLD = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data.parquet"
DATA_PATH_NEW = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_02_27.01/data.parquet"
# RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/"
# PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_24.01/"
RESULTS_PATH_OLD = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_02_27.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2026_02_27.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

## READ IN DATA

# Raw data
old_raw = pd.read_parquet(DATA_PATH_OLD)
new_raw = pd.read_parquet(DATA_PATH_NEW)

# Modeled data

df_model = pd.read_parquet(
    RESULTS_PATH + "predictions_cm_v7_factored_birth_year_means.parquet"
)
df_model_old = pd.read_parquet(
    RESULTS_PATH_OLD + "predictions_cm_v7_repredicted.parquet"
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
        # plt.savefig(os.path.join(PLOT_PATH, f"{outfile}_{col}.png"))
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

custom_x_bins = [0, 1, 14, 46, 98, 283]

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

# Make plots for cumulative estimates ###############################################
# Define custom bin edges

multiply_by_val = 1000  # for easier to read heatmaps

heatmap_df = df_model.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
    # heatmap_df[f"{col}_bin"] = pd.cut(heatmap_df[col], bins=10, include_lowest=True)
heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
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
            heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])[version]
            .sum()
            .unstack()
            / heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["age_month"]
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
    outfile="raw_heatmap_child_mortality_2026_02_27_ppt",
    title="",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=1000,
    vmin=vmin,  # 0.5
    vmax=vmax,  # 4
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "cumhaz_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_child_mortality_2026_02_27",
    title="",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=1000,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

plot_heat_map_person_time(
    data=df_model.rename(
        columns={
            "cumhaz_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_child_mortality_2026_02_27_means",
    title="",
    bin_cols=columns_to_bin,
    format=".2f",
    multiply_by=1000,
    vmin=vmin,
    vmax=vmax,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

model_name = "Child Mortality"
versions = [
    ("child_mortality", "Child Mortality Data"),
    ("cumhaz_me", "Predicted ME"),
    ("cumhaz_fe", "Predicted FE"),
]

# Create side-by-side heatmaps
# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "child_mortality_version_2026_02_27.pdf")
with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

    for col, (version, version_label) in enumerate(versions):
        # Prepare the data for the current model and version
        data = df_model.rename(columns={version: "model_predictions"})

        """
        model = models[1]
        model_name = '1-month'
        data = model.rename(columns={version: "model_predictions"})
        bin_cols=[bin_col_dict[model_name]]
        custom_bins=custom_bins_fixed
        """
        # custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

        # Plot on the specific Axes
        plot_heat_map_person_time_grid(
            data=data,
            bin_cols=columns_to_bin,
            ax=axes[col],
            title=f"{model_name} - {version_label}",
            multiply_by=multiply_by_val,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=(col == 2),  # Show colorbar only for the last column
            x_bins=custom_x_bins,
            y_bins=custom_y_bins,
        )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")


# Make heatmaps of data points per cell for raw data


# get vmax and vmin
all_maxes = []
all_mins = []

heatmap_df = old_raw.copy()

col = columns_to_bin[0]
heatmap_df[f"{col}_bin"] = pd.cut(
    heatmap_df[col], bins=custom_x_bins, include_lowest=True, right=False
)

heatmap_df["consumption_pd_bin"] = pd.cut(
    heatmap_df.consumption_pd,
    bins=custom_y_bins,
    include_lowest=True,
    right=False,
)

heatmap_data = (
    heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["child_mortality"]
    .size()
    .unstack()
)

all_maxes.append(heatmap_data.values.max())
all_mins.append(heatmap_data.values.min())

# repeat for new:
heatmap_df = new_raw.copy()

col = columns_to_bin[0]
heatmap_df[f"{col}_bin"] = pd.cut(
    heatmap_df[col], bins=custom_x_bins, include_lowest=True, right=False
)

heatmap_df["consumption_pd_bin"] = pd.cut(
    heatmap_df.consumption_pd,
    bins=custom_y_bins,
    include_lowest=True,
    right=False,
)

heatmap_data = (
    heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"])["child_mortality"]
    .size()
    .unstack()
)

all_maxes.append(heatmap_data.values.max())
all_mins.append(heatmap_data.values.min())


vmin = min(all_mins)
vmax = max(all_maxes)


plot_data_pts_per_cell(
    data=old_raw.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="data_pts_per_cell_heatmap_child_mortality_version_10_30",
    title="Data Points per Cell - Version 10/30/2025",
    vmin=vmin,
    vmax=vmax,
    bin_cols=columns_to_bin,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)

plot_data_pts_per_cell(
    data=new_raw.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="data_pts_per_cell_heatmap_child_mortality_version_03_06",
    title="Data Points per Cell - Version 03/06/2026",
    vmin=vmin,
    vmax=vmax,
    bin_cols=columns_to_bin,
    show_colorbar=True,
    x_bins=custom_x_bins,
    y_bins=custom_y_bins,
)


## Make coefficient table
# make table of summaries
SUMMARY_DIR = RESULTS_PATH + "model_summaries/"
os.listdir(SUMMARY_DIR)
coef_txt_file = "cm_v7_factored_birth_year.txt"
coef_outfile_name = "child_mortality_3_6_26_coefs.csv"


climate_var_interest = "days_over_30C"
vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "days_over_30C",
    "total_precipitation",
]

results_table = pd.DataFrame(columns=["Variable", "Estimate", "p_value"])
f = coef_txt_file

import re


def asterix_from_p_value(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


with open(SUMMARY_DIR + f, "r") as infile:
    s = infile.read().split("\n")

for l in s:
    if l.startswith(tuple(vars_of_interest)):
        # test if all info on single line or if it overflowed:
        if len(l.split()) == 5:
            coef = l.split()[0]
            estimate = l.split()[1]

            results_table = pd.concat(
                [
                    results_table,
                    pd.DataFrame.from_records(
                        [
                            {
                                "Variable": coef,
                                "Estimate": estimate,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        elif len(l.split()) == 3:  # variable and p-val
            coef = l.split()[0]
            # significance = re.findall(r"[\d]{1}\.[\d]*", l)[0]
            p_value = l.split()[2]
            results_table.loc[results_table["Variable"] == coef, "p_value"] = p_value

results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Significance"] = results_table["p_value"].apply(asterix_from_p_value)

results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(4)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table["Estimate"] = results_table["Estimate"] + results_table["Significance"]

results_table.drop(columns=["p_value", "Significance"], inplace=True)
results_table["Variable"].unique()

var_order = [
    "sex_idFemale",
    "consumption_pd",
    climate_var_interest,
    "total_precipitation",
]

results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)


results_table.to_csv(PLOT_PATH + coef_outfile_name, index=False)


# Replot old results using same min/max values
# Create side-by-side heatmaps
# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "child_mortality_version_2025_10_30_replotted.pdf")
with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

    for col, (version, version_label) in enumerate(versions):
        # Prepare the data for the current model and version
        data = df_model_old.rename(columns={version: "model_predictions"})

        """
        model = models[1]
        model_name = '1-month'
        data = model.rename(columns={version: "model_predictions"})
        bin_cols=[bin_col_dict[model_name]]
        custom_bins=custom_bins_fixed
        """
        # custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

        # Plot on the specific Axes
        plot_heat_map_person_time_grid(
            data=data,
            bin_cols=columns_to_bin,
            ax=axes[col],
            title=f"{model_name} - {version_label}",
            multiply_by=multiply_by_val,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=(col == 2),  # Show colorbar only for the last column
            x_bins=custom_x_bins,
            y_bins=custom_y_bins,
        )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")


## Make before and after data maps

# Get locations for shape files
cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "lbw")
loc_meta = cm_data.load_fhs_hierarchy()

global_vmin = 1
global_vmax = 9


gbd_shapefile_path = Path(
    "/snfs1/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2021/master/shapefiles/GBD2021_mapping_final.shp"
)
gbd_shapefile = gpd.read_file(gbd_shapefile_path)
gbd_shapefile = gbd_shapefile[
    gbd_shapefile["loc_id"].isin(loc_meta.query("level == 3").location_id.unique())
]

other_shapefile = gpd.read_file(
    "/snfs1/WORK/11_geospatial/admin_shapefiles/2023_10_30/lbd_standard_admin_0_simplified.shp"
)
other_shapefile = other_shapefile[
    ~other_shapefile["loc_id"].isin(loc_meta.location_id.unique())
]
other_shapefile = other_shapefile.rename(columns={"ADM0_NAME": "loc_name"})
other_shapefile["parent_id"] = 1
other_shapefile["level"] = 3
shapefile = pd.concat([gbd_shapefile, other_shapefile])

# Before:
train = old_raw.copy()
output_path = PLOT_PATH + "child_mortality_data_availability_map_2025_10_30.pdf"


train_counts = (
    train[["nid", "ihme_loc_id", "year_start"]]
    .drop_duplicates()
    .groupby(["ihme_loc_id"])
    .agg({"nid": "count", "year_start": "count"})
)
plot_gdf = shapefile.query("level == 3")[["ihme_lc_id", "loc_name", "geometry"]].merge(
    train_counts,
    left_on="ihme_lc_id",
    right_index=True,
    how="outer",
    suffixes=("_shapefile", "_train"),
)

## Make plot:
fig, ax = plt.subplots(figsize=(10, 6))
plot_gdf.plot(
    column="nid",
    ax=ax,
    categorical=True,  # Treat the column as categorical for distinct colors/legend entries
    legend=True,
    cmap="summer",
    vmin=global_vmin,
    vmax=global_vmax,
    missing_kwds={
        "color": "lightgray",
        "label": "NA",  # This label will appear in the legend
    },
    legend_kwds={
        "title": "Number of Surveys Extracted",  # Title for the legend
        "title_fontsize": "14",  # Font size for the legend title
        "fontsize": "14",  # Font size for legend items
        # 'frameon': False,         # No frame around the legend
        # --- Add these lines for positioning and columns ---
        "loc": "lower center",  # Anchor legend at its lower-center
        "bbox_to_anchor": (0.5, -0.30),  # Position legend: 0.5 for horizontal center,
        # -0.15 to place it below the plot. Adjust -0.15 as needed.
        "ncol": 4,  # Number of columns for legend items
        # --- End of added lines ---
    },
)
legend = ax.get_legend()
if legend:  # Check if legend exists
    for text_obj in legend.get_texts():
        current_label = text_obj.get_text()
        # Check if the current label is NOT the one used for missing/NA values
        if current_label != "NA":  # Use the exact string from missing_kwds['label']
            try:
                # Convert to float first (in case it's "1.0"), then to int, then to string
                formatted_label = str(int(float(current_label)))
                text_obj.set_text(formatted_label)
            except ValueError:
                # If conversion fails (e.g., it's already some other string), leave it as is
                pass
ax.set_axis_off()
ax.set_title(f"Child Mortality Survey Data Availability, Previous Version", fontsize=18)

# Add tight_layout or subplots_adjust to make space for the legend
plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)  # rect=[left, bottom, right, top] reserves space at bottom

# Save the figure
# output_path = Path(paper_plots_loc) / f"figs1_data_availability.pdf"
# print(f"\nSaving combined plot to {output_path}")
fig.savefig(output_path, bbox_inches="tight")
print(len(train))
print(train.nid.nunique())
print(train.ihme_loc_id.nunique())


# After
train = new_raw.copy()
output_path = PLOT_PATH + "child_mortality_data_availability_map_2026_2_27.pdf"


train_counts = (
    train[["nid", "ihme_loc_id", "year_start"]]
    .drop_duplicates()
    .groupby(["ihme_loc_id"])
    .agg({"nid": "count", "year_start": "count"})
)
plot_gdf = shapefile.query("level == 3")[["ihme_lc_id", "loc_name", "geometry"]].merge(
    train_counts,
    left_on="ihme_lc_id",
    right_index=True,
    how="outer",
    suffixes=("_shapefile", "_train"),
)

## Make plot:
fig, ax = plt.subplots(figsize=(10, 6))
plot_gdf.plot(
    column="nid",
    ax=ax,
    categorical=True,  # Treat the column as categorical for distinct colors/legend entries
    legend=True,
    cmap="summer",
    vmin=global_vmin,
    vmax=global_vmax,
    missing_kwds={
        "color": "lightgray",
        "label": "NA",  # This label will appear in the legend
    },
    legend_kwds={
        "title": "Number of Surveys Extracted",  # Title for the legend
        "title_fontsize": "14",  # Font size for the legend title
        "fontsize": "14",  # Font size for legend items
        # 'frameon': False,         # No frame around the legend
        # --- Add these lines for positioning and columns ---
        "loc": "lower center",  # Anchor legend at its lower-center
        "bbox_to_anchor": (0.5, -0.30),  # Position legend: 0.5 for horizontal center,
        # -0.15 to place it below the plot. Adjust -0.15 as needed.
        "ncol": 4,  # Number of columns for legend items
        # --- End of added lines ---
    },
)
legend = ax.get_legend()
if legend:  # Check if legend exists
    for text_obj in legend.get_texts():
        current_label = text_obj.get_text()
        # Check if the current label is NOT the one used for missing/NA values
        if current_label != "NA":  # Use the exact string from missing_kwds['label']
            try:
                # Convert to float first (in case it's "1.0"), then to int, then to string
                formatted_label = str(int(float(current_label)))
                text_obj.set_text(formatted_label)
            except ValueError:
                # If conversion fails (e.g., it's already some other string), leave it as is
                pass
ax.set_axis_off()
ax.set_title(f"Child Mortality Survey Data Availability, Updated Version", fontsize=18)

# Add tight_layout or subplots_adjust to make space for the legend
plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)  # rect=[left, bottom, right, top] reserves space at bottom

# Save the figure
# output_path = Path(paper_plots_loc) / f"figs1_data_availability.pdf"
# print(f"\nSaving combined plot to {output_path}")
fig.savefig(output_path, bbox_inches="tight")
print(len(train))
print(train.nid.nunique())
print(train.ihme_loc_id.nunique())
