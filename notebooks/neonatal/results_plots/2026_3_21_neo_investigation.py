"""
Plot neonatal mortality model latest

"""

import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import re
import os
from tqdm import tqdm


RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2026_03_19.04/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2026_03_19.04/plots/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)


## FUNCTIONS


def plot_heat_map(
    data: pd.DataFrame,
    # outfile: str,
    title: str,
    bin_cols: list,
    format: str = ".2f",
    multiply_by: int = 1,
    vmin: float = None,
    vmax: float = None,
    show_colorbar: bool = True,
    custom_bins: list = None,
    custom_y_bins: list = None,
):
    """
    data = model6.copy()
    data.rename(columns={
        "child_mortality": "model_predictions",
    }, inplace=True)
    outfile="raw_heatmap_neonatal_11_21"
    title=""
    bin_cols=["days_over_30C_prev_6_mo_avg"]
    format=".2f"
    multiply_by=multiply_by_val
    vmin=vmin
    vmax=vmax
    show_colorbar=False
    col = "days_over_30C_prev_6_mo_avg"
    """
    heatmap_df = data.copy()
    for col in bin_cols:
        new_bin_col = f"{col}_bin"
        # heatmap_df[new_bin_col] = pd.qcut(
        #     heatmap_df[col], 10, retbins=False, duplicates="drop"
        # )
        if custom_bins is not None:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=custom_bins, include_lowest=True, right=False
            )
        else:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=10, include_lowest=True
            )
    if custom_y_bins is None:
        heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
            heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
        )
    else:
        heatmap_df["consumption_pd_bin"] = pd.cut(
            heatmap_df.consumption_pd,
            bins=custom_y_bins,
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

        x_tick_vals = heatmap_df.groupby([new_bin_col])[col].min().values.tolist() + [
            heatmap_df.groupby([new_bin_col])[col].max().max()
        ]
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

        # plt.tight_layout()
        # plt.savefig(os.path.join(PLOT_PATH, f"{outfile}_{col}.png"))
        # plt.close()
        plt.show()


def plot_heat_map_grid(
    data: pd.DataFrame,
    bin_cols: list,
    ax=None,
    title: str = "",
    format: str = ".2f",
    multiply_by: int = 1,
    vmin: float = None,
    vmax: float = None,
    show_colorbar: bool = True,
    custom_bins: list = None,
    custom_y_bins: list = None,
    y_axis_label=None,
    x_axis_label=None,
):
    """
    Modified plot_heat_map to return the Axes object for use in subplots.
    data=data
    bin_cols=[bin_col_dict[model_name]]
    ax=axes[col]
    title=f"{model_name} - {version_label}"
    multiply_by=multiply_by_val
    vmin=vmin
    vmax=vmax
    show_colorbar=(col == 2) # Show colorbar only for the last column
    custom_bins=custom_bins_fixed  # custom_bins_for_model,
    custom_y_bins=custom_y_bins
    """
    heatmap_df = data.copy()
    for col in bin_cols:
        new_bin_col = f"{col}_bin"
        if custom_bins is not None:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=custom_bins, include_lowest=True, right=False
            )
        else:
            heatmap_df[f"{col}_bin"] = pd.cut(
                heatmap_df[col], bins=10, include_lowest=True
            )
    if custom_y_bins is None:
        heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
            heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
        )
    else:
        heatmap_df["consumption_pd_bin"] = pd.cut(
            heatmap_df.consumption_pd,
            bins=custom_y_bins,
            include_lowest=True,
            right=False,
        )

    for col in bin_cols:
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

        # Create a new figure if no Axes object is provided
        if ax is None:
            figsize = (6, 6) if not show_colorbar else (7, 6)
            fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=format,
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            cbar=show_colorbar,
            ax=ax,
            annot_kws={"size": 8, "weight": "regular"},
        )

        # Set tick values and labels
        # x_tick_vals = heatmap_df.groupby([new_bin_col])[col].min().values.tolist() + [
        #     heatmap_df.groupby([new_bin_col])[col].max().max()
        # ]
        # x_ticks = range(len(x_tick_vals))
        # x_labs = [f"{x_tick_vals[i]:.1f}" for i in range(len(x_tick_vals))]
        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels(x_labs, rotation=45, ha="right", fontsize=10)
        x_tick_vals = custom_bins_fixed  # Use the precomputed bin edges
        x_ticks = range(len(x_tick_vals))
        x_labs = [f"{x_tick_vals[i]:.1f}" for i in range(len(x_tick_vals))]

        # Set the x-axis ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labs, rotation=45, ha="right", fontsize=10)

        y_tick_vals = heatmap_df.groupby(
            ["consumption_pd_bin"]
        ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]
        y_ticks = range(len(y_tick_vals))
        if custom_y_bins is None:
            y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]
        else:
            y_labs = [f"{i:.1f}" for i in custom_y_bins]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=10)

        # Set axis labels and title
        if x_axis_label is None:
            ax.set_xlabel(new_bin_col, fontsize=11)
        else:
            ax.set_xlabel(x_axis_label, fontsize=13)
        if y_axis_label is None:
            ax.set_ylabel("Daily Consumption (Binned)", fontsize=13)
        else:
            ax.set_ylabel(y_axis_label, fontsize=13)
        ax.set_title(title, fontsize=15, pad=20)

        # Format the colorbar
        # if show_colorbar:
        #     cbar = ax.collections[0].colorbar
        #     cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    return ax


def create_custom_bins(data: pd.DataFrame, col: str):
    """
    Create custom bins for a given column in the data.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        col (str): The column name for which to create custom bins.

    Returns:
        list: Custom bin edges.
    """
    # Extract the column
    column_data = data[col]

    # Separate zero and non-zero values
    zero_values = column_data[column_data == 0]  # All zero values
    non_zero_values = column_data[column_data > 0]  # All non-zero values

    # Get the first non-zero value
    first_non_zero = non_zero_values.min()

    # Calculate quartiles for non-zero values
    quartiles = np.percentile(non_zero_values, [25, 50, 75, 100])

    # Define custom bin edges
    custom_bins = [0, first_non_zero] + list(quartiles)

    return custom_bins


################################################################################

# 1. Scam package  ##############################################

model1 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_1_mo_do30_scam_summarywith_psu.parquet"
)

# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    # "q9_prev_0_mo",
    # "q9_prev_3_mo_avg",
    # "q9_prev_6_mo_avg",
    # "q9_prev_9_mo_avg",
    # "q95_prev_0_mo",
    # "q95_prev_3_mo_avg",
    # "q95_prev_6_mo_avg",
    # "q95_prev_9_mo_avg",
    "days_over_30C_prev_0_mo"
]

# data = pd.concat(
#     [
#         model1["q9_prev_0_mo"],
#         model3["q9_prev_3_mo_avg"],
#         model6["q9_prev_6_mo_avg"],
#         model9["q9_prev_9_mo_avg"],
#     ],
#     ignore_index=True,
# )


# # Calculate the frequency of each unique value
# value_counts = data.value_counts(normalize=True) * 100  # Normalize to get percentages

# # Sort the values for better visualization
# value_counts = value_counts.sort_index()

# # make custom bins for days-over30:
# # first will be 0 to the first non-zero value
# # the next 4 bins will be quartiles of the non-zero values
# # Separate zero and non-zero values
# zero_values = data[data == 0]  # All zero values
# non_zero_values = data[data > 0]  # All non-zero values

# # Get the first non-zero value
# first_non_zero = non_zero_values.min()

# # Calculate quartiles for non-zero values
# quartiles = np.percentile(non_zero_values, [25, 50, 75, 100])

# Define custom bin edges
custom_bins_fixed = [0, 0.1, 2, 4, 9, 31]

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
# get mix/maxes for plots
multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for model in [model1]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.cut(
            heatmap_df[col], bins=custom_bins_fixed, include_lowest=True, right=False
        )
    # heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
    #     heatmap_df.consumption_pd, 10, retbins=True
    # )
    heatmap_df["consumption_pd"] = pd.cut(
        heatmap_df.consumption_pd, bins=custom_y_bins, include_lowest=True, right=False
    )
    versions = [
        "child_mortality",
        "pred_fe",
        "pred_me",
    ]
    for col in columns_to_bin:
        for version in versions:
            vals = (
                heatmap_df.groupby(["consumption_pd", f"{col}_bin"])[version]
                .mean()
                .values
            )
            all_values.append(vals)

all_values = np.concatenate(all_values)

vmin = all_values.min()
vmax = all_values.max()

vmin *= multiply_by_val
vmax *= multiply_by_val
print(vmax)

# Plot all on same PDF
# Define models and versions
model = model1.copy()
model_name = "1-month"
versions = [
    ("child_mortality", "Child Mortality"),
    ("pred_me", "Predicted ME"),
    ("pred_fe", "Predicted FE"),
]


bin_col_dict = {
    "1-month": "days_over_30C_prev_0_mo"
    # "1-month": "q95_prev_0_mo",
    # "3-month": "q95_prev_3_mo_avg",
    # "6-month": "q95_prev_6_mo_avg",
    # "9-month": "q95_prev_9_mo_avg",
}


# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_do30_scam_v1.pdf")
with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

    for col, (version, version_label) in enumerate(versions):
        # Prepare the data for the current model and version
        data = model.rename(columns={version: "model_predictions"})

        """
        model = models[1]
        model_name = '1-month'
        data = model.rename(columns={version: "model_predictions"})
        bin_cols=[bin_col_dict[model_name]]
        custom_bins=custom_bins_fixed
        """
        # custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

        # Plot on the specific Axes
        plot_heat_map_grid(
            data=data,
            bin_cols=[bin_col_dict[model_name]],
            ax=axes[col],
            title=f"{model_name} - {version_label}",
            multiply_by=multiply_by_val,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=(col == 2),  # Show colorbar only for the last column
            custom_bins=custom_bins_fixed,  # custom_bins_for_model,
            custom_y_bins=custom_y_bins,
        )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")
