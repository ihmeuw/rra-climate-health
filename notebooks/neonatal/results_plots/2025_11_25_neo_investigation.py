"""
Plot different time horizons side-by-side for neonatal mortality model.
"""

import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

import os

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2025_11_20.01/neonatal/neonatal_data.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_11_20.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/plots/2025_11_20.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

## READ IN DATA

# Raw data
data = pd.read_parquet(DATA_PATH)


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
    custom_bins: list = None,
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
):
    """
    Modified plot_heat_map to return the Axes object for use in subplots.
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
    heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
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
        x_tick_vals = heatmap_df.groupby([new_bin_col])[col].min().values.tolist() + [
            heatmap_df.groupby([new_bin_col])[col].max().max()
        ]
        y_tick_vals = heatmap_df.groupby(
            ["consumption_pd_bin"]
        ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]

        x_ticks = range(len(x_tick_vals))
        y_ticks = range(len(y_tick_vals))
        x_labs = [f"{x_tick_vals[i]:.1f}" for i in range(len(x_tick_vals))]
        y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labs, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=10)

        # Set axis labels and title
        ax.set_xlabel(new_bin_col, fontsize=11)
        ax.set_ylabel("Daily Consumption (Binned)", fontsize=13)
        ax.set_title(title, fontsize=15)

        # Format the colorbar
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    return ax


def plot_data_pts_grid(
    data: pd.DataFrame,
    bin_cols: list,
    ax=None,
    title: str = "",
    vmin: float = None,
    vmax: float = None,
    show_colorbar: bool = True,
):
    """
    Modified plot_heat_map to return the Axes object for use in subplots.

    data = model6.rename(columns={version: "model_predictions"})
    model_name = "6-month"
    bin_cols=[bin_col_dict[model_name]]
    """
    heatmap_df = data.copy()
    for col in bin_cols:
        new_bin_col = f"{col}_bin"
        heatmap_df[f"{col}_bin"] = pd.cut(heatmap_df[col], bins=10, include_lowest=True)
    heatmap_df["consumption_pd_bin"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True, duplicates="drop"
    )

    for col in bin_cols:
        heatmap_data = (
            heatmap_df.groupby(["consumption_pd_bin", f"{col}_bin"]).size().unstack()
        )

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
            fmt="d",
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            cbar=show_colorbar,
            ax=ax,
            annot_kws={"size": 6, "weight": "regular"},
        )

        # Set tick values and labels
        x_tick_vals = heatmap_df.groupby([new_bin_col])[col].min().astype(
            int
        ).values.tolist() + [int(heatmap_df.groupby([new_bin_col])[col].max().max())]
        y_tick_vals = heatmap_df.groupby(
            ["consumption_pd_bin"]
        ).consumption_pd.min().values.tolist() + [heatmap_df.consumption_pd.max()]

        x_ticks = range(len(x_tick_vals))
        y_ticks = range(len(y_tick_vals))
        x_labs = [f"{x_tick_vals[i]}" for i in range(len(x_tick_vals))]
        y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labs, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labs, rotation=0, fontsize=10)

        # Set axis labels and title
        ax.set_xlabel(new_bin_col, fontsize=11)
        ax.set_ylabel("Daily Consumption (Binned)", fontsize=13)
        ax.set_title(title, fontsize=15)

        # Format the colorbar
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

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


# Plot updated days over 30 with 50% of data ###################################
# Modeled data

# Neonatal predictions
model1 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_1_mo_model_summary.parquet"
)
model3 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_3_mo_model_summary.parquet"
)
model6 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_6_mo_model_summary.parquet"
)
model9 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_9_mo_model_summary.parquet"
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
    # "days_over_30C",
    # "days_over_26C",
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
]


# get min and max values for color scale consistency across plots

# Explore bins
# Extract the column
col = "days_over_30C_prev_6_mo_avg"
data = model6[col]  # Replace `model6` with the appropriate DataFrame

# Calculate the frequency of each unique value
value_counts = data.value_counts(normalize=True) * 100  # Normalize to get percentages

# Sort the values for better visualization
value_counts = value_counts.sort_index()

# make custom bins for days-over30:
# first will be 0 to the first non-zero value
# the next 4 bins will be quartiles of the non-zero values
# Separate zero and non-zero values
zero_values = data[data == 0]  # All zero values
non_zero_values = data[data > 0]  # All non-zero values

# Get the first non-zero value
first_non_zero = non_zero_values.min()

# Calculate quartiles for non-zero values
quartiles = np.percentile(non_zero_values, [25, 50, 75, 100])

# Define custom bin edges
custom_bins = [0, first_non_zero] + list(quartiles)

# get mix/maxes for plots
multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.cut(
            heatmap_df[col], bins=custom_bins, include_lowest=True, right=False
        )
    heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True
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


plot_heat_map(
    data=model3.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_neonatal_11_24",
    title="",
    bin_cols=["days_over_30C_prev_3_mo_avg"],
    format=".2f",
    multiply_by=multiply_by_val,
    # vmin=vmin,
    # vmax=vmax,
    show_colorbar=False,
)


plot_heat_map(
    data=model3.rename(
        columns={
            "pred_me": "model_predictions",
        }
    ),
    outfile="me_heatmap_neonatal_11_24",
    title="",
    bin_cols=["days_over_30C_prev_3_mo_avg"],
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
    show_colorbar=False,
)

plot_heat_map(
    data=model3.rename(
        columns={
            "pred_fe": "model_predictions",
        }
    ),
    outfile="fe_heatmap_neonatal_11_24",
    title="",
    bin_cols=["days_over_30C_prev_3_mo_avg"],
    format=".2f",
    multiply_by=multiply_by_val,
    vmin=vmin,
    vmax=vmax,
)

# Plot all on same PDF
# Define models and versions
models = [model1, model3, model6, model9]
model_names = ["1-month", "3-month", "6-month", "9-month"]
versions = [
    ("child_mortality", "Child Mortality"),
    ("pred_me", "Predicted ME"),
    ("pred_fe", "Predicted FE"),
]

bin_col_dict = {
    "1-month": "days_over_30C_prev_0_mo",
    "3-month": "days_over_30C_prev_3_mo_avg",
    "6-month": "days_over_30C_prev_6_mo_avg",
    "9-month": "days_over_30C_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_100pc_time_comparisons.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
        for col, (version, version_label) in enumerate(versions):
            # Prepare the data for the current model and version
            data = model.rename(columns={version: "model_predictions"})

            custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

            # Plot on the specific Axes
            plot_heat_map_grid(
                data=data,
                bin_cols=[bin_col_dict[model_name]],
                ax=axes[row, col],
                title=f"{model_name} - {version_label}",
                multiply_by=multiply_by_val,
                vmin=vmin,
                vmax=vmax,
                show_colorbar=(col == 2),  # Show colorbar only for the last column
                custom_bins=custom_bins_for_model,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

################################################################################

# Plot mean temperature ########################################################

# Neonatal predictions
model1 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_1_mo_mean_temp_summary.parquet"
)
model3 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_3_mo_mean_temp_summary.parquet"
)
model6 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_6_mo_mean_temp_summary.parquet"
)
model9 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_9_mo_mean_temp_summary.parquet"
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
    # "days_over_30C",
    # "days_over_26C",
    "mean_temperature_prev_0_mo",
    "mean_temperature_prev_3_mo_avg",
    "mean_temperature_prev_6_mo_avg",
    "mean_temperature_prev_9_mo_avg",
]


# get min and max values for color scale consistency across plots

multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.qcut(
            heatmap_df[col], 10, retbins=False, duplicates="drop"
        )
    heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True
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


# Plot all on same PDF
# Define models and versions
models = [model1, model3, model6, model9]
model_names = ["1-month", "3-month", "6-month", "9-month"]
versions = [
    ("child_mortality", "Child Mortality"),
    ("pred_me", "Predicted ME"),
    ("pred_fe", "Predicted FE"),
]

bin_col_dict = {
    "1-month": "mean_temperature_prev_0_mo",
    "3-month": "mean_temperature_prev_3_mo_avg",
    "6-month": "mean_temperature_prev_6_mo_avg",
    "9-month": "mean_temperature_prev_9_mo_avg",
}


# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_full_mean_temp_time_comparisons.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
        for col, (version, version_label) in enumerate(versions):
            # Prepare the data for the current model and version
            data = model.rename(columns={version: "model_predictions"})

            # Plot on the specific Axes
            plot_heat_map_grid(
                data=data,
                bin_cols=[bin_col_dict[model_name]],
                ax=axes[row, col],
                title=f"{model_name} - {version_label}",
                multiply_by=multiply_by_val,
                vmin=vmin,
                vmax=vmax,
                show_colorbar=(col == 2),  # Show colorbar only for the last column
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")


# plot data points heatmap to see distribution of data points ##################

all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.qcut(
            heatmap_df[col], 10, retbins=False, duplicates="drop"
        )
    heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True
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
                .size()
                .values
            )
            all_values.append(vals)

all_values = np.concatenate(all_values)


vmin = all_values.min()
vmax = all_values.max()

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_mean_temp_data_points.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 1 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=1, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
        version = "child_mortality"
        version_label = "Child Mortality"

        # Prepare the data for the current model and version
        data = model.rename(columns={version: "model_predictions"})

        # Plot on the specific Axes
        plot_data_pts_grid(
            data=data,
            bin_cols=[bin_col_dict[model_name]],
            ax=axes[row],
            title=f"{model_name} - {version_label}",
            vmin=vmin,
            vmax=vmax,
            show_colorbar=True,  # Show colorbar only for the last column
        )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

################################################################################


# Plot days over 28 ############################################################

# Neonatal predictions
model1 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_1_mo_do28_summary.parquet"
)
model3 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_3_mo_do28_summary.parquet"
)
model6 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_6_mo_do28_summary.parquet"
)
model9 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_9_mo_do28_summary.parquet"
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
    # "days_over_30C",
    # "days_over_26C",
    "days_over_28C_prev_0_mo",
    "days_over_28C_prev_3_mo_avg",
    "days_over_28C_prev_6_mo_avg",
    "days_over_28C_prev_9_mo_avg",
]


# get min and max values for color scale consistency across plots

multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.qcut(
            heatmap_df[col], 10, retbins=False, duplicates="drop"
        )
    heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True
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


# Plot all on same PDF
# Define models and versions
models = [model1, model3, model6, model9]
model_names = ["1-month", "3-month", "6-month", "9-month"]
versions = [
    ("child_mortality", "Child Mortality"),
    ("pred_me", "Predicted ME"),
    ("pred_fe", "Predicted FE"),
]

bin_col_dict = {
    "1-month": "days_over_28C_prev_0_mo",
    "3-month": "days_over_28C_prev_3_mo_avg",
    "6-month": "days_over_28C_prev_6_mo_avg",
    "9-month": "days_over_28C_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_do28_time_comparisons.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
        for col, (version, version_label) in enumerate(versions):
            # Prepare the data for the current model and version
            data = model.rename(columns={version: "model_predictions"})

            # Plot on the specific Axes
            plot_heat_map_grid(
                data=data,
                bin_cols=[bin_col_dict[model_name]],
                ax=axes[row, col],
                title=f"{model_name} - {version_label}",
                multiply_by=multiply_by_val,
                vmin=vmin,
                vmax=vmax,
                show_colorbar=(col == 2),  # Show colorbar only for the last column
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

# plot data points heatmap to see distribution of data points ##################

all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.qcut(
            heatmap_df[col], 10, retbins=False, duplicates="drop"
        )
    heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True
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
                .size()
                .values
            )
            all_values.append(vals)

all_values = np.concatenate(all_values)


vmin = all_values.min()
vmax = all_values.max()

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_do28_data_points.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 1 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=1, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
        version = "child_mortality"
        version_label = "Child Mortality"

        # Prepare the data for the current model and version
        data = model.rename(columns={version: "model_predictions"})

        # Plot on the specific Axes
        plot_data_pts_grid(
            data=data,
            bin_cols=[bin_col_dict[model_name]],
            ax=axes[row],
            title=f"{model_name} - {version_label}",
            vmin=vmin,
            vmax=vmax,
            show_colorbar=True,  # Show colorbar only for the last column
        )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

################################################################################

# scatter outcome rate against each of the days-over-30C variables #############


bin_col_dict = {
    "1-month": "days_over_30C_prev_0_mo",
    "3-month-avg": "days_over_30C_prev_3_mo_avg",
    "6-month-avg": "days_over_30C_prev_6_mo_avg",
    "9-month-avg": "days_over_30C_prev_9_mo_avg",
    "yearly-days": "days_over_30C",
}

# Create a PDF to save the plots
# pdf_path = os.path.join(PLOT_PATH, "neonatal_50pc_time_comparisons.pdf")

pdf_path = os.path.join(PLOT_PATH, "neonatal_scatter_days_over_30C.pdf")
with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(
        nrows=5, ncols=1, figsize=(10, 30), constrained_layout=True
    )

    # Iterate over the bin_col_dict and assign each plot to an axis
    for ax, (title, bin_col) in zip(axes.flatten(), bin_col_dict.items()):
        """
        title = "1-month"
        bin_col = "days_over_30C_prev_0_mo"
        """
        version = "child_mortality"
        version_label = "Child Mortality"

        # Prepare the data for the current model and version
        time_range = data[[version, bin_col]].copy()
        time_range_grouped = time_range.groupby(bin_col)[version].mean().reset_index()

        # Create scatter plot on the current axis
        sns.scatterplot(
            data=time_range_grouped,
            x=bin_col,
            y=version,
            alpha=0.9,
            ax=ax,
        )

        # Set titles and labels for the current axis
        ax.set_title(f"{title} - {version_label}", fontsize=14)
        ax.set_xlabel(bin_col, fontsize=12)
        ax.set_ylabel(version_label, fontsize=12)

    # Show the plot
    # plt.show()
    pdf.savefig(fig)


## agg by psu instead
data["agg_psu"] = data["nid"].astype(str) + "_" + data["psu"].astype(str)
data_psu_agg = (
    data.groupby("agg_psu")
    .agg(
        {
            "child_mortality": "mean",
            "days_over_30C_prev_0_mo": "mean",
            "days_over_30C_prev_3_mo_avg": "mean",
            "days_over_30C_prev_6_mo_avg": "mean",
            "days_over_30C_prev_9_mo_avg": "mean",
            "days_over_30C": "mean",
        }
    )
    .reset_index()
)

## agg by country-year instead, and use log mortality rate
data_country_year_agg = (
    data.groupby(["ihme_loc_id", "birth_year"])
    .agg(
        {
            "child_mortality": lambda x: np.log(np.mean(x + 1e-6)),
            "days_over_30C_prev_0_mo": "mean",
            "days_over_30C_prev_3_mo_avg": "mean",
            "days_over_30C_prev_6_mo_avg": "mean",
            "days_over_30C_prev_9_mo_avg": "mean",
            "days_over_30C": "mean",
        }
    )
    .reset_index()
)

pdf_path = os.path.join(
    PLOT_PATH, "neonatal_scatter_days_over_30C_country_year_grouped.pdf"
)
with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(
        nrows=5, ncols=1, figsize=(10, 30), constrained_layout=True
    )

    # Iterate over the bin_col_dict and assign each plot to an axis
    for ax, (title, bin_col) in zip(axes.flatten(), bin_col_dict.items()):
        """
        title = "1-month"
        bin_col = "days_over_30C_prev_0_mo"
        """
        version = "child_mortality"
        version_label = "Child Mortality"

        # Prepare the data for the current model and version
        time_range = data_country_year_agg[[version, bin_col]].copy()

        # Create scatter plot on the current axis
        sns.scatterplot(
            data=time_range,
            x=bin_col,
            y=version,
            alpha=0.5,
            ax=ax,
        )

        # Set titles and labels for the current axis
        ax.set_title(f"{title} - {version_label}", fontsize=14)
        ax.set_xlabel(bin_col, fontsize=12)
        ax.set_ylabel(version_label, fontsize=12)

    # Show the plot
    # plt.show()
    pdf.savefig(fig)
