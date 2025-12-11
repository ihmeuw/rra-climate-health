"""
Plot different time horizons side-by-side for neonatal mortality model.

1. Zones with full specification (mostly failed to converge)
2. Quantile thresholds for 90th percentile
3. Quantile thresholds for 95th percentile
4. Simplified zones specifications
5. Simplified zones with quantile thresholds for 90th percentile
6. Simplified zones with quantile thresholds for 95th percentile
"""

import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import re
import os
from tqdm import tqdm

RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/plots/2025_12_09.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

## CONSTANTS

# Heat maps of variables
columns_to_bin = [
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
]

## FUNCTIONS


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
    data = data
    col = bin_cols[0]
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
        y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]
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


# Plot data ####################################################################

## 1. Zones with full specification (mostly failed to converge) ################

# Load and concat results from all models
# model 1-month
zone_results_path_1m = os.path.join(RESULTS_PATH, "zones/mo_1/")
results_files_1m = os.listdir(zone_results_path_1m)
results_files_1m = [f for f in results_files_1m if f.endswith(".parquet")]
sorted(results_files_1m)
results_dfs_1m = [
    pd.read_parquet(os.path.join(zone_results_path_1m, f)) for f in results_files_1m
]
model1 = pd.concat(results_dfs_1m, ignore_index=True)
model1["zone"] = model1["zone"].astype(int)
model1["zone"].value_counts()

# model 3-months - error. Zone number not included but can retrieve from filename
zone_results_path_3m = os.path.join(RESULTS_PATH, "zones/mo_3/")
results_files_3m = os.listdir(zone_results_path_3m)
results_files_3m = [f for f in results_files_3m if f.endswith(".parquet")]
results_dfs_3m = []
for f in tqdm(results_files_3m):
    zone = int(re.findall(r"(?<=zone_).*(?=_summary)", f)[0])
    tmp_df = pd.read_parquet(os.path.join(zone_results_path_3m, f))
    tmp_df["zone"] = zone
    results_dfs_3m.append(tmp_df)

model3 = pd.concat(results_dfs_3m, ignore_index=True)
model3["zone"] = model3["zone"].astype(int)

# model 6-months - error. Zone number not included but can retrieve from filename
zone_results_path_6m = os.path.join(RESULTS_PATH, "zones/mo_6/")
results_files_6m = os.listdir(zone_results_path_6m)
results_files_6m = [f for f in results_files_6m if f.endswith(".parquet")]
results_dfs_6m = []
for f in tqdm(results_files_6m):
    zone = int(re.findall(r"(?<=zone_).*(?=_summary)", f)[0])
    tmp_df = pd.read_parquet(os.path.join(zone_results_path_6m, f))
    tmp_df["zone"] = zone
    results_dfs_6m.append(tmp_df)
model6 = pd.concat(results_dfs_6m, ignore_index=True)
model6["zone"] = model6["zone"].astype(int)

# model 9-months - error. Zone number not included but can retrieve from filename
zone_results_path_9m = os.path.join(RESULTS_PATH, "zones/mo_9/")
results_files_9m = os.listdir(zone_results_path_9m)
results_files_9m = [f for f in results_files_9m if f.endswith(".parquet")]
results_dfs_9m = []
for f in tqdm(results_files_9m):
    zone = int(re.findall(r"(?<=zone_).*(?=_summary)", f)[0])
    tmp_df = pd.read_parquet(os.path.join(zone_results_path_9m, f))
    tmp_df["zone"] = zone
    results_dfs_9m.append(tmp_df)
model9 = pd.concat(results_dfs_9m, ignore_index=True)
model9["zone"] = model9["zone"].astype(int)


# get min and max values for color scale consistency across plots
# as well as custom bins
data = pd.concat(
    [
        model1["days_over_30C_prev_0_mo"],
        model3["days_over_30C_prev_3_mo_avg"],
        model6["days_over_30C_prev_6_mo_avg"],
        model9["days_over_30C_prev_9_mo_avg"],
    ],
    ignore_index=True,
)


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
custom_bins_fixed = [0, first_non_zero] + list(quartiles)

# get mix/maxes for plots
multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.cut(
            heatmap_df[col], bins=custom_bins_fixed, include_lowest=True, right=False
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
    "1-month": "days_over_30C_prev_0_mo",
    "3-month": "days_over_30C_prev_3_mo_avg",
    "6-month": "days_over_30C_prev_6_mo_avg",
    "9-month": "days_over_30C_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_zones_exp_full_spec.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
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
                ax=axes[row, col],
                title=f"{model_name} - {version_label}",
                multiply_by=multiply_by_val,
                vmin=vmin,
                vmax=vmax,
                show_colorbar=(col == 2),  # Show colorbar only for the last column
                custom_bins=custom_bins_fixed,  # custom_bins_for_model,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

################################################################################

## 2. Quantile thresholds for 90th percentile ##################################
################################################################################

## 3. Quantile thresholds for 95th percentile ##################################
################################################################################

## 4. Simplified zones specifications ##########################################
################################################################################

## 5. Simplified zones with quantile thresholds for 90th percentile ############
################################################################################

## 6. Simplified zones with quantile thresholds for 95th percentile ############
################################################################################

# Set versions

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
# as well as custom bins
data = pd.concat(
    [
        model1["days_over_30C_prev_0_mo"],
        model3["days_over_30C_prev_3_mo_avg"],
        model6["days_over_30C_prev_6_mo_avg"],
        model9["days_over_30C_prev_9_mo_avg"],
    ],
    ignore_index=True,
)


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
custom_bins_fixed = [0, first_non_zero] + list(quartiles)

# get mix/maxes for plots
multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"] = pd.cut(
            heatmap_df[col], bins=custom_bins_fixed, include_lowest=True, right=False
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
pdf_path = os.path.join(PLOT_PATH, "neonatal_100pc_time_comparisons_same_axis.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
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
                ax=axes[row, col],
                title=f"{model_name} - {version_label}",
                multiply_by=multiply_by_val,
                vmin=vmin,
                vmax=vmax,
                show_colorbar=(col == 2),  # Show colorbar only for the last column
                custom_bins=custom_bins_fixed,  # custom_bins_for_model,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

################################################################################


# Plot mean temp ###############################################################

# Set versions

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
    # "days_over_30C_prev_0_mo",
    # "days_over_30C_prev_3_mo_avg",
    # "days_over_30C_prev_6_mo_avg",
    # "days_over_30C_prev_9_mo_avg",
    "mean_temperature_prev_0_mo",
    "mean_temperature_prev_3_mo_avg",
    "mean_temperature_prev_6_mo_avg",
    "mean_temperature_prev_9_mo_avg",
    # "days_over_28C_prev_0_mo",
    # "days_over_28C_prev_3_mo_avg",
    # "days_over_28C_prev_6_mo_avg",
    # "days_over_28C_prev_9_mo_avg",
    # "days_over_30C_prev_0_mo",
    # "days_over_30C_prev_3_mo_avg",
    # "days_over_30C_prev_6_mo_avg",
    # "days_over_30C_prev_9_mo_avg",
]


# get min and max values for color scale consistency across plots

# Use default bins for mean temperature
# Extract the column
col = "mean_temperature_prev_6_mo_avg"
data = model6[col]  # Replace `model6` with the appropriate DataFrame


# get mix/maxes for plots
multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for model in [model1, model3, model6, model9]:
    heatmap_df = model.copy()
    for col in columns_to_bin:
        heatmap_df[f"{col}_bin"], bin_edges = pd.qcut(heatmap_df[col], 10, retbins=True)
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
pdf_path = os.path.join(PLOT_PATH, "neonatal_100pc_mean_temp_time_comparisons.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
        for col, (version, version_label) in enumerate(versions):
            # Prepare the data for the current model and version
            data = model.rename(columns={version: "model_predictions"})

            # custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

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
                # custom_bins=custom_bins_for_model,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

################################################################################
# Plot days over 28 ############################################################

# Set versions

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
    # "days_over_30C_prev_0_mo",
    # "days_over_30C_prev_3_mo_avg",
    # "days_over_30C_prev_6_mo_avg",
    # "days_over_30C_prev_9_mo_avg",
    # "mean_temperature_prev_0_mo",
    # "mean_temperature_prev_3_mo_avg",
    # "mean_temperature_prev_6_mo_avg",
    # "mean_temperature_prev_9_mo_avg",
    "days_over_28C_prev_0_mo",
    "days_over_28C_prev_3_mo_avg",
    "days_over_28C_prev_6_mo_avg",
    "days_over_28C_prev_9_mo_avg",
    # "days_over_30C_prev_0_mo",
    # "days_over_30C_prev_3_mo_avg",
    # "days_over_30C_prev_6_mo_avg",
    # "days_over_30C_prev_9_mo_avg",
]


# get min and max values for color scale consistency across plots

# Explore bins
# Extract the column
col = "days_over_28C_prev_6_mo_avg"
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
pdf_path = os.path.join(PLOT_PATH, "neonatal_100pc_days_over_28C_time_comparisons.pdf")

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
# Plot days over 32 ############################################################

# Set versions

# Neonatal predictions
model1 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_1_mo_do32_summary.parquet"
)
model3 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_3_mo_do32_summary.parquet"
)
model6 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_6_mo_do32_summary.parquet"
)
model9 = pd.read_parquet(
    RESULTS_PATH + "predictions_nnm_full_9_mo_do32_summary.parquet"
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
    # "days_over_30C_prev_0_mo",
    # "days_over_30C_prev_3_mo_avg",
    # "days_over_30C_prev_6_mo_avg",
    # "days_over_30C_prev_9_mo_avg",
    # "mean_temperature_prev_0_mo",
    # "mean_temperature_prev_3_mo_avg",
    # "mean_temperature_prev_6_mo_avg",
    # "mean_temperature_prev_9_mo_avg",
    # "days_over_28C_prev_0_mo",
    # "days_over_28C_prev_3_mo_avg",
    # "days_over_28C_prev_6_mo_avg",
    # "days_over_28C_prev_9_mo_avg",
    # "days_over_30C_prev_0_mo",
    # "days_over_30C_prev_3_mo_avg",
    # "days_over_30C_prev_6_mo_avg",
    # "days_over_30C_prev_9_mo_avg",
    "days_over_32C_prev_0_mo",
    "days_over_32C_prev_3_mo_avg",
    "days_over_32C_prev_6_mo_avg",
    "days_over_32C_prev_9_mo_avg",
]


# get min and max values for color scale consistency across plots

# Explore bins
# Extract the column
col = "days_over_32C_prev_6_mo_avg"
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


models = [model1, model3, model6, model9]
model_names = ["1-month", "3-month", "6-month", "9-month"]


bin_col_dict = {
    "1-month": "days_over_32C_prev_0_mo",
    "3-month": "days_over_32C_prev_3_mo_avg",
    "6-month": "days_over_32C_prev_6_mo_avg",
    "9-month": "days_over_32C_prev_9_mo_avg",
}

# temp versions
versions = [
    "child_mortality",
    "pred_fe",
    "pred_me",
]

# get mix/maxes for plots
multiply_by_val = 1000  # for easier to read heatmaps
all_values = []

for row, (model, model_name) in enumerate(zip(models, model_names)):
    # for col, (version, version_label) in enumerate(versions):

    custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

    heatmap_df = model.copy()

    heatmap_df[f"{bin_col_dict[model_name]}_bin"] = pd.cut(
        heatmap_df[bin_col_dict[model_name]],
        bins=custom_bins_for_model,
        include_lowest=True,
        right=False,
    )
    heatmap_df["consumption_pd"], ldi_bins = pd.qcut(
        heatmap_df.consumption_pd, 10, retbins=True
    )

    for version in versions:

        vals = (
            heatmap_df.groupby(["consumption_pd", f"{bin_col_dict[model_name]}_bin"])[
                version
            ]
            .mean()
            .values
        )

        vals = [v for v in vals if not np.isnan(v)]
        all_values.append(vals)

all_values = np.concatenate(all_values)


vmin = all_values.min()
vmax = all_values.max()

vmin *= multiply_by_val
vmax *= multiply_by_val

# Plot all on same PDF
# Update versions
versions = [
    ("child_mortality", "Child Mortality"),
    ("pred_me", "Predicted ME"),
    ("pred_fe", "Predicted FE"),
]

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_100pc_days_over_32C_time_comparisons.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(15, 20), constrained_layout=True
    )

    for row, (model, model_name) in enumerate(zip(models, model_names)):
        for col, (version, version_label) in enumerate(versions):
            # Prepare the data for the current model and version
            data = model.rename(columns={version: "model_predictions"})

            # custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

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
                custom_bins=custom_bins,  # custom_bins_for_model,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

################################################################################
