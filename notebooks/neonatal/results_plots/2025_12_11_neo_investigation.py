"""
Plot different time horizons side-by-side for neonatal mortality model.

1. Zones with full specification (mostly failed to converge)
2. Quantile thresholds for 90th percentile
3. Quantile thresholds for 95th percentile
4. Simplified zones specifications
5. Simplified zones with quantile thresholds for 90th percentile
6. Simplified zones with quantile thresholds for 95th percentile
7. Quantile thresholds for 99th percentile
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
    custom_y_bins: list = None,
    y_axis_label=None,
    x_axis_label=None,
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
        y_labs = [f"{y_tick_vals[i]:.1f}" for i in range(len(y_tick_vals))]
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
# Heat maps of variables
columns_to_bin = [
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
]

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

# make table of summaries
results_table = pd.DataFrame(
    columns=["Zone", "Time", "Variable", "Estimate", "p_value"]
)
SUMMARY_DIRS = [
    RESULTS_PATH + "zones/mo_1/model_summaries/",
    RESULTS_PATH + "zones/mo_3/model_summaries/",
    RESULTS_PATH + "zones/mo_6/model_summaries/",
    RESULTS_PATH + "zones/mo_9/model_summaries/",
]

vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]

"""
MO_SUMMARY_DIR = RESULTS_PATH + "zones/mo_1/model_summaries/"
f = 'nnm_1_mo_zone_18_summary.txt'
"""
for MO_SUMMARY_DIR in SUMMARY_DIRS:

    summaries = [f for f in os.listdir(MO_SUMMARY_DIR) if f.endswith(".txt")]
    for f in summaries:

        zone = re.findall(r"(?<=zone_).*(?=_summary)", f)[0]
        # print(zone)
        zone_v = f"Z_{zone}"
        time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
        # print(time_period)
        time_period_v = f"{time_period}-month"

        coef_table = pd.DataFrame(
            columns=["Zone", "Time", "Variable", "Estimate", "p_value"]
        )
        with open(MO_SUMMARY_DIR + f, "r") as infile:
            s = infile.read().split("\n")
        # [l for l in s]
        for l in s:
            if l.startswith(tuple(vars_of_interest)) and bool(
                re.findall(r"[\d]{3}", l)
            ):
                # test if all info on single line or if it overflowed:
                if len(l.split()) == 5:
                    coef = l.split()[0]
                    estimate = l.split()[1]
                    p_value = l.split()[-1]
                    coef_table = pd.concat(
                        [
                            coef_table,
                            pd.DataFrame.from_records(
                                [
                                    {
                                        "Zone": zone_v,
                                        "Time": time_period_v,
                                        "Variable": coef,
                                        "Estimate": estimate,
                                        "p_value": p_value,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                elif (
                    len(re.findall(r"[\d]{1}\.[\d]*", l)) == 3
                ):  # p-val missing from end
                    coef = l.split()[0]
                    estimate = l.split()[1]
                    coef_table = pd.concat(
                        [
                            coef_table,
                            pd.DataFrame.from_records(
                                [
                                    {
                                        "Zone": zone_v,
                                        "Time": time_period_v,
                                        "Variable": coef,
                                        "Estimate": estimate,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 1:  # variable and p-val
                    coef = l.split()[0]
                    p_value = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                    coef_table.loc[coef_table["Variable"] == coef, "p_value"] = p_value

        results_table = pd.concat([results_table, coef_table], ignore_index=True)

results_table.drop(columns="Time", inplace=True)
results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(2)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table.loc[results_table["p_value"] < 0.05, "Estimate"] = (
    results_table.loc[results_table["p_value"] < 0.05, "Estimate"] + "*"
)
results_table.drop(columns="p_value", inplace=True)

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]
zone_order = [
    "Z_6",
    "Z_7",
    "Z_8",
    "Z_9",
    "Z_10",
    "Z_11",
    "Z_12",
    "Z_13",
    "Z_14",
    "Z_15",
    "Z_16",
    "Z_17",
    "Z_18",
    "Z_19",
    "Z_20",
    "Z_21",
    "Z_22",
    "Z_23",
    "Z_24",
    "Z_25",
    "Z_26",
    "Z_27",
    "Z_28",
    "Z_29",
]
results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)
results_table["Zone"] = pd.Categorical(
    results_table["Zone"], categories=zone_order, ordered=True
)
results_table.sort_values(by=["Variable", "Zone"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Zone"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(PLOT_PATH + "neonatal_zones_full_spec_coefs.csv", index=False)

################################################################################

## 2. Quantile thresholds for 90th percentile ##################################

model1 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_1_mo_q9_summary.parquet")
model3 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_3_mo_q9_summary.parquet")
model6 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_6_mo_q9_summary.parquet")
model9 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_9_mo_q9_summary.parquet")

# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    "q9_prev_0_mo",
    "q9_prev_3_mo_avg",
    "q9_prev_6_mo_avg",
    "q9_prev_9_mo_avg",
    # "q95_prev_0_mo",
    # 'q95_prev_3_mo_avg',
    # 'q95_prev_6_mo_avg',
    # 'q95_prev_9_mo_avg',
]

data = pd.concat(
    [
        model1["q9_prev_0_mo"],
        model3["q9_prev_3_mo_avg"],
        model6["q9_prev_6_mo_avg"],
        model9["q9_prev_9_mo_avg"],
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
# custom_bins_fixed = [0, first_non_zero] + list(quartiles)
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

for model in [model1, model3, model6, model9]:
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
    "1-month": "q9_prev_0_mo",
    "3-month": "q9_prev_3_mo_avg",
    "6-month": "q9_prev_6_mo_avg",
    "9-month": "q9_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q9.pdf")

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
                custom_y_bins=custom_y_bins,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

# make table of summaries
results_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
SUMMARY_DIR = RESULTS_PATH + "model_summaries/"

vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q9_prev_0_mo",
    "q9_prev_3_mo_avg",
    "q9_prev_6_mo_avg",
    "q9_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]

"""
f = 'nnm_9_mo_q9_summary.txt'
"""

summaries = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".txt")]
# only look at q9
summaries = [f for f in summaries if "_q9_summary" in f]

for f in summaries:

    time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
    # print(time_period)
    time_period_v = f"{time_period}-month"

    coef_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
    with open(SUMMARY_DIR + f, "r") as infile:
        s = infile.read().split("\n")
    # [l for l in s]
    for l in s:
        if l.startswith(tuple(vars_of_interest)) and bool(re.findall(r"[\d]{3}", l)):
            # test if all info on single line or if it overflowed:
            if len(l.split()) == 5:
                coef = l.split()[0]
                estimate = l.split()[1]
                p_value = l.split()[-1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                    "p_value": p_value,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 3:  # p-val missing from end
                coef = l.split()[0]
                estimate = l.split()[1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 1:  # variable and p-val
                coef = l.split()[0]
                p_value = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                coef_table.loc[coef_table["Variable"] == coef, "p_value"] = p_value

    results_table = pd.concat([results_table, coef_table], ignore_index=True)

results_table["Variable"] = results_table["Variable"].str.replace("_0_mo", "_X_mo_avg")
results_table["Variable"] = results_table["Variable"].str.replace("_3_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_6_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_9_mo", "_X_mo")

results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(4)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table.loc[results_table["p_value"] < 0.05, "Estimate"] = (
    results_table.loc[results_table["p_value"] < 0.05, "Estimate"] + "*"
)
results_table.drop(columns="p_value", inplace=True)
results_table["Variable"].unique()

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q9_prev_X_mo_avg",
    "total_precipitation_prev_X_mo_avg",
]

results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)

results_table.sort_values(by=["Variable", "Time"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Time"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(PLOT_PATH + "neonatal_q9_full_spec_coefs.csv", index=False)

################################################################################

## 3. Quantile thresholds for 95th percentile ##################################

model1 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_1_mo_q95_summary.parquet")
model3 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_3_mo_q95_summary.parquet")
model6 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_6_mo_q95_summary.parquet")
model9 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_9_mo_q95_summary.parquet")

# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    # "q9_prev_0_mo",
    # "q9_prev_3_mo_avg",
    # "q9_prev_6_mo_avg",
    # "q9_prev_9_mo_avg",
    "q95_prev_0_mo",
    "q95_prev_3_mo_avg",
    "q95_prev_6_mo_avg",
    "q95_prev_9_mo_avg",
]

data = pd.concat(
    [
        model1["q95_prev_0_mo"],
        model3["q95_prev_3_mo_avg"],
        model6["q95_prev_6_mo_avg"],
        model9["q95_prev_9_mo_avg"],
    ],
    ignore_index=True,
)
# data = model1["q95_prev_0_mo"]


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
# custom_bins_fixed = [0, first_non_zero] + list(quartiles)
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

for model in [model1, model3, model6, model9]:
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
all_values = [v for v in all_values if not np.isnan(v)]

vmin = min(all_values)
vmax = max(all_values)

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
    "1-month": "q95_prev_0_mo",
    "3-month": "q95_prev_3_mo_avg",
    "6-month": "q95_prev_6_mo_avg",
    "9-month": "q95_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q95.pdf")
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
                custom_y_bins=custom_y_bins,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

## Make final presentation-style figure
data = model1["q95_prev_0_mo"]


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
# custom_bins_fixed = [0, first_non_zero] + list(quartiles)
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
all_values = [v for v in all_values if not np.isnan(v)]

vmin = min(all_values)
vmax = max(all_values)

vmin *= multiply_by_val
vmax *= multiply_by_val
print(vmax)

# Plot all on same PDF
# Define models and versions
models = [model1]
model_names = ["1-month"]
versions = [
    ("child_mortality", "Child Mortality Data"),
    ("pred_fe", "Predicted Child Mortality"),
]


bin_col_dict = {
    "1-month": "q95_prev_0_mo",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q95_full_spec_data_fe_only_new_bins.pdf")
with PdfPages(pdf_path) as pdf:
    # Create a figure with 1 rows and 2 columns
    # fig, axes = plt.subplots(
    #     nrows=1, ncols=2, figsize=(13, 5)
    # )  # , constrained_layout=True)
    fig = plt.figure(figsize=(13, 5))  # Total figure size
    spec = GridSpec(nrows=1, ncols=2, width_ratios=[6, 7], figure=fig)  # Column sizes

    # Create Axes for the two plots
    ax1 = fig.add_subplot(spec[0])  # First plot (width 6)
    ax2 = fig.add_subplot(spec[1])  # Second plot (width 7)

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
                # ax=axes[row, col],
                ax=[ax1, ax2][col],
                # ax=axes[col],
                # title=f"{model_name} - {version_label}",
                title=f"{version_label}",
                multiply_by=multiply_by_val,
                vmin=vmin,
                vmax=vmax,
                show_colorbar=(col == 1),  # Show colorbar only for the last column
                custom_bins=custom_bins_fixed,  # custom_bins_for_model,
                custom_y_bins=custom_y_bins,
                y_axis_label="",
                x_axis_label="",
            )

    # Add shared x-axis and y-axis labels
    fig.text(
        0.5,
        0.01,  # Adjusted to move the x-axis label further down
        "Days over 95th Percentile Temperature during Birth Month",
        ha="center",
        fontsize=14,
    )
    fig.text(
        0.01,  # Adjusted to move the y-axis label further left
        0.5,
        "Daily Consumption",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    # Adjust layout to prevent overlap
    fig.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.2, wspace=0.3)

    # Save the figure to the PDF
    pdf.savefig(
        fig, bbox_inches="tight"
    )  # Use bbox_inches to ensure nothing is cut off
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

## Same as above but comparing against days over 30C
## Make final presentation-style figure

modeldo30 = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_11_20.01/predictions_nnm_full_1_mo_model_summary.parquet"
)


# Define custom bin edges
# custom_bins_fixed = [0, first_non_zero] + list(quartiles)
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
        # "child_mortality",
        "pred_fe",
        # "pred_me",
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
all_values = [v for v in all_values if not np.isnan(v)]

vmin = min(all_values)
vmax = max(all_values)

vmin *= multiply_by_val
vmax *= multiply_by_val
print(vmax)

# Plot all on same PDF
# Define models and versions
models = [modeldo30, model1]
model_names = ["1-month do30C", "1-month"]
versions = [
    ("pred_fe", "Predicted Child Mortality"),
]


bin_col_dict = {
    "1-month do30C": "days_over_30C_prev_0_mo",
    "1-month": "q95_prev_0_mo",
}

x_labels = [
    "Days over 30C during Birth Month",
    "Days over 95th Percentile Temperature during Birth Month",
]

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q95_vs_do30C.pdf")

with PdfPages(pdf_path) as pdf:
    # Create a figure with 1 rows and 2 columns
    # fig, axes = plt.subplots(
    #     nrows=1, ncols=2, figsize=(13, 5)
    # )  # , constrained_layout=True)
    fig = plt.figure(figsize=(13, 5))  # Total figure size
    spec = GridSpec(nrows=1, ncols=2, width_ratios=[6, 7], figure=fig)  # Column sizes

    # Create Axes for the two plots
    ax1 = fig.add_subplot(spec[0])  # First plot (width 6)
    ax2 = fig.add_subplot(spec[1])  # Second plot (width 7)

    for col, (model, model_name, x_label) in enumerate(
        zip(models, model_names, x_labels)
    ):
        # for col, (version, version_label) in enumerate(versions):
        version = "pred_fe"
        version_label = "Predicted Child Mortality"
        # Prepare the data for the current model and version
        data = model.rename(columns={version: "model_predictions"})

        """
        model = models[0]
        model_name = "1-month"
        data = model.rename(columns={version: "model_predictions"})
        bin_cols=[bin_col_dict[model_name]]
        """
        # custom_bins_for_model = create_custom_bins(model, bin_col_dict[model_name])

        # Plot on the specific Axes
        plot_heat_map_grid(
            data=data,
            bin_cols=[bin_col_dict[model_name]],
            # ax=axes[row, col],
            ax=[ax1, ax2][col],
            # ax=axes[col],
            # title=f"{model_name} - {version_label}",
            title=f"{version_label}",
            multiply_by=multiply_by_val,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=(col == 1),  # Show colorbar only for the last column
            custom_bins=custom_bins_fixed,  # custom_bins_for_model,
            custom_y_bins=custom_y_bins,
            y_axis_label="",
            x_axis_label=x_label,
        )

    # Add shared x-axis and y-axis labels
    # fig.text(
    #     0.5,
    #     0.01,  # Adjusted to move the x-axis label further down
    #     "Days over 95th Percentile Temperature during Birth Month",
    #     ha="center",
    #     fontsize=14,
    # )
    fig.text(
        0.01,  # Adjusted to move the y-axis label further left
        0.5,
        "Daily Consumption",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    # Adjust layout to prevent overlap
    fig.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.2, wspace=0.3)

    # Save the figure to the PDF
    pdf.savefig(
        fig, bbox_inches="tight"
    )  # Use bbox_inches to ensure nothing is cut off
    plt.close(fig)

print(f"PDF saved to {pdf_path}")


# make table of summaries
results_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
SUMMARY_DIR = RESULTS_PATH + "model_summaries/"

vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q95_prev_0_mo",
    "q95_prev_3_mo_avg",
    "q95_prev_6_mo_avg",
    "q95_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]


summaries = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".txt")]
# only look at q95
summaries = [f for f in summaries if "_q95_summary" in f]

for f in summaries:

    time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
    # print(time_period)
    time_period_v = f"{time_period}-month"

    coef_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
    with open(SUMMARY_DIR + f, "r") as infile:
        s = infile.read().split("\n")
    # [l for l in s]
    for l in s:
        if l.startswith(tuple(vars_of_interest)) and bool(re.findall(r"[\d]{3}", l)):
            # test if all info on single line or if it overflowed:
            if len(l.split()) == 5:
                coef = l.split()[0]
                estimate = l.split()[1]
                p_value = l.split()[-1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                    "p_value": p_value,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 3:  # p-val missing from end
                coef = l.split()[0]
                estimate = l.split()[1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 1:  # variable and p-val
                coef = l.split()[0]
                p_value = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                coef_table.loc[coef_table["Variable"] == coef, "p_value"] = p_value

    results_table = pd.concat([results_table, coef_table], ignore_index=True)

results_table["Variable"] = results_table["Variable"].str.replace("_0_mo", "_X_mo_avg")
results_table["Variable"] = results_table["Variable"].str.replace("_3_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_6_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_9_mo", "_X_mo")

results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(4)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table.loc[results_table["p_value"] < 0.05, "Estimate"] = (
    results_table.loc[results_table["p_value"] < 0.05, "Estimate"] + "*"
)
results_table.drop(columns="p_value", inplace=True)
results_table["Variable"].unique()

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q95_prev_X_mo_avg",
    "total_precipitation_prev_X_mo_avg",
]

results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)

results_table.sort_values(by=["Variable", "Time"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Time"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(PLOT_PATH + "neonatal_q95_full_spec_coefs.csv", index=False)

################################################################################

## 4. Simplified zones specifications ##########################################
################################################################################

## 1. Zones with full specification (mostly failed to converge) ################

# Load and concat results from all models
# model 1-month
zone_results_path_1m = os.path.join(RESULTS_PATH, "zones/simplified/mo_1/")
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
zone_results_path_3m = os.path.join(RESULTS_PATH, "zones/simplified/mo_3/")
results_files_3m = os.listdir(zone_results_path_3m)
results_files_3m = [f for f in results_files_3m if f.endswith(".parquet")]
results_dfs_3m = []
for f in tqdm(results_files_3m):
    zone = int(re.findall(r"(?<=zone_).*(?=_simplified)", f)[0])
    tmp_df = pd.read_parquet(os.path.join(zone_results_path_3m, f))
    tmp_df["zone"] = zone
    results_dfs_3m.append(tmp_df)

model3 = pd.concat(results_dfs_3m, ignore_index=True)
model3["zone"] = model3["zone"].astype(int)

# model 6-months - error. Zone number not included but can retrieve from filename
zone_results_path_6m = os.path.join(RESULTS_PATH, "zones/simplified/mo_6/")
results_files_6m = os.listdir(zone_results_path_6m)
results_files_6m = [f for f in results_files_6m if f.endswith(".parquet")]
results_dfs_6m = []
for f in tqdm(results_files_6m):
    zone = int(re.findall(r"(?<=zone_).*(?=_simplified)", f)[0])
    tmp_df = pd.read_parquet(os.path.join(zone_results_path_6m, f))
    tmp_df["zone"] = zone
    results_dfs_6m.append(tmp_df)
model6 = pd.concat(results_dfs_6m, ignore_index=True)
model6["zone"] = model6["zone"].astype(int)

# model 9-months - error. Zone number not included but can retrieve from filename
zone_results_path_9m = os.path.join(RESULTS_PATH, "zones/simplified/mo_9/")
results_files_9m = os.listdir(zone_results_path_9m)
results_files_9m = [f for f in results_files_9m if f.endswith(".parquet")]
results_dfs_9m = []
for f in tqdm(results_files_9m):
    zone = int(re.findall(r"(?<=zone_).*(?=_simplified)", f)[0])
    tmp_df = pd.read_parquet(os.path.join(zone_results_path_9m, f))
    tmp_df["zone"] = zone
    results_dfs_9m.append(tmp_df)
model9 = pd.concat(results_dfs_9m, ignore_index=True)
model9["zone"] = model9["zone"].astype(int)


# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
]

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
pdf_path = os.path.join(PLOT_PATH, "neonatal_zones_exp_simplified_spec.pdf")

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

# make table of summaries
results_table = pd.DataFrame(
    columns=["Zone", "Time", "Variable", "Estimate", "p_value"]
)
SUMMARY_DIRS = [
    RESULTS_PATH + "zones/simplified/mo_1/model_summaries/",
    RESULTS_PATH + "zones/simplified/mo_3/model_summaries/",
    RESULTS_PATH + "zones/simplified/mo_6/model_summaries/",
    RESULTS_PATH + "zones/simplified/mo_9/model_summaries/",
]

vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]

"""
MO_SUMMARY_DIR = RESULTS_PATH + "zones/mo_1/model_summaries/"
f = 'nnm_1_mo_zone_18_summary.txt'
"""
for MO_SUMMARY_DIR in SUMMARY_DIRS:

    summaries = [f for f in os.listdir(MO_SUMMARY_DIR) if f.endswith(".txt")]
    for f in summaries:

        zone = re.findall(r"(?<=zone_).*(?=_simplified)", f)[0]
        # print(zone)
        zone_v = f"Z_{zone}"
        time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
        # print(time_period)
        time_period_v = f"{time_period}-month"

        coef_table = pd.DataFrame(
            columns=["Zone", "Time", "Variable", "Estimate", "p_value"]
        )
        with open(MO_SUMMARY_DIR + f, "r") as infile:
            s = infile.read().split("\n")
        # [l for l in s]
        for l in s:
            if l.startswith(tuple(vars_of_interest)) and bool(
                re.findall(r"[\d]{3}", l)
            ):
                # test if all info on single line or if it overflowed:
                if len(l.split()) == 5:
                    coef = l.split()[0]
                    estimate = l.split()[1]
                    p_value = l.split()[-1]
                    coef_table = pd.concat(
                        [
                            coef_table,
                            pd.DataFrame.from_records(
                                [
                                    {
                                        "Zone": zone_v,
                                        "Time": time_period_v,
                                        "Variable": coef,
                                        "Estimate": estimate,
                                        "p_value": p_value,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                elif (
                    len(re.findall(r"[\d]{1}\.[\d]*", l)) == 3
                ):  # p-val missing from end
                    coef = l.split()[0]
                    estimate = l.split()[1]
                    coef_table = pd.concat(
                        [
                            coef_table,
                            pd.DataFrame.from_records(
                                [
                                    {
                                        "Zone": zone_v,
                                        "Time": time_period_v,
                                        "Variable": coef,
                                        "Estimate": estimate,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 1:  # variable and p-val
                    coef = l.split()[0]
                    p_value = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                    coef_table.loc[coef_table["Variable"] == coef, "p_value"] = p_value

        results_table = pd.concat([results_table, coef_table], ignore_index=True)

results_table.drop(columns="Time", inplace=True)
results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["p_value"] = results_table["p_value"].str.replace("<", "")
results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(2)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table.loc[results_table["p_value"] < 0.05, "Estimate"] = (
    results_table.loc[results_table["p_value"] < 0.05, "Estimate"] + "*"
)
results_table.drop(columns="p_value", inplace=True)

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "days_over_30C_prev_0_mo",
    "days_over_30C_prev_3_mo_avg",
    "days_over_30C_prev_6_mo_avg",
    "days_over_30C_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]
zone_order = [
    "Z_6",
    "Z_7",
    "Z_8",
    "Z_9",
    "Z_10",
    "Z_11",
    "Z_12",
    "Z_13",
    "Z_14",
    "Z_15",
    "Z_16",
    "Z_17",
    "Z_18",
    "Z_19",
    "Z_20",
    "Z_21",
    "Z_22",
    "Z_23",
    "Z_24",
    "Z_25",
    "Z_26",
    "Z_27",
    "Z_28",
    "Z_29",
]
results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)
results_table["Zone"] = pd.Categorical(
    results_table["Zone"], categories=zone_order, ordered=True
)
results_table.sort_values(by=["Variable", "Zone"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Zone"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(
    PLOT_PATH + "neonatal_zones_simplified_spec_coefs.csv", index=False
)

## 5. Simplified quantile thresholds for 90th percentile #######################

model1 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_1_mo_q9_simplified_summary.parquet"
)
model3 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_3_mo_q9_simplified_summary.parquet"
)
model6 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_6_mo_q9_simplified_summary.parquet"
)
model9 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_9_mo_q9_simplified_summary.parquet"
)

# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    "q9_prev_0_mo",
    "q9_prev_3_mo_avg",
    "q9_prev_6_mo_avg",
    "q9_prev_9_mo_avg",
    # "q95_prev_0_mo",
    # 'q95_prev_3_mo_avg',
    # 'q95_prev_6_mo_avg',
    # 'q95_prev_9_mo_avg',
]

data = pd.concat(
    [
        model1["q9_prev_0_mo"],
        model3["q9_prev_3_mo_avg"],
        model6["q9_prev_6_mo_avg"],
        model9["q9_prev_9_mo_avg"],
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
    "1-month": "q9_prev_0_mo",
    "3-month": "q9_prev_3_mo_avg",
    "6-month": "q9_prev_6_mo_avg",
    "9-month": "q9_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q9_simplified_spec.pdf")

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

# make table of summaries
results_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
SUMMARY_DIR = RESULTS_PATH + "simplified_quantiles/model_summaries/"

vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q9_prev_0_mo",
    "q9_prev_3_mo_avg",
    "q9_prev_6_mo_avg",
    "q9_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]

"""
f = 'nnm_9_mo_q9_summary.txt'
"""

summaries = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".txt")]
# only look at q9
summaries = [f for f in summaries if "_q9_simplified" in f]

for f in summaries:

    time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
    # print(time_period)
    time_period_v = f"{time_period}-month"

    coef_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
    with open(SUMMARY_DIR + f, "r") as infile:
        s = infile.read().split("\n")
    # [l for l in s]
    for l in s:
        if l.startswith(tuple(vars_of_interest)) and bool(re.findall(r"[\d]{3}", l)):
            # test if all info on single line or if it overflowed:
            if len(l.split()) == 5:
                if not l.endswith("  "):
                    coef = l.split()[0]
                    estimate = l.split()[1]
                    p_value = l.split()[-1]
                    coef_table = pd.concat(
                        [
                            coef_table,
                            pd.DataFrame.from_records(
                                [
                                    {
                                        "Time": time_period_v,
                                        "Variable": coef,
                                        "Estimate": estimate,
                                        "p_value": p_value,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 3:  # p-val missing from end
                coef = l.split()[0]
                estimate = l.split()[1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 1:  # variable and p-val
                coef = l.split()[0]
                p_value = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                coef_table.loc[coef_table["Variable"] == coef, "p_value"] = p_value

    results_table = pd.concat([results_table, coef_table], ignore_index=True)

results_table["Variable"] = results_table["Variable"].str.replace("_0_mo", "_X_mo_avg")
results_table["Variable"] = results_table["Variable"].str.replace("_3_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_6_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_9_mo", "_X_mo")

results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(4)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table.loc[results_table["p_value"] < 0.05, "Estimate"] = (
    results_table.loc[results_table["p_value"] < 0.05, "Estimate"] + "*"
)
results_table.drop(columns="p_value", inplace=True)
results_table["Variable"].unique()

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q9_prev_X_mo_avg",
    "total_precipitation_prev_X_mo_avg",
]

results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)

results_table.sort_values(by=["Variable", "Time"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Time"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(
    PLOT_PATH + "neonatal_q9_simplified_spec_coefs.csv", index=False
)

################################################################################

## 6. Simplified quantile thresholds for 95th percentile #######################

model1 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_1_mo_q95_simplified_summary.parquet"
)
model3 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_3_mo_q95_simplified_summary.parquet"
)
model6 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_6_mo_q95_simplified_summary.parquet"
)
model9 = pd.read_parquet(
    RESULTS_PATH
    + "simplified_quantiles/predictions_nnm_9_mo_q95_simplified_summary.parquet"
)

# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    "q95_prev_0_mo",
    "q95_prev_3_mo_avg",
    "q95_prev_6_mo_avg",
    "q95_prev_9_mo_avg",
    # "q95_prev_0_mo",
    # 'q95_prev_3_mo_avg',
    # 'q95_prev_6_mo_avg',
    # 'q95_prev_9_mo_avg',
]

data = pd.concat(
    [
        model1["q95_prev_0_mo"],
        model3["q95_prev_3_mo_avg"],
        model6["q95_prev_6_mo_avg"],
        model9["q95_prev_9_mo_avg"],
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
all_values = [v for v in all_values if not np.isnan(v)]

vmin = min(all_values)
vmax = max(all_values)

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
    "1-month": "q95_prev_0_mo",
    "3-month": "q95_prev_3_mo_avg",
    "6-month": "q95_prev_6_mo_avg",
    "9-month": "q95_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q95_simplified_spec.pdf")

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

# make table of summaries
results_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
SUMMARY_DIR = RESULTS_PATH + "simplified_quantiles/model_summaries/"

vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q95_prev_0_mo",
    "q95_prev_3_mo_avg",
    "q95_prev_6_mo_avg",
    "q95_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]

"""
f = 'nnm_9_mo_q9_summary.txt'
"""

summaries = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".txt")]
# only look at q9
summaries = [f for f in summaries if "_q95_simplified" in f]

for f in summaries:

    time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
    # print(time_period)
    time_period_v = f"{time_period}-month"

    coef_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
    with open(SUMMARY_DIR + f, "r") as infile:
        s = infile.read().split("\n")
    # [l for l in s]
    for l in s:
        if l.startswith(tuple(vars_of_interest)) and bool(re.findall(r"[\d]{3}", l)):
            # test if all info on single line or if it overflowed:
            if len(l.split()) == 5:
                if not l.endswith("  "):
                    coef = l.split()[0]
                    estimate = l.split()[1]
                    p_value = l.split()[-1]
                    coef_table = pd.concat(
                        [
                            coef_table,
                            pd.DataFrame.from_records(
                                [
                                    {
                                        "Time": time_period_v,
                                        "Variable": coef,
                                        "Estimate": estimate,
                                        "p_value": p_value,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 3:  # p-val missing from end
                coef = l.split()[0]
                estimate = l.split()[1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 1:  # variable and p-val
                coef = l.split()[0]
                p_value = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                coef_table.loc[coef_table["Variable"] == coef, "p_value"] = p_value

    results_table = pd.concat([results_table, coef_table], ignore_index=True)

results_table["Variable"] = results_table["Variable"].str.replace("_0_mo", "_X_mo_avg")
results_table["Variable"] = results_table["Variable"].str.replace("_3_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_6_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_9_mo", "_X_mo")

results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(4)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table.loc[results_table["p_value"] < 0.05, "Estimate"] = (
    results_table.loc[results_table["p_value"] < 0.05, "Estimate"] + "*"
)
results_table.drop(columns="p_value", inplace=True)
results_table["Variable"].unique()

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q95_prev_X_mo_avg",
    "total_precipitation_prev_X_mo_avg",
]

results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)

results_table.sort_values(by=["Variable", "Time"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Time"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(
    PLOT_PATH + "neonatal_q95_simplified_spec_coefs.csv", index=False
)

################################################################################

## 7. Quantile thresholds for 99th percentile ##################################

RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_12.01/"
model1 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_1_mo_q99_summary.parquet")
model3 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_3_mo_q99_summary.parquet")
model6 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_6_mo_q99_summary.parquet")
model9 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_9_mo_q99_summary.parquet")

# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    "q99_prev_0_mo",
    "q99_prev_3_mo_avg",
    "q99_prev_6_mo_avg",
    "q99_prev_9_mo_avg",
    # "q95_prev_0_mo",
    # 'q95_prev_3_mo_avg',
    # 'q95_prev_6_mo_avg',
    # 'q95_prev_9_mo_avg',
]

# Define custom bin edges
# custom_bins_fixed = [0, first_non_zero] + list(quartiles)
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

for model in [model1, model3, model6, model9]:
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
all_values = [v for v in all_values if not np.isnan(v)]

vmin = min(all_values)
vmax = max(all_values)

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
    "1-month": "q99_prev_0_mo",
    "3-month": "q99_prev_3_mo_avg",
    "6-month": "q99_prev_6_mo_avg",
    "9-month": "q99_prev_9_mo_avg",
}

# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q99.pdf")
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
                custom_y_bins=custom_y_bins,
            )

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF saved to {pdf_path}")

# make table of summaries
results_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
SUMMARY_DIR = RESULTS_PATH + "model_summaries/"

vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q99_prev_0_mo",
    "q99_prev_3_mo_avg",
    "q99_prev_6_mo_avg",
    "q99_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    "total_precipitation_prev_3_mo_avg",
    "total_precipitation_prev_6_mo_avg",
    "total_precipitation_prev_9_mo_avg",
]


summaries = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".txt")]
# only look at q99
summaries = [f for f in summaries if "_q99_summary" in f]

for f in summaries:

    time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
    # print(time_period)
    time_period_v = f"{time_period}-month"

    coef_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "p_value"])
    with open(SUMMARY_DIR + f, "r") as infile:
        s = infile.read().split("\n")
    # [l for l in s]
    for l in s:
        if l.startswith(tuple(vars_of_interest)) and bool(re.findall(r"[\d]{3}", l)):
            # test if all info on single line or if it overflowed:
            if len(l.split()) == 5:
                coef = l.split()[0]
                estimate = l.split()[1]
                p_value = l.split()[-1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                    "p_value": p_value,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 3:  # p-val missing from end
                coef = l.split()[0]
                estimate = l.split()[1]
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            elif len(re.findall(r"[\d]{1}\.[\d]*", l)) == 1:  # variable and p-val
                coef = l.split()[0]
                p_value = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                coef_table.loc[coef_table["Variable"] == coef, "p_value"] = p_value

    results_table = pd.concat([results_table, coef_table], ignore_index=True)

results_table["Variable"] = results_table["Variable"].str.replace("_0_mo", "_X_mo_avg")
results_table["Variable"] = results_table["Variable"].str.replace("_3_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_6_mo", "_X_mo")
results_table["Variable"] = results_table["Variable"].str.replace("_9_mo", "_X_mo")

results_table["Estimate"] = results_table["Estimate"].astype(float)
results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(4)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table.loc[results_table["p_value"] < 0.05, "Estimate"] = (
    results_table.loc[results_table["p_value"] < 0.05, "Estimate"] + "*"
)
results_table.drop(columns="p_value", inplace=True)
results_table["Variable"].unique()

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q99_prev_X_mo_avg",
    "total_precipitation_prev_X_mo_avg",
]

results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)

results_table.sort_values(by=["Variable", "Time"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Time"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(PLOT_PATH + "neonatal_q99_full_spec_coefs.csv", index=False)
