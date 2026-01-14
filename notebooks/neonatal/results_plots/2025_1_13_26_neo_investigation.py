"""
Plot neonatal mortality model with 5-year cutoffs.

1. Factor year 95th percentile
2. Factor year days over 30

"""

import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import re
import os
from tqdm import tqdm

RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/plots/"

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


################################################################################

# 1. Scam package  ##############################################

model1 = pd.read_parquet(RESULTS_PATH + "predictions_nnm_1_mo_q95_mgcv_summary.parquet")

# get min and max values for color scale consistency across plots
# as well as custom bins
# Heat maps of variables
columns_to_bin = [
    # "q9_prev_0_mo",
    # "q9_prev_3_mo_avg",
    # "q9_prev_6_mo_avg",
    # "q9_prev_9_mo_avg",
    "q95_prev_0_mo",
    # "q95_prev_3_mo_avg",
    # "q95_prev_6_mo_avg",
    # "q95_prev_9_mo_avg",
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
    "1-month": "q95_prev_0_mo",
    # "3-month": "q95_prev_3_mo_avg",
    # "6-month": "q95_prev_6_mo_avg",
    # "9-month": "q95_prev_9_mo_avg",
}


# Create a PDF to save the plots
pdf_path = os.path.join(PLOT_PATH, "neonatal_q95_scam.pdf")
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

# make table of summaries
coef_file_name = "neonatal_q95_cutoff_coefs.csv"

SUMMARY_DIR = RESULTS_PATH + "model_summaries/"

climate_var_interest = "q95_prev_0_mo"
vars_of_interest = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    "q95_prev_0_mo",
    # "q95_prev_3_mo_avg",
    # "q95_prev_6_mo_avg",
    # "q95_prev_9_mo_avg",
    "total_precipitation_prev_0_mo",
    # "total_precipitation_prev_3_mo_avg",
    # "total_precipitation_prev_6_mo_avg",
    # "total_precipitation_prev_9_mo_avg",
    # "birth_year",
]

"""
f = 'nnm_9_mo_q95_ly_summary.txt'
"""

summaries = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".txt")]
# only look at q9
summaries = [f for f in summaries if "_q95_5yr_cutoff_summary" in f]
print(summaries)

results_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "significance"])
for f in summaries:

    time_period = re.findall(r"(?<=nnm_).*(?=_mo)", f)[0]
    # print(time_period)
    time_period_v = f"{time_period}-month"

    coef_table = pd.DataFrame(columns=["Time", "Variable", "Estimate", "significance"])
    with open(SUMMARY_DIR + f, "r") as infile:
        s = infile.read().split("\n")
    # remove all lines between 'Correlation of Fixed Effects:' and 'optimizer'
    # to prevent parsing errors.
    start_idx = None
    end_idx = None
    for i, l in enumerate(s):
        if "Correlation of Fixed Effects:" in l:
            start_idx = i
        if "optimizer" in l and start_idx is not None and end_idx is None:
            end_idx = i
    if start_idx is not None and end_idx is not None:
        s = s[:start_idx] + s[end_idx + 1 :]
    # [l for l in s]
    for l in s:
        if l.startswith(tuple(vars_of_interest)) and bool(re.findall(r"[\d]{3}", l)):
            # test if all info on single line or if it overflowed:
            if len(l.split()) == 5:
                coef = l.split()[0]
                estimate = l.split()[1]
                significance = re.findall(r"\*.*", l)
                significance = significance[0] if significance else ""
                coef_table = pd.concat(
                    [
                        coef_table,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Time": time_period_v,
                                    "Variable": coef,
                                    "Estimate": estimate,
                                    "significance": significance,
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
                # significance = re.findall(r"[\d]{1}\.[\d]*", l)[0]
                significance = re.findall(r"\*.*", l)
                significance = significance[0] if significance else ""
                coef_table.loc[coef_table["Variable"] == coef, "significance"] = (
                    significance
                )

    results_table = pd.concat([results_table, coef_table], ignore_index=True)

# results_table["Variable"] = results_table["Variable"].str.replace("_0_mo", "_X_mo_avg")
# results_table["Variable"] = results_table["Variable"].str.replace("_3_mo", "_X_mo")
# results_table["Variable"] = results_table["Variable"].str.replace("_6_mo", "_X_mo")
# results_table["Variable"] = results_table["Variable"].str.replace("_9_mo", "_X_mo")

results_table["Estimate"] = results_table["Estimate"].astype(float)
# results_table["p_value"] = results_table["p_value"].astype(float)
results_table["Estimate"] = results_table["Estimate"].round(4)
results_table["Estimate"] = results_table["Estimate"].astype(str)
results_table["Estimate"] = results_table["Estimate"] + results_table["significance"]

results_table.drop(columns="significance", inplace=True)
results_table["Variable"].unique()

var_order = [
    "(Intercept)",
    "consumption_pd",
    "sex_id",
    climate_var_interest,
    "total_precipitation_prev_0_mo",
    # "birth_year",
]

results_table["Variable"] = pd.Categorical(
    results_table["Variable"], categories=var_order, ordered=True
)

results_table.sort_values(by=["Variable", "Time"], inplace=True)
results_table_wide = results_table.pivot_table(
    index=["Variable"], columns=["Time"], values="Estimate", aggfunc="first"
)
results_table_wide.reset_index(inplace=True)

results_table_wide.to_csv(PLOT_PATH + coef_file_name, index=False)

################################################################################

# 2. Make scatter plots with fit overlaying ####################################################


# get average child_mortality, consumption_pd, 'q95_prev_0_mo', s_consumption_pd_contribution,
# s_consumption_pd_contribution_upper, s_consumption_pd_contribution_lower,
# s_q95_prev_0_mo_contribution, s_q95_prev_0_mo_contribution_upper, s_q95_prev_0_mo_contribution_lower
# by psu
model_grouped_psu = (
    model1.groupby("psu")[
        [
            "child_mortality",
            "consumption_pd",
            "q95_prev_0_mo",
            "s_consumption_pd_contribution",
            "s_consumption_pd_contribution_upper",
            "s_consumption_pd_contribution_lower",
            "s_q95_prev_0_mo_contribution",
            "s_q95_prev_0_mo_contribution_upper",
            "s_q95_prev_0_mo_contribution_lower",
        ]
    ]
    .mean()
    .reset_index()
)

model_grouped_psu["log_mortality"] = np.log(model_grouped_psu["child_mortality"] + 1e-6)
model_grouped_psu["logit_mortality"] = np.log(
    model_grouped_psu["child_mortality"]
    / (1 - model_grouped_psu["child_mortality"] + 1e-6)
)
model_grouped_psu["log_s_consumption_pd_contribution"] = np.log(
    model_grouped_psu["s_consumption_pd_contribution"] + 1e-6
)
model_grouped_psu["log_s_consumption_pd_contribution_upper"] = np.log(
    model_grouped_psu["s_consumption_pd_contribution_upper"] + 1e-6
)
model_grouped_psu["log_s_consumption_pd_contribution_lower"] = np.log(
    model_grouped_psu["s_consumption_pd_contribution_lower"] + 1e-6
)
model_grouped_psu["logit_s_consumption_pd_contribution"] = np.log(
    model_grouped_psu["s_consumption_pd_contribution"]
    / (1 - model_grouped_psu["s_consumption_pd_contribution"] + 1e-6)
)
model_grouped_psu["logit_s_consumption_pd_contribution_upper"] = np.log(
    model_grouped_psu["s_consumption_pd_contribution_upper"]
    / (1 - model_grouped_psu["s_consumption_pd_contribution_upper"] + 1e-6)
)
model_grouped_psu["logit_s_consumption_pd_contribution_lower"] = np.log(
    model_grouped_psu["s_consumption_pd_contribution_lower"]
    / (1 - model_grouped_psu["s_consumption_pd_contribution_lower"] + 1e-6)
)


model_grouped_psu["log_s_q95_prev_0_mo_contribution"] = np.log(
    model_grouped_psu["s_q95_prev_0_mo_contribution"] + 1e-6
)
model_grouped_psu["log_s_q95_prev_0_mo_contribution_upper"] = np.log(
    model_grouped_psu["s_q95_prev_0_mo_contribution_upper"] + 1e-6
)
model_grouped_psu["log_s_q95_prev_0_mo_contribution_lower"] = np.log(
    model_grouped_psu["s_q95_prev_0_mo_contribution_lower"] + 1e-6
)
model_grouped_psu["logit_s_q95_prev_0_mo_contribution"] = np.log(
    model_grouped_psu["s_q95_prev_0_mo_contribution"]
    / (1 - model_grouped_psu["s_q95_prev_0_mo_contribution"] + 1e-6)
)
model_grouped_psu["logit_s_q95_prev_0_mo_contribution_upper"] = np.log(
    model_grouped_psu["s_q95_prev_0_mo_contribution_upper"]
    / (1 - model_grouped_psu["s_q95_prev_0_mo_contribution_upper"] + 1e-6)
)
model_grouped_psu["logit_s_q95_prev_0_mo_contribution_lower"] = np.log(
    model_grouped_psu["s_q95_prev_0_mo_contribution_lower"]
    / (1 - model_grouped_psu["s_q95_prev_0_mo_contribution_lower"] + 1e-6)
)

# Plot q95
# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the density of data points using hist2d
hist = ax.hist2d(
    model_grouped_psu["q95_prev_0_mo"],
    model_grouped_psu["child_mortality"],
    bins=100,
    cmap="Blues",
    norm=mcolors.LogNorm(),
)

# Add a colorbar to show density
cbar = plt.colorbar(hist[3], ax=ax)
cbar.set_label("Density")

# Overlay the line for s_q95_prev_0_mo_contribution
sorted_data = model_grouped_psu.sort_values("q95_prev_0_mo")
ax.plot(
    sorted_data["q95_prev_0_mo"],
    sorted_data["s_q95_prev_0_mo_contribution"],
    color="red",
    label="Spline Contribution",
    linewidth=2,
)

# Add labels, title, and legend
ax.set_xlabel("q95_prev_0_mo", fontsize=12)
ax.set_ylabel("Child Mortality", fontsize=12)
ax.set_title("Child Mortality vs q95_prev_0_mo with Spline Contribution", fontsize=14)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

# plot consumption
# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the density of data points using hist2d
hist = ax.hist2d(
    model_grouped_psu["consumption_pd"],
    model_grouped_psu["child_mortality"],
    bins=100,
    cmap="Blues",
    norm=mcolors.LogNorm(),
)

# Add a colorbar to show density
cbar = plt.colorbar(hist[3], ax=ax)
cbar.set_label("Density")

# Overlay the line for s_consumption_pd_contribution
sorted_data = model_grouped_psu.sort_values("consumption_pd")
ax.plot(
    sorted_data["consumption_pd"],
    sorted_data["s_consumption_pd_contribution"],
    color="red",
    label="Spline Contribution",
    linewidth=2,
)

# Add labels, title, and legend
ax.set_xlabel("consumption_pd", fontsize=12)
ax.set_ylabel("Child Mortality", fontsize=12)
ax.set_title("Child Mortality vs consumption_pd with Spline Contribution", fontsize=14)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
################################################################################
