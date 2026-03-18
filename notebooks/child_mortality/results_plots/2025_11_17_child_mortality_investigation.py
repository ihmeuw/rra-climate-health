"""
Tasks:
- Compare distribution by age-month in raw data, processed data, and low birth-weight
data
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

RAW_DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_matched_10_06_2025.parquet"
PROCESSED_DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data.parquet"
LBW_DATA_PATH = "/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/wasting_stunting_combined_2024-10-11.csv"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_24.01/"

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
    # sort by age_month
    agg_age["age_month"] = agg_age["age_month"].astype(int)
    agg_age = agg_age.sort_values(by="age_month")
    plt.figure(figsize=(30, 5))
    ax = agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False)
    plt.title("Unique Individuals per Age Month in Raw Data")
    months = agg_age["age_month"].values
    tick_positions = range(0, len(months), 12)  # Every 6 months
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([int(months[i]) for i in tick_positions])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.tight_layout()


def plot_indvs_by_age_group_by_status(data):
    # Ensure "indv_id" exists
    if "indv_id" not in data.columns:
        data = data.copy()
        data["indv_id"] = (
            data[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
        )

    # Group by age_month and child_alive, then count unique individuals
    agg_age = (
        data.groupby(["age_month", "child_alive"])["indv_id"]
        .nunique()
        .reset_index()
        .rename(columns={"indv_id": "unique_individuals"})
    )

    # Pivot the data to create a stacked bar chart
    agg_age_pivot = agg_age.pivot(
        index="age_month", columns="child_alive", values="unique_individuals"
    ).fillna(0)

    # Sort by age_month
    agg_age_pivot = agg_age_pivot.sort_index()

    # Plot the stacked bar chart
    plt.figure(figsize=(30, 5))
    ax = agg_age_pivot.plot(
        kind="bar", stacked=True, figsize=(30, 5), colormap="viridis", legend=True
    )
    plt.title("Unique Individuals per Age Month (Stacked by Child Alive)")
    plt.xlabel("Age Month")
    plt.ylabel("Unique Individuals")

    # Set tick positions and labels every 6 months
    months = agg_age_pivot.index.values
    tick_positions = range(0, len(months), 6)  # Every 6 months
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([int(months[i]) for i in tick_positions])

    # Format y-axis with commas
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.tight_layout()


# make side-by-side versions:


def plot_indvs_by_age_group_on_ax(data, ax):
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
    agg_age["age_month"] = agg_age["age_month"].astype(int)
    agg_age = agg_age.sort_values(by="age_month")
    agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False, ax=ax)
    ax.set_title("Unique Individuals per Age Month")
    months = agg_age["age_month"].values
    tick_positions = range(0, len(months), 6)  # Every 6 months
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([int(months[i]) for i in tick_positions])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.set_xlabel("Age Month")
    ax.set_ylabel("Unique Individuals")


def plot_indvs_by_age_group_by_status_on_ax(data, ax):
    # Ensure "indv_id" exists
    if "indv_id" not in data.columns:
        data = data.copy()
        data["indv_id"] = (
            data[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
        )

    agg_age = (
        data.groupby(["age_month", "child_alive"])["indv_id"]
        .nunique()
        .reset_index()
        .rename(columns={"indv_id": "unique_individuals"})
    )
    agg_age_pivot = agg_age.pivot(
        index="age_month", columns="child_alive", values="unique_individuals"
    ).fillna(0)
    agg_age_pivot = agg_age_pivot.sort_index()
    agg_age_pivot.plot(kind="bar", stacked=True, colormap="viridis", legend=True, ax=ax)
    ax.set_title("Unique Individuals per Age Month (Stacked by Child Alive)")
    months = agg_age_pivot.index.values
    tick_positions = range(0, len(months), 6)  # Every 6 months
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([int(months[i]) for i in tick_positions])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.set_xlabel("Age Month", fontsize=15)
    ax.set_ylabel("Unique Individuals", fontsize=15)


## READ IN DATA

df_proc = pd.read_parquet(PROCESSED_DATA_PATH)
# Raw data
df_raw = pd.read_parquet(RAW_DATA_PATH)
df_raw = df_raw[df_raw["age_month"].notnull()]
df_raw["age_month"] = df_raw["age_month"].astype(int)

# lbw data
df_lbw = pd.read_csv(LBW_DATA_PATH)

# lbw raw - no permissions
path = "/ihme/mnch/lbwsg/data/extraction/microdata/bw_only/ubcov/DHS_for_imputation/gbd2019"
df_lbw_raw = pd.concat(
    [
        pd.read_csv(f"{path}/{file}")
        for file in os.listdir(path)
        if file.endswith(".csv")
    ],
    ignore_index=True,
)

## PLOTTING
plot_indvs_by_age_group(df_proc)
plot_indvs_by_age_group(df_proc[df_proc["age_month"] < 60])

plot_indvs_by_age_group(df_raw)
plot_indvs_by_age_group(df_raw[df_raw["age_month"] <= 60])

plot_indvs_by_age_group(df_lbw)

plot_indvs_by_age_group_by_status(df_proc[df_proc["age_month"] < 60])

plot_indvs_by_age_group_by_status(df_raw[df_raw["age_month"] <= 60])

# Make side-by-side plots
# raw mortality versus lbw data
with PdfPages(PLOT_PATH + "age_month_mortality_raw_vs_cgf.pdf") as pdf:
    fig, axes = plt.subplots(
        1, 2, figsize=(20, 5), sharey=True
    )  # 1 row, 2 columns, share y-axis
    plot_indvs_by_age_group_on_ax(df_raw, axes[0])
    plot_indvs_by_age_group_on_ax(df_lbw, axes[1])
    axes[0].axvline(x=60, color="red", linestyle="--", linewidth=2)  # Add vertical line
    axes[0].set_title("Raw Mortality Data", fontsize=18)
    axes[1].set_title("Pre-Treatment CGF Data", fontsize=18)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# make three rows for mortality: raw by status, processed by status, processed <60 months by status

# Make three rows for mortality: raw by status, processed by status, processed <60 months by status
with PdfPages(PLOT_PATH + "mortality_by_status_shared_x.pdf") as pdf:
    fig, axes = plt.subplots(
        3, 1, figsize=(20, 25), sharex=True  # Share x-axis instead of y-axis
    )  # 3 rows, 1 column
    plot_indvs_by_age_group_by_status_on_ax(df_raw, axes[0])
    axes[0].set_xlim(-0.5, 60.5)

    plot_indvs_by_age_group_by_status_on_ax(df_proc, axes[1])
    axes[1].set_xlim(-0.5, 60.5)

    plot_indvs_by_age_group_by_status_on_ax(df_proc[df_proc["age_month"] < 60], axes[2])
    axes[2].set_xlim(-0.5, 60.5)

    # Set titles for each subplot
    axes[0].set_title("Raw Data (0-60 Months)", fontsize=18)
    axes[1].set_title("Processed Data (0-60 Months)", fontsize=18)
    axes[2].set_title("Processed Data (<60 Months)", fontsize=18)

    # Add a shared x-axis label

    # Adjust layout
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
