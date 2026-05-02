LDI_VERSION = "v6"
import multiprocessing as mp
from functools import partial
from pathlib import Path

import re
import click
import geopandas as gpd
import logging
import numpy as np
import os
import pandas as pd
import polars as pl
import rioxarray
from tqdm import tqdm
import sys
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
from scipy.interpolate import PchipInterpolator


import rra_climate_health.cli_options as clio
from rra_climate_health import paths

# from rra_climate_health.data_prep import upstream_paths
from rra_climate_health.data import (
    DEFAULT_ROOT,
    ClimateMalnutritionData,
)

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

WEALTH_DATA_ROOT = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions"
)

WEALTH_DATA_PATHS = {
    "LSMS": WEALTH_DATA_ROOT / "LSMS_wealth.parquet",
    "DHS": WEALTH_DATA_ROOT / "DHS_wealth.parquet",
    "MICS": WEALTH_DATA_ROOT / "MICS_wealth.parquet",
}

SURVEY_DATA_ROOT = Path(
    "/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data"
)

EXTRACTIONS_ROOT = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/"
)

SDI_PATH = Path("/mnt/share/forecasting/data/7/past/sdi/20240531_gk24/sdi.nc")
# /mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/wasting_stunting_combined_2024-10-11.csv
SURVEY_DATA_PATHS = {
    "bmi": {"gbd": SURVEY_DATA_ROOT / "bmi" / "bmi_data_outliered_wealth_rex.csv"},
    "cgf": {
        "gbd": SURVEY_DATA_ROOT
        / "wasting_stunting"
        / "wasting_stunting_combined_2024-10-11.csv",
        "lsae": "/mnt/share/limited_use/LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/pre_collapse/cgf_lbw_2020_06_15.csv",
    },
    "wealth": {
        "LSMS": WEALTH_DATA_ROOT / "LSMS_wealth.parquet",
        "DHS": WEALTH_DATA_ROOT / "DHS_wealth.parquet",
        "MICS": WEALTH_DATA_ROOT / "MICS_wealth.parquet",
    },
    "anemia": EXTRACTIONS_ROOT / "anemia" / "anemia_extracts_compiled_09_02_2025.csv",
    "child_mortality": {
        "dem_br": EXTRACTIONS_ROOT / "dem_br",
        "dem_vr": EXTRACTIONS_ROOT / "dem_vr",
        "dem_br_vr": EXTRACTIONS_ROOT / "dem_br_vr",
    },
}

DATA_SOURCE_TYPE = {
    "stunting": "cgf",
    "wasting": "cgf",
    "underweight": "cgf",
    "low_adult_bmi": "bmi",
    "anemia": "anemia",
}
MEASURES_IN_SOURCE = {
    "cgf": ["stunting", "wasting", "underweight"],
    "bmi": ["low_adult_bmi"],
    "anemia": ["anemia"],
    "child_mortality": ["child_alive"],
}

PLOT_DIR = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2026_04_28.01/"
os.makedirs(PLOT_DIR, exist_ok=True)

## Load data ###################################################################

stunting = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/stunting/training_data/2026_04_28.01/data.parquet"
)
stunting_indv = stunting[stunting["line_id"].notna()]
stunting_indv["indv_id"] = stunting_indv["indv_id"].str.replace(".0", "", regex=False)

# stunting_sub = pd.read_parquet(
#     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/inspect_data/stunting_sub.parquet"
# )

df_subset = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/inspect_data/cm_subset.parquet"
)

stunting_match = stunting_indv[stunting_indv["indv_id"].isin(df_subset["indv_id"])]
len(stunting_match)
stunting_match["days_over_30C_month"].describe()
df_subset["days_over_30C_monthly_within_bin"].describe()


# df_month_sub = pd.read_parquet(
#     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/inspect_data/df_month_sub.parquet"
# )

# df_month_sub["days_over_30C_monthly"].describe()


df_monthly = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_13.01/data_monthly_expanded_rel_thresholds.parquet"
)

bins = [0, 1, 3, 6, 12, 24, 36, 48, 60]
labels = ["0-1m", "2-3m", "4-6m", "7-12m", "13-24m", "25-36m", "37-48m", "49-60m"]

df_monthly["age_group"] = pd.cut(
    df_monthly["age_month"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False,
)

for ag in labels:
    df_monthly[df_monthly["age_group"] == ag].to_parquet(
        f"/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_13.01/bin_subsets/data_monthly_age_group_{ag}.parquet",
        index=False,
    )

#


## Latest data set with all days over meaures available. By age bin, distributions
# of each measure to see how they differ. Plot avg deaths per unit of threshold change
df_cumulative = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_28.01/data_cumulative_bins.parquet"
)

# days_over_30C (annual)
# days_over_30C_monthly (monthly average for since birth)
# days_over_30C_monthly_constant (monthly avg for last age of child)
# days_over_30C_constant ?

df_within_bin = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/data_within_bin.parquet"
)
# days_over_30C_monthly_within_bin

# Combine datasets
for ag in [
    "age_1_m",
    "age_3_m",
    "age_6_m",
    "age_12_m",
    "age_24_m",
    "age_36_m",
    "age_48_m",
    "age_60_m",
]:
    df_within_bin.loc[df_within_bin[ag] == 1, "age_group"] = ag
    df_cumulative.loc[df_cumulative[ag] == 1, "age_group"] = ag

df_combined = pd.merge(
    df_within_bin[
        [
            "indv_id",
            "sex_id",
            # "int_month",
            "int_birth_year_diff_months",
            "age_month",
            "age_month_original",
            "age_group",
            "days_over_30C_within_bin",
            "days_over_30C_monthly_within_bin",
            "child_mortality",
        ]
    ],
    df_cumulative[
        [
            "indv_id",
            "sex_id",
            # "int_month",
            "int_birth_year_diff_months",
            "age_month",
            "age_month_original",
            "age_group",
            "days_over_30C_monthly_cumul",
            "days_over_30C_monthly_constant",
            "child_mortality",
        ]
    ],
    on=[
        "indv_id",
        "sex_id",
        # "int_month",
        "int_birth_year_diff_months",
        "age_month",
        "age_month_original",
        "age_group",
        "child_mortality",
    ],
    how="inner",
)

df_combined = df_combined[df_combined["int_birth_year_diff_months"] <= 120]


stunting_monthly = pd.read_parquet(
    "/mnt/share/scratch/users/victorvt/for/elyeb/intermediate_stunting_monthly_data.parquet"
)

## EXAMIN OVERLAPPING DATA #####################################################

# Check stunting for duplicate indv_id
print(f"{len(stunting_indv['indv_id']):,} total rows in stunting_indv")
print(f"{stunting_indv['indv_id'].nunique():,} unique indv_id in stunting_indv")

stunting_dup = stunting_indv[stunting_indv.duplicated(subset=["indv_id"], keep=False)]
print(f"{len(stunting_dup):,} duplicate rows in stunting_indv")
stunting_dup = stunting_dup.sort_values("indv_id").reset_index(drop=True)
stunting_dup_check = stunting_dup[
    stunting_dup["indv_id"] == stunting_dup["indv_id"].iloc[0]
]

stunting_indv["sex_id"] = stunting_indv["sex_id"].astype(int)
stunting_indv["indv_id"] += "_" + stunting_indv["sex_id"].astype(str)

df_combined["indv_id"] += "_" + df_combined["sex_id"].astype(str)

df_cm_stunting_match = df_combined[
    df_combined["indv_id"].isin(stunting_indv["indv_id"])
]
print(
    f"Number of overlapping individuals in combined and stunting_indv: {df_cm_stunting_match['indv_id'].nunique():,}"
)  # 95,995
stunting_match = stunting_indv[stunting_indv["indv_id"].isin(df_combined["indv_id"])]
print(
    f"Number of overlapping individuals in combined and stunting_indv: {stunting_match['indv_id'].nunique():,}"
)  # 95,995

# get max age_months per indv_id
df_cm_stunting_match_max_age = (
    df_cm_stunting_match.sort_values("age_month")
    .groupby("indv_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)


df_cm_stunting_match_max_age["days_over_30C_monthly_cumul"].describe()
stunting_match["days_over_30C_month"].describe()

stunting_merge = stunting_match.rename(
    columns={
        "days_over_30C_month": "days_over_30C_month_stunting",
        "age_month": "age_month_stunting",
    }
)
stunting_merge = stunting_merge[
    ["indv_id", "age_month_stunting", "days_over_30C_month_stunting"]
]
# drop any duplicates
stunting_merge_unique = stunting_merge[
    ~stunting_merge.duplicated(subset=["indv_id"], keep=False)
]
stunting_merge_unique["indv_id"].nunique()  # 81427

df_cm_stunting_merge = df_cm_stunting_match_max_age.merge(
    stunting_merge_unique,
    on="indv_id",
    how="inner",
)
f"{len(df_cm_stunting_merge):,} rows in merged dataset"
f"{df_cm_stunting_merge['indv_id'].nunique():,} unique individuals in merged dataset"

# Compare age_months
df_cm_stunting_merge[
    [
        "indv_id",
        "age_month",
        "age_month_original",
        "age_month_stunting",
        "child_mortality",
    ]
].head()
df_cm_stunting_merge[
    df_cm_stunting_merge["age_month_original"]
    != df_cm_stunting_merge["age_month_stunting"]
]

# If age_month_original != age_month_stunting, probably not same indv
df_cm_stunting_merge = df_cm_stunting_merge[
    df_cm_stunting_merge["age_month_original"]
    == df_cm_stunting_merge["age_month_stunting"]
]
f"{df_cm_stunting_merge['indv_id'].nunique():,} unique individuals in merged dataset"  # 72,222
len(df_cm_stunting_merge)

# Sanity check. Why would I have age_month 0 and age_month_original 10 for child_mortality 0?
# This is max age...
# Test case indv_id 20649_252_561_1
test_monthly = df_monthly[df_monthly["indv_id"] == "20649_252_561_1"]
test_monthly["age_month"]  # 0 through 10
test_df_combined = df_combined[df_combined["indv_id"] == "20649_252_561_1_1"]
test_df_cumul = df_cumulative[df_cumulative["indv_id"] == "20649_252_561_1"]
test_df_cumul[["age_month", "age_month_original", "child_mortality"]]
test_df_within = df_within_bin[df_within_bin["indv_id"] == "20649_252_561_1"]
test_df_within[["age_month", "age_month_original", "child_mortality"]]
test_max_age = df_cm_stunting_match_max_age[
    df_cm_stunting_match_max_age["indv_id"] == "20649_252_561_1_1"
]
df_cm_stunting_match[df_cm_stunting_match["indv_id"] == "20649_252_561_1_1"][
    ["age_month", "age_month_original", "child_mortality"]
]
# Why doesn't int_year and int_month increment along with age_month in df_cumulative?
# int_year/month weren't in indentity_vars while grouping in data processing, so they
# are no longer correct. However, I can manually see if calculations were done right
# try non-zero-days over example, 19167_452_7_1_1 or 19167_452_7_1
test_monthly = df_monthly[df_monthly["indv_id"] == "19167_452_7_1"]
test_monthly["age_month"]  # 0 through 10
test_df_combined = df_combined[df_combined["indv_id"] == "19167_452_7_1_1"]
test_df_cumul = df_cumulative[df_cumulative["indv_id"] == "19167_452_7_1"]
test_df_cumul[["age_month", "age_month_original", "child_mortality"]]
test_df_within = df_within_bin[df_within_bin["indv_id"] == "19167_452_7_1"]
test_df_within[["age_month", "age_month_original", "child_mortality"]]
pd.set_option("display.max_columns", None)
# create scatter below of df_cm_stunting_merge


# Try same but overall, not just among subset
stunting_overlap = stunting_indv[
    stunting_indv["indv_id"].isin(df_cumulative["indv_id"])
]
df_cumul_stunting_match_max_age = (
    df_cumulative.sort_values("age_month")
    .groupby("indv_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)
df_cumul_stunting_match_max_age_overlap = df_cumul_stunting_match_max_age[
    df_cumul_stunting_match_max_age["indv_id"].isin(stunting_overlap["indv_id"])
]
df_cumul_stunting_match_max_age_overlap["days_over_30C_monthly_cumul"].describe()
stunting_overlap["days_over_30C_month"].describe()

# test out few examples:
df_cumul_exam = df_cumul_stunting_match_max_age_overlap[
    df_cumul_stunting_match_max_age_overlap["days_over_30C_monthly_cumul"] == 31
]
df_cumul_full_exam = df_cumulative[
    df_cumulative["indv_id"].isin(df_cumul_exam["indv_id"])
]
stunting_exam_overlap = stunting_overlap[
    stunting_overlap["indv_id"].isin(df_cumul_exam["indv_id"])
]


# Look at monthly
stunting_monthly_indv = stunting_monthly[
    ~stunting_monthly["indv_id"].str.endswith("_nan")
]
stunting_monthly_indv["indv_id"] = stunting_monthly_indv["indv_id"].str.replace(
    ".0", "", regex=False
)

stunting_monthly_exam_overlap = stunting_monthly_indv[
    stunting_monthly_indv["indv_id"].isin(df_cumulative["indv_id"])
]

## PLOTS #######################################################################


## Scatters

# Compare overlapping individuals in df_monthly and stunting_monthly_indv
df_monthly["cm"] = "child_mortality"
stunting_monthly_indv["stunting"] = "stunting"

df_monthly_overlap = df_monthly[
    df_monthly["indv_id"].isin(stunting_monthly_indv["indv_id"])
]
stunting_monthly_indv_merge = stunting_monthly_indv[
    ["indv_id", "climate_year", "climate_month", "days_over_30C"]
]
stunting_monthly_indv_merge.rename(
    columns={
        "climate_year": "int_year",
        "climate_month": "int_month",
        "days_over_30C": "days_over_30C_stunting",
    },
    inplace=True,
)

df_monthly_overlap.rename(
    columns={"days_over_30C_monthly": "days_over_30C_child_mortality"}, inplace=True
)
df_monthly_overlap = df_monthly_overlap.merge(
    stunting_monthly_indv_merge,
    on=["indv_id", "int_year", "int_month"],
    how="inner",
)
print(f"Number of overlapping individuals: {df_monthly_overlap['indv_id'].nunique()}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    df_monthly_overlap["days_over_30C_child_mortality"],
    df_monthly_overlap["days_over_30C_stunting"],
    alpha=0.3,
)
ax.set_xlabel("Days Over 30C Child Mortality")
ax.set_ylabel("Days Over 30C Stunting")
ax.set_title("Scatter Plot of Days Over 30C (Child Mortality vs Stunting)")
ax.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR + "scatter_days_over_30C_child_mortality_vs_stunting.png")
plt.show()

# Repeat but for processed average days
age_group_order = {
    "age_1_m": 1,
    "age_3_m": 2,
    "age_6_m": 3,
    "age_12_m": 4,
    "age_24_m": 5,
    "age_36_m": 6,
    "age_48_m": 7,
    "age_60_m": 8,
}
df_combined["age_group_order"] = df_combined["age_group"].map(age_group_order)
df_combined_max_age = (
    df_combined.sort_values("age_group_order")
    .groupby("indv_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

# df_combined[df_combined["age_group"] == "age_60_m"]
df_combined_max_age_overlap = df_combined_max_age[
    df_combined_max_age["indv_id"].isin(stunting_indv["indv_id"])
]


len(df_combined_max_age_overlap)
df_combined_max_age_overlap["indv_id"].nunique()
df_processed_merged = (
    df_cm_stunting_match[
        [
            "indv_id",
            "child_mortality",
            "days_over_30C_monthly_cumul",
            "days_over_30C_monthly_constant",
            "days_over_30C_monthly_within_bin",
        ]
    ]
    .rename(
        columns={
            "days_over_30C_monthly_cumul": "days_over_30C_monthly_cumul_child_mortality",
            "days_over_30C_monthly_constant": "days_over_30C_monthly_constant_child_mortality",
            "days_over_30C_monthly_within_bin": "days_over_30C_monthly_within_bin_child_mortality",
        }
    )
    .merge(
        stunting_indv[["indv_id", "days_over_30C_month"]]
        .drop_duplicates()
        .rename(columns={"days_over_30C_month": "days_over_30C_month_stunting"}),
        on="indv_id",
        how="inner",
    )
)
# make 3 plots: days_over_30C_monthly_cumul_child_mortality vs days_over_30C_month_stunting
# days_over_30C_monthly_within_bin_child_mortality vs days_over_30C_month_stunting
# days_over_30C_monthly_constant_child_mortality vs days_over_30C_month_stunting

df_cm_stunting_merge.rename(
    columns={
        "days_over_30C_monthly_cumul": "days_over_30C_monthly_cumul_child_mortality",
        "days_over_30C_monthly_within_bin": "days_over_30C_monthly_within_bin_child_mortality",
        "days_over_30C_monthly_constant": "days_over_30C_monthly_constant_child_mortality",
    },
    inplace=True,
)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

colors = df_cm_stunting_merge["child_mortality"].map({0: "blue", 1: "red"})


axes[0].scatter(
    df_cm_stunting_merge["days_over_30C_monthly_cumul_child_mortality"],
    df_cm_stunting_merge["days_over_30C_month_stunting"],
    alpha=0.3,
    # c=colors,
)
axes[0].set_xlabel("Days Over 30C Monthly Cumulative (Child Mortality)")
axes[0].set_ylabel("Days Over 30C Month (Stunting)")
axes[0].set_title("Scatter Plot of Days Over 30C (Monthly Cumulative vs Stunting)")
axes[0].grid(True)
axes[1].scatter(
    df_cm_stunting_merge["days_over_30C_monthly_within_bin_child_mortality"],
    df_cm_stunting_merge["days_over_30C_month_stunting"],
    alpha=0.3,
    # c=colors,
)
axes[1].set_xlabel("Days Over 30C Monthly Within Bin (Child Mortality)")
axes[1].set_ylabel("Days Over 30C Month (Stunting)")
axes[1].set_title("Scatter Plot of Days Over 30C (Monthly Within Bin vs Stunting)")
axes[1].grid(True)
# axes[2].scatter(
#     df_cm_stunting_merge["days_over_30C_monthly_constant_child_mortality"],
#     df_cm_stunting_merge["days_over_30C_month_stunting"],
#     alpha=0.3,
#     c=colors,
# )
# axes[2].set_xlabel("Days Over 30C Monthly Constant (Child Mortality)")
# axes[2].set_ylabel("Days Over 30C Month (Stunting)")
# axes[2].set_title("Scatter Plot of Days Over 30C (Monthly Constant vs Stunting)")
# axes[2].grid(True)
plt.tight_layout()
plt.savefig(
    PLOT_DIR
    + "scatter_days_over_30C_child_mortality_vs_stunting_processed_all_ages.png"
)
plt.show()

investigate_df = df_cm_stunting_merge[
    (df_cm_stunting_merge["days_over_30C_monthly_cumul_child_mortality"] <= 1)
    & (df_cm_stunting_merge["days_over_30C_month_stunting"] >= 10)
]
investigate_df = investigate_df[investigate_df["child_mortality"] == 0]
investigate_df["age_group"].value_counts()
investigate_df["indv_id"].iloc[0]  # 398033_53_129_1_1

investigate_cm = df_monthly[df_monthly["indv_id"] == "398033_53_129_1"]
investigate_stunting = stunting_monthly[
    stunting_monthly["indv_id"] == "398033_53_129_1.0"
]

# df_combined_upper = df_combined[df_combined["age_group"] == "age_60_m"]

# Make age-group-specific plots
for i, ag in enumerate(
    [
        "age_1_m",
        "age_3_m",
        "age_6_m",
        "age_12_m",
        "age_24_m",
        "age_36_m",
        "age_48_m",
        "age_60_m",
    ]
):
    subset = df_combined[df_combined["age_group"] == ag]

    # Create a figure for each age group
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed

    # Scatter days_over_30C_within_bin and days_over_30C_monthly_cumul
    ax.scatter(
        subset["days_over_30C_monthly_cumul"],
        subset["days_over_30C_monthly_within_bin"],
        alpha=0.3,
    )
    ax.set_xlabel("Days Over 30C Monthly Cumulative")
    ax.set_ylabel("Days Over 30C Within Bin")
    ax.set_title(
        f"Scatter Plot of Days Over 30C (Monthly Cumulative vs Within Bin) for {ag}"
    )
    ax.grid(True)

    # Save the figure as a PNG file
    png_path = PLOT_DIR + f"scatter_days_over_30C_{ag}.png"
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    print(f"PNG saved to {png_path}")


## Make plots of average mortality per days over 30 bin
# for each version
# custom_bins = [0, 5, 10, 15, 20, 25, 30]
custom_bins = [0, 1.5, 5.25, 9.3, 15.5, 31]
# custom_bins = [r for r in range(0, 31, 1)]

df_combined["do30_cumul_bin"] = pd.cut(
    df_combined["days_over_30C_monthly_cumul"],
    bins=custom_bins,
    # labels=True,
    include_lowest=True,
    right=False,
)
df_combined["do30_cumul_bin"].value_counts()

df_combined["do30_within_bin"] = pd.cut(
    df_combined["days_over_30C_monthly_within_bin"],
    bins=custom_bins,
    # labels=True,
    include_lowest=True,
    right=False,
)
df_combined["do30_within_bin"].value_counts()


# Get percentage of deaths for each age group
deaths_by_age_group = (
    df_combined.groupby("age_group")[["child_mortality"]].sum().reset_index()
)
deaths_by_age_group["percentage_deaths"] = (
    deaths_by_age_group["child_mortality"]
    / deaths_by_age_group["child_mortality"].sum()
) * 100
deaths_by_age_group["percentage_deaths"] = round(
    deaths_by_age_group["percentage_deaths"], 1
)
deaths_by_age_group["percentage_deaths"] = (
    deaths_by_age_group["percentage_deaths"].astype(str) + "%"
)
deaths_by_age_group["percentage_deaths"] = (
    deaths_by_age_group["percentage_deaths"] + " of total deaths"
)

df_cumul_grouped = (
    df_combined.groupby(["age_group", "do30_cumul_bin"])[["child_mortality"]]
    .mean()
    .reset_index()
)

df_within_grouped = (
    df_combined.groupby(["age_group", "do30_within_bin"])[["child_mortality"]]
    .mean()
    .reset_index()
)

# Make line plots of avg mortality by days over 30 bin, faceted by age group, for each version
# in two columns of 8 plots each. Save results to PDF
age_labels = [
    "age_1_m",
    "age_3_m",
    "age_6_m",
    "age_12_m",
    "age_24_m",
    "age_36_m",
    "age_48_m",
    "age_60_m",
]
bin_labels = [str(b) for b in df_combined["do30_cumul_bin"].cat.categories]


pct_deaths_lookup = deaths_by_age_group.set_index("age_group")[
    "percentage_deaths"
].to_dict()

fig, axes = plt.subplots(len(age_labels), 2, figsize=(16, 6 * len(age_labels) + 10))

for i, ag in enumerate(age_labels):
    age_data_cumul = df_cumul_grouped[df_cumul_grouped["age_group"] == ag].copy()
    age_data_within = df_within_grouped[df_within_grouped["age_group"] == ag].copy()

    age_data_cumul["bin_mid"] = age_data_cumul["do30_cumul_bin"].apply(lambda x: x.mid)
    age_data_within["bin_mid"] = age_data_within["do30_within_bin"].apply(
        lambda x: x.mid
    )

    pct_deaths = pct_deaths_lookup.get(ag, "")

    # Plot cumulative bins (column 0)
    axes[i, 0].plot(
        age_data_cumul["bin_mid"],
        age_data_cumul["child_mortality"],
        marker="o",
        label="Cumulative Bins",
        color="blue",
    )
    axes[i, 0].set_title(f"Age Group: {ag} ({pct_deaths}) (Cumulative)")
    axes[i, 0].set_xlabel("Days Over 30C (Cumulative Bin)")
    axes[i, 0].set_ylabel("Average Mortality")
    axes[i, 0].set_ylim(0, 1)
    axes[i, 0].set_xticks(custom_bins)
    axes[i, 0].tick_params(axis="x", rotation=45)
    axes[i, 0].legend()
    axes[i, 0].grid()

    # Plot within bins (column 1)
    axes[i, 1].plot(
        age_data_within["bin_mid"],
        age_data_within["child_mortality"],
        marker="o",
        label="Within Bins",
        color="green",
    )
    axes[i, 1].set_title(f"Age Group: {ag} (Within)")
    axes[i, 1].set_xlabel("Days Over 30C (Within Bin)")
    axes[i, 1].set_ylabel("Average Mortality")
    axes[i, 1].set_ylim(0, 1)
    axes[i, 1].set_xticks(custom_bins)
    axes[i, 1].tick_params(axis="x", rotation=45)
    axes[i, 1].legend()
    axes[i, 1].grid()

# Save the entire figure as a PNG file
plt.tight_layout()
output_path = PLOT_DIR + "average_mortality_by_days_over_30_bins_1_increments.png"
plt.savefig(output_path)
plt.close(fig)

print(f"Plot saved to {output_path}")

# Scatter average mortality within each climate bin, faceted by age group
cumul_grouped = (
    df_combined.groupby(["age_group", "do30_cumul_bin"])[["child_mortality"]]
    .mean()
    .reset_index()
)
within_grouped = (
    df_combined.groupby(["age_group", "do30_within_bin"])[["child_mortality"]]
    .mean()
    .reset_index()
)
cumul_grouped.rename(
    columns={"child_mortality": "mortality_cumul_avg", "do30_cumul_bin": "do30_bin"},
    inplace=True,
)
within_grouped.rename(
    columns={"child_mortality": "mortality_within_avg", "do30_within_bin": "do30_bin"},
    inplace=True,
)
bin_grouped = cumul_grouped.merge(
    within_grouped, on=["age_group", "do30_bin"], how="inner"
)


fig, axes = plt.subplots(len(age_labels), 1, figsize=(16, 6 * len(age_labels) + 10))

for i, ag in enumerate(age_labels):

    age_data = bin_grouped[bin_grouped["age_group"] == ag].copy()
    age_data = age_data.dropna(subset=["mortality_cumul_avg", "mortality_within_avg"])
    max_val = max(
        age_data["mortality_cumul_avg"].max(), age_data["mortality_within_avg"].max()
    )

    # scatter mortality_cumul_avg and mortality_within_avg
    axes[i].scatter(
        age_data["mortality_cumul_avg"],
        age_data["mortality_within_avg"],
        color="blue",
        # alpha=0.7,
    )
    axes[i].set_title(f"Age Group: {ag}")
    axes[i].set_xlabel("Average Mortality (Cumulative Bin)")
    axes[i].set_ylabel("Average Mortality (Within Bin)")
    axes[i].set_xlim(0, max_val)
    axes[i].set_ylim(0, max_val)
    axes[i].set_aspect("equal", adjustable="box")
    # axes[i].set_xticks(custom_bins)
    axes[i].tick_params(axis="x", rotation=45)
    axes[i].legend()
    axes[i].grid()


# scatter days_over_30C against average mortality, grouped by consumption bin
consumption_bins = [
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
df_within_bin["consumption_bin"] = pd.cut(
    df_within_bin["consumption_pd_within_bin"],
    bins=consumption_bins,
    include_lowest=True,
    right=False,
)
df_cumulative["consumption_bin"] = pd.cut(
    df_cumulative["consumption_pd_cumul"],
    bins=consumption_bins,
    include_lowest=True,
    right=False,
)


fig, axes = plt.subplots(len(age_labels), 2, figsize=(16, 6 * len(age_labels) + 10))

for i, ag in enumerate(age_labels):
    age_data_cumul = df_cumulative[df_cumulative["age_group"] == ag].copy()
    age_data_within = df_within_bin[df_within_bin["age_group"] == ag].copy()

    age_data_cumul_grouped = (
        age_data_cumul.groupby("consumption_bin")[
            ["child_mortality", "days_over_30C_monthly_cumul"]
        ]
        .mean()
        .reset_index()
    )
    age_data_within_grouped = (
        age_data_within.groupby("consumption_bin")[
            ["child_mortality", "days_over_30C_monthly_within_bin"]
        ]
        .mean()
        .reset_index()
    )

    x_axis_max = max(
        age_data_cumul_grouped["days_over_30C_monthly_cumul"].max(),
        age_data_within_grouped["days_over_30C_monthly_within_bin"].max(),
    )
    y_axis_max = max(
        age_data_cumul_grouped["child_mortality"].max(),
        age_data_within_grouped["child_mortality"].max(),
    )
    axes[i, 0].set_xlim(0, x_axis_max)
    axes[i, 0].set_ylim(0, y_axis_max)
    axes[i, 1].set_xlim(0, x_axis_max)
    axes[i, 1].set_ylim(0, y_axis_max)
    # Plot cumulative bins (column 0)
    axes[i, 0].scatter(
        age_data_cumul_grouped["days_over_30C_monthly_cumul"],
        age_data_cumul_grouped["child_mortality"],
        color="blue",
        # alpha=0.7,
    )
    axes[i, 0].set_title(f"Age Group: {ag} (Cumulative)")
    axes[i, 0].set_xlabel("Days Over 30C Monthly Cumulative")
    axes[i, 0].set_ylabel("Average Mortality")
    # axes[i, 0].set_ylim(0, 1)
    # axes[i, 0].set_xticks(custom_bins)
    axes[i, 0].tick_params(axis="x", rotation=45)
    axes[i, 0].legend()
    axes[i, 0].grid()
    # Plot within bins (column 1)
    axes[i, 1].scatter(
        age_data_within_grouped["days_over_30C_monthly_within_bin"],
        age_data_within_grouped["child_mortality"],
        color="blue",
        # alpha=0.7,
    )
    axes[i, 1].set_title(f"Age Group: {ag} (Within)")
    axes[i, 1].set_xlabel("Days Over 30C Monthly Within Bin")
    axes[i, 1].set_ylabel("Average Mortality")
    # axes[i, 1].set_ylim(0, 1)
    # axes[i, 1].set_xticks(custom_bins)
    axes[i, 1].tick_params(axis="x", rotation=45)
    axes[i, 1].legend()
    axes[i, 1].grid()


# Save the entire figure as a PNG file
plt.tight_layout()
output_path = PLOT_DIR + "average_mortality_by_days_over_30_bins_1_increments.png"
plt.savefig(output_path)
plt.close(fig)

print(f"Plot saved to {output_path}")


# Save the entire figure as a PNG file
plt.tight_layout()
output_path = PLOT_DIR + "average_mortality_by_days_over_30_bins_1_increments.png"
plt.savefig(output_path)
plt.close(fig)

print(f"Plot saved to {output_path}")


## Make plots of average mortality per days over 30 bin
# for each version
custom_bins = [0, 5, 10, 15, 20, 25, 30]
custom_bins = [0, 1.5, 5.25, 9.3, 15.5, 31]
custom_bins = [r for r in range(0, 31, 1)]

df_combined["do30_cumul_bin"] = pd.cut(
    df_combined["days_over_30C_monthly_cumul"],
    bins=custom_bins,
    # labels=True,
    include_lowest=True,
    right=False,
)
df_combined["do30_cumul_bin"].value_counts()

df_combined["do30_within_bin"] = pd.cut(
    df_combined["days_over_30C_monthly_within_bin"],
    bins=custom_bins,
    # labels=True,
    include_lowest=True,
    right=False,
)
df_combined["do30_within_bin"].value_counts()


df_cumul_rows_grouped = (
    df_combined.groupby(["age_group", "do30_cumul_bin"])
    .agg(
        child_mortality_sum=("child_mortality", "sum"),
        total_rows=("child_mortality", "count"),
    )
    .reset_index()
)

df_within_rows_grouped = (
    df_combined.groupby(["age_group", "do30_within_bin"])
    .agg(
        child_mortality_sum=("child_mortality", "sum"),
        total_rows=("child_mortality", "count"),
    )
    .reset_index()
)

# Convert to percents
df_cumul_rows_grouped["child_mortality_percent"] = round(
    (
        df_cumul_rows_grouped["child_mortality_sum"]
        / df_cumul_rows_grouped["child_mortality_sum"].sum()
    )
    * 100,
    1,
)
df_cumul_rows_grouped["total_rows_percent"] = round(
    (df_cumul_rows_grouped["total_rows"] / df_cumul_rows_grouped["total_rows"].sum())
    * 100,
    1,
)
df_within_rows_grouped["child_mortality_percent"] = round(
    (
        df_within_rows_grouped["child_mortality_sum"]
        / df_within_rows_grouped["child_mortality_sum"].sum()
    )
    * 100,
    1,
)
df_within_rows_grouped["total_rows_percent"] = round(
    (df_within_rows_grouped["total_rows"] / df_within_rows_grouped["total_rows"].sum())
    * 100,
    1,
)


# Make line plots of avg mortality by days over 30 bin, faceted by age group, for each version
# in two columns of 8 plots each. Save results to PDF
age_labels = [
    "age_1_m",
    "age_3_m",
    "age_6_m",
    "age_12_m",
    "age_24_m",
    "age_36_m",
    "age_48_m",
    "age_60_m",
]
bin_labels = [str(b) for b in df_combined["do30_cumul_bin"].cat.categories]


pct_deaths_lookup = deaths_by_age_group.set_index("age_group")[
    "percentage_deaths"
].to_dict()

fig, axes = plt.subplots(len(age_labels), 2, figsize=(16, 6 * len(age_labels) + 10))

for i, ag in enumerate(age_labels):
    age_data_cumul = df_cumul_rows_grouped[
        df_cumul_rows_grouped["age_group"] == ag
    ].copy()
    age_data_within = df_within_rows_grouped[
        df_within_rows_grouped["age_group"] == ag
    ].copy()

    age_data_cumul["bin_mid"] = age_data_cumul["do30_cumul_bin"].apply(lambda x: x.mid)
    age_data_within["bin_mid"] = age_data_within["do30_within_bin"].apply(
        lambda x: x.mid
    )

    pct_deaths = pct_deaths_lookup.get(ag, "")
    bar_width = 0.4

    # Plot cumulative bins (column 0)
    x_cumul = np.arange(len(age_data_cumul))
    axes[i, 0].bar(
        x_cumul - bar_width / 2,
        age_data_cumul["child_mortality_percent"],
        width=bar_width,
        label="Deaths (%)",
        color="blue",
        alpha=0.7,
    )
    axes[i, 0].bar(
        x_cumul + bar_width / 2,
        age_data_cumul["total_rows_percent"],
        width=bar_width,
        label="Rows (%)",
        color="orange",
        alpha=0.7,
    )
    axes[i, 0].set_title(f"Age Group: {ag} ({pct_deaths}) (Cumulative)")
    axes[i, 0].set_xlabel("Days Over 30C (Cumulative Bin)")
    axes[i, 0].set_ylabel("Percentage of Total")
    axes[i, 0].set_ylim(0, 100)
    axes[i, 0].set_xticks(x_cumul)
    axes[i, 0].set_xticklabels(
        [str(b.mid) for b in age_data_cumul["do30_cumul_bin"]], rotation=45
    )
    axes[i, 0].legend()
    axes[i, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot within bins (column 1)
    x_within = np.arange(len(age_data_within))
    axes[i, 1].bar(
        x_within - bar_width / 2,
        age_data_within["child_mortality_percent"],
        width=bar_width,
        label="Deaths (%)",
        color="blue",
        alpha=0.7,
    )
    axes[i, 1].bar(
        x_within + bar_width / 2,
        age_data_within["total_rows_percent"],
        width=bar_width,
        label="Rows (%)",
        color="orange",
        alpha=0.7,
    )
    axes[i, 1].set_title(f"Age Group: {ag} (Within)")
    axes[i, 1].set_xlabel("Days Over 30C (Within Bin)")
    axes[i, 1].set_ylabel("Percentage of Total")
    axes[i, 1].set_ylim(0, 100)
    axes[i, 1].set_xticks(x_within)
    axes[i, 1].set_xticklabels(
        [str(b.mid) for b in age_data_within["do30_within_bin"]], rotation=45
    )
    axes[i, 1].legend()
    axes[i, 1].grid(axis="y", linestyle="--", alpha=0.7)

# Save the entire figure as a PNG file
plt.tight_layout()
output_path = PLOT_DIR + "average_mortality_by_days_over_30_bins_1_increments.png"
plt.savefig(output_path)
plt.close(fig)

print(f"Plot saved to {output_path}")

# Per time bin plot:
# 1. monthly distribution of mortality rates
# 2. histogram distribution of days_over_30C_monthly
# 3. monthly distribution of days_over_30C_monthly among deaths only
# 4. monthly distribution of days_over_30C_monthly among both
df_monthly_grouped = (
    df_monthly.groupby(["age_group", "int_month"])[
        ["child_mortality", "days_over_30C_monthly"]
    ]
    .mean()
    .reset_index()
)
df_monthly_grouped.to_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_13.01/data_monthly_grouped_by_age_bin_int_month.parquet",
    index=False,
)
fig, axes = plt.subplots(len(labels), 1, figsize=(12, 6 * len(labels)))
for i, ag in enumerate(labels):
    age_data = df_monthly_grouped[df_monthly_grouped["age_group"] == ag]
    axes[i].hist(
        age_data["days_over_30C_monthly"],
        bins=30,
        alpha=0.7,
        label="Days over 30C Monthly",
        color="blue",
        edgecolor="black",
    )
    axes[i].set_title(f"Age Group: {ag}")
    axes[i].set_xlabel("Days over 30C Monthly")
    axes[i].set_ylabel("Frequency")
    axes[i].legend()
    axes[i].grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "histogram_days_over_30C_by_age_group.png")
plt.show()

# 2. histogram distribution of days_over_30C_monthly
fig, axes = plt.subplots(len(labels), 1, figsize=(12, 6 * len(labels)))
for i, ag in enumerate(labels):
    age_data = df_monthly[df_monthly["age_group"] == ag]
    axes[i].hist(
        age_data["days_over_30C_monthly"],
        bins=30,
        alpha=0.7,
        label="Days over 30C Monthly",
        color="blue",
        edgecolor="black",
    )
    axes[i].set_title(f"Age Group: {ag}")
    axes[i].set_xlabel("Days over 30C Monthly")
    axes[i].set_ylabel("Frequency")
    axes[i].legend()
    axes[i].grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "histogram_days_over_30C_by_age_group.png")
plt.show()

fig, axes = plt.subplots(len(labels), 1, figsize=(12, 6 * len(labels)))
for i, ag in enumerate(labels):
    age_data = df_monthly_grouped[df_monthly_grouped["age_group"] == ag]
    axes[i].plot(
        age_data["int_month"],
        age_data["days_over_30C_monthly"],
        marker="o",
        label="Avg Days over 30C Monthly",
    )
    axes[i].set_title(f"Age Group: {ag}")
    axes[i].set_xlabel("Month")
    axes[i].set_ylabel("Value")
    axes[i].tick_params(axis="x", rotation=90)
    axes[i].legend()
    axes[i].grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "seasonality_days_over_30C_by_age_group.png")
plt.show()


# 3. monthly distribution of days_over_30C_monthly among deaths only
df_monthly_grouped_deaths = (
    df_monthly[df_monthly["child_mortality"] == 1]
    .groupby(["age_group", "int_month"])[["days_over_30C_monthly"]]
    .mean()
    .reset_index()
)

fig, axes = plt.subplots(len(labels), 1, figsize=(12, 6 * len(labels)))
for i, ag in enumerate(labels):
    age_data = df_monthly_grouped_deaths[df_monthly_grouped_deaths["age_group"] == ag]
    axes[i].plot(
        age_data["int_month"],
        age_data["days_over_30C_monthly"],
        marker="o",
        label="Avg Days over 30C Monthly",
    )
    axes[i].set_title(f"Age Group: {ag}")
    axes[i].set_xlabel("Month")
    axes[i].set_ylabel("Value")
    axes[i].tick_params(axis="x", rotation=90)
    axes[i].legend()
    axes[i].grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "seasonality_days_over_30C_by_age_group_deaths_only.png")
plt.show()


# Get data points by int_month
df_monthly["int_month"].value_counts()  # pretty even
# how about within age bins?
fig, axes = plt.subplots(len(labels), 1, figsize=(12, 6 * len(labels)))
for i, ag in enumerate(labels):
    age_dist = df_monthly[df_monthly["age_group"] == ag]["int_month"].value_counts()
    axes[i].bar(age_dist.index, age_dist.values, color="skyblue", edgecolor="black")
    axes[i].set_title(f"Distribution of Data Points by Month for Age Group: {ag}")
    axes[i].set_xlabel("Month")
    axes[i].set_ylabel("Number of Data Points")
    axes[i].tick_params(axis="x", rotation=90)
    axes[i].grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "distribution_of_data_points_by_month_and_age_group.png")
plt.show()  # looks very similar

# mortality only
fig, axes = plt.subplots(len(labels), 1, figsize=(12, 6 * len(labels)))
for i, ag in enumerate(labels):
    age_dist = df_monthly[
        (df_monthly["age_group"] == ag) & (df_monthly["child_mortality"] == 1)
    ]["int_month"].value_counts()
    axes[i].bar(age_dist.index, age_dist.values, color="skyblue", edgecolor="black")
    axes[i].set_title(f"Distribution of Data Points by Month for Age Group: {ag}")
    axes[i].set_xlabel("Month")
    axes[i].set_ylabel("Number of Data Points")
    axes[i].tick_params(axis="x", rotation=90)
    axes[i].grid()
plt.tight_layout()
plt.savefig(
    PLOT_DIR + "distribution_of_data_points_by_month_and_age_group_deaths_only.png"
)
plt.show()  # few data points in later bins

# Get distribution of days_over_30C_monthly
fig, axes = plt.subplots(len(labels), 1, figsize=(12, 6 * len(labels)))
for i, ag in enumerate(labels):
    age_dist = df_monthly[(df_monthly["age_group"] == ag)]

    axes[i].hist(
        age_dist["days_over_30C_monthly"],
        bins=30,
        alpha=0.7,
        label="Days over 30C Monthly",
        color="blue",
        edgecolor="black",
    )
    axes[i].set_title(f"Distribution of Days over 30C Monthly: {ag}")
    axes[i].set_xlabel("Days over 30C Monthly")
    axes[i].set_ylabel("Number of Data Points")
    axes[i].tick_params(axis="x", rotation=90)
    axes[i].grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "distribution_of_days_over_30C_monthly_by_age_group.png")
plt.show()  #


# sanity checks

# Spot check calculation
indv_z = df_monthly[
    (df_monthly["age_month"] == 59) & (df_monthly["days_over_30C_monthly"] > 20)
].iloc[0]
indv_z_monthly = df_monthly[df_monthly["indv_id"] == indv_z["indv_id"]]
indv_z_within_bin = df_combined[df_combined["indv_id"] == indv_z["indv_id"]]
indv_z_cumulative = df_cumulative[df_cumulative["indv_id"] == indv_z["indv_id"]]

# check calculations
indv_z_monthly["days_over_30C_monthly"]
indv_z_monthly["days_over_30C_monthly"].describe()
indv_z_monthly[["days_over_30C_monthly", "age_month", "age_group"]]
indv_z_within_bin[["days_over_30C_monthly_within_bin", "age_group"]]
indv_z_cumulative[
    ["age_group", "days_over_30C_monthly_cumulative", "days_over_30C_monthly_constant"]
]
## Is the shape different between under 20 days over 30C and above 20 days over 30C?
df_monthly["over_20_days_over_30C"] = df_monthly["days_over_30C_monthly"] >= 20
mortality_by_month_over_20 = (
    df_monthly[df_monthly["over_20_days_over_30C"]]
    .groupby("int_month")[["child_mortality"]]
    .mean()
    .reset_index()
)
mortality_by_month_under_20 = (
    df_monthly[~df_monthly["over_20_days_over_30C"]]
    .groupby("int_month")[["child_mortality"]]
    .mean()
    .reset_index()
)
plt.figure(figsize=(12, 6))
plt.plot(
    mortality_by_month_over_20["int_month"],
    mortality_by_month_over_20["child_mortality"],
    marker="o",
    label=">= 20 Days over 30C",
)
plt.plot(
    mortality_by_month_under_20["int_month"],
    mortality_by_month_under_20["child_mortality"],
    marker="o",
    label="< 20 Days over 30C",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Child Mortality")
plt.title("Mortality by Month for Different Levels of Days over 30C")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR + "mortality_by_month_over_under_20_days_over_30C.png")


# Check for seasonality trends in df_month_sub
df_month_sub["int_month_fmt"] = df_month_sub["int_month"].astype(str).str.zfill(2)
df_month_sub["int_year_month"] = (
    df_month_sub["int_year"].astype(str) + "-" + df_month_sub["int_month_fmt"]
)

df_month_sub = df_month_sub.sort_values("int_year_month")

# Plot with int_month on x axis and days_over_30C_monthly on y axis
df_month_sub_group = (
    df_month_sub.groupby("int_month")[["days_over_30C_monthly", "child_mortality"]]
    .mean()
    .reset_index()
)


plt.figure(figsize=(12, 6))
plt.plot(
    df_month_sub_group["int_month"],
    df_month_sub_group["days_over_30C_monthly"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Days over 30C Monthly")
plt.title("Seasonality Trends in Days over 30C Monthly")
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(
    df_month_sub_group["int_month"],
    df_month_sub_group["child_mortality"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Child Mortality")
plt.title("Seasonality Trends in Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()

# Make a scatter
plt.figure(figsize=(8, 6))
plt.scatter(
    df_month_sub_group["days_over_30C_monthly"],
    df_month_sub_group["child_mortality"],
)
plt.xlabel("Avg Days over 30C Monthly")
plt.ylabel("Avg Child Mortality")
plt.title("Relationship between Days over 30C Monthly and Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()


# break down by country?
df_month_sub_group_cntry = (
    df_month_sub.groupby(["int_month", "ihme_loc_id"])[
        ["days_over_30C_monthly", "child_mortality"]
    ]
    .mean()
    .reset_index()
)


# Group data by country
countries = df_month_sub_group_cntry["ihme_loc_id"].unique()

# Initialize PDF writer
pdf_path = PLOT_DIR + "df_subset_country_plots.pdf"
with PdfPages(pdf_path) as pdf:
    for i, country in enumerate(countries):
        country_data = df_month_sub_group_cntry[
            df_month_sub_group_cntry["ihme_loc_id"] == country
        ]

        # Create figure for the three plots
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # Plot 1: Avg Child Mortality by Month
        avg_child_mortality = country_data.groupby("int_month")[
            "child_mortality"
        ].mean()
        axes[0].plot(avg_child_mortality.index, avg_child_mortality.values, marker="o")
        axes[0].set_title(f"{country}: Avg Child Mortality by Month")
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Avg Child Mortality")
        axes[0].tick_params(axis="x", rotation=90)

        # Plot 2: Days Over 30C Monthly by Month
        avg_days_over_30C = country_data.groupby("int_month")[
            "days_over_30C_monthly"
        ].mean()
        axes[1].plot(
            avg_days_over_30C.index,
            avg_days_over_30C.values,
            marker="o",
            color="orange",
        )
        axes[1].set_title(f"{country}: Days Over 30C Monthly by Month")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Days Over 30C")
        axes[1].tick_params(axis="x", rotation=90)

        # Plot 3: Scatter Plot of Child Mortality vs Days Over 30C
        axes[2].scatter(
            country_data["days_over_30C_monthly"],
            country_data["child_mortality"],
            alpha=0.7,
        )
        axes[2].set_title(f"{country}: Child Mortality vs Days Over 30C")
        axes[2].set_xlabel("Days Over 30C Monthly")
        axes[2].set_ylabel("Child Mortality")

        # Adjust layout and save page
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# Do same but for neonatal

df_neo = df_monthly[df_monthly["age_month"] == 0]
df_neo_above_20 = df_neo[df_neo["days_over_30C_monthly"] >= 20]


# Plot with int_month on x axis and days_over_30C_monthly on y axis
df_neo_above_20_group = (
    df_neo_above_20.groupby("int_month")[["days_over_30C_monthly", "child_mortality"]]
    .mean()
    .reset_index()
)


plt.figure(figsize=(12, 6))
plt.plot(
    df_neo_above_20_group["int_month"],
    df_neo_above_20_group["days_over_30C_monthly"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Days over 30C Monthly")
plt.title("Seasonality Trends in Days over 30C Monthly")
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(
    df_neo_above_20_group["int_month"],
    df_neo_above_20_group["child_mortality"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Child Mortality")
plt.title("Seasonality Trends in Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()

# Make a scatter
plt.figure(figsize=(8, 6))
plt.scatter(
    df_neo_above_20_group["days_over_30C_monthly"],
    df_neo_above_20_group["child_mortality"],
)
plt.xlabel("Avg Days over 30C Monthly")
plt.ylabel("Avg Child Mortality")
plt.title("Relationship between Days over 30C Monthly and Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()


# Look at full datasets
df_neo_group = (
    df_neo.groupby("int_month")[["days_over_30C_monthly", "child_mortality"]]
    .mean()
    .reset_index()
)

import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 10000  # Set to a higher value
plt.figure(figsize=(12, 6))
plt.plot(
    df_neo_group["int_month"],
    df_neo_group["days_over_30C_monthly"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Days over 30C Monthly")
plt.title("Seasonality Trends in Days over 30C Monthly")
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(
    df_neo_group["int_month"],
    df_neo_group["child_mortality"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Child Mortality")
plt.title("Seasonality Trends in Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()

# Make a scatter
plt.figure(figsize=(8, 6))
plt.scatter(
    df_neo_group["days_over_30C_monthly"],
    df_neo_group["child_mortality"],
)
plt.xlabel("Avg Days over 30C Monthly")
plt.ylabel("Avg Child Mortality")
plt.title("Relationship between Days over 30C Monthly and Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()


## Same for above 48 months
df_above_48 = df_monthly[df_monthly["age_month"] >= 48]

df_above_48_group = (
    df_above_48.groupby("int_month")[["days_over_30C_monthly", "child_mortality"]]
    .mean()
    .reset_index()
)


plt.plot(
    df_above_48_group["int_month"],
    df_above_48_group["days_over_30C_monthly"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Days over 30C Monthly")
plt.title("Seasonality Trends in Days over 30C Monthly")
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(
    df_above_48_group["int_month"],
    df_above_48_group["child_mortality"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Child Mortality")
plt.title("Seasonality Trends in Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()

# Make a scatter
plt.figure(figsize=(8, 6))
plt.scatter(
    df_above_48_group["days_over_30C_monthly"],
    df_above_48_group["child_mortality"],
)
plt.xlabel("Avg Days over 30C Monthly")
plt.ylabel("Avg Child Mortality")
plt.title("Relationship between Days over 30C Monthly and Child Mortality")
plt.grid()
plt.tight_layout()
plt.show()

# The higher the days over 30, and the higher the age_month, the higher the correlation
#  between days_over_30C_monthly and child_mortality
df_monthly["days_over_30C_monthly_int"] = df_monthly["days_over_30C_monthly"].astype(
    int
)

# Correlation table: rows = age_month, columns = cumulative threshold (>= X)
# cells = corr(days_over_30C_monthly, child_mortality) among obs at that age
# with days_over_30C_monthly >= threshold
thresholds = [0, 1, 5, 10, 15, 20, 25, 30]
age_months = sorted(df_monthly["age_month"].unique())

corr_rows = []
for age in age_months:
    age_data = df_monthly[df_monthly["age_month"] == age]
    row = {"age_month": age}
    for t in thresholds:
        subset = age_data[age_data["days_over_30C_monthly"] >= t]
        if len(subset) >= 10 and subset["child_mortality"].nunique() > 1:
            row[f">={t}"] = subset["days_over_30C_monthly"].corr(
                subset["child_mortality"]
            )
        else:
            row[f">={t}"] = np.nan
    corr_rows.append(row)

corr_table = pd.DataFrame(corr_rows).set_index("age_month")
print("Correlation(days_over_30C_monthly, child_mortality) by age and threshold:")
print(corr_table.to_string())


corr_table.to_csv(PLOT_DIR + "correlation_table_by_age_month_and_threshold.csv")


## Plot overall mortality by month
monthly_mortality = (
    df_monthly.groupby("int_month")[["child_mortality"]].mean().reset_index()
)
plt.figure(figsize=(12, 6))
plt.plot(
    monthly_mortality["int_month"],
    monthly_mortality["child_mortality"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Avg Child Mortality")
plt.title("Overall Mortality by Month")
plt.grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "overall_mortality_by_month.png")
plt.show()


## Plot monthly distribution among deaths only
deaths_monthly = (
    df_monthly[df_monthly["child_mortality"] == 1]
    .groupby("int_month")[["child_mortality"]]
    .count()
    .reset_index()
)
plt.figure(figsize=(12, 6))
plt.plot(
    deaths_monthly["int_month"],
    deaths_monthly["child_mortality"],
    marker="o",
)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Number of Deaths")
plt.title("Monthly Distribution of Deaths")
plt.grid()
plt.tight_layout()
plt.savefig(PLOT_DIR + "monthly_distribution_of_deaths.png")
plt.show()


# plt.show()
