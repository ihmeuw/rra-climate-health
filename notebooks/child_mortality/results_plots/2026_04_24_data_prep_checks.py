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

############################
# Wasting/Stunting columns #
############################


## Load data ###################################################################
stunting_sub = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/inspect_data/stunting_sub.parquet"
)

df_subset = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/inspect_data/cm_subset.parquet"
)

df_month_sub = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_21.01/inspect_data/df_month_sub.parquet"
)

df_month_sub["days_over_30C_monthly"].describe()


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


PLOT_DIR = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2026_04_21.01/"

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
df_monthly = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_13.01/data_monthly_expanded_rel_thresholds.parquet"
)
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

# Correlation table: rows = age_month, columns = days_over_30C_monthly_int
# cells = correlation between days_over_30C_monthly and child_mortality
corr_table = (
    df_monthly.groupby(["age_month", "days_over_30C_monthly_int"])
    .apply(lambda g: g["days_over_30C_monthly"].corr(g["child_mortality"]))
    .unstack(level="days_over_30C_monthly_int")
)
corr_table.columns.name = "days_over_30C_monthly_int"
corr_table.index.name = "age_month"
print(corr_table.to_string())
