"""
To incorporate updated consumption data into child mortality data without
full rerun
"""

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
import rioxarray
from tqdm import tqdm
import sys
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
from scipy.interpolate import PchipInterpolator
import polars as pl

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
# Load last updated data and format from there
############################

df = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_03_23.01/child_mortality_exploded_updated_wealth.parquet"
)
len_before = len(df)


df["indv_id"].nunique()
df["age_month"].max()

## Step 0: Need to fix the fence-post problem. Original age_month should be cut
# off at 59 months before increment, 60 months after increment.
df[df["child_mortality"] == 1][["age_month", "aod_months", "age_month_original"]].head()

# 0.a. decrement age_month by 1
df["age_month"] -= 1

# 0.b. for any child at new age_month 60, which we assume to be 60 to 61 after
# decrementing, if their aod_months is equal or greater than 60,
# then their child_mortality should be set to 0.
# convert aod_months
df.loc[df["aod_months"].isna(), "aod_months"] = "99999"
df["aod_months"] = df["aod_months"].astype(int)
df.loc[(df["age_month"] == 59) & (df["aod_months"] >= 59), "child_mortality"] = 0

# cut off data at 59 months
df = df[df["age_month"] <= 59].reset_index(drop=True)

# 0.c. save out corrected raw data
df.to_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_03_23.01/child_mortality_exploded_updated_wealth_decremented_age_mo.parquet",
    index=False,
)

## Step 1: clean any nonsensical aod_month versus age_month
df_max_age = (
    df.sort_values("age_month")
    .groupby("indv_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

df_max_age["aod_months"].unique()
df_max_age["age_month"].value_counts()

nonsense_ids_df = df_max_age[
    ((df_max_age["child_alive"] == 1) & (df_max_age["aod_months"].notna()))
    | ((df_max_age["child_alive"] == 0) & (df_max_age["aod_months"].isna()))
]
len(nonsense_ids_df)  # 63913
nonsense_ids_df[
    "age_month_original"
].describe()  # min is 61... this is not actually an issue
nonsense_ids_df[["age_month_original", "aod_months", "child_alive"]].head(20)
nonsense_ids_df[
    "aod_months"
].value_counts()  # many have aod_months of 99999, which we set for missing values

# Step 2: Add time bins to the data as dummy variables if the age_month falls within the bin

# Note the bin time below. The original age_month would have 0 for neonatal, and
# was incremented +1 for the original survival analyses. This means that age_month
# 59 became 60, and we are assuming the age is UP TO, but not including, the
# new age_month.
time_bin_dict = {
    "age_1_m": (0, 1),
    "age_3_m": (1, 3),
    "age_6_m": (3, 6),
    "age_12_m": (6, 12),
    "age_24_m": (12, 24),
    "age_36_m": (24, 36),
    "age_48_m": (36, 48),
    "age_60_m": (48, 60),
}

# bin_name will define what range of time the month falls into
for bin_name, bin_month in time_bin_dict.items():
    df[bin_name] = (
        (df["age_month"] >= bin_month[0]) & (df["age_month"] < bin_month[1])
    ).astype(int)


"""
'year_start', - get max
 'year_end', - get max
 'nid', - get max
 'survey_name', - get max
 'int_year', - get max
 'int_month', - get max
 'sex_id', - get max
 'mothers_age_year', - get max
 'aod_months', - get max
 'age_month', - get max (will not be used, since bins are used instead)
 'ihme_loc_id', - group by and get max
 'geospatial_id', - group by and get max
 'psu', - group by and get max
 'strata', - group by and get max
 'line_id', - group by and get max
 'hh_id', - group by and get max
 'hhweight', - get max
 'pweight', - get max
 'birth_year', - get max
 'birth_month', - get max
 'child_alive', - get max
 'lat', - group by and get max
 'long', - group by and get max
 'wealth_index_dhs_x', - drop 
 'location_id_x', - drop 
 'int_birth_year_diff_months', - get max
 'age_month_original', - get max
 'months_to_expand', - drop 
 'age_month_pre_exploded', - drop 
 'int_year_original', - get max
 'int_month_original', - get max
 'age_group_id', -drop 
 'age_year', - drop
 'age_group_id_agg', - get max
 'mean_temperature', - get avg
 'days_over_30C', - get avg
 'precipitation_days', - get avg
 'total_precipitation', - get avg
 'mean_low_temperature', - get avg
 'mean_high_temperature', - get avg
 'relative_humidity', - get avg
 'elevation', - get avg
 'lbd_admin2_id', - get max
 'indv_id', - group by and get max
 'child_mortality', - get max
 'consumption', - get avg
 'consumption_pd', - get avg
 'any_days_over_30C', - get max
 'wealth_index_dhs_y', - drop 
 'location_id_y', - drop 
 'unweighted_population_percentile', - drop 
 'cum_weight', - drop 
 'weighted_population_percentile',  - drop 
 'ldipc_unweighted_no_match',  - drop 
 'ldipc_weighted_no_match',  - drop 
 'ldipc_unweighted_match',  - drop 
 'ldipc_weighted_match', - drop 
 'age_1_m', - group by and get max
 'age_3_m',- group by and get max
 'age_6_m',- group by and get max
 'age_12_m',- group by and get max
 'age_24_m',- group by and get max
 'age_36_m',- group by and get max
 'age_48_m',- group by and get max
 'age_60_m',- group by and get max
"""

## Get weighted averages (by number of months in bin) of explanatory variables, grouping by binned age_month and child_alive status
get_max_vars = [
    "year_start",
    "year_end",
    "nid",
    # "survey_name",
    "int_year",
    "int_month",
    "sex_id",
    # "mothers_age_year",
    # "aod_months",
    "age_month",
    # "hhweight",
    "pweight",
    "birth_year",
    "birth_month",
    "int_birth_year_diff_months",
    "age_month_original",
    "int_year_original",
    "int_month_original",
    # "age_group_id_agg",
    "any_days_over_30C",
    "child_alive",
    "child_mortality",
]

get_avg_vars = [
    "mean_temperature",
    "days_over_30C",
    "precipitation_days",
    "total_precipitation",
    "mean_low_temperature",
    "mean_high_temperature",
    "relative_humidity",
    "elevation",
    "consumption",
    "consumption_pd",
]

group_by_vars = [
    "ihme_loc_id",
    "geospatial_id",
    "psu",
    "strata",
    "line_id",
    "hh_id",
    "lat",
    "long",
    "lbd_admin2_id",
    "indv_id",
    "age_1_m",
    "age_3_m",
    "age_6_m",
    "age_12_m",
    "age_24_m",
    "age_36_m",
    "age_48_m",
    "age_60_m",
]

# check for anything missing before performing groupby:
set(df.columns) - set(get_max_vars) - set(get_avg_vars) - set(group_by_vars)

len(df)
len(df.dropna(subset=get_max_vars))
nan_columns = [col for col in get_max_vars if df[col].isna().any()]
print(nan_columns)
nan_columns = [col for col in group_by_vars if df[col].isna().any()]
print(nan_columns)
nan_columns = [col for col in get_avg_vars if df[col].isna().any()]
print(nan_columns)
df = df.dropna(subset=get_avg_vars)  # drop rows with missing values in these columns
print(len(df))

print("Performing groupby to get binned time data...")
# df_grouped = (
#     df.groupby(group_by_vars)
#     .agg(
#         {
#             **{var: "max" for var in get_max_vars},
#             **{var: "mean" for var in get_avg_vars},
#         }
#     )
#     .reset_index()
# )

# print("Saving data...")

# Attempt polars
df_pl = pl.from_pandas(df)

df_grouped = (
    df_pl.group_by(group_by_vars)
    .agg(
        [pl.col(var).max() for var in get_max_vars]
        + [pl.col(var).mean() for var in get_avg_vars]
    )
    .to_pandas()
)


# Add the aod_months back onto data:
aod_df = df_max_age[["indv_id", "aod_months"]].drop_duplicates()
df_grouped = df_grouped.merge(aod_df, on="indv_id", how="left")

df_grouped.to_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_03_23.01/child_mortality_exploded_binned_age_month_decremented.parquet",
    index=False,
)

print("done")

# df_grouped = pd.read_parquet(
#     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_03_23.01/child_mortality_exploded_binned_age_month.parquet"
# )


# Get obs by sample size
sample_percent = 10000 / len(df_grouped)

# Get unique individuals and their locations
indv_dt = df_grouped[["indv_id", "ihme_loc_id"]].drop_duplicates()

# Count individuals per location
indv_counts = indv_dt.groupby("ihme_loc_id").size().reset_index(name="N")

# Merge counts back to individual data
indv_dt = indv_dt.merge(indv_counts, on="ihme_loc_id", suffixes=("", "_total"))

# Calculate the number of samples to take per location
indv_dt["n_sample"] = np.floor(sample_percent * indv_dt["N"]).astype(int)

# Set random seed for reproducibility
np.random.seed(42)

# Sample individuals within each location
sampled_indv = (
    indv_dt.groupby("ihme_loc_id")
    .apply(lambda group: group.sample(n=min(len(group), group["n_sample"].iloc[0])))[
        "indv_id"
    ]
    .values
)

# Filter the original DataFrame to include only sampled individuals
df_sample = df_grouped[df_grouped["indv_id"].isin(sampled_indv)]
f"{df_sample["indv_id"].nunique():,}"


### Make histograms of age_month distribution and child_mortality status

PLOT_DIR = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/"
# Raw age_month distribution
plt.figure(figsize=(10, 6))
plt.hist(df_max_age["age_month"], bins=60, edgecolor="black")
plt.title("Distribution of age_month at last observation")
plt.xlabel("age_month")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(12)
)  # Set x-axis labels every 12
plt.savefig(
    os.path.join(PLOT_DIR, "child_mortality_binned_age_month_distribution_all.png")
)
plt.close()

# Plot mortality only
plt.figure(figsize=(10, 6))
plt.hist(
    df_max_age[df_max_age["child_mortality"] == 1]["age_month"],
    bins=60,
    edgecolor="black",
)
plt.title("Distribution of age_month at last observation for child_mortality=1")
plt.xlabel("age_month")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.gca().xaxis.set_major_locator(
    mticker.MultipleLocator(12)
)  # Set x-axis labels every 12
plt.savefig(
    os.path.join(
        PLOT_DIR, "child_mortality_binned_age_month_distribution_mortality_only.png"
    )
)
plt.close()

# Plot percentages of total for age_month distribution
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(
    df_max_age["age_month"], bins=60, edgecolor="black", weights=np.ones(len(df_max_age)) / len(df_max_age)
)
plt.title("Percentage Distribution of age_month at last observation")
plt.xlabel("age_month")
plt.ylabel("Percentage")
plt.grid(axis="y", alpha=0.75)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(12))  # Set x-axis labels every 12
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))  # Format y-axis as percentages
plt.savefig(
    os.path.join(PLOT_DIR, "child_mortality_binned_age_month_percentage_all.png")
)
plt.close()

# Plot percentages of total for mortality only
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(
    df_max_age[df_max_age["child_mortality"] == 1]["age_month"],
    bins=60,
    edgecolor="black",
    weights=np.ones(len(df_max_age[df_max_age["child_mortality"] == 1])) / len(df_max_age)
)
plt.title("Percentage Distribution of age_month at last observation for child_mortality=1")
plt.xlabel("age_month")
plt.ylabel("Percentage")
plt.grid(axis="y", alpha=0.75)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(12))  # Set x-axis labels every 12
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))  # Format y-axis as percentages
plt.savefig(
    os.path.join(
        PLOT_DIR, "child_mortality_binned_age_month_percentage_mortality_only.png"
    )
)
plt.close()


