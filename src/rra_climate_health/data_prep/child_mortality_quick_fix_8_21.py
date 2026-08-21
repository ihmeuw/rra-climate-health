# fix problematic rows
# 1. Ensure that child_mortality is 1 only once, corresponding to the
# age_month_original if child_alive = 0.
# For child_alive == 0, get max age_month

# 2. Keep rows leading up to max age, then drop thereafter

LDI_VERSION = "v8"
import gc
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


def examine_survey_schema(df: pd.DataFrame, columns: list[str]) -> None:
    print("Records:", len(df))
    print()

    template = "{:<20} {:>10} {:>10} {:>10}"
    header = template.format("COLUMN", "N_UNIQUE", "N_NULL", "DTYPE")

    print(header)
    print("=" * len(header))
    for col in columns:
        unique = df[col].nunique()
        nulls = df[col].isna().sum()
        dtype = str(df[col].dtype)
        print(template.format(col, unique, nulls, dtype))


COLUMN_NAME_TRANSLATOR = {
    "country": "ihme_loc_id",
    "year_start": "year_start",
    "end_year": "year_end",
    "psu_id": "psu",
    "strata_id": "strata",
    "sex": "sex_id",
    "age_mo": "age_month",
    "stunting_mod_b": "stunting",
    "wasting_mod_b": "wasting",
    "underweight_mod_b": "underweight",
    "HAZ_b2": "stunting",
    "WHZ_b2": "wasting",
    "WAZ_b2": "underweight",
    "latnum": "lat",
    "longnum": "long",
    "latitude": "lat",
    "longitude": "long",
}


# 1. Load data #################################################################

# Load last version
df_latest = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_08_03.02/data.parquet"
)

# add to indv_id of df_latest
df_latest["indv_id"] = (
    df_latest[["indv_id", "sex_id", "birth_year", "birth_month"]]
    .astype(str)
    .agg("_".join, axis=1)
)

# Load version with original aod_months
data_raw = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_08_21.01/child_mortality_indv_id.parquet"
)
# data_raw["line_id"] = data_raw["line_id"].astype(int)

# data_raw["indv_id"] = (
#     data_raw[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
# )

# check all indv_id in df_latest are in data_raw
missing_indv_ids = list(set(df_latest["indv_id"]) - set(data_raw["indv_id"]))
len(missing_indv_ids)  # how can this be >0?

# resolve duplicate indv_id values
dup_data = data_raw[data_raw.duplicated(subset=["indv_id"], keep=False)]
len(dup_data)

df_to_merge = data_raw[
    [
        "indv_id",
        "lat",
        "long",
        "birth_year",
        "birth_month",
        "age_month_original",
        "aod_months",
        "child_alive",
    ]
].drop_duplicates()
df_to_merge.rename(columns={"child_alive": "child_alive_original"}, inplace=True)

# 2. Merge data ################################################################

df_merged = df_latest.merge(
    df_to_merge,
    on=["indv_id", "lat", "long", "birth_year", "birth_month", "age_month_original"],
    how="left",
)

assert len(df_merged) == len(
    df_latest
), "Merged dataframe has different length than original"


# 3. Perform filters ###########################################################

# reset child alive to original values
df_merged["child_mortality"] = 1 - df_merged["child_alive_original"]

# separate alive for all survey
df_alive = df_merged[df_merged["aod_months"].isna()]
df_alive["child_mortality"].unique()  # all 0

df_deaths = df_merged[df_merged["aod_months"].notna()]
df_deaths["aod_months"] = df_deaths["aod_months"].astype(int)
df_deaths["child_mortality"].unique()  # all 1

# This should give a clear distiction of individuals
set(df_alive["indv_id"]).intersection(set(df_deaths["indv_id"]))  # should be empty

# investigate overlap
test = df_merged[df_merged["indv_id"] == "19670_3226_7_3"]
# 1. Ensure that child_mortality is 1 only once, corresponding to the
# age_month_original if child_alive = 0.
# For child_alive == 0, get max age_month


# 4. Save updated ##############################################################
