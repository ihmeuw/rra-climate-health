LDI_VERSION = "v9"
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


from rra_tools import jobmon

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


def merge_left_without_inflating(df_left, df_right, **kwargs):
    """Merge left without inflating the left dataframe."""
    df = df_left.merge(df_right, how="left", **kwargs)
    if len(df) != len(df_left):
        msg = "Mismatch in length of data and merged data."
        raise RuntimeError(msg)
    return df


def get_lookup_months(birth_year, birth_month):
    """Returns list of year-month tuples to lookup climate data for."""

    return_tuple = []
    return_tuple.append((birth_year, birth_month, birth_year, birth_month, "prev_0_mo"))
    for i in range(1, 10):
        lookup_month = birth_month - i
        lookup_year = birth_year
        if lookup_month <= 0:
            lookup_month += 12
            lookup_year -= 1
        return_tuple.append(
            (birth_year, birth_month, lookup_year, lookup_month, f"prev_{i}_mo")
        )

    return_df = pd.DataFrame(
        return_tuple, columns=["birth_year", "birth_month", "year", "month", "suffix"]
    )
    return return_df


def get_prev_climate_var_months(climate_var: str, num_months: int) -> list[str]:
    """
    Helper function to retrieve relavant previous monthly climate variable names
    to construct cumulative exposures.
    """
    return_vars = []
    for i in range(num_months):
        return_vars.append(f"{climate_var}_prev_{i}_mo")
    return return_vars


def get_prev_climate_threshold_months(threshold: str, num_months: int) -> list[str]:
    """
    Helper function to retrieve relavant previous monthly climate variable names
    to construct cumulative exposures.
    """
    return_vars = []
    for i in range(num_months):
        return_vars.append(f"q{threshold}_prev_{i}_mo")
    return return_vars


def get_climate_vars_for_months(
    month_df: pd.DataFrame,
    climate_variables: list[str],
    look_up_year: int,
    look_up_month: int,
    prev_time_suffix: str,
    lat_col: str = "lat",
    long_col: str = "long",
) -> pd.DataFrame:
    """
    returns data for previous months
    """
    temp_df = month_df.copy()
    lats = xr.DataArray(temp_df[lat_col], dims="point")
    lons = xr.DataArray(temp_df[long_col], dims="point")

    for climate_variable in climate_variables:
        # climate_ds = ClimateMalnutritionData(Path(DEFAULT_ROOT)/'stunting').load_climate_raster(climate_variable, 'ssp245', yr, 0)
        # Temporary workaround for climate data loading
        climate_ds = xr.open_dataset(
            f"/mnt/share/erf/climate_downscale/results/monthly/raw/historical/{climate_variable}/{look_up_year}_era5.nc"
        ).sel(month=look_up_month)["value"]
        temp_df[f"{climate_variable}_{prev_time_suffix}"] = (
            climate_ds.sel(latitude=lats, longitude=lons, method="nearest")
            .to_numpy()
            .flatten()  # the flatten also wasn't there before
        )
    return temp_df


def get_climate_vars_for_prev_months(
    month_df: pd.DataFrame,
    climate_variables: list[str],
    year_col: str = "birth_year",
    month_col: str = "birth_month",
    lat_col: str = "lat",
    long_col: str = "long",
) -> pd.DataFrame:
    """
    returns data for previous months
    """
    temp_df = month_df.copy()
    lats = xr.DataArray(temp_df[lat_col], dims="point")
    lons = xr.DataArray(temp_df[long_col], dims="point")

    lookup_year_months = get_lookup_months(
        temp_df[year_col].iloc[0], temp_df[month_col].iloc[0]
    )

    for climate_variable in climate_variables:
        for look_up_year, look_up_month, prev_time_suffix in lookup_year_months:

            climate_ds = xr.open_dataset(
                f"/mnt/share/erf/climate_downscale/results/monthly/raw/historical/{climate_variable}/{look_up_year}_era5.nc"
            ).sel(month=look_up_month)["value"]
            temp_df[f"{climate_variable}_{prev_time_suffix}"] = (
                climate_ds.sel(latitude=lats, longitude=lons, method="nearest")
                .to_numpy()
                .flatten()  # the flatten also wasn't there before
            )
    return temp_df


def get_prev_monthly_climate_vars_for_dataframe(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
) -> pd.DataFrame:
    var_names = [
        "mean_temperature",
        # "precipitation_days",
        "total_precipitation",
        # "mean_low_temperature",
        # "mean_high_temperature",
        # "relative_humidity",
        "days_over_24C",
        "days_over_25C",
        "days_over_26C",
        "days_over_27C",
        "days_over_28C",
        "days_over_29C",
        "days_over_30C",
        "days_over_31C",
        "days_over_32C",
        # "days_over_33C",
    ]

    unique_coords = df[
        [lat_col, long_col, "birth_year", "birth_month"]
    ].drop_duplicates()
    unique_coords_grouped = unique_coords.groupby(["birth_year", "birth_month"])
    df_splits = []
    for (birth_year, birth_month), group in unique_coords_grouped:
        df_split = group.copy()
        df_splits.append(df_split)

    p = mp.Pool(processes=25)
    results_df = pd.concat(
        p.map(
            partial(
                get_climate_vars_for_prev_months,
                climate_variables=var_names,
                year_col="birth_year",
                month_col="birth_month",
            ),
            df_splits,
        )
    )
    p.close()
    p.join()
    return results_df


def get_climate_vars_all_locs(
    year_df: pd.DataFrame,
    climate_variables: list[str],
    year_col: str = "lookup_year",
    lat_col: str = "lat",
    long_col: str = "long",
) -> pd.DataFrame:
    """
    returns data for all lat/longs for all months for a given year
    year_df = df_split.copy()
    year_col = "lookup_year"
    climate_variables = var_names
    climate_variable = climate_variables[0]
    """
    temp_df = year_df.copy()
    lats = xr.DataArray(temp_df[lat_col], dims="point")
    lons = xr.DataArray(temp_df[long_col], dims="point")
    lookup_yr = temp_df[year_col].iloc[0]

    return_arrays = []
    for climate_variable in climate_variables:
        try:
            climate_da = xr.open_dataarray(
                f"/mnt/share/erf/climate_downscale/results/monthly/raw/historical/{climate_variable}/{lookup_yr}_era5.nc"
            )
            climate_da = climate_da.load()
            # Select nearest latitude and longitude
            climate_da = climate_da.sel(latitude=lats, longitude=lons, method="nearest")
            # Add original latitude and longitude as coordinates
            climate_da = climate_da.assign_coords(lat_orig=("point", lats.values))
            climate_da = climate_da.assign_coords(long_orig=("point", lons.values))

            # Add climate variable dimension
            climate_da = climate_da.expand_dims(dim="climate_var")
            climate_da = climate_da.assign_coords(climate_var=[climate_variable])
            # Drop the "point" dimension if not needed
            climate_da = climate_da.drop_vars("point")
            return_arrays.append(climate_da)
        except:
            print(
                f"Error loading climate data for year {lookup_yr} and variable {climate_variable}"
            )
            print("check file path:")
            print(
                f"/mnt/share/erf/climate_downscale/results/monthly/raw/historical/{climate_variable}/{lookup_yr}_era5.nc"
            )
            pass
    # Concatenate all climate variables along the "climate_var" dimension
    if len(return_arrays) > 0:
        result = xr.concat(return_arrays, dim="climate_var")
        return result
    else:
        pass


def get_all_climate_vars_year_months_for_latlongs(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
    year_var: str = "birth_year",
    month_var: str = "birth_month",
) -> pd.DataFrame:
    """
    df = df_climate.copy()
    lat_col = "lat"
    long_col= "long"
    year_var="int_year"
    month_var="int_month"
    """
    var_names = [
        "mean_temperature",
        # "precipitation_days",
        "total_precipitation",
        # "mean_low_temperature",
        # "mean_high_temperature",
        # "relative_humidity",
        "days_over_24C",
        "days_over_25C",
        "days_over_26C",
        "days_over_27C",
        "days_over_28C",
        "days_over_29C",
        "days_over_30C",
        "days_over_31C",
        "days_over_32C",
        # "days_over_33C",
    ]

    unique_coords = df[[lat_col, long_col]].drop_duplicates()

    min_year = df[year_var].min() - 1  # need prior year
    max_year = df[year_var].max()

    # get all years and months for each climate variable for all coordinates
    df_splits = []
    for year in range(min_year, max_year + 1):
        df_split = unique_coords.copy()
        df_split["lookup_year"] = year
        df_splits.append(df_split)

    p = mp.Pool(processes=25)
    # Wrap the p.map call with tqdm for progress tracking
    results_xarrays = list(
        tqdm(
            p.imap(
                partial(
                    get_climate_vars_all_locs,
                    climate_variables=var_names,
                ),
                df_splits,
            ),
            total=len(df_splits),  # Total number of tasks for tqdm
            desc="Processing climate variables",  # Description for the progress bar
        )
    )
    p.close()
    p.join()

    # Concatenate the xarrays along the "lookup_year" dimension
    results_xarrays = [da for da in results_xarrays if da is not None]
    results_da = xr.concat(results_xarrays, dim="year")
    return results_da


def merge_dfs_in_parallel(df1, df2, merge_cols, year):
    """
    Helper function to merge datasets in parallel
    """
    df2 = df2.query(f"lookup_year == {year}")
    result = df1.merge(df2, how="left", on=merge_cols)

    return result


def get_climate_thresholds_all_locs(
    year_df: pd.DataFrame,
    year_col: str = "lookup_year",
    lat_col: str = "lat",
    long_col: str = "long",
) -> pd.DataFrame:
    """
    get thresholds get_climate_thresholds_for_prev_months

    returns data for all lat/longs for all months for a given year
    year_df = df_split.copy()
    year_col = "lookup_year"

    """
    temp_df = year_df.copy()
    lats = xr.DataArray(temp_df[lat_col], dims="point")
    lons = xr.DataArray(temp_df[long_col], dims="point")
    lookup_yr = temp_df[year_col].iloc[0]

    # for climate_variable in climate_variables:
    try:
        climate_da = xr.open_dataarray(
            f"/mnt/share/erf/climate_downscale/results/monthly/raw/historical/days_over_relative_threshold/{lookup_yr}_era5.nc"
        )
        climate_da = climate_da.load()
        # Select nearest latitude and longitude
        climate_da = climate_da.sel(latitude=lats, longitude=lons, method="nearest")
        # Add original latitude and longitude as coordinates
        climate_da = climate_da.assign_coords(lat_orig=("point", lats.values))
        climate_da = climate_da.assign_coords(long_orig=("point", lons.values))

        return climate_da
    except:
        print(f"Error loading climate data for year {lookup_yr}")
        pass


def get_all_climate_thresholds_year_months_for_latlongs(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
    year_var: str = "birth_year",
    month_var: str = "birth_month",
) -> pd.DataFrame:
    """
    df = df_min_age.copy()
    lat_col = "lat"
    long_col= "long"
    """

    unique_coords = df[[lat_col, long_col]].drop_duplicates()

    min_year = df[year_var].min() - 1  # need prior year
    max_year = df[year_var].max()

    # get all years and months for each climate variable for all coordinates
    df_splits = []
    for year in range(min_year, max_year + 1):
        df_split = unique_coords.copy()
        df_split["lookup_year"] = year
        df_splits.append(df_split)

    p = mp.Pool(processes=25)
    # Wrap the p.map call with tqdm for progress tracking
    results_xarrays = list(
        tqdm(
            p.imap(
                partial(
                    get_climate_thresholds_all_locs,
                ),
                df_splits,
            ),
            total=len(df_splits),  # Total number of tasks for tqdm
            desc="Processing climate variables",  # Description for the progress bar
        )
    )
    p.close()
    p.join()

    # remove any None results due to loading errors
    results_xarrays = [da for da in results_xarrays if da is not None]
    # Concatenate the xarrays along the "lookup_year" dimension
    results_da = xr.concat(results_xarrays, dim="year")
    return results_da


##


def get_climate_vars_for_year(
    year_df: pd.DataFrame,
    climate_variables: list[str],
    lat_col: str = "lat",
    long_col: str = "long",
    year_col: str = "int_year",
) -> pd.DataFrame:
    if year_df[year_col].nunique() != 1:
        msg = "Multiple years in climate data."
        raise ValueError(msg)

    yr = year_df[year_col].iloc[0]

    temp_df = year_df.copy()
    lats = xr.DataArray(temp_df[lat_col], dims="point")
    lons = xr.DataArray(temp_df[long_col], dims="point")
    years = xr.DataArray(temp_df[year_col], dims="point")
    for climate_variable in climate_variables:
        # climate_ds = ClimateMalnutritionData(Path(DEFAULT_ROOT)/'stunting').load_climate_raster(climate_variable, 'ssp245', yr, 0)
        # Temporary workaround for climate data loading
        climate_ds = xr.open_dataset(
            f"/mnt/share/erf/climate_downscale/results/annual/raw/historical/{climate_variable}/{yr}_era5.nc"
        )
        climate_ds = climate_ds.load()
        climate_ds = climate_ds["value"]
        temp_df[climate_variable] = (
            climate_ds.sel(latitude=lats, longitude=lons, method="nearest")
            .to_numpy()
            .flatten()  # the flatten also wasn't there before
        )
    return temp_df


CLIMATE_VAR_NAMES = [
        'mean_temperature', 'days_over_30C', 'precipitation_days', 
        'total_precipitation', 'mean_low_temperature', 'mean_high_temperature',
        'relative_humidity']# 'days_over_26C', 'days_over_27C', 'days_over_28C',


def get_climate_vars_for_year_month(
    year_df: pd.DataFrame,
    climate_variables: list[str],
    lat_col: str = "lat",
    long_col: str = "long",
    year_col: str = "climate_year",
    month_col: str = "climate_month",
) -> pd.DataFrame:
    if year_df[year_col].nunique() != 1:
        msg = "Multiple years in climate data."
        raise ValueError(msg)

    yr = year_df[year_col].iloc[0]

    temp_df = year_df.copy()
    lats = xr.DataArray(temp_df[lat_col], dims="point")
    lons = xr.DataArray(temp_df[long_col], dims="point")
    months = xr.DataArray(temp_df[month_col], dims="point")
    for climate_variable in climate_variables:
        # climate_ds = ClimateMalnutritionData(Path(DEFAULT_ROOT)/'stunting').load_climate_raster(climate_variable, 'ssp245', yr, 0)
        # Temporary workaround for climate data loading
        climate_ds = xr.open_dataset(
            f"/mnt/share/erf/climate_downscale/results/monthly/raw/historical/{climate_variable}/{yr}_era5.nc"
        )["value"].load()
        if 'quantile' in climate_ds.coords:
            thresholds = climate_ds['quantile'].values
            for threshold in thresholds:
                temp_df[f"days_over_{threshold}_quant"] = (
                    climate_ds.sel(latitude=lats, longitude=lons, month=months, quantile=threshold, method="nearest")
                    .to_numpy()
                    .flatten()  
                )
        else:
            temp_df[climate_variable] = (
                climate_ds.sel(latitude=lats, longitude=lons, month=months, method="nearest")
                .to_numpy()
                .flatten()  # the flatten also wasn't there before
            )
    return temp_df


def get_lifetime_climate_vars(
        df: pd.DataFrame,
        lat_col: str = "lat",
        long_col: str = "long",
        year_col: str = "int_year",
        month_col: str = "int_month",
        age_year_col: str = "age_year",
        age_month_col: str = "age_month",
        include_gestation: bool = False,
) -> pd.DataFrame:
    df['row_id'] = df.index
    # Calculate birth year
    df['birth_year'] = df[year_col] + (df[month_col]/12) - df[age_year_col] - (0.75 if include_gestation else 0)
    lifetime_climate_dfs = []
    # get all years of life to explode later
    df['years_of_life'] = df.apply(lambda row: list(range(int(np.floor(row['birth_year'])), int(row[year_col] + 1))), axis=1)
    df_exploded = df[[lat_col, long_col, 'years_of_life', 'row_id']].explode('years_of_life').rename(columns={'years_of_life': 'life_year'})

    unique_coords = df_exploded[[lat_col, long_col, 'life_year']].drop_duplicates()
    df_splits = [year_df for _, year_df in unique_coords.groupby('life_year')]
    p = mp.Pool(processes=25)
    results_df = pd.concat(
        p.map(
            partial(get_climate_vars_for_year, climate_variables=CLIMATE_VAR_NAMES, year_col='life_year'), df_splits
        )
    )
    p.close()
    p.join()
    # merge back to exploded df
    df_merged = df_exploded.merge(results_df, how='left', validate="many_to_one")#merge_left_without_inflating(df_exploded, results_df, on=[lat_col, long_col, 'life_year'])
    # average over lifetime
    lifetime_climate_df = df_merged.groupby('row_id', as_index=False)[CLIMATE_VAR_NAMES].mean()
    df_final = df.merge(lifetime_climate_df, how='left', on='row_id', validate="one_to_one")
    #df_final = df_final.drop(columns=['row_id', 'birth_year', 'years_of_life'])
    return df_final




def get_climate_vars_for_dataframe(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
    year_col: str = "int_year",
) -> pd.DataFrame:
    var_names = [
        "mean_temperature",
        # "precipitation_days",
        "total_precipitation",
        # "mean_low_temperature",
        # "mean_high_temperature",
        # "relative_humidity",
        # "days_over_24C",
        # "days_over_25C",
        # "days_over_26C",
        # "days_over_27C",
        # "days_over_28C",
        # "days_over_29C",
        "days_over_30C",
        # "days_over_31C",
        # "days_over_32C",
        # "days_over_33C",
    ]

    unique_coords = df[[lat_col, long_col, year_col]].drop_duplicates()

    df_splits = [year_df for _, year_df in unique_coords.groupby(year_col)]
    p = mp.Pool(processes=25)
    results_df = pd.concat(
        p.map(
            partial(
                get_climate_vars_for_year,
                climate_variables=var_names,
                year_col=year_col,
            ),
            df_splits,
        )
    )
    p.close()
    p.join()
    return results_df


ELEVATION_FILEPATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/elevation/GLOBE_DEM_MOSAIC_Y2016M02D09.TIF"


def get_elevation_for_dataframe(
    df: pd.DataFrame, lat_col: str = "lat", long_col: str = "long"
) -> pd.DataFrame:
    unique_coords = df[[lat_col, long_col]].drop_duplicates()
    elevation_ds: xr.Dataset = rioxarray.open_rasterio(ELEVATION_FILEPATH, masked=True)  # type: ignore[assignment]
    unique_coords["elevation"] = elevation_ds.sel(
        x=xr.DataArray(unique_coords.long, dims="z"),
        y=xr.DataArray(unique_coords.lat, dims="z"),
        band=1,
        method="nearest",
    ).to_numpy()

    unique_coords["elevation"] = unique_coords["elevation"].fillna(0)
    unique_coords["elevation"] = unique_coords["elevation"].astype(int)
    results_df = merge_left_without_inflating(df, unique_coords, on=[lat_col, long_col])
    if results_df["elevation"].isna().any():
        msg = "Null elevation values."
        raise RuntimeError(msg)
    return results_df


def get_ldipc_from_asset_score(
    asset_df: pd.DataFrame,
    cm_data,  # instance providing the load_ldi_distributions() method
    asset_score_col: str = "wealth_index_dhs",
    year_df_col: str = "year_start",
    *,
    ldi_version="",
    weights_col="pweight",
    plot: bool = True,
    plot_pdf_path: str = "./nid_plots.pdf",
) -> pd.DataFrame:
    """
    Calculates LDI-PC from asset scores using four methods:
      (1) Unweighted percentiles, direct interpolation (no matching)
      (2) Weighted percentiles, direct interpolation (no matching)
      (3) Unweighted percentiles with distribution matching
      (4) Weighted percentiles with distribution matching

    Additionally, if plotting is enabled, a single PDF is produced (at plot_pdf_path)
    containing one page per NID. Each page shows:
        - The reference (target) LDI distribution (which is always 10 points)
        - The 4 computed LDI-PC curves.

    The dynamic grid used for matching and plotting is determined based on the number
    of observations in the NID (nid_df).

    Parameters:
      asset_df         : DataFrame containing asset scores and related columns.
      cm_data          : Object providing the load_ldi_distributions() method.
      asset_score_col  : Column name for the asset score.
      year_df_col      : Column name for the year.
      ldi_version      : Version identifier for LDI distribution.
      weights_col      : Column name for weights.
      plot             : If True, generate a PDF with NID plots.
      plot_pdf_path    : File path for the output PDF containing the plots.

    Returns:
      A DataFrame with additional columns:
        - ldi_pc_unweighted_no_match
        - ldi_pc_weighted_no_match
        - ldi_pc_unweighted_match
        - ldi_pc_weighted_match
    """
    # Load LDI distribution data
    ldi = cm_data.load_ldi_distributions(geospecificity="national", version=ldi_version)
    if "scenario" in ldi.columns:
        if 0 in ldi.scenario.unique():
            ldi = ldi.loc[ldi["scenario"] == 0].drop("scenario", axis=1)
        elif "reference" in ldi.scenario.unique():
            ldi = ldi.loc[ldi["scenario"] == "reference"].drop("scenario", axis=1)
        elif 4.5 in ldi.scenario.unique():
            ldi = ldi.loc[ldi["scenario"] == 4.5].drop("scenario", axis=1)
        else:
            raise ValueError("No valid scenario in LDI data.")
    print("Calculating four versions of LDI-PC (weighted/unweighted, match/no-match).")

    # Optionally initialize PdfPages for a single PDF with all plots.
    if plot:
        pdf_pages = PdfPages(plot_pdf_path)

    results = []

    for nid in asset_df.nid.unique():
        nid_df = (
            asset_df.loc[asset_df.nid == nid]
            .copy()
            .sort_values(["nid", "ihme_loc_id", year_df_col, asset_score_col])
        )

        # Choose year: if multiple, use the earliest
        if nid_df[year_df_col].nunique() > 1:
            print(f"Multiple years for NID {nid}: {nid_df[year_df_col].unique()}")
            year = nid_df[year_df_col].min()
        else:
            year = nid_df[year_df_col].iloc[0]

        # Ensure a single location for this NID
        if nid_df.ihme_loc_id.nunique() > 1:
            raise ValueError(f"Multiple locations for NID {nid}")
        ihme_loc_id = nid_df.ihme_loc_id.iloc[0]

        # Compute percentiles: unweighted and weighted.
        nid_df["unweighted_population_percentile"] = nid_df[asset_score_col].rank(
            pct=True
        )
        nid_df["cum_weight"] = nid_df[weights_col].cumsum()
        total_weight = nid_df[weights_col].sum()
        nid_df["weighted_population_percentile"] = nid_df["cum_weight"] / total_weight

        # Get LDI data for this location and year; sort by population_percentile.
        ldi_df = ldi.loc[(ldi.ihme_loc_id == ihme_loc_id) & (ldi.year_id == year)]
        ldi_df = ldi_df.sort_values("population_percentile")
        if ldi_df.empty:
            raise ValueError(f"No LDI data for NID {nid} in year {year}")

        # Determine dynamic grid resolution based on survey data.
        # Since the target distribution always has 10 points, we use the NID data resolution.
        grid_points = max(1000, min(1000000, len(nid_df) * 100))
        # print(f"Using {grid_points} points for NID {nid}.")
        dynamic_grid = np.linspace(0, 1, grid_points)

        # Build the interpolator from the target (LDI) distribution.
        ldi_interpolator = PchipInterpolator(
            ldi_df["population_percentile"], ldi_df["ldipc"]
        )
        # Evaluate reference LDI distribution on the dynamic grid.
        ldi_ref = ldi_interpolator(dynamic_grid)

        # Direct interpolation (no matching) for both unweighted and weighted.
        unweighted_no_match = ldi_interpolator(
            nid_df["unweighted_population_percentile"]
        )
        weighted_no_match = ldi_interpolator(nid_df["weighted_population_percentile"])

        # Evaluate LDI distribution on the grid for matching purposes.
        interpolated_income_distrib = ldi_interpolator(dynamic_grid)

        # ldi_value_at_80th_percentile = ldi_interpolator(0.8)

        # Define the matching procedure.
        def match_distribution(pop_pct, asset_scores):
            """
            For a given set of population percentiles and asset scores,
            iterate over a range of clipping thresholds of the interpolated LDI
            distribution. The candidate curve that minimizes the absolute difference
            between its scaled version and the scaled asset scores is returned.
            """
            scaled_asset = (asset_scores - asset_scores.min()) / (
                asset_scores.max() - asset_scores.min()
            )
            best_candidate = None
            minimum_difference = np.inf
            # Try thresholds from 25% to 100% in 76 steps.
            for threshold_quantile in np.linspace(0.8, 1, 21):
                threshold_value = np.quantile(
                    interpolated_income_distrib, threshold_quantile
                )
                clipped_income = interpolated_income_distrib[
                    interpolated_income_distrib <= threshold_value
                ]
                if len(clipped_income) < 2:
                    continue
                clipped_interpolator = PchipInterpolator(
                    np.linspace(0, 1, len(clipped_income)), clipped_income
                )
                candidate = clipped_interpolator(pop_pct)
                scaled_candidate = (candidate - candidate.min()) / (
                    candidate.max() - candidate.min()
                )
                diff = np.abs(scaled_candidate - scaled_asset).sum()
                if diff < minimum_difference:
                    minimum_difference = diff
                    best_candidate = candidate
            return best_candidate

        unweighted_match = match_distribution(
            nid_df["unweighted_population_percentile"], nid_df[asset_score_col]
        )
        weighted_match = match_distribution(
            nid_df["weighted_population_percentile"], nid_df[asset_score_col]
        )

        # Store the four computed versions in new columns.
        nid_df["ldipc_unweighted_no_match"] = unweighted_no_match
        nid_df["ldipc_weighted_no_match"] = weighted_no_match
        nid_df["ldipc_unweighted_match"] = unweighted_match
        nid_df["ldipc_weighted_match"] = weighted_match

        # Plotting per NID if enabled.
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot the reference LDI distribution (flipped axes).
            ax.plot(
                ldi_ref,
                dynamic_grid,
                label="Reference LDI",
                linestyle="--",
                color="black",
            )

            # Plot the computed LDI-PC curves with flipped axes.
            sorted_unw = sorted(
                zip(unweighted_no_match, nid_df["unweighted_population_percentile"])
            )
            ax.plot(*zip(*sorted_unw), label="Unweighted No Match")

            sorted_w = sorted(
                zip(weighted_no_match, nid_df["weighted_population_percentile"])
            )
            ax.plot(*zip(*sorted_w), label="Weighted No Match")

            sorted_unw_match = sorted(
                zip(unweighted_match, nid_df["unweighted_population_percentile"])
            )
            ax.plot(*zip(*sorted_unw_match), label="Unweighted Match")

            sorted_w_match = sorted(
                zip(weighted_match, nid_df["weighted_population_percentile"])
            )
            ax.plot(*zip(*sorted_w_match), label="Weighted Match")

            # Compute scaling parameters for wealth index.
            wealth_min = nid_df[asset_score_col].min()
            wealth_max = nid_df[asset_score_col].max()
            ldi_min = ldi_ref.min()
            ldi_max = ldi_ref.max()

            # Define transformation functions for mapping wealth index to LDI range and vice versa.
            def wealth_to_ldi(w):
                return (w - wealth_min) / (wealth_max - wealth_min) * (
                    ldi_max - ldi_min
                ) + ldi_min

            def ldi_to_wealth(x):
                return (x - ldi_min) / (ldi_max - ldi_min) * (
                    wealth_max - wealth_min
                ) + wealth_min

            # Plot the original wealth index distribution, scaled to LDI range.
            sorted_wealth = sorted(
                zip(nid_df[asset_score_col], nid_df["unweighted_population_percentile"])
            )
            scaled_sorted_wealth = [(wealth_to_ldi(w), p) for w, p in sorted_wealth]
            ax.plot(
                *zip(*scaled_sorted_wealth),
                label="Wealth Index",
                linestyle=":",
                color="blue",
            )

            sorted_wealth = sorted(
                zip(nid_df[asset_score_col], nid_df["weighted_population_percentile"])
            )
            scaled_sorted_wealth = [(wealth_to_ldi(w), p) for w, p in sorted_wealth]
            ax.plot(
                *zip(*scaled_sorted_wealth),
                label="Wealth Index (weighted)",
                linestyle=":",
                color="green",
            )

            # Set labels and title.
            ax.set_xlabel("LDI-PC")
            ax.set_ylabel("Population Percentile")
            ax.set_title(f"NID {nid} | Year: {year} | ihme_loc_id: {ihme_loc_id}")
            ax.legend()
            ax.grid(True)

            # Add a secondary x-axis for Wealth Index.
            secax = ax.secondary_xaxis("top", functions=(ldi_to_wealth, wealth_to_ldi))
            secax.set_xlabel("Wealth Index")

            pdf_pages.savefig(fig)
            plt.close(fig)

        results.append(nid_df)

    if plot:
        pdf_pages.close()

    asset_df_ldipc = pd.concat(results)
    new_cols = [
        "ldipc_unweighted_no_match",
        "ldipc_weighted_no_match",
        "ldipc_unweighted_match",
        "ldipc_weighted_match",
    ]
    if asset_df_ldipc[new_cols].isna().sum().sum() != 0:
        raise RuntimeError("Null LDI-PC values in one of the methods.")
    if len(asset_df_ldipc) != len(asset_df):
        raise RuntimeError("Mismatch in length of asset data and LDI-PC data.")
    return asset_df_ldipc  # type: ignore[no-any-return]


LDIPC_METHOD_COLUMNS = [
    "ldipc_unweighted_no_match",
    "ldipc_weighted_no_match",
    "ldipc_unweighted_match",
    "ldipc_weighted_match",
]

# Default household grain of the wealth extractions. Percentiles are computed on
# this grain so that siblings sharing a household are not counted twice.
DEFAULT_HOUSEHOLD_COLS = ["nid", "ihme_loc_id", "psu", "hh_id"]


def load_reference_ldi_distributions(
    cm_data,  # instance providing the load_ldi_distributions() method
    ldi_version: str = "",
) -> pd.DataFrame:
    """Load national LDI-PC distributions, collapsed to the reference scenario.

    Different LDI versions encode the reference scenario differently, so try the
    known encodings in turn.
    """
    ldi = cm_data.load_ldi_distributions(geospecificity="national", version=ldi_version)
    if "scenario" in ldi.columns:
        scenarios = ldi.scenario.unique()
        for reference_scenario in (0, "reference", 4.5):
            if reference_scenario in scenarios:
                return ldi.loc[ldi["scenario"] == reference_scenario].drop(
                    "scenario", axis=1
                )
        raise ValueError("No valid scenario in LDI data.")
    return ldi


def find_best_clip_quantile(
    reference_curve: np.ndarray,
    pop_pct: np.ndarray,
    asset_scores: np.ndarray,
) -> float:
    """Find the tail truncation of an income distribution that best matches asset scores.

    Iterates over clipping thresholds of `reference_curve` and returns the quantile
    whose truncated, min-max-scaled curve is closest to the min-max-scaled asset
    scores. This is the search half of the `match_distribution` closure in
    `get_ldipc_from_asset_score`, split out so the chosen threshold can be applied
    to a *different* year's distribution than the one it was fit on.
    """
    asset_span = asset_scores.max() - asset_scores.min()
    if asset_span == 0:
        return 1.0
    scaled_asset = (asset_scores - asset_scores.min()) / asset_span

    best_quantile = 1.0
    minimum_difference = np.inf
    for threshold_quantile in np.linspace(0.8, 1, 21):
        candidate = apply_clip_quantile(reference_curve, threshold_quantile, pop_pct)
        if candidate is None:
            continue
        candidate_span = candidate.max() - candidate.min()
        if candidate_span == 0:
            continue
        scaled_candidate = (candidate - candidate.min()) / candidate_span
        diff = np.abs(scaled_candidate - scaled_asset).sum()
        if diff < minimum_difference:
            minimum_difference = diff
            best_quantile = threshold_quantile
    return best_quantile


def apply_clip_quantile(
    reference_curve: np.ndarray,
    clip_quantile: float,
    pop_pct: np.ndarray,
) -> np.ndarray | None:
    """Evaluate `pop_pct` against `reference_curve` truncated at `clip_quantile`.

    Returns None if the truncation leaves too few points to interpolate.
    """
    threshold_value = np.quantile(reference_curve, clip_quantile)
    clipped_income = reference_curve[reference_curve <= threshold_value]
    if len(clipped_income) < 2:
        return None
    clipped_interpolator = PchipInterpolator(
        np.linspace(0, 1, len(clipped_income)), clipped_income
    )
    return clipped_interpolator(pop_pct)


PERCENTILE_COLUMNS = [
    "unweighted_population_percentile",
    "weighted_population_percentile",
    "unweighted_clip_quantile",
    "weighted_clip_quantile",
]

# Grid used to represent an income curve when applying a clip quantile. The curve
# has only 10 knots and is smooth, so this is far denser than needed.
CLIP_GRID_POINTS = 100000


def drop_stale_percentile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip percentile columns left behind by an earlier pass.

    `get_ldipc_from_asset_score` computes `unweighted_population_percentile` and
    `weighted_population_percentile` for its own use and leaves them on its
    output, so any frame descended from it arrives already carrying two of these
    names. `get_asset_score_percentiles` is the function that defines them, and
    every merge downstream of it keys on them, so a stale copy is not harmless:
    a merge would rename both sides to *_x / *_y and the names would vanish.

    Dropping is safe -- these are recomputed from the wealth extract, which is
    where they came from in the first place.
    """
    stale_cols = [col for col in PERCENTILE_COLUMNS if col in df.columns]
    return df.drop(columns=stale_cols) if stale_cols else df


def get_asset_score_percentiles(
    asset_df: pd.DataFrame,
    cm_data,  # instance providing the load_ldi_distributions() method
    asset_score_col: str = "wealth_index_dhs",
    year_df_col: str = "year_start",
    *,
    ldi_version: str = "",
    weights_col: str = "pweight",
    household_cols: list[str] | None = None,
    grid_points: int | None = None,
) -> pd.DataFrame:
    """Step 1 of birth-year LDI-PC: rank households within each survey.

    Run this on the *wealth extract*, before it is merged onto survey data. That
    matters: ranking has to cover every household the survey enumerated, not just
    the ones that turned out to contain a child. Ranking after the merge would
    rank a child-bearing subset and shift everybody's percentile.

    Adds four columns, all constant within a household:

      unweighted_population_percentile : rank of the asset score in the survey
      weighted_population_percentile   : same, weighted by `weights_col`
      unweighted_clip_quantile         : tail truncation for the *match* method
      weighted_clip_quantile           : same, for the weighted percentile

    On the clip quantiles: the *match* method exists because the income
    distribution has a long rich tail that the asset index does not -- an asset
    index built from "owns a fridge, owns a motorbike" saturates at the top. So
    rather than reading a household's income off its raw rank, `match` shrinks
    everyone's rank by a factor first, which is equivalent to lopping off the top
    of the income distribution and restretching what remains over the full 0-100%
    range. This function searches factors from 0.80 to 1.00 in 21 steps and keeps
    whichever produces an income curve whose shape best matches the shape of the
    asset scores. The chosen factor is what gets stored, so that
    `assign_ldipc_by_birth_year` can apply it to a different year's distribution.

    That search needs *an* income distribution to fit against, and uses the
    survey's own `year_df_col`. It is only shape being compared, after min-max
    scaling, and adjacent years of a country's LDI distribution are close to a
    pure rescaling of each other, so the fitted factor is nearly invariant to
    which year is used.

    Parameters:
      asset_df        : Wealth extract. One row per household, or set household_cols.
      cm_data         : Object providing the load_ldi_distributions() method.
      asset_score_col : Column name for the asset score.
      year_df_col     : Survey year. Only used to fit the clip quantiles.
      ldi_version     : Version identifier for LDI distribution.
      weights_col     : Column name for household weights.
      household_cols  : Grain to rank on. Defaults to DEFAULT_HOUSEHOLD_COLS; pass
                        [] if `asset_df` is already one row per household.
      grid_points     : Resolution of the clip search grid. Defaults to the
                        heuristic used by `get_ldipc_from_asset_score`.

    Returns:
      `asset_df` with PERCENTILE_COLUMNS added.
    """
    household_cols = (
        DEFAULT_HOUSEHOLD_COLS if household_cols is None else list(household_cols)
    )
    required_cols = {
        "nid",
        "ihme_loc_id",
        asset_score_col,
        year_df_col,
        weights_col,
        *household_cols,
    }
    missing_cols = sorted(required_cols - set(asset_df.columns))
    if missing_cols:
        msg = (
            f"asset_df is missing required columns: {missing_cols}. Pass "
            "household_cols=[] if each row is already one household."
        )
        raise ValueError(msg)

    asset_df = drop_stale_percentile_columns(asset_df)

    ldi = load_reference_ldi_distributions(cm_data, ldi_version)
    ldi = ldi.sort_values(["ihme_loc_id", "year_id", "population_percentile"])

    print("Ranking households within each survey and fitting match clip quantiles.")

    results = []
    for nid in asset_df.nid.unique():
        nid_df = asset_df.loc[asset_df.nid == nid].copy()

        if nid_df.ihme_loc_id.nunique() > 1:
            msg = f"Multiple locations for NID {nid}"
            raise ValueError(msg)
        ihme_loc_id = nid_df.ihme_loc_id.iloc[0]

        if nid_df[year_df_col].nunique() > 1:
            print(f"Multiple years for NID {nid}: {nid_df[year_df_col].unique()}")
        survey_year = int(nid_df[year_df_col].min())

        # --- Survey-wide ranking, on household grain ---
        if household_cols:
            household_df = nid_df.groupby(household_cols, as_index=False).agg(
                **{
                    asset_score_col: (asset_score_col, "first"),
                    weights_col: (weights_col, "first"),
                    "n_asset_scores": (asset_score_col, "nunique"),
                    "n_weights": (weights_col, "nunique"),
                }
            )
            if (household_df.n_asset_scores != 1).any() or (
                household_df.n_weights != 1
            ).any():
                msg = (
                    f"NID {nid} has households with more than one {asset_score_col} "
                    f"or {weights_col} value; the ranking grain is wrong."
                )
                raise RuntimeError(msg)
            household_df = household_df.drop(columns=["n_asset_scores", "n_weights"])
        else:
            household_df = nid_df.copy()

        # Sort explicitly by asset score: the weighted percentile is a running sum
        # of weights, so it is only meaningful in asset-score order.
        household_df = household_df.sort_values(asset_score_col, kind="mergesort")
        total_weight = household_df[weights_col].sum()
        if total_weight <= 0:
            msg = f"Non-positive total {weights_col} for NID {nid}."
            raise RuntimeError(msg)
        household_df["unweighted_population_percentile"] = household_df[
            asset_score_col
        ].rank(pct=True)
        household_df["weighted_population_percentile"] = (
            household_df[weights_col].cumsum() / total_weight
        )

        # --- Fit the match method's tail truncation, against the survey year ---
        survey_ldi = ldi.loc[
            (ldi.ihme_loc_id == ihme_loc_id) & (ldi.year_id == survey_year)
        ]
        if survey_ldi.empty:
            msg = f"No LDI data for NID {nid} ({ihme_loc_id}) in year {survey_year}."
            raise ValueError(msg)
        n_grid = (
            grid_points
            if grid_points is not None
            else max(1000, min(1000000, len(household_df) * 100))
        )
        survey_curve = PchipInterpolator(
            survey_ldi["population_percentile"], survey_ldi["ldipc"]
        )(np.linspace(0, 1, n_grid))

        asset_scores = household_df[asset_score_col].to_numpy()
        household_df["unweighted_clip_quantile"] = find_best_clip_quantile(
            survey_curve,
            household_df["unweighted_population_percentile"].to_numpy(),
            asset_scores,
        )
        household_df["weighted_clip_quantile"] = find_best_clip_quantile(
            survey_curve,
            household_df["weighted_population_percentile"].to_numpy(),
            asset_scores,
        )

        if household_cols:
            # merge resets the index; put the caller's back so that concatenating
            # the per-NID frames does not manufacture a duplicated index.
            original_index = nid_df.index
            nid_df = merge_left_without_inflating(
                nid_df,
                household_df[household_cols + PERCENTILE_COLUMNS],
                on=household_cols,
            )
            nid_df.index = original_index
        else:
            nid_df = household_df

        results.append(nid_df)

    asset_df_ranked = pd.concat(results)
    if asset_df_ranked[PERCENTILE_COLUMNS].isna().sum().sum() != 0:
        msg = "Null percentiles; check for null asset scores or weights."
        raise RuntimeError(msg)
    if len(asset_df_ranked) != len(asset_df):
        msg = "Mismatch in length of asset data and ranked data."
        raise RuntimeError(msg)
    return asset_df_ranked  # type: ignore[no-any-return]


def assign_ldipc_by_birth_year(
    df: pd.DataFrame,
    cm_data,  # instance providing the load_ldi_distributions() method
    birth_year_col: str = "birth_year",
    *,
    ldi_version: str = "",
    missing_year_policy: str = "null",
    grid_points: int = CLIP_GRID_POINTS,
    col_suffix: str = "",
) -> pd.DataFrame:
    """Step 2 of birth-year LDI-PC: look each row's rank up in its birth year.

    Run this *after* the wealth extract has been merged onto survey data, which is
    where `birth_year_col` first exists. `df` must already carry
    PERCENTILE_COLUMNS from `get_asset_score_percentiles`.

    Each row's survey-wide percentile is read off the national LDI distribution
    for that row's birth year rather than the survey year, so two siblings in one
    household share a percentile but get different LDI-PC values. The *match*
    columns apply the clip quantile fitted in step 1 to the birth year's curve.

    Birth years the LDI series does not cover -- it starts in 1980, and surveys
    hold birth histories going back further -- are handled by
    `missing_year_policy`. The default leaves those rows null rather than
    inventing a value for them: there is no defensible income distribution to read
    a 1979 birth off, and clamping to 1980 would quietly assert that 1979 and 1980
    were the same, with the error growing the further back the birth year goes.

    Parameters:
      df                  : Merged survey data carrying PERCENTILE_COLUMNS.
      cm_data             : Object providing the load_ldi_distributions() method.
      birth_year_col      : Per-row year used to select the income distribution.
      ldi_version         : Version identifier for LDI distribution.
      missing_year_policy : What to do with a birth year the LDI series misses --
                            'null' (default) leaves the LDI-PC columns NaN and
                            `ldi_year_used` 0; 'clamp' falls back to the nearest
                            year with data; 'raise' refuses.
      grid_points         : Resolution used when applying a clip quantile.
      col_suffix          : Appended to the four output column names.

    Returns:
      `df` with `ldipc_{unweighted,weighted}_{no_match,match}` (plus `col_suffix`)
      and `ldi_year_used`, the LDI year each row resolved to (0 where no year
      could be resolved).
    """
    if missing_year_policy not in ("raise", "clamp", "null"):
        msg = "missing_year_policy must be 'null', 'clamp' or 'raise'."
        raise ValueError(msg)

    required_cols = {"ihme_loc_id", birth_year_col, *PERCENTILE_COLUMNS}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        msg = (
            f"df is missing required columns: {missing_cols}. Run "
            "get_asset_score_percentiles on the wealth extract first and carry "
            "its output columns through the merge."
        )
        raise ValueError(msg)

    df = df.copy()
    if df[birth_year_col].isna().any():
        n_null = int(df[birth_year_col].isna().sum())
        msg = (
            f"{n_null:,} of {len(df):,} rows have a null {birth_year_col}. "
            "Recover or drop them before calling this function."
        )
        raise ValueError(msg)
    if df[PERCENTILE_COLUMNS].isna().any().any():
        n_null = int(df[PERCENTILE_COLUMNS].isna().any(axis=1).sum())
        msg = (
            f"{n_null:,} of {len(df):,} rows have null percentile columns, most "
            "likely rows that failed to merge against the wealth extract. Drop "
            "them before calling this function."
        )
        raise ValueError(msg)
    df[birth_year_col] = df[birth_year_col].astype(int)

    ldi = load_reference_ldi_distributions(cm_data, ldi_version)
    ldi = ldi.sort_values(["ihme_loc_id", "year_id", "population_percentile"])
    available_years = {
        loc: np.sort(years)
        for loc, years in ldi.groupby("ihme_loc_id")["year_id"].unique().items()
    }
    clamped_years: dict[tuple[str, int], int] = {}
    null_years: dict[tuple[str, int], int] = {}

    def resolve_ldi_year(ihme_loc_id: str, year: int) -> int | None:
        """Map a requested year onto a year the LDI distribution actually covers.

        Returns None when there is no such year and the policy is to null out.
        """
        years = available_years.get(ihme_loc_id)
        if years is None:
            msg = f"No LDI data at all for location {ihme_loc_id}."
            raise ValueError(msg)
        if year in years:
            return year
        if missing_year_policy == "raise":
            msg = (
                f"No LDI data for {ihme_loc_id} in year {year} (available: "
                f"{years.min()}-{years.max()}). Pass missing_year_policy='clamp' "
                "to fall back to the nearest available year, or 'null' to leave "
                "these rows empty."
            )
            raise ValueError(msg)
        if missing_year_policy == "null":
            null_years[(ihme_loc_id, year)] = 0
            return None
        nearest = int(years[np.abs(years - year).argmin()])
        clamped_years[(ihme_loc_id, year)] = nearest
        return nearest

    print(f"Assigning LDI-PC from each row's {birth_year_col} income distribution.")

    out_cols = [col + col_suffix for col in LDIPC_METHOD_COLUMNS]
    ldipc = {col: np.full(len(df), np.nan) for col in out_cols}
    ldi_year_used = np.zeros(len(df), dtype=int)
    grid = np.linspace(0, 1, grid_points)

    unweighted_pct = df["unweighted_population_percentile"].to_numpy()
    weighted_pct = df["weighted_population_percentile"].to_numpy()
    unweighted_clip = df["unweighted_clip_quantile"].to_numpy()
    weighted_clip = df["weighted_clip_quantile"].to_numpy()

    # .indices gives positional row numbers, so this holds up whatever index the
    # caller's frame carries -- including a duplicated one.
    group_positions = df.groupby(["ihme_loc_id", birth_year_col], sort=False).indices
    for (ihme_loc_id, birth_year), idx in group_positions.items():
        resolved_year = resolve_ldi_year(ihme_loc_id, int(birth_year))
        if resolved_year is None:
            # Nothing to read this group's percentiles off. Leave the LDI-PC
            # columns NaN and ldi_year_used 0.
            null_years[(ihme_loc_id, int(birth_year))] += len(idx)
            continue
        ldi_year_used[idx] = resolved_year

        ldi_df = ldi.loc[
            (ldi.ihme_loc_id == ihme_loc_id) & (ldi.year_id == resolved_year)
        ]
        interpolator = PchipInterpolator(
            ldi_df["population_percentile"], ldi_df["ldipc"]
        )

        ldipc["ldipc_unweighted_no_match" + col_suffix][idx] = interpolator(
            unweighted_pct[idx]
        )
        ldipc["ldipc_weighted_no_match" + col_suffix][idx] = interpolator(
            weighted_pct[idx]
        )

        # The clip quantile is fitted per survey, so one location-year can hold
        # several. Apply each to the shared curve for this location-year.
        year_curve = interpolator(grid)
        for clips, pct, col in (
            (unweighted_clip, unweighted_pct, "ldipc_unweighted_match"),
            (weighted_clip, weighted_pct, "ldipc_weighted_match"),
        ):
            for clip in np.unique(clips[idx]):
                clip_idx = idx[clips[idx] == clip]
                matched = apply_clip_quantile(year_curve, clip, pct[clip_idx])
                # Fall back to the untruncated curve if the clip left too few points.
                if matched is None:
                    matched = interpolator(pct[clip_idx])
                ldipc[col + col_suffix][clip_idx] = matched

    for col, values in ldipc.items():
        df[col] = values
    df["ldi_year_used"] = ldi_year_used

    if clamped_years:
        logging.warning(
            f"Clamped {len(clamped_years)} location-years to the nearest year with "
            f"LDI data, e.g. {list(clamped_years.items())[:5]}"
        )
    if null_years:
        n_null_rows = sum(null_years.values())
        year_range = f"{min(y for _, y in null_years)}-{max(y for _, y in null_years)}"
        logging.warning(
            f"Left {n_null_rows:,} of {len(df):,} rows null across "
            f"{len(null_years)} location-years the LDI series does not cover "
            f"(birth years {year_range})."
        )

    # Rows that resolved to a year must have come out with a value; rows that did
    # not resolve are null on purpose.
    resolved = ldi_year_used != 0
    if df.loc[resolved, out_cols].isna().sum().sum() != 0:
        msg = "Null LDI-PC values in one of the methods."
        raise RuntimeError(msg)
    return df


def get_ldipc_from_asset_score_by_birth_year(
    asset_df: pd.DataFrame,
    cm_data,  # instance providing the load_ldi_distributions() method
    child_df: pd.DataFrame,
    asset_score_col: str = "wealth_index_dhs",
    year_df_col: str = "year_start",
    birth_year_col: str = "birth_year",
    *,
    ldi_version: str = "",
    weights_col: str = "pweight",
    household_cols: list[str] | None = None,
    missing_year_policy: str = "null",
    grid_points: int | None = None,
    col_suffix: str = "",
) -> pd.DataFrame:
    """Calculate LDI-PC from asset scores, matched to each child's *birth* year.

    Same four methods as `get_ldipc_from_asset_score`:
      (1) Unweighted percentiles, direct interpolation (no matching)
      (2) Weighted percentiles, direct interpolation (no matching)
      (3) Unweighted percentiles with distribution matching
      (4) Weighted percentiles with distribution matching

    The one difference is *which year's income distribution a percentile is read
    off*. In both functions a household's place in the asset score distribution is
    fixed by the survey it came from, so a household at asset percentile X in a
    survey run in year Y stays at percentile X. What changes is the next step:

      get_ldipc_from_asset_score : percentile X -> the consumption percentile X
          estimated for year Y (the survey year). Every child in the household
          gets that one value.
      this function             : percentile X -> the consumption percentile X
          estimated for year Z, where Z is that child's birth year. Siblings in
          one household share percentile X but land on different LDI-PC values,
          because they were born into different years of the country's income
          distribution.

    So a household at the 40th asset percentile in a 2015 survey, with children
    born in 2005 and 2012, gets the 40th percentile of the 2005 consumption
    distribution for the first child and of the 2012 distribution for the second,
    rather than the 40th percentile of the 2015 distribution for both.

    One wrinkle carried over from the original: the *match* methods do not read a
    percentile straight off the curve, they first shrink everyone's percentile by
    a factor fitted so that the resulting income curve has the same shape as the
    asset scores (see `get_asset_score_percentiles` for why). That factor is
    fitted once per survey against the *survey* year's curve, then applied to each
    birth year's curve. Only shape enters the fit, and neighbouring years of a
    country's LDI distribution are close to rescalings of one another, so which
    year it is fitted on barely matters.

    Why this takes two frames where `get_ldipc_from_asset_score` takes one: the
    two things it needs live in two different places. The ranking has to run over
    the whole wealth extract, because a household's percentile only means anything
    relative to every household the survey enumerated -- including the ones that
    turned out to contain no child. The birth year only exists on the child data.
    So `asset_df` supplies the ranking and `child_df` supplies the birth years,
    and the function joins them on `household_cols`.

    Rows of `child_df` with no matching household in `asset_df` come back with the
    LDI-PC columns left null, the same way a left merge of
    `get_ldipc_from_asset_score` output onto child data would leave them. So do
    rows whose birth year predates the LDI series, which starts in 1980 -- see
    `missing_year_policy`.

    Parameters:
      asset_df            : Wealth extract, one row per household (or set
                            household_cols). Used only for ranking; not returned.
      cm_data             : Object providing the load_ldi_distributions() method.
      child_df            : Child-level data carrying `birth_year_col` and the
                            `household_cols` needed to join it to `asset_df`.
      asset_score_col     : Column name for the asset score, in `asset_df`.
      year_df_col         : Survey year in `asset_df`. Only used to fit the match
                            factors.
      birth_year_col      : Per-child year in `child_df`, used to pick the income
                            distribution.
      ldi_version         : Version identifier for LDI distribution.
      weights_col         : Column name for household weights, in `asset_df`.
      household_cols      : Household key, used both as the ranking grain and as
                            the join key. Defaults to DEFAULT_HOUSEHOLD_COLS.
      missing_year_policy : What to do with a birth year the LDI series misses
                            (it starts in 1980) -- 'null' (default) leaves those
                            rows empty, 'clamp' uses the nearest year with data,
                            'raise' refuses.
      grid_points         : Resolution of the interpolation grids.
      col_suffix          : Appended to the four output column names.

    Returns:
      `child_df` with additional columns:
        - ldipc_unweighted_no_match
        - ldipc_weighted_no_match
        - ldipc_unweighted_match
        - ldipc_weighted_match
        - ldi_year_used  (the LDI year each row resolved to, = birth year unless
                          clamped; 0 for rows that did not match `asset_df` and
                          for birth years the LDI series does not cover)
    """
    household_cols = (
        DEFAULT_HOUSEHOLD_COLS if household_cols is None else list(household_cols)
    )
    if not household_cols:
        msg = (
            "household_cols cannot be empty: it is the key used to carry each "
            "household's percentile from asset_df over to the children in child_df."
        )
        raise ValueError(msg)
    missing_cols = sorted(
        {birth_year_col, *household_cols} - set(child_df.columns)
    )
    if missing_cols:
        msg = f"child_df is missing required columns: {missing_cols}."
        raise ValueError(msg)

    # Refuse to quietly replace survey-year results with birth-year ones under the
    # same names. Reachable when child_df carries the output of
    # `get_ldipc_from_asset_score` and no col_suffix is given to tell them apart.
    clashing_cols = sorted(
        {col + col_suffix for col in LDIPC_METHOD_COLUMNS} & set(child_df.columns)
    )
    if clashing_cols:
        msg = (
            f"child_df already has these output columns: {clashing_cols}. Pass a "
            "col_suffix (e.g. col_suffix='_birth_year') to keep both sets, or drop "
            "the existing ones first."
        )
        raise ValueError(msg)

    # Step 1: rank each household within its own survey. Runs on the full wealth
    # extract, so the percentiles are the same ones get_ldipc_from_asset_score
    # would produce.
    asset_df_ranked = get_asset_score_percentiles(
        asset_df,
        cm_data,
        asset_score_col=asset_score_col,
        year_df_col=year_df_col,
        ldi_version=ldi_version,
        weights_col=weights_col,
        household_cols=household_cols,
        grid_points=grid_points,
    )

    # Carry those percentiles onto the children. One row per household first, or
    # the merge would inflate child_df. child_df is very likely descended from
    # `get_ldipc_from_asset_score` and carrying two of these column names already,
    # which would collide in the merge, so clear them out.
    household_percentiles = asset_df_ranked[
        household_cols + PERCENTILE_COLUMNS
    ].drop_duplicates(subset=household_cols)
    original_index = child_df.index
    child_df = merge_left_without_inflating(
        drop_stale_percentile_columns(child_df).reset_index(drop=True),
        household_percentiles,
        on=household_cols,
    )

    # Step 2: read each child's household percentile off the income distribution
    # for that child's birth year. This is the step that differs from
    # get_ldipc_from_asset_score, which reads them all off the survey year.
    # Children whose household is not in the extract have no percentile to look
    # up, so they sit out and keep null LDI-PC.
    matched = child_df[PERCENTILE_COLUMNS].notna().all(axis=1)
    n_unmatched = int((~matched).sum())
    if n_unmatched:
        print(
            f"{n_unmatched:,} of {len(child_df):,} child rows have no household in "
            "the wealth extract; leaving their LDI-PC null."
        )
    assigned = assign_ldipc_by_birth_year(
        child_df.loc[matched],
        cm_data,
        birth_year_col=birth_year_col,
        ldi_version=ldi_version,
        missing_year_policy=missing_year_policy,
        grid_points=CLIP_GRID_POINTS if grid_points is None else grid_points,
        col_suffix=col_suffix,
    )

    new_cols = [col + col_suffix for col in LDIPC_METHOD_COLUMNS] + ["ldi_year_used"]
    for col in new_cols:
        child_df[col] = assigned[col].reindex(child_df.index)
    child_df["ldi_year_used"] = child_df["ldi_year_used"].fillna(0).astype(int)
    child_df.index = original_index
    return child_df



WEALTH_DATASET_COMMON_COLUMNS = [
    "ihme_loc_id",
    "nid",
    "psu",
    "strata",
    "hh_id",
    "hhweight",
    "lat",
    "long",
    "year_start",
    "geospatial_id",
]


def filter_point_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with point == 1."""
    return df[df["point"] == 1]


def validate_lat_long_uniqueness(df: pd.DataFrame) -> None:
    """Ensure that for each household (nid, psu, hh_id), lat and long are unique."""
    grouped = df.groupby(["nid", "psu", "hh_id"])
    if not (grouped["lat"].nunique().eq(1) & grouped["long"].nunique().eq(1)).all():
        raise ValueError(
            "Multiple latitudes or longitudes for the same nid, psu, and hh_id."
        )


def validate_no_duplicate_entries(
    df: pd.DataFrame, group_cols: list, error_msg: str
) -> None:
    """Ensure that grouping by group_cols gives a single entry per group."""
    if df.groupby(group_cols).size().max() != 1:
        raise ValueError(error_msg)


def common_validations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform validations common to all wealth datasets:
      - Check lat/long uniqueness.
      - Check no duplicate entries by (nid, psu, hh_id) and related groupings.
      - Merge with location metadata and cast types.
    """
    validate_lat_long_uniqueness(df)
    validate_no_duplicate_entries(
        df,
        ["nid", "psu", "hh_id"],
        "Multiple entries for the same nid, psu, and hh_id.",
    )

    if (
        df.groupby(["nid", "hh_id", "year_start", "psu", "geospatial_id"]).size().max()
        != 1
    ):
        raise ValueError("Multiple entries for the same nid, psu, hh_id, and year.")
    if df.groupby(["nid", "hh_id", "year_start", "psu"]).size().max() != 1:
        raise ValueError("Multiple entries for the same nid, psu, hh_id, and year.")

    # Load location metadata here (only used within these validations)
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)
    df = merge_left_without_inflating(
        df, loc_meta[["location_id", "ihme_loc_id"]], on="ihme_loc_id"
    )
    if df.location_id.isna().any():
        raise RuntimeError("Null location IDs.")

    df["year_start"] = df["year_start"].astype(int)
    df["nid"] = df["nid"].astype(int)
    return df


def get_DHS_wealth_dataset() -> pd.DataFrame:
    wealth_raw = pd.read_parquet(SURVEY_DATA_PATHS["wealth"]["DHS"])
    df = filter_point_data(wealth_raw)

    # Specific renaming for DHS
    df = df.rename(
        columns={
            "wealth_score": "wealth_index_dhs",
            "iso3": "ihme_loc_id",
            "weight": "hhweight",
        }
    )
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)

    # Use the global column order and remove duplicates
    df = df[WEALTH_DATASET_COMMON_COLUMNS + ["wealth_index_dhs"]].drop_duplicates()

    # update variable data types besides hh_id
    int_cols = [
        "nid",
        "psu",
        "geospatial_id",
    ]
    df[int_cols] = df[int_cols].astype("int")
    # performing hh_id cleaning and data type conversions before returning wealth
    # data, in order to prevent merge issues downstream.
    # Apply cleaning function to each group and update hh_id
    df["old_hh_id"] = df["hh_id"]
    df["hh_id"] = df.groupby(["nid", "ihme_loc_id", "psu"], group_keys=False).apply(
        clean_hh_id_subset
    )

    assert len(df[df["hh_id"].isna()]) == len(
        df[df["old_hh_id"].isna()]
    ), "NAs introduced by cleaning"
    df.drop(columns=["old_hh_id"], inplace=True)

    df["hh_id"] = df["hh_id"].astype(int)
    df["psu"] = df["psu"].astype(int)
    df["hhweight"] = df["hhweight"].astype(int)
    df["wealth_index_dhs"] = df["wealth_index_dhs"].astype(float) # prevent precision loss

    df.dropna(subset=["strata"], inplace=True)
    df["strata"] = df["strata"].astype(int)

    # Remove problematic NIDs
    bad_entries = df.groupby(["nid", "hh_id", "year_start", "psu"]).size()
    bad_nids = list(bad_entries[bad_entries > 1].reset_index().nid.unique())
    bad_nids = [*bad_nids, 20315, 20301, 20537]
    df = df[~df.nid.isin(bad_nids)]

    df = common_validations(df)
    return df


def get_MICS_wealth_dataset() -> pd.DataFrame:
    wealth_raw = pd.read_parquet(SURVEY_DATA_PATHS["wealth"]["MICS"])
    df = filter_point_data(wealth_raw)

    # Specific renaming for MICS
    df = df.rename(
        columns={
            "wealth_score": "wealth_index_dhs",
            "iso3": "ihme_loc_id",
        }
    )
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)

    df = df[WEALTH_DATASET_COMMON_COLUMNS + ["wealth_index_dhs"]].drop_duplicates()

    # For NID 7618, use geospatial_id as the hh_id
    df.loc[df.nid == 7618, "hh_id"] = df.loc[df.nid == 7618, "geospatial_id"]
    # Truncate ihme_loc_id to country-level (first 3 characters)
    df["ihme_loc_id"] = df["ihme_loc_id"].str[:3]

    df = common_validations(df)
    return df


def get_LSMS_wealth_dataset() -> pd.DataFrame:
    wealth_raw = pd.read_parquet(SURVEY_DATA_PATHS["wealth"]["LSMS"])
    df = filter_point_data(wealth_raw)

    # Exclude NIDs with no wealth measure available
    wealth_measure_availability = df.groupby("nid").apply(
        lambda x: x[["income", "expenditure", "consumption"]].isna().all().all(),
        include_groups=False,
    )
    nids_without_wealth = wealth_measure_availability[
        wealth_measure_availability
    ].index.astype(int)
    df = df[~df.nid.isin(nids_without_wealth)].reset_index(drop=True)

    # Choose the best available wealth measure, preferring income, then consumption, then expenditure
    wealth_measure = (
        df.groupby("nid")
        .apply(
            lambda x: x[["income", "expenditure", "consumption"]]
            .isna()
            .mean()
            .idxmin(),
            include_groups=False,
        )
        .reset_index()
    )
    wealth_measure.columns = ["nid", "wealth_measure"]
    df = df.merge(wealth_measure, on="nid", how="left")
    df["wealth_measurement"] = df.apply(lambda row: row[row["wealth_measure"]], axis=1)

    # Remove problematic NID with multiple measures
    df = df[~df.nid.isin([283013])].reset_index(drop=True)

    # Use hhweight if available; otherwise, use pweight
    df["hhweight"] = df["hhweight"].combine_first(df["pweight"]).astype(float)

    # Validate that each household has a unique wealth measurement
    grouped = df.groupby(["nid", "hh_id", "psu", "geospatial_id"])
    if not (grouped["wealth_measurement"].nunique().eq(1)).all():
        raise ValueError("Multiple wealth measurements for the same household.")

    df = df.rename(
        columns={
            "iso3": "ihme_loc_id",
        }
    )
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)

    df = df[
        WEALTH_DATASET_COMMON_COLUMNS + ["wealth_measurement", "wealth_measure"]
    ].drop_duplicates()

    # Truncate ihme_loc_id to country-level (first 3 characters)
    df["ihme_loc_id"] = df["ihme_loc_id"].str[:3]

    df = common_validations(df)
    return df


def assign_age_group(df: pd.DataFrame, indicator="cgf") -> pd.DataFrame:
    age_group_spans = pd.read_parquet(paths.AGE_SPANS_FILEPATH)
    if indicator in ["cgf"]:
        age_group_spans = age_group_spans.query(
            "age_group_id in [2,3,388, 389, 238, 34]"
        )
    elif indicator in ["child_mortality"]:
        age_group_spans = age_group_spans.query(
            "age_group_id in [2,3,388, 389, 238, 34]"
        )
    df["age_group_id"] = np.nan
    if (indicator == "child_mortality") & ("age_year" not in df.columns):
        # aod_months should replace age_month for child_alive==0
        df["age_year"] = df["age_month"] / 12  # keep as float
    for _, row in age_group_spans.iterrows():
        df.loc[
            (df.age_year >= row.age_group_years_start.round(5))
            & (df.age_year < row.age_group_years_end),
            "age_group_id",
        ] = row.age_group_id
    for _, row in age_group_spans.iterrows():
        df.loc[
            (df.age_group_id.isna())
            & (df.age_year == 0)
            & (df.age_month / 12 >= row.age_group_years_start)
            & (df.age_month / 12 < row.age_group_years_end),
            "age_group_id",
        ] = row.age_group_id
    # Fix for some kids that had age_year = 0.076 but month == 1
    age_one_month, age_id_one_month = 0.076, 388
    df.loc[
        (df.age_group_id.isna()) & (df.age_year > age_one_month) & (df.age_month == 1),
        "age_group_id",
    ] = age_id_one_month

    age_id_map = {
        2: 2,
        3: 3,
        388: 4,
        389: 4,
        238: 5,
        34: 5,
    }
    df["age_group_id_agg"] = df["age_group_id"].map(age_id_map)
    return df


def assign_lbd_admin2_location_id(
    data: gpd.GeoDataFrame, lat_col: str = "lat", long_col: str = "long"
) -> pd.DataFrame:
    lbd_admin2_metadata_filepath = (
        ClimateMalnutritionData._PROCESSED_DATA_ROOT
        / "ihme"
        / "lbd_admin2.parquet"  # noqa: SLF001
    )
    admin2_shapes = (
        gpd.read_parquet(lbd_admin2_metadata_filepath)[["loc_id", "geometry"]]
        .rename(columns={"loc_id": "lbd_admin2_id"})
        .to_crs("ESRI:54009")
    )
    if "lbd_admin2_id" in data.columns:
        msg = "lbd_admin2_id column already in data."
        raise ValueError(msg)

    cgf_coords = data[[lat_col, long_col]].drop_duplicates()
    cgf_coords = gpd.GeoDataFrame(
        cgf_coords, geometry=gpd.points_from_xy(cgf_coords.long, cgf_coords.lat)
    )
    cgf_coords = cgf_coords.set_crs("WGS84").to_crs("ESRI:54009")

    lencheck = len(cgf_coords)
    cgf_coords = cgf_coords.sjoin_nearest(
        admin2_shapes, how="left", distance_col="distance"
    )
    # There's a point in Colombia that is included in two different admin2s so it gets duplicated; dropping the wrong location for it
    cgf_coords = cgf_coords.query(
        f"not({lat_col} == 10.935365 and {long_col} == -74.764815 and lbd_admin2_id == 55124)"
    )
    if len(cgf_coords) != lencheck:
        msg = "Mismatch in length of data and merged data."
        raise RuntimeError(msg)

    result = merge_left_without_inflating(
        data, cgf_coords[[lat_col, long_col, "lbd_admin2_id"]], on=[lat_col, long_col]
    )
    return result  # type: ignore[no-any-return]


def assign_sdi(df: pd.DataFrame, year_col: str = "year_start") -> pd.DataFrame:
    sdi = xr.open_dataset(SDI_PATH)
    sdi = sdi.mean(dim="draw")
    return df.merge(
        sdi.to_dataframe(),
        left_on=["location_id", year_col],
        right_index=True,
        how="left",
    )


def clean_hh_id(row):
    if pd.isna(row["hh_id"]):
        return row["hh_id"]
    hh_id_str = str(row["hh_id"])
    geo_str = str(row["geospatial_id"])
    # Match one or more leading zeros followed by the geospatial_id at the start
    pattern = r"^0+" + re.escape(geo_str)
    # Remove the matched pattern (if found)
    cleaned = re.sub(pattern, "", hh_id_str)
    # Strip remaining leading zeros and handle empty results
    cleaned = cleaned.lstrip("0") or "0"
    return cleaned


def clean_hh_id_subset(data: pd.DataFrame | pd.Series) -> float:
    """
    Function to clean household IDs (hh_id) for anemia and child mortality data.

    Key confounding examples:
    - geospatial ID 1 and hh_id 1558 needs to become 558, but
    - geospatial ID 1 and hh_id 15 needs to become 15

    Whether or not the leading digit is the geospatial ID is usually obvious by
    underscores or spaces, but if these are not present, that does not mean the
    leading digit is not the geospatial ID. This can still be seen by comparing
    corresponding hh_ids in DHS data that have values such as '1 61', '1137', '1146'.
    Therefore, detecting these must be done using the presences of sequences at
    the group level of hh_ids within each nid, psu, geospatial_id combination.
    Rule: if leading geospatial_id found for any of the hh_id values due to spaces
    or underscores, the leading geospatial_id must be removed for all hh_id values
    in the group.
    """

    clean_hh_id_list = []
    geospatial_id_detected = False

    # first check if any hh_id in the group has a space or underscore
    for i in range(0, len(data["hh_id"])):
        hh_id = data["hh_id"].iloc[i]

        # Convert to string and trim leading/trailing whitespace
        hh_id = str(hh_id).strip()

        # Replace multiple spaces with a single space
        hh_id = re.sub(r"\s{2,}", " ", hh_id)

        # If the hh_id is already clean (no spaces or underscores), return it
        if len(re.split(r"[_ ]", hh_id)) > 1:
            geospatial_id_detected = True
            break

    for i in range(0, len(data["hh_id"])):
        hh_id = data["hh_id"].iloc[i]
        geo_str = str(data["geospatial_id"].iloc[i])

        if pd.isna(hh_id):
            clean_hh_id_list.append(hh_id)
        else:
            # Convert to string and trim leading/trailing whitespace
            hh_id = str(hh_id).strip()

            # Replace multiple spaces with a single space
            hh_id = re.sub(r"\s{2,}", " ", hh_id)

            # Handle cases with spaces or underscores
            if " " in hh_id:
                hh_id = hh_id.split(" ")[-1]
            elif "_" in hh_id:
                hh_id = hh_id.split("_")[-1]
            elif geospatial_id_detected:
                # If any hh_id in the group had a space or underscore, remove leading geospatial_id
                pattern = r"^" + re.escape(geo_str)
                hh_id = re.sub(pattern, "", hh_id)

            # Match and remove leading zeros followed by the geospatial_id
            pattern = r"^0+" + re.escape(geo_str)
            hh_id = re.sub(pattern, "", hh_id)

            # Strip remaining leading zeros and handle empty results
            hh_id = hh_id.lstrip("0") or "0"

            clean_hh_id_list.append(hh_id)

    return pd.Series(clean_hh_id_list, index=data.index)


def concat_valid_extractions(file_path: str) -> pd.DataFrame:
    # Concatenate all valid extraction files into a single DataFrame
    # extraction_files = [f for f in os.listdir(file_path) if f.endswith("_dataset.csv")]
    extraction_files = [f for f in os.listdir(file_path) if f.endswith(".dta")]
    dfs = []
    for f in extraction_files:
        # df = pd.read_csv(os.path.join(file_path, f), low_memory=False)
        df = pd.read_stata(os.path.join(file_path, f))
        df["source_file"] = f  # Keep track of the source file
        # Perform any necessary validation on the DataFrame
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def check_columns(df: pd.DataFrame, module: str) -> pd.DataFrame:
    if module == "dem_br":
        # concatenated data contains 2 versions of hh_id, psu_id and strata_id
        # df.drop(columns="hhid", inplace=True)  # duplicate to hh_id

        # give preverance to psu over psu_id, which is sometimes not integerable.
        # e.g. psu_id= "0_15", psu = 6. psu_id values for which psu is na are
        # also not clear enough to fill for missing values.
        df.drop(columns="psu_id", inplace=True)

        # same with strata and strata_id
        df.drop(columns="strata_id", inplace=True)

        # latitude/longitude is empty, but lat/long is not
        # df.drop(columns="latitude", inplace=True)
        # df.drop(columns="longitude", inplace=True)
    elif module == "dem_vr":
        pass
    return df


def get_age_month_at_year_end(row):
    """
    For child mortality, we want to know the age in months at the end of the
    observation year. If the child died during that year, we return the age at
    death.
    """
    obs_year = row["years_to_expand"]
    birth_year = row["birth_year"]
    birth_month = row["birth_month"]
    age_month = row["age_month"]

    age_months_at_year_end = (obs_year + 1 - birth_year) * 12 - birth_month
    return min(age_month, age_months_at_year_end)

## Main indicator-specific data prep methods:
def run_training_data_prep_cgf(  # noqa: PLR0915
    output_root: str | Path,
    data_source_type: str,
) -> None:
    survey_data_path = SURVEY_DATA_PATHS[data_source_type]
    print(f"Running training data prep for {data_source_type}...")

    print("Processing gbd extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    lsae_cgf_data_raw = pd.read_csv(
        survey_data_path["lsae"],
        dtype={"hh_id": str, "year_start": int, "int_year": int, "year_end": int},
    )
    gbd_cgf_data_raw = pd.read_csv(
        survey_data_path["gbd"],
        dtype={"hh_id": str, "year_start": int, "int_year": int, "year_end": int},
    )

    # Start with the GBD CGF data
    gbd_cgf_data = gbd_cgf_data_raw[
        [
            "nid",
            "ihme_loc_id",
            "year_start",
            "year_end",
            "geospatial_id",
            "psu_id",
            "pweight",
            "strata_id",
            "hh_id",
            "urban",
            "sex_id",
            "age_year",
            "age_month",
            "int_year",
            "int_month",
            "HAZ_b2",
            "WHZ_b2",
            "WAZ_b2",
            "wealth_index_dhs",
            "latnum",
            "longnum",
        ]
    ]
    gbd_cgf_data = gbd_cgf_data.rename(columns=COLUMN_NAME_TRANSLATOR)

    # Take away NIDs without wealth information, to see later if we can add it from second source
    print(len(gbd_cgf_data))
    gbd_nids_without_wealth = gbd_cgf_data[
        gbd_cgf_data.wealth_index_dhs.isna()
    ].nid.unique()

    lsms_wealth_data = get_LSMS_wealth_dataset()
    dhs_wealth_data = get_DHS_wealth_dataset()
    mics_wealth_data = get_MICS_wealth_dataset()

    wealth_nids = set(dhs_wealth_data.nid.unique()) | set(mics_wealth_data.nid.unique()) | set(lsms_wealth_data.nid.unique())
    cgf_nids = set(lsae_cgf_data_raw.nid.unique()) | set(gbd_cgf_data_raw.nid.unique())
    common_nids = wealth_nids.intersection(cgf_nids)

    # Subset to common nids
    lsms_wealth_data = lsms_wealth_data.query('nid in @common_nids')
    dhs_wealth_data = dhs_wealth_data.query('nid in @common_nids')
    mics_wealth_data = mics_wealth_data.query('nid in @common_nids')

    # All of GBD's come from DHS. For GBD, we prefer the wealth data from the wealth team,
    # so subset to the nids in the DHS wealth data
    gbd_cgf_data_to_match = gbd_cgf_data.query('nid in @dhs_wealth_data.nid')
    #gbd_cgf_data_own_wealth = gbd_cgf_data.query('nid not in @dhs_wealth_data.nid and nid not in @nids_without_wealth')
    gbd_cgf_data_own_wealth = gbd_cgf_data.query('nid not in @gbd_nids_without_wealth')

    # If there are any NIDs that aren't in the wealth data but don't have their own wealth, then they're missing wealth altogether
    nids_without_wealth_in_any_dataset = set(gbd_cgf_data.nid.unique()) - set(gbd_cgf_data_to_match.nid.unique()) - set(gbd_cgf_data_own_wealth.nid.unique())
    print("NIDs without wealth in any dataset:", nids_without_wealth_in_any_dataset)

    # for the lsae cgf nids, subset ot the nids in each of the wealth datasets
    lsae_cgf_data_mics = lsae_cgf_data_raw.query('nid in @mics_wealth_data.nid')
    lsae_cgf_data_lsms = lsae_cgf_data_raw.query('nid in @lsms_wealth_data.nid')
    lsae_cgf_data_dhs = lsae_cgf_data_raw.query('nid in @dhs_wealth_data.nid')

    # First the GBD CGF data with its own wealth data
    # We get wealth data to be merged with GBD CGF data and merge
    print("Processing GBD CGF data with its own wealth data...")
    gbd_data_wealth_distribution = (
        gbd_cgf_data_own_wealth.groupby(["nid", "ihme_loc_id", "year_start", "psu", "hh_id"])
        .agg(
            wealth_index_dhs=("wealth_index_dhs", "first"),
            pweight=("pweight", "first"),
            check=("wealth_index_dhs", "nunique"),
        )
        .reset_index()
    )

    if (gbd_data_wealth_distribution.check != 1).any():
        msg = "Multiple wealth index values for the same household."
        raise RuntimeError(msg)

    gbd_data_wealth_distribution = merge_left_without_inflating(gbd_data_wealth_distribution, loc_meta, on="ihme_loc_id")

    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / MEASURES_IN_SOURCE[data_source_type][0])

    gbd_data_wealth_distribution = get_ldipc_from_asset_score(
        gbd_data_wealth_distribution, cm_data, asset_score_col="wealth_index_dhs",
        plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "gbd_plots.pdf",
        ldi_version = LDI_VERSION,
    )
    ldi_cols = ['ldipc_unweighted_no_match', 'ldipc_weighted_no_match', 'ldipc_unweighted_match', 'ldipc_weighted_match']

    gbd_data_wealth_distribution = gbd_data_wealth_distribution[
        ["nid", "ihme_loc_id", "location_id", "year_start", "psu", "hh_id"] + ldi_cols
    ]

    gbd_cgf_data_own_wealth = merge_left_without_inflating(gbd_cgf_data_own_wealth, gbd_data_wealth_distribution, on=["nid", "ihme_loc_id", "year_start", "psu", "hh_id"])

    # Getting income distributions
    dhs_wealth_data = get_ldipc_from_asset_score(
        dhs_wealth_data, cm_data, asset_score_col="wealth_index_dhs", weights_col="hhweight",
        plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "dhs_plots.pdf",
        ldi_version = LDI_VERSION,
    )

    mics_wealth_data = get_ldipc_from_asset_score(
        mics_wealth_data, cm_data, asset_score_col="wealth_index_dhs", weights_col="hhweight",
        plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "mics_plots.pdf",
        ldi_version = LDI_VERSION,
    )

    lsms_wealth_data = get_ldipc_from_asset_score(
        lsms_wealth_data, cm_data, asset_score_col="wealth_measurement", weights_col="hhweight",
        plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "lsms_plots.pdf",
        ldi_version = LDI_VERSION,
    )

    wealth_cols = ['ihme_loc_id', 'location_id', 'nid', 'psu', 'hh_id', 'geospatial_id', 'lat', 'long', 'year_start'] + ldi_cols
    all_wealth_data = pd.concat([dhs_wealth_data[wealth_cols], mics_wealth_data[wealth_cols], lsms_wealth_data[wealth_cols]])
    wealth_lsae_df = all_wealth_data.copy()


    # Now the LSAE CGF data
    lsae_cgf_data = lsae_cgf_data_raw[
        [
            "nid",
            "country",
            "year_start",
            "end_year",
            "geospatial_id",
            "psu",
            "pweight",
            #"strata",
            "hh_id",
            "sex",
            "age_year",
            "age_mo",
            "int_year",
            "int_month",
            "stunting_mod_b",
            "wasting_mod_b",
            "underweight_mod_b",
        ]
    ]
    lsae_cgf_data = lsae_cgf_data.rename(columns=COLUMN_NAME_TRANSLATOR)

    print("Processing LSAE data...")
    # Take away bad NIDs, without household information
    print(len(lsae_cgf_data))
    no_hhid_nids = (
        lsae_cgf_data.groupby("nid")
        .filter(lambda x: x["hh_id"].isna().all())
        .nid.unique()
    )
    lsae_cgf_data = lsae_cgf_data[~lsae_cgf_data.nid.isin(no_hhid_nids)]
    #TODO print these NIDS to a file

    # Try to make household id usable to merge on
    # For some reason in some extractions hh_id is a string with household id and psu, in others it's just the household id
    lsae_cgf_data = lsae_cgf_data[lsae_cgf_data["nid"].isin(common_nids)].copy()
    lsae_cgf_data = lsae_cgf_data.rename(
        columns={"latitude": "lat", "longitude": "long", "hh_id": "old_hh_id"}
    )
    wealth_lsae_df = wealth_lsae_df.rename(columns={"hh_id": "old_hh_id"})
    lsae_cgf_data["hh_id"] = lsae_cgf_data["old_hh_id"].str.split(r"[_ ]").str[-1]
    wealth_lsae_df["hh_id"] = wealth_lsae_df["old_hh_id"].str.split(r"[_ ]").str[-1]
    lsae_cgf_data["psu"] = lsae_cgf_data["psu"].astype(int)
    lsae_cgf_data["hh_id"] = lsae_cgf_data.apply(clean_hh_id, axis=1)
    wealth_lsae_df["hh_id"] = wealth_lsae_df.apply(clean_hh_id, axis=1)
    print(len(lsae_cgf_data))

    # Some NIDs need extra cleaning so that hh_id can be merged.
    # Take those out and merge LSAE CGF data with wealth
    print(len(lsae_cgf_data))

    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]
    # maybe_fixable_df.loc[maybe_fixable_df.sex_id == 0, "sex_id"] = 2

    lsae_merged = merge_left_without_inflating(lsae_cgf_data.drop(columns=["old_hh_id"]), wealth_lsae_df, on=merge_cols)
    print(len(lsae_cgf_data))
    print(len(lsae_merged))

    # Take out NIDs with more than 5% of missing wealth data
    allowed_wealth_nan_proportion = 0.05
    nan_proportion = lsae_merged.groupby('nid').apply(lambda x: x.ldipc_unweighted_no_match.isna().mean(), include_groups=False).reset_index().rename(columns={0: 'nan_proportion'})
    bad_wealth_merge_nids = nan_proportion.query('nan_proportion > @allowed_wealth_nan_proportion').nid.unique()
    print(len(lsae_merged))
    lsae_merged = lsae_merged[~lsae_merged.nid.isin(bad_wealth_merge_nids)]
    print(len(lsae_merged))
    # Take out rows with missing wealth data but in NIDs with less than 5% of missing wealth data
    lsae_merged = lsae_merged[~lsae_merged.ldipc_unweighted_no_match.isna()]
    print(len(lsae_merged))

    maybe_fixable_df = lsae_cgf_data[
        lsae_cgf_data["nid"].isin(bad_wealth_merge_nids)#([157057, 286780, 341838])
    ].copy()

    # Only interested in rows have have either stunting, wasting or underweight information, with both wealth and location
    print(len(lsae_merged))
    lsae_merged = lsae_merged.dropna(
        subset=["stunting", "wasting", "underweight"], how="all"
    )
    print(len(lsae_merged))
    lsae_merged = lsae_merged.dropna(subset=["lat", "long"], how="any")
    lsae_merged.loc[lsae_merged.sex_id == 0, "sex_id"] = 2
    print(len(lsae_merged))


    extra_nids = gbd_cgf_data_to_match.copy().drop(columns=['lat', 'long'])

    extra_nids["hh_id"] = extra_nids["hh_id"].str.split(r"[_ ]").str[-1]
    extra_nids["hh_id"] = extra_nids.apply(clean_hh_id, axis=1)
    extra_nids_nids = extra_nids.nid.unique()

    extra_nids_wealth = all_wealth_data.query("nid in @extra_nids_nids").copy()
    extra_nids_wealth["hh_id"] = extra_nids_wealth["hh_id"].str.split(r"[_ ]").str[-1]
    extra_nids_wealth["hh_id"] = extra_nids_wealth.apply(clean_hh_id, axis=1)
    extra_nids = merge_left_without_inflating(extra_nids, extra_nids_wealth, on=["nid", "ihme_loc_id", "hh_id", "psu", "year_start"])
    print(len(extra_nids))

    # Take out NIDs with more than 5% of missing wealth data
    allowed_wealth_nan_proportion = 0.05
    nan_proportion = extra_nids.groupby('nid').apply(lambda x: x.ldipc_unweighted_no_match.isna().mean(), include_groups=False).reset_index().rename(columns={0: 'nan_proportion'})
    bad_wealth_merge_nids_extra = nan_proportion.query('nan_proportion > @allowed_wealth_nan_proportion').nid.unique()
    print(len(extra_nids))
    extra_nids = extra_nids[~extra_nids.nid.isin(bad_wealth_merge_nids_extra)]
    print(len(extra_nids))
    # Take out rows with missing wealth data but in NIDs with less than 5% of missing wealth data
    extra_nids = extra_nids[~extra_nids.ldipc_unweighted_no_match.isna()]
    print(len(extra_nids))


    # Bring the two datasets (LSAE and GBD) together, giving preference to the LSAE extractions
    gbd_extraction_nids = set(gbd_cgf_data_own_wealth.nid.unique()) | set(extra_nids.nid.unique())
    lsae_only = lsae_merged.loc[~lsae_merged.nid.isin(gbd_extraction_nids)]
    gbd_only = pd.concat([extra_nids.loc[~extra_nids.nid.isin(lsae_merged.nid.unique())],
        gbd_cgf_data_own_wealth.loc[~gbd_cgf_data_own_wealth.nid.isin(lsae_merged.nid.unique())]
    ])


    # Merge the two datasets and the NIDs that needed wealth information
    cgf_consolidated = pd.concat(
        [lsae_merged, gbd_only], ignore_index=True
    ).reset_index(drop=True)

    cgf_consolidated = cgf_consolidated.drop(columns=["strata", "geospatial_id"])

    # Assign age group
    cgf_consolidated = assign_age_group(cgf_consolidated, )
    cgf_consolidated = cgf_consolidated.dropna(subset=["age_group_id"])

    # Take out data with invalid lat and long
    cgf_consolidated = cgf_consolidated.dropna(subset=["lat", "long"])
    cgf_consolidated = cgf_consolidated.query("lat != 0 and long != 0")

    # NID 275090 is a very long survey in Peru, 2003-2008 that is coded as having
    # multiple year_starts. Removing it
    cgf_consolidated = cgf_consolidated.query("nid != 275090")

    # NID 411301 is a Zambia survey in which the prevalences end up being 0
    # after removing data with invalid age columns, remove it
    cgf_consolidated = cgf_consolidated.query("nid != 411301")

    # Get the month of the interview, and then calculate the month of pregnancy for each row, to merge with monthly climate data
    cgf_consolidated['computed_age_month'] = (cgf_consolidated.age_year * 12).round().astype(int)
    cgf_consolidated.loc[cgf_consolidated.computed_age_month == 0, 'computed_age_month'] = cgf_consolidated.loc[cgf_consolidated.computed_age_month == 0, 'age_month'].round().astype(int)
    cgf_consolidated['computed_age_month_preg'] = cgf_consolidated['computed_age_month'] + 9

    # cases:
    # age_year and age_month are concordant, age year includes fraction, age month is the total number of months
    # age_year has years, age_month has 0
    # age_year has 0, age_month has months
    cgf_consolidated.loc[(cgf_consolidated.nid == 425283) & (cgf_consolidated.int_month.isna()), 'int_month'] = 1
    cgf_consolidated = cgf_consolidated.reset_index(drop=True)
    cgf_consolidated['row_id'] = cgf_consolidated.index

    nids_with_missing_int_month = cgf_consolidated[cgf_consolidated['int_month'].isna()].nid.unique()
    for nid in nids_with_missing_int_month:
        print(cgf_consolidated.query("nid == @nid and int_month.notnull()").int_month.mode())
        mode_month = cgf_consolidated.query("nid == @nid and int_month.notnull()").int_month.mode()[0]
        cgf_consolidated.loc[(cgf_consolidated['nid'] == nid) & (cgf_consolidated['int_month'].isna()), 'int_month'] = mode_month

    for col in ['int_year', 'year_start', 'year_end', 'int_month']:
        cgf_consolidated[col] = cgf_consolidated[col].astype(int)

    # Calculate the month and year of the interview for each row, 
    # and then explode the dataframe so that there's one row per month of pregnancy for each child, 
    # to merge with monthly climate data
    df = cgf_consolidated[['row_id', 'int_year', 'int_month', 'computed_age_month_preg', 'lat', 'long']].drop_duplicates().reset_index(drop=True)

    df_exploded = df.loc[df.index.repeat(df['computed_age_month_preg'])]

    # Helper column to calculate the month of the interview for each row after exploding
    df_exploded['i'] = df_exploded.groupby(df_exploded.index).cumcount()

    total_months_calc = (
        df_exploded['int_year'] * 12 + 
        df_exploded['int_month'] - 
        df_exploded['i'] - 1
    )

    df_exploded['climate_year'] = total_months_calc // 12
    df_exploded['climate_month'] = 1 + (total_months_calc % 12)
    df_exploded = df_exploded.drop(columns=['i'])

    unique_year_month_loc = df_exploded[['climate_year', 'climate_month', 'lat', 'long']].drop_duplicates().reset_index(drop=True)
    year_col = 'climate_year'

    climate_variables =['days_over_24C', 'days_over_25C', 'days_over_26C', 'days_over_27C', 'days_over_28C', 
        'days_over_29C', 'days_over_30C', 'days_over_31C', 'days_over_32C', 'mean_temperature', 
        'total_precipitation', 'days_over_relative_threshold']
    n_threads = 25
    df_splits =  [year_df for _, year_df in unique_year_month_loc.groupby(year_col)]

    with mp.Pool(n_threads) as pool:
        climate_month_df = pd.concat(list(tqdm.tqdm(pool.imap_unordered(partial(get_climate_vars_for_year_month, climate_variables=climate_variables), df_splits), total=len(df_splits))))

    results_df = df_exploded.merge(
        climate_month_df,
        on=['climate_year', 'climate_month', 'lat', 'long'],
        how='left',
        validate='many_to_one'
    )

    print("Adding elevation data...")
    cgf_consolidated = get_elevation_for_dataframe(cgf_consolidated)

    cgf_consolidated = assign_lbd_admin2_location_id(cgf_consolidated)

    relative_thresh_vars = [x for x in results_df.columns if x.startswith('days_over_') and x.endswith('quant')]
    vars_to_agg = [x for x in climate_variables if x != 'days_over_relative_threshold'] + relative_thresh_vars
    climate_month_df_agg = results_df.groupby('row_id')[vars_to_agg].mean().reset_index()
    climate_month_df_agg = climate_month_df_agg.rename(columns={col: f"{col}_month" for col in vars_to_agg})
    merged_df = cgf_consolidated.merge(
        climate_month_df_agg,
        on='row_id',
        how='left',
        validate='one_to_one'
    )
    cgf_consolidated = merged_df


    #Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = cgf_consolidated[cgf_consolidated[measure].notna()].copy().drop(columns=[x for x in cgf_consolidated.columns if x.endswith('_x') or x.endswith('_y')])
        measure_df["cgf_measure"] = measure
        measure_df["cgf_value"] = measure_df[measure]
        measure_root = Path(output_root) / measure
        cm_data = ClimateMalnutritionData(measure_root)
        print(f"Saving data for {measure} to {measure_root} {len(measure_df)} rows")
        for ldi_col in ['ldipc_weighted_no_match']: #ldi_cols:
            measure_df['ldi_pc_pd'] = measure_df[ldi_col] / 365
            version = cm_data.new_training_version()
            print(f"Saving data for {measure} to version {version} with {ldi_col} as LDI")
            cm_data.save_training_data(measure_df, version)
            message = "Used " + ldi_col + " as LDI"
            # Save a small file with a record of which ldi column was used for this version
            with open(cm_data.training_data / version / "ldi_col.txt", "w") as f:
                f.write(message)


def run_training_data_prep_anemia(
    output_root: str | Path,
    data_source_type: str,
) -> None:

    # Set up logging and versioned output path
    measure_root = Path(output_root) / data_source_type
    os.makedirs(Path(measure_root) / "training_data", exist_ok=True, mode=0o777)
    cm_data = ClimateMalnutritionData(measure_root)
    version = cm_data.new_training_version()
    output_path_version = Path(measure_root) / "training_data" / version
    os.makedirs(
        output_path_version,
        exist_ok=True,
        mode=0o777,
    )

    # Set up logging
    dataprep_log_path = Path(output_path_version) / "data_prep_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(dataprep_log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    survey_data_path = SURVEY_DATA_PATHS[data_source_type]
    logging.info(f"Running training data prep for {data_source_type}...")

    logging.info(f"Creating new version stored under version: {version}")

    logging.info("Processing extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    anemia_data_raw = pd.read_csv(survey_data_path, low_memory=False)

    anemia_data = anemia_data_raw[
        [
            "nid",
            "ihme_loc_id",
            "survey_name",
            "year_start",
            "year_end",
            "urban",
            "geospatial_id",
            "psu_id",
            "pweight",
            "strata",
            "hh_id",
            "line_id",
            "sex_id",
            "age_year",
            "age_month",
            "int_year",
            "anemia_anemic_brinda",
            "anemia_mod_sev_brinda",
            "latnum",
            "longnum",
        ]
    ]

    # Using anemia_anemic_brinda as outcome variable and renaming to'anemia'
    anemia_data = anemia_data.rename(columns={"anemia_anemic_brinda": "anemia"})

    anemia_data = anemia_data.rename(columns=COLUMN_NAME_TRANSLATOR)

    # Apply cleaning function to each group and update hh_id
    anemia_data["old_hh_id"] = anemia_data["hh_id"]
    anemia_data["hh_id"] = anemia_data.groupby(
        ["nid", "ihme_loc_id", "psu"], group_keys=False
    ).apply(clean_hh_id_subset)

    assert len(anemia_data[anemia_data["hh_id"].isna()]) == len(
        anemia_data[anemia_data["old_hh_id"].isna()]
    ), "NAs introduced by cleaning"

    # Prepping wealth dataset
    dhs_wealth_data_raw = get_DHS_wealth_dataset()
    dhs_wealth_data = dhs_wealth_data_raw.copy()

    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "anemia")
    dhs_wealth_data = get_ldipc_from_asset_score(
        dhs_wealth_data,
        cm_data,
        asset_score_col="wealth_index_dhs",
        weights_col="hhweight",
        ldi_version=LDI_VERSION,
    )

    missing_hh_rows = anemia_data[anemia_data["hh_id"].isna()]
    logging.info(
        f"Dropping {len(missing_hh_rows)} rows from anemia data with missing hh_id"
    )
    anemia_data = anemia_data[anemia_data["hh_id"].notna()]
    anemia_data["hh_id"] = anemia_data["hh_id"].astype(int)

    # Find out percent of anemia nids and hh_ids that can be matched in wealth data
    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]

    dhs_wealth_data["hh_id"] = dhs_wealth_data["hh_id"].astype(int)
    anemia_data["psu"] = anemia_data["psu"].astype(int)
    dhs_wealth_data["psu"] = dhs_wealth_data["psu"].astype(int)

    wealth_nids = set(dhs_wealth_data.nid.unique())
    anemia_nids = set(anemia_data.nid.unique())
    common_nids = wealth_nids.intersection(anemia_nids)

    nid_with_wealth_pc = 100 * len(common_nids) / len(anemia_nids)
    logging.info(
        f"{nid_with_wealth_pc:.1f}% of anemia NIDs - {len(common_nids)} out of "
        f"{len(anemia_nids)} in wealth data NIDs"
    )

    dhs_wealth_data = dhs_wealth_data.query("nid in @anemia_nids")

    anemia_data.drop(columns=["old_hh_id"], inplace=True)

    # Merge data
    anemia_data_wealth = merge_left_without_inflating(
        anemia_data,
        dhs_wealth_data.drop(columns=["geospatial_id", "strata", "lat", "long"]),
        on=merge_cols,
    )

    # Calculate proportion of NA and filter out nids with too much wealth missingness (bad merges)
    merged_na_props = (
        anemia_data_wealth.groupby(["nid"]).ldipc_weighted_no_match.count()
        / anemia_data_wealth.groupby(["nid"]).ldipc_weighted_no_match.size()
    )
    merged_nids = merged_na_props[merged_na_props > 0.95].index.to_list()
    anemia_df = anemia_data_wealth.query("nid in @merged_nids").copy()
    dropped_too_missingness = len(anemia_data_wealth) - len(anemia_df)
    logging.info(
        f"Dropped {dropped_too_missingness:,} rows from {len(anemia_data_wealth):,} due to excessive missingness"
    )

    # drop other unmerged
    unmergable_rows = anemia_df[anemia_df["wealth_index_dhs"].isna()]
    anemia_df = anemia_df[anemia_df["wealth_index_dhs"].notna()]

    # Assign age group
    before_rows = len(anemia_df)
    anemia_df = assign_age_group(anemia_df, indicator="anemia")
    anemia_df = anemia_df.dropna(subset=["age_group_id"])
    dropped_due_to_age = before_rows - len(anemia_df)
    logging.info(f"Dropped {dropped_due_to_age:,} rows due to age groups not found")

    # Take out data with invalid lat and long
    before_rows = len(anemia_df)
    anemia_df = anemia_df.dropna(subset=["lat", "long"])
    anemia_df = anemia_df.query("lat != 0 and long != 0")
    dropped_due_to_coords = before_rows - len(anemia_df)
    logging.info(
        f"Dropped {dropped_due_to_coords:,} rows due to invalid lat and long values"
    )

    # NID 275090 is a very long survey in Peru, 2003-2008 that is coded as having
    # multiple year_starts. Removing it.
    # NID 411301 - update: not problematic for anemia
    problematic_nids = [275090]
    before_rows = len(anemia_df)
    anemia_df = anemia_df.query("nid not in @problematic_nids")
    dropped_problematic_nids = before_rows - len(anemia_df)
    logging.info(f"Dropped {dropped_problematic_nids:,} rows due to problematic NIDs")

    # missing outcome variables
    measure_columns = MEASURES_IN_SOURCE[data_source_type]
    rows_with_na_outcomes = anemia_df[measure_columns].isna().any(axis=1).sum()
    rows_with_na_outcomes = int(rows_with_na_outcomes)

    full_data_rows = len(anemia_df) - rows_with_na_outcomes
    logging.info(
        f"Dropped {rows_with_na_outcomes:,} rows with missing outcome variables"
    )
    logging.info(f"Data contains {full_data_rows:,} rows after cleaning")

    anemia_df = anemia_df.dropna(subset=measure_columns)
    # update variable data types
    int_cols = [
        "anemia",
        "anemia_mod_sev_brinda",
        "age_group_id",
    ]
    anemia_df[int_cols] = anemia_df[int_cols].astype("int")

    # Merge with climate data
    logging.info("Processing climate data...")
    climate_vars = get_climate_vars_for_dataframe(anemia_df)
    anemia_df = merge_left_without_inflating(
        anemia_df, climate_vars, on=["int_year", "lat", "long"]
    )

    logging.info("Adding elevation data...")
    anemia_df = get_elevation_for_dataframe(anemia_df)

    anemia_df = assign_lbd_admin2_location_id(anemia_df)

    # Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = anemia_df[anemia_df[measure].notna()].copy()
        measure_df["measure"] = measure
        measure_df["value"] = measure_df[measure]
        logging.info(
            f"Saving data for {measure} to {output_path_version} {len(measure_df)} rows"
        )
        for ldi_col in ["ldipc_weighted_no_match"]:  # ldi_cols:
            measure_df["ldi_pc_pd"] = measure_df[ldi_col] / 365
            logging.info(
                f"Saving data for {measure} to version {version} with {ldi_col} as LDI"
            )
            cm_data.save_training_data(measure_df, version)
            message = "Used " + ldi_col + " as LDI"
            # Save a small file with a record of which ldi column was used for this version
            with open(cm_data.training_data / version / "ldi_col.txt", "w") as f:
                f.write(message)


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
    plt.figure(figsize=(50, 5))
    ax = agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False)
    plt.title("Unique Individuals per Age Month in Raw Data")
    months = agg_age["age_month"].values
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.tight_layout()


def run_training_data_prep_child_mortality(
    output_root: str | Path, data_source_type: str, module: str
) -> None:
    # TODO: Update TOCs
    # India = location_id 163, ihme_loc_id IND

    """
    Overall structure of child_mortality data prep:
    1. Load and format child_mortality data from DEM_BR module
    2. Extracting and merging wealth dataset
    3. Extract and merging annual climate variables
    4. Extract monthly climate variables for previous months
    5. Merge in monthly climate variables for previous months
    6. Calculate averages over time periods analyzed for monthly climate variables
    7. Extract monthly relative climate thresholds for previous months
    8. Merge monthly relative climate thresholds with child_mortality data
    9. Calculate averages over time periods analyzed for monthly relative climate thresholds
    10. Add in temperature zones
    """

    ## 1. Load and format child_mortality data from DEM_BR module

    # Set up logging and versioned output path
    measure_root = Path(output_root) / data_source_type
    os.makedirs(Path(measure_root) / "training_data", exist_ok=True, mode=0o777)
    cm_data = ClimateMalnutritionData(measure_root)
    version = cm_data.new_training_version()
    output_path_version = Path(measure_root) / "training_data" / version
    os.makedirs(
        output_path_version,
        exist_ok=True,
        mode=0o777,
    )

    dataprep_log_path = Path(output_path_version) / "data_prep_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(dataprep_log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    survey_data_path = SURVEY_DATA_PATHS[data_source_type][module]
    logging.info(f"Running training data prep for {data_source_type}...")

    logging.info(f"Creating new version stored under version: {version}")

    logging.info("Processing extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    # data_raw = concat_valid_extractions(survey_data_path)
    # data_raw = pd.read_csv(
    #     survey_data_path / "dem_br_matched_latlong.csv", encoding="latin1"
    # )
    data_raw = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_matched_2025_10_14.parquet"
    )

    logging.info(f"Total rows in concatenated raw data: {len(data_raw):,}")
    logging.info(
        f"Total unique NIDs in concatenated raw data: {data_raw['nid'].nunique():,}"
    )

    df = data_raw.copy()
    df = check_columns(df, module)

    # drop rows with missing key variables
    # Special note on India. India surveys contain age_month, which is the difference
    # between time of birth and time of interview in months. This is important because
    # int_year and int_month are missing. Calculate the variables using available
    # data as much as possible to avoid unnecessary data loss.

    # Attemp 1: int_year recovery via birth_year/month and age_month
    def recover_int_year(row):
        try:
            birth_mo = int(row["birth_month"])
            birth_yr = int(row["birth_year"])
            age_mo = int(row["age_month"])
            if pd.notna(birth_mo) and pd.notna(birth_yr) and pd.notna(age_mo):
                int_yr = birth_yr + (age_mo // 12)
                int_mo = (birth_mo + (age_mo % 12)) % 12
                return int_yr
            else:
                return row["int_year"]
        except:
            return row["int_year"]

    def recover_int_mo(row):
        try:
            birth_mo = int(row["birth_month"])
            birth_yr = int(row["birth_year"])
            age_mo = int(row["age_month"])
            if pd.notna(birth_mo) and pd.notna(birth_yr) and pd.notna(age_mo):
                int_yr = birth_yr + (age_mo // 12)
                int_mo = (birth_mo + (age_mo % 12)) % 12
                return int_mo
            else:
                return row["int_month"]
        except:
            return row["int_month"]

    df["int_year_recovered"] = df.apply(recover_int_year, axis=1)
    df["int_month_recovered"] = df.apply(recover_int_mo, axis=1)

    df["int_year"].fillna(df["int_year_recovered"], inplace=True)
    df["int_month"].fillna(df["int_month_recovered"], inplace=True)

    # Are there any rows with nonempty int_year/month and age-Month but no birth_month?
    len(
        df[
            (df["int_year"].notna())
            & (df["int_month"].notna())
            & (df["age_month"].notna())
            & (df["birth_month"].isna())
        ]
    )  # 207270

    def recover_birth_year(row):
        try:
            int_mo = int(row["int_month"])
            int_yr = int(row["int_year"])
            age_mo = int(row["age_month"])
            if pd.notna(int_mo) and pd.notna(int_yr) and pd.notna(age_mo):
                birth_yr = int_yr - (age_mo // 12)
                return birth_yr
            else:
                return row["birth_year"]
        except:
            return row["birth_year"]

    def recover_birth_mo(row):
        try:
            int_mo = int(row["int_month"])
            int_yr = int(row["int_year"])
            age_mo = int(row["age_month"])
            if pd.notna(int_mo) and pd.notna(int_yr) and pd.notna(age_mo):
                birth_mo = (int_mo - (age_mo % 12)) % 12
                return birth_mo
            else:
                return row["birth_month"]
        except:
            return row["birth_month"]

    df["birth_year_recovered"] = df.apply(recover_birth_year, axis=1)
    df["birth_month_recovered"] = df.apply(recover_birth_mo, axis=1)
    df["birth_year"].fillna(df["birth_year_recovered"], inplace=True)
    df["birth_month"].fillna(df["birth_month_recovered"], inplace=True)

    df.drop(
        columns=[
            "int_year_recovered",
            "int_month_recovered",
            "birth_year_recovered",
            "birth_month_recovered",
        ],
        inplace=True,
    )

    rows_before_na_drop = len(df)
    key_vars = [
        "nid",
        "psu",
        "birth_year",
        "birth_month",
        "int_year",
        "int_month",
        "age_month",
        "hh_id",
        "geospatial_id",
        "line_id",
        "lat",
        "long",
        "child_alive",
    ]
    df_tmp = df.dropna(subset=key_vars)
    na_rows_dropped = rows_before_na_drop - len(df_tmp)
    logging.info(
        f"Dropped {na_rows_dropped:,} rows with missing key variables: {key_vars}"
    )
    df = df_tmp.copy()

    for var in key_vars:
        logging.info(f"- {var}: {data_raw[var].isna().sum():,} missing values")
    logging.info(f"NIDs with incomplete (some NAs) key variables: {key_vars}")
    for var in key_vars:
        logging.info(
            f"-{data_raw[data_raw[var].isna()]["nid"].nunique():,} contains missing {var} values"
        )

    logging.info(f"Total unique NIDs after dropped NA values: {df['nid'].nunique():,}")
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)

    # update variable data types besides hh_id
    int_cols = [
        "nid",
        "psu",
        "birth_year",
        "birth_month",
        "int_year",
        "int_month",
        "age_month",
        "child_alive",
        "geospatial_id",
    ]
    df[int_cols] = df[int_cols].astype("int")

    # Apply cleaning function to each group and update hh_id
    df["old_hh_id"] = df["hh_id"]
    df["hh_id"] = df.groupby(["nid", "ihme_loc_id", "psu"], group_keys=False).apply(
        clean_hh_id_subset
    )

    assert len(df[df["hh_id"].isna()]) == len(
        df[df["old_hh_id"].isna()]
    ), "NAs introduced by cleaning"

    df["hh_id"] = df["hh_id"].astype("int")
    df.drop(columns=["old_hh_id"], inplace=True)

    # Prepping wealth dataset
    dhs_wealth_data_raw = get_DHS_wealth_dataset()
    dhs_wealth_data = dhs_wealth_data_raw.copy()

    # Find out percent of anemia nids and hh_ids that can be matched in wealth data
    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]

    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "child_mortality")
    dhs_wealth_data_test = get_ldipc_from_asset_score(
        dhs_wealth_data,
        cm_data,
        asset_score_col="wealth_index_dhs",
        weights_col="hhweight",
        # plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "dhs_plots.pdf",
        ldi_version=LDI_VERSION,
    )

    wealth_nids = set(dhs_wealth_data_test.nid.unique())
    df_nids = set(df.nid.unique())
    common_nids = wealth_nids.intersection(df_nids)

    nid_with_wealth_pc = 100 * len(common_nids) / len(df_nids)
    logging.info(
        f"{nid_with_wealth_pc:.1f}% of df NIDs - {len(common_nids)} out of "
        f"{len(df_nids)} in wealth data NIDs"
    )

    dhs_wealth_data_test = dhs_wealth_data_test.query("nid in @df_nids")

    # Merge data

    # fix df columns
    df.rename(columns={"iso3": "ihme_loc_id"}, inplace=True)
    df["ihme_loc_id"] = df["ihme_loc_id"].str.replace("KEN_.*", "KEN", regex=True)
    df[["year_start", "year_end", "int_year"]] = df[
        ["year_start", "year_end", "int_year"]
    ].astype(int)

    df_wealth = merge_left_without_inflating(
        df,
        dhs_wealth_data_test.drop(
            columns=["geospatial_id", "strata", "lat", "long", "hhweight"]
        ),
        on=merge_cols,
    )

    merged_percent = len(df_wealth[~df_wealth["ldipc_weighted_no_match"].isna()]) / len(
        df_wealth
    )
    print(f"Merged rows percent: {100*merged_percent:.1f}%")

    # Calculate proportion of NA and filter out nids with too much wealth missingness (bad merges)
    merged_na_props = (
        df_wealth.groupby(["nid"]).ldipc_weighted_no_match.count()
        / df_wealth.groupby(["nid"]).ldipc_weighted_no_match.size()
    )
    merged_nids = merged_na_props[merged_na_props > 0.95].index.to_list()
    df_merged = df_wealth.query("nid in @merged_nids").copy()
    dropped_too_missingness = len(df_wealth) - len(df_merged)
    logging.info(
        f"Dropped {dropped_too_missingness:,} rows from {len(df_wealth):,} due to excessive wealth missingness in NIDs"
    )
    logging.info(f"Total unique NIDs remaining: {df_merged['nid'].nunique():,}")

    # Include difference between int_year and birth_year for sensitivity analysis
    df_merged["int_birth_year_diff_months"] = 12 * (
        df_merged["int_year"] - df_merged["birth_year"]
    ) + (df_merged["int_month"] - df_merged["birth_month"])

    # Assign age group
    before_rows = len(df_merged)

    # replace age_month with aod_months for rows with child_alive==0
    df_merged["age_month_original"] = df_merged["age_month"]  # keep copy of original
    df_merged.loc[df_merged.child_alive == 0, "age_month"] = df_merged.loc[
        df_merged.child_alive == 0, "aod_months"
    ]

    # drop data with no age_month
    before_rows = len(df_merged)
    df_merged = df_merged[df_merged["age_month"].notna()]
    logging.info(
        f"Dropped {before_rows - len(df_merged):,} rows with missing age_month or aod_months"
    )

    # create list of years between birth year and year that the age_month lands on.
    df_merged["age_month"] = df_merged["age_month"].astype(int)
    df_merged["year_of_recorded_age"] = (
        df_merged["birth_year"] * 12 + df_merged["birth_month"] + df_merged["age_month"]
    ) // 12
    df_merged["year_of_recorded_age"] = df_merged["year_of_recorded_age"].astype(int)

    # filter to up to 6 years to expand
    df_merged["years_to_expand"] = df_merged.apply(
        lambda x: [
            y
            for y in range(
                x["birth_year"], min(x["year_of_recorded_age"] + 1, x["birth_year"] + 6)
            )
        ],
        axis=1,
    )

    # check lengths of resulting lists
    df_merged["n_years_to_expand"] = df_merged["years_to_expand"].apply(len)

    # explode data on years_to_expand
    df_exploded = df_merged.explode("years_to_expand")

    df_exploded["age_month_at_year_end"] = df_exploded.apply(
        get_age_month_at_year_end, axis=1
    )

    # For rows with age_month >59, set to 59. These are remainder months after 5 years,
    # but the 5 year cutoff was already implemented above when exploding to max 6 years.
    df_exploded.loc[df_exploded.age_month > 59, "age_month"] = 59
    df_exploded.loc[df_exploded.age_month_at_year_end > 59, "age_month_at_year_end"] = (
        59
    )

    # override age_month
    df_exploded["age_month"] = df_exploded["age_month_at_year_end"]

    logging.info(
        f"Exploded data to {len(df_exploded):,} rows by expanding on years between child birth and either age of death or age at interview"
    )

    df_exploded = assign_age_group(df_exploded, indicator="child_mortality")
    before_dropping_unused_age_groups = len(df_exploded)
    df_exploded = df_exploded.dropna(subset=["age_group_id"])
    dropped_due_to_age = before_dropping_unused_age_groups - len(df_exploded)

    # age_days and aod_days are empty, but we assume that age_month and aod_months
    # are rounded down, such that age_month 0 is not stillborns, but deaths between
    # 0 and 1 month. This is required for a survival modeling approach, for which
    # time to event cannot be 0.
    df_exploded["age_month"] += 1
    df_exploded["age_month_at_year_end"] += 1

    logging.info(
        f"Dropped {dropped_due_to_age:,} rows due to age groups not found among 2, 3, 388, 389, 238, 34"
    )
    logging.info(f"Total unique NIDs remaining: {df_exploded['nid'].nunique():,}")

    df_exploded["age_group_id"] = df_exploded["age_group_id"].astype(int)
    df_exploded["age_group_id_agg"] = df_exploded["age_group_id_agg"].astype(int)
    df_exploded["age_year"] = df_exploded["age_year"].astype(int)

    df_exploded["age_year_at_year_end"] = df_exploded["age_month_at_year_end"] / 12
    rows_before = len(df_exploded)
    df_exploded = df_exploded[
        df_exploded["age_month_at_year_end"] > 0
    ]  # nothing to model at time=0
    logging.info(
        f"Dropped {rows_before - len(df_exploded):,} rows with age_month_at_year_end=0"
    )

    # save temp merged:
    df_merged.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_tmp_merged.parquet",
        index=False,
    )
    # save temp exploded
    df_exploded.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_tmp_exploded.parquet",
        index=False,
    )

    # read back in if necessary
    # df_exploded = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_tmp_exploded.parquet")

    # override int_year, used to get climate vars
    df_exploded["int_year_original"] = df_exploded["int_year"]  # keep copy of original
    df_exploded["int_year"] = df_exploded["years_to_expand"].astype(int)

    def get_months_child_alive_in_year(row):
        """
        Get the number of months child was alive in the int_year, to be used for
        taken weighted averages of climate vars.
        """
        int_year = row["int_year"]
        birth_year = row["birth_year"]
        birth_month = row["birth_month"]
        age_month = row["age_month"]
        # get num months alive in birth_year
        remaining_months_in_birth_year = 12 - birth_month + 1
        if (age_month <= remaining_months_in_birth_year) & (birth_year == int_year):
            return age_month

        else:
            months_at_beginning_of_year = (
                12 * (int_year - birth_year - 1) + remaining_months_in_birth_year
            )
            months_in_year = min(age_month - months_at_beginning_of_year, 12)
            return months_in_year

    # Calculate number of months child alive for given year (to be used for taking
    # weighted averages of climate variables)
    df_exploded["months_child_alive_in_year"] = df_exploded.apply(
        get_months_child_alive_in_year, axis=1
    )

    df_exploded["years_child_alive_in_year"] = (
        df_exploded["months_child_alive_in_year"] / 12
    )

    # remove rows with months_child_alive_in_year =0
    before_rows = len(df_exploded)
    df_exploded = df_exploded[df_exploded["months_child_alive_in_year"] > 0]
    logging.info(
        f"Dropped {before_rows - len(df_exploded):,} rows with months_child_alive_in_year=0"
    )

    # for rows with child_alive==0, replace with child_alive=1 if int_year < year_of_recorded_age
    df_exploded["child_alive"] = df_exploded["child_alive"].astype(int)
    df_exploded.loc[
        (df_exploded.child_alive == 0)
        & (df_exploded.int_year < df_exploded.year_of_recorded_age),
        "child_alive",
    ] = 1

    # Take out data with invalid lat and long
    before_rows = len(df_exploded)
    df_exploded = df_exploded.dropna(subset=["lat", "long"])
    df_exploded = df_exploded.query("lat != 0 and long != 0")
    dropped_due_to_coords = before_rows - len(df_exploded)
    logging.info(
        f"Dropped {dropped_due_to_coords:,} rows due to invalid lat and long values"
    )

    # NID 275090 is a very long survey in Peru, 2003-2008 that is coded as having
    # multiple year_starts. Removing it.
    # NID 411301 - updated: not in BR data extractions
    problematic_nids = [275090]
    before_rows = len(df_exploded)
    df_exploded = df_exploded.query("nid not in @problematic_nids")
    dropped_problematic_nids = before_rows - len(df_exploded)
    logging.info(
        f"Dropped {dropped_problematic_nids:,} rows due to problematic NIDs: {problematic_nids}"
    )

    # missing outcome variables
    measure_columns = MEASURES_IN_SOURCE[data_source_type]
    rows_with_na_outcomes = df_exploded[measure_columns].isna().any(axis=1).sum()
    rows_with_na_outcomes = int(rows_with_na_outcomes)
    logging.info(
        f"Dropped {rows_with_na_outcomes:,} rows with missing outcome variables ({measure_columns})"
    )

    # save temp files
    df_exploded.to_csv(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.csv",
        index=False,
    )

    # df_exploded = pd.read_csv("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.csv")

    # Merge with climate data 
    logging.info("Processing climate data...")
    climate_vars = get_climate_vars_for_dataframe(df_exploded)
    df_climate = merge_left_without_inflating(
        df_exploded, climate_vars, on=["int_year", "lat", "long"]
    )

    logging.info("Adding elevation data...")
    df_climate = get_elevation_for_dataframe(df_climate)
    df_climate = assign_lbd_admin2_location_id(df_climate)

    # save temp files
    df_climate.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_with_climate.parquet",
        index=False,
    )

    # df_climate = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_with_climate.parquet")

    # get unique invidiuals and clean variables
    df_climate["line_id"] = df_climate["line_id"].astype(int)

    df_climate["indv_id"] = (
        df_climate[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
    )
    logging.info(f"{df_climate['indv_id'].nunique():,} unique individuals in data")

    # flip child_alive so 1 = died, 0 = alive for easier interpretation
    df_climate["child_mortality"] = 1 - df_climate["child_alive"]

    df_climate["consumption"] = df_climate["ldipc_weighted_no_match"]

    # collapse by average climate var exposure for each child
    climate_vars = [
        "mean_temperature",
        "days_over_30C",
        "precipitation_days",
        "total_precipitation",
        "mean_low_temperature",
        "mean_high_temperature",
        "relative_humidity",
        # "days_over_26C",
        # "days_over_27C",
        # "days_over_28C",
        # "days_over_29C",
        # "days_over_31C",
        # "days_over_32C",
        # "days_over_33C",
        "elevation",
    ]

    # Neonatal
    # get the index of the row with the min age_month_at_year_end for each indv_id
    # df_min_age = df_climate.copy()
    # df_min_age = (
    #     df_climate.sort_values("age_month")
    #     .groupby("indv_id", as_index=False)
    #     .head(1)
    #     .reset_index(drop=True)
    # )

    # # for any child with age_month > 1, set their age_month to 1,
    # # child_mortality to 0, and child_alive to 1. This should get true neonatal
    # # mortality for all individuals.
    # df_min_age.loc[df_min_age.age_month > 1, "child_alive"] = 1
    # df_min_age.loc[df_min_age.age_month > 1, "child_mortality"] = 0
    # df_min_age.loc[df_min_age.age_month > 1, "age_month"] = 1

    # # make version of consumption that is per day
    # df_min_age["consumption_pd"] = df_min_age["consumption"] / 365

    # # make any day over 30C variable binary
    # df_min_age["any_days_over_30C"] = np.where(df_min_age["days_over_30C"] > 0, 1, 0)

    # # save out neonatal data set
    # df_min_age.to_parquet(Path(output_path_version) / "neonatal_data.parquet")

    # df_min_age = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/neonatal_data.parquet")

    # Get the index of the row with the max age_month_at_year_end for each indv_id
    df_max_age = df_climate.copy()
    df_max_age = (
        df_climate.sort_values("age_month")
        .groupby("indv_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # For each climate variable, replace its value in df_max_age with the average
    # for that indv_id. This should be weighted by 'years_child_alive_in_year'
    # def weighted_mean(group, value_cols, weight_col):
    #     return pd.DataFrame({
    #         col: np.average(group[col], weights=group[weight_col]) for col in value_cols
    #     }, index=[group.name])
    # weighted_climate_means = (
    #     df_climate
    #     .groupby("indv_id")
    #     .apply(weighted_mean, value_cols=climate_vars, weight_col="years_child_alive_in_year")
    #     .reset_index()
    # )
    def weighted_avg(group):
        d = {}
        w = group["years_child_alive_in_year"]
        for col in climate_vars:
            d[col] = np.average(group[col], weights=w)
        return pd.Series(d)

    weighted_climate_means = (
        df_climate.groupby("indv_id", group_keys=False)
        .apply(weighted_avg)
        .reset_index()
    )

    # climate_means = df_climate.groupby("indv_id")[climate_vars].mean()
    df_max_age = df_max_age.drop(columns=climate_vars).merge(
        weighted_climate_means, on="indv_id", how="left"
    )

    # df_max_age.to_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_13.01/data_avg_climate.parquet")
    # df_max_age = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_22.01/data.parquet")

    # make version of consumption that is per day
    df_max_age["consumption_pd"] = df_max_age["consumption"] / 365

    # make any day over 30C variable binary
    df_max_age["any_days_over_30C"] = np.where(df_max_age["days_over_30C"] > 0, 1, 0)

    # Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = df_max_age[df_max_age[measure].notna()].copy()
        measure_df["measure"] = measure
        measure_df["value"] = measure_df[measure]
        logging.info(
            f"Saving data for {measure} to {output_path_version} {len(measure_df)} rows"
        )
        for ldi_col in ["ldipc_weighted_no_match"]:  # ldi_cols:
            measure_df["ldi_pc_pd"] = measure_df[ldi_col] / 365
            logging.info(
                f"Saving data for {measure} to version {version} with {ldi_col} as LDI"
            )
            cm_data.save_training_data(measure_df, version)
            message = "Used " + ldi_col + " as LDI"
            # Save a small file with a record of which ldi column was used for this version
            with open(cm_data.training_data / version / "ldi_col.txt", "w") as f:
                f.write(message)


def run_training_data_prep_child_mortality_monthly(
    output_root: str | Path, data_source_type: str, module: str
) -> None:
    # TODO: Update TOCs
    # India = location_id 163, ihme_loc_id IND

    """
    Updates:
    - Load previously-processed monthly data for time
    - Merge on monthly absolute thresholds
    - Merge on monthly relative thresholds
    - Create bins
    """

    ## 1. Load and format child_mortality data from DEM_BR module

    # Set up logging and versioned output path
    measure_root = Path(output_root) / data_source_type
    os.makedirs(Path(measure_root) / "training_data", exist_ok=True, mode=0o777)
    cm_data = ClimateMalnutritionData(measure_root)
    version = cm_data.new_training_version()
    output_path_version = Path(measure_root) / "training_data" / version
    os.makedirs(
        output_path_version,
        exist_ok=True,
        mode=0o777,
    )

    dataprep_log_path = Path(output_path_version) / "data_prep_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(dataprep_log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    survey_data_path = SURVEY_DATA_PATHS[data_source_type][module]
    logging.info(f"Running training data prep for {data_source_type}...")

    logging.info(f"Creating new version stored under version: {version}")

    logging.info("Processing extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    # data_raw = concat_valid_extractions(survey_data_path)
    # data_raw = pd.read_csv(
    #     survey_data_path / "dem_br_matched_latlong.csv", encoding="latin1"
    # )
    data_raw = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_matched_2025_10_14.parquet"
    )
    # data_raw = pd.read_parquet(
    #     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_03_23.01/child_mortality_exploded_updated_wealth_decremented_age_mo.parquet"
    # )
    print(f"num rows = {len(data_raw):,}")

    logging.info(f"Total rows in concatenated raw data: {len(data_raw):,}")
    logging.info(
        f"Total unique NIDs in concatenated raw data: {data_raw['nid'].nunique():,}"
    )

    df = data_raw.copy()
    df = check_columns(df, module)

    for c in df.columns:
        if c.endswith("_x"):
            root = c[:-2]
            df.rename(columns={c: root}, inplace=True)
            y_version = f"{root}_y"
            if y_version in df.columns:
                df.drop(columns=y_version, inplace=True)

    # drop rows with missing key variables
    # Special note on India. India surveys contain age_month, which is the difference
    # between time of birth and time of interview in months. This is important because
    # int_year and int_month are missing. Calculate the variables using available
    # data as much as possible to avoid unnecessary data loss.

    # Attemp 1: int_year recovery via birth_year/month and age_month
    def recover_int_year(row):
        try:
            birth_mo = int(row["birth_month"])
            birth_yr = int(row["birth_year"])
            age_mo = int(row["age_month"])
            if pd.notna(birth_mo) and pd.notna(birth_yr) and pd.notna(age_mo):
                int_yr = birth_yr + (age_mo // 12)
                int_mo = (birth_mo + (age_mo % 12)) % 12
                return int_yr
            else:
                return row["int_year"]
        except:
            return row["int_year"]

    def recover_int_mo(row):
        try:
            birth_mo = int(row["birth_month"])
            birth_yr = int(row["birth_year"])
            age_mo = int(row["age_month"])
            if pd.notna(birth_mo) and pd.notna(birth_yr) and pd.notna(age_mo):
                int_yr = birth_yr + (age_mo // 12)
                int_mo = (birth_mo + (age_mo % 12)) % 12
                return int_mo
            else:
                return row["int_month"]
        except:
            return row["int_month"]

    df["int_year_recovered"] = df.apply(recover_int_year, axis=1)
    df["int_month_recovered"] = df.apply(recover_int_mo, axis=1)

    df["int_year"].fillna(df["int_year_recovered"], inplace=True)
    df["int_month"].fillna(df["int_month_recovered"], inplace=True)

    # Are there any rows with nonempty int_year/month and age-Month but no birth_month?
    len(
        df[
            (df["int_year"].notna())
            & (df["int_month"].notna())
            & (df["age_month"].notna())
            & (df["birth_month"].isna())
        ]
    )  # 207270

    def recover_birth_year(row):
        try:
            int_mo = int(row["int_month"])
            int_yr = int(row["int_year"])
            age_mo = int(row["age_month"])
            if pd.notna(int_mo) and pd.notna(int_yr) and pd.notna(age_mo):
                birth_yr = int_yr - (age_mo // 12)
                return birth_yr
            else:
                return row["birth_year"]
        except:
            return row["birth_year"]

    def recover_birth_mo(row):
        try:
            int_mo = int(row["int_month"])
            int_yr = int(row["int_year"])
            age_mo = int(row["age_month"])
            if pd.notna(int_mo) and pd.notna(int_yr) and pd.notna(age_mo):
                birth_mo = (int_mo - (age_mo % 12)) % 12
                return 12 if birth_mo == 0 else birth_mo  # prevent 0
            else:
                return row["birth_month"]
        except:
            return row["birth_month"]

    df["birth_year_recovered"] = df.apply(recover_birth_year, axis=1)
    df["birth_month_recovered"] = df.apply(recover_birth_mo, axis=1)
    df["birth_year"].fillna(df["birth_year_recovered"], inplace=True)
    df["birth_month"].fillna(df["birth_month_recovered"], inplace=True)

    df.drop(
        columns=[
            "int_year_recovered",
            "int_month_recovered",
            "birth_year_recovered",
            "birth_month_recovered",
        ],
        inplace=True,
    )

    rows_before_na_drop = len(df)
    key_vars = [
        "nid",
        "psu",
        "birth_year",
        "birth_month",
        "int_year",
        "int_month",
        "age_month",
        "hh_id",
        "geospatial_id",
        "line_id",
        "lat",
        "long",
        "child_alive",
    ]
    df_tmp = df.dropna(subset=key_vars)
    na_rows_dropped = rows_before_na_drop - len(df_tmp)
    logging.info(
        f"Dropped {na_rows_dropped:,} rows with missing key variables: {key_vars}"
    )
    df = df_tmp.copy()

    for var in key_vars:
        logging.info(f"- {var}: {data_raw[var].isna().sum():,} missing values")
    logging.info(f"NIDs with incomplete (some NAs) key variables: {key_vars}")
    for var in key_vars:
        logging.info(
            f"-{data_raw[data_raw[var].isna()]["nid"].nunique():,} contains missing {var} values"
        )

    logging.info(f"Total unique NIDs after dropped NA values: {df['nid'].nunique():,}")
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)

    # update variable data types besides hh_id
    int_cols = [
        "nid",
        "psu",
        "birth_year",
        "birth_month",
        "int_year",
        "int_month",
        "age_month",
        "child_alive",
        "geospatial_id",
    ]
    df[int_cols] = df[int_cols].astype("int")

    # Apply cleaning function to each group and update hh_id
    df["old_hh_id"] = df["hh_id"]
    df["hh_id"] = df.groupby(["nid", "ihme_loc_id", "psu"], group_keys=False).apply(
        clean_hh_id_subset
    )

    assert len(df[df["hh_id"].isna()]) == len(
        df[df["old_hh_id"].isna()]
    ), "NAs introduced by cleaning"

    df["hh_id"] = df["hh_id"].astype("int")
    df.drop(columns=["old_hh_id"], inplace=True)

    # create indv_id
    # get unique invidiuals and clean variables
    df["line_id"] = df["line_id"].astype(int)

    # create temp mothers_age_year and aod_months
    df["mothers_age_year_tmp"] = df["mothers_age_year"].replace(np.nan, "NA")
    df["aod_months_tmp"] = df["aod_months"].replace(np.nan, "NA")

    df["indv_id"] = (
        df[["nid", "psu", "hh_id", "line_id","sex_id","birth_year","birth_month","int_year","int_month","age_month","child_alive","mothers_age_year_tmp","aod_months_tmp"]].astype(str).agg("_".join, axis=1)
    )

    len(df)-df["indv_id"].nunique() # 12. Have to assume these are true duplicates for remainder of script.
    # dups = df[df.duplicated(subset=["indv_id"], keep=False)]

    df = df[~df.duplicated(subset=["indv_id"], keep="first")]
    assert len(df)==df["indv_id"].nunique(), "indv_id not unique"

    df.to_parquet(output_path_version / "child_mortality_indv_id.parquet", index=False)

    # Prepping wealth dataset
    dhs_wealth_data_raw = get_DHS_wealth_dataset()
    dhs_wealth_data = dhs_wealth_data_raw.copy()

    # Find out percent of anemia nids and hh_ids that can be matched in wealth data
    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]

    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "child_mortality")
    dhs_wealth_data_test = get_ldipc_from_asset_score(
        dhs_wealth_data,
        cm_data,
        asset_score_col="wealth_index_dhs",
        weights_col="hhweight",
        # plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "dhs_plots.pdf",
        ldi_version=LDI_VERSION,
    )

    wealth_nids = set(dhs_wealth_data_test.nid.unique())
    df_nids = set(df.nid.unique())
    common_nids = wealth_nids.intersection(df_nids)

    nid_with_wealth_pc = 100 * len(common_nids) / len(df_nids)
    logging.info(
        f"{nid_with_wealth_pc:.1f}% of df NIDs - {len(common_nids)} out of "
        f"{len(df_nids)} in wealth data NIDs"
    )

    dhs_wealth_data_test = dhs_wealth_data_test.query("nid in @df_nids")

    # Merge data

    # fix df columns
    df.rename(columns={"iso3": "ihme_loc_id"}, inplace=True)
    df["ihme_loc_id"] = df["ihme_loc_id"].str.replace("KEN_.*", "KEN", regex=True)
    df[["year_start", "year_end", "int_year"]] = df[
        ["year_start", "year_end", "int_year"]
    ].astype(int)

    df_wealth = merge_left_without_inflating(
        df,
        dhs_wealth_data_test.drop(
            columns=["geospatial_id", "strata", "lat", "long", "hhweight"]
        ),
        on=merge_cols,
    )

    merged_percent = len(df_wealth[~df_wealth["ldipc_weighted_no_match"].isna()]) / len(
        df_wealth
    )
    print(f"Merged rows percent: {100*merged_percent:.1f}%")

    # Calculate proportion of NA and filter out nids with too much wealth missingness (bad merges)
    merged_na_props = (
        df_wealth.groupby(["nid"]).ldipc_weighted_no_match.count()
        / df_wealth.groupby(["nid"]).ldipc_weighted_no_match.size()
    )
    merged_nids = merged_na_props[merged_na_props > 0.95].index.to_list()
    df_merged = df_wealth.query("nid in @merged_nids").copy()
    dropped_too_missingness = len(df_wealth) - len(df_merged)
    logging.info(
        f"Dropped {dropped_too_missingness:,} rows from {len(df_wealth):,} due to excessive wealth missingness in NIDs"
    )
    logging.info(f"Total unique NIDs remaining: {df_merged['nid'].nunique():,}")

    # Add the birth-year version of LDI-PC alongside the survey-year one above,
    # matching what run_training_data_prep_neonatal does. It needs both frames:
    # the raw extract to rank households against the whole survey, and df_merged
    # for each child's birth year. col_suffix keeps both sets of columns so they
    # can be compared.
    #
    # This runs BEFORE the monthly explode below, so it is assigned once per
    # child and then copied onto every month of that child's exposure: a child
    # observed at 48 months carries the income distribution of its BIRTH year,
    # not of the year it was living through. For neonatal those are the same
    # year by construction; here they are not. To use exposure-year income
    # instead, this call has to move after the explode and read the per-row
    # int_year the explode computes.
    df_merged = get_ldipc_from_asset_score_by_birth_year(
        dhs_wealth_data,
        cm_data,
        df_merged,
        asset_score_col="wealth_index_dhs",
        weights_col="hhweight",
        ldi_version=LDI_VERSION,
        col_suffix="_birth_year",
    )

    # Include difference between int_year and birth_year for sensitivity analysis
    df_merged["int_birth_year_diff_months"] = 12 * (
        df_merged["int_year"] - df_merged["birth_year"]
    ) + (df_merged["int_month"] - df_merged["birth_month"])

    # Assign age group
    before_rows = len(df_merged)

    # replace age_month with aod_months for rows with child_alive==0
    df_merged["age_month_original"] = df_merged["age_month"]  # keep copy of original
    df_merged.loc[df_merged.child_alive == 0, "age_month"] = df_merged.loc[
        df_merged.child_alive == 0, "aod_months"
    ]

    # drop data with no age_month
    before_rows = len(df_merged)
    df_merged = df_merged[df_merged["age_month"].notna()]
    logging.info(
        f"Dropped {before_rows - len(df_merged):,} rows with missing age_month or aod_months"
    )

    # create list of years between birth year and year that the age_month lands on.
    df_merged["age_month"] = df_merged["age_month"].astype(int)

    # save temp merged:
    df_merged.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_tmp_wealth_merged.parquet",
        index=False,
    )

    # Drop the survey-year LDI-PC columns now that the birth-year ones replace
    # them downstream. This has to happen BEFORE the explode below: the explode
    # turns each child into one row per month of exposure, so every column still
    # on the frame is multiplied by ~50. At 400m+ exploded rows one float64
    # column costs roughly 3Gb resident, and this job already peaks at its
    # 900Gb ceiling. The survey-year values are not lost - the temp parquet
    # written just above is saved pre-explode and still carries them.
    df_merged = df_merged.drop(columns=LDIPC_METHOD_COLUMNS, errors="ignore")

    # we want an int_year and int_month, and a new age_month incrementing for all
    # months up until age_month. Cut off age limit here at 60 months to avoid
    # giant dataset
    df_merged["months_to_expand"] = df_merged.apply(
        lambda x: list(range(0, min(x["age_month"] + 1, 61))), axis=1
    )

    df_exploded = df_merged.explode("months_to_expand")
    df_exploded["age_month_pre_exploded"] = df_exploded["age_month"]
    df_exploded["age_month"] = df_exploded["months_to_expand"]

    df_exploded["int_year_original"] = df_exploded["int_year"]
    df_exploded["int_month_original"] = df_exploded["int_month"]

    # make all outcome values pre-age-month be child_alive=1
    df_exploded.loc[
        df_exploded.age_month < df_exploded.age_month_pre_exploded, "child_alive"
    ] = 1

    # get int_year for each row
    df_exploded["int_year"] = df_exploded.apply(
        lambda x: x["birth_year"] + (x["birth_month"] + x["age_month"] - 1) // 12,
        axis=1,
    )
    df_exploded["int_month"] = df_exploded.apply(
        lambda x: (x["birth_month"] + (x["age_month"] % 12)) % 12, axis=1
    )
    df_exploded.loc[df_exploded["int_month"] == 0, "int_month"] = 12

    # save temp merged:
    df_exploded.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_monthly.parquet",
        index=False,
    )

    # df_exploded = pd.read_parquet(
    #     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_monthly.parquet"
    # )

    logging.info(
        f"Exploded data to {len(df_exploded):,} rows by expanding on years between child birth and either age of death or age at interview"
    )

    df_exploded = assign_age_group(df_exploded, indicator="child_mortality")

    # age_days and aod_days are empty, but we assume that age_month and aod_months
    # are rounded down, such that age_month 0 is not stillborns, but deaths between
    # 0 and 1 month. This is required for a survival modeling approach, for which
    # time to event cannot be 0.
    # df_exploded["age_month"] += 1

    # for rows with child_alive==0, replace with child_alive=1 if int_year < year_of_recorded_age
    df_exploded["child_alive"] = df_exploded["child_alive"].astype(int)

    # Take out data with invalid lat and long
    before_rows = len(df_exploded)
    df_exploded = df_exploded.dropna(subset=["lat", "long"])
    df_exploded = df_exploded.query("lat != 0 and long != 0")
    dropped_due_to_coords = before_rows - len(df_exploded)
    logging.info(
        f"Dropped {dropped_due_to_coords:,} rows due to invalid lat and long values"
    )

    # NID 275090 is a very long survey in Peru, 2003-2008 that is coded as having
    # multiple year_starts. Removing it.
    # NID 411301 - updated: not in BR data extractions
    problematic_nids = [275090]
    before_rows = len(df_exploded)
    df_exploded = df_exploded.query("nid not in @problematic_nids")
    dropped_problematic_nids = before_rows - len(df_exploded)
    logging.info(
        f"Dropped {dropped_problematic_nids:,} rows due to problematic NIDs: {problematic_nids}"
    )

    # missing outcome variables
    measure_columns = MEASURES_IN_SOURCE[data_source_type]
    rows_with_na_outcomes = df_exploded[measure_columns].isna().any(axis=1).sum()
    rows_with_na_outcomes = int(rows_with_na_outcomes)
    logging.info(
        f"Dropped {rows_with_na_outcomes:,} rows with missing outcome variables ({measure_columns})"
    )

    # save temp files
    df_exploded.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.parquet",
        index=False,
    )

    # df_exploded = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.parquet")

    # Merge with climate data - skip if pressed for time. Annual variables not currently
    logging.info("Processing climate data...")
    climate_vars = get_climate_vars_for_dataframe(df_exploded)
    df_climate = merge_left_without_inflating(
        df_exploded, climate_vars, on=["int_year", "lat", "long"]
    )

    # save temp files
    # df_climate.to_parquet(
    #     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_with_climate.parquet",
    #     index=False,
    # )

    logging.info("Adding elevation data...")
    df_climate = get_elevation_for_dataframe(df_climate)

    # save temp files
    # df_climate.to_parquet(
    #     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_with_climate.parquet",
    #     index=False,
    # )
    
    df_climate = assign_lbd_admin2_location_id(df_climate)

    # save temp files
    # df_climate.to_parquet(
    #     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_with_climate.parquet",
    #     index=False,
    # )

    # df_climate = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_exploded_with_climate.parquet")

    # get unique invidiuals and clean variables
    # df_climate["line_id"] = df_climate["line_id"].astype(int)

    # df_climate["indv_id"] = (
    #     df_climate[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
    # )
    logging.info(f"{df_climate['indv_id'].nunique():,} unique individuals in data")

    # avg number of months per individual
    avg_months_per_indv = len(df_climate) / df_climate["indv_id"].nunique()
    print(f"Average number of months per individual: {avg_months_per_indv:.2f}")

    # flip child_alive so 1 = died, 0 = alive for easier interpretation
    df_climate["child_mortality"] = 1 - df_climate["child_alive"]

    # Birth-year income, matching what run_training_data_prep_neonatal does.
    # The survey-year columns are dropped before the explode above, so this is
    # the only LDI-PC left on the frame.
    df_climate["consumption"] = df_climate["ldipc_weighted_no_match_birth_year"]

    # make version of consumption that is per day
    df_climate["consumption_pd"] = df_climate["consumption"] / 365

    # make any day over 30C variable binary - skip if pressed for time. Annual variables not currently
    df_climate["any_days_over_30C"] = np.where(df_climate["days_over_30C"] > 0, 1, 0)

    # Write to output
    df_climate.to_parquet(Path(output_path_version) / "data.parquet", index=False)
    del df_climate; gc.collect()

    # Add absolute monthly climate vars thresholds
    climate_vars_da = get_all_climate_vars_year_months_for_latlongs(
        pd.read_parquet(Path(output_path_version) / "data.parquet", columns=["lat", "long", "int_year", "int_month"]),
        year_var="int_year", month_var="int_month"
    )
    climate_vars_da.to_netcdf(Path(output_path_version) / "abs_month_climate_vars.nc")
    # climate_vars_da = xr.open_dataarray(Path(output_path_version) / "abs_month_climate_vars.nc")
    climate_vars_df = climate_vars_da.to_dataframe().reset_index()
    del climate_vars_da; gc.collect()

    # set names to merge
    climate_vars_df.drop(columns=["point", "longitude", "latitude"], inplace=True)
    climate_vars_df.rename(
        columns={
            "year": "int_year",
            "month": "int_month",
            "lat_orig": "lat",
            "long_orig": "long",
        },
        inplace=True,
    )
    # pivot wide
    climate_vars_wide_df = climate_vars_df.pivot_table(
        index=["int_year", "int_month", "lat", "long"],
        columns="climate_var",
        values="value",
    ).reset_index()
    del climate_vars_df; gc.collect()

    climate_vars_wide_df.rename(
        columns={
            "mean_temperature": "mean_temperature_monthly",
            "total_precipitation": "total_precipitation_monthly",
            "days_over_24C": "days_over_24C_monthly",
            "days_over_25C": "days_over_25C_monthly",
            "days_over_26C": "days_over_26C_monthly",
            "days_over_27C": "days_over_27C_monthly",
            "days_over_28C": "days_over_28C_monthly",
            "days_over_29C": "days_over_29C_monthly",
            "days_over_30C": "days_over_30C_monthly",
            "days_over_31C": "days_over_31C_monthly",
            "days_over_32C": "days_over_32C_monthly",
        },
        inplace=True,
    )

    # Streaming merge abs thresholds using Polars (avoids doubling 400M-row DF in memory)
    abs_lookup_pl = pl.from_pandas(climate_vars_wide_df)
    del climate_vars_wide_df; gc.collect()
    abs_out = Path(output_path_version) / "data_monthly_expanded_abs_thresholds.parquet"
    (
        pl.scan_parquet(Path(output_path_version) / "data.parquet")
        .join(abs_lookup_pl.lazy(), on=["int_year", "int_month", "lat", "long"], how="left")
        .sink_parquet(abs_out)
    )
    del abs_lookup_pl; gc.collect()

    # TODO: Add relative monthly climate vars thresholds
    rel_lookup_parts_dir = Path(output_path_version) / "rel_lookup_parts"
    rel_lookup_parts_dir.mkdir(parents=True, exist_ok=True)
    for part_path in rel_lookup_parts_dir.glob("*.parquet"):
        part_path.unlink()

    coords_df = (
        pl.scan_parquet(abs_out)
        .select(["lat", "long", "int_year", "int_month"])
        .unique()
        .collect()
    )
    coords_pd = coords_df.to_pandas()
    del coords_df; gc.collect()

    years = sorted(coords_pd["int_year"].dropna().unique().tolist())
    for year in years:
        year_coords = coords_pd.loc[
            coords_pd["int_year"] == year, ["lat", "long"]
        ].drop_duplicates().copy()
        year_coords["lookup_year"] = year

        thresholds_da = get_climate_thresholds_all_locs(
            year_coords,
            year_col="lookup_year",
            lat_col="lat",
            long_col="long",
        )
        if thresholds_da is None:
            del year_coords; gc.collect()
            continue

        thresholds_df = thresholds_da.to_dataframe().reset_index()
        del thresholds_da; gc.collect()

        thresholds_df.drop(columns=["point", "longitude", "latitude"], inplace=True)
        thresholds_df.rename(
            columns={
                "year": "int_year",
                "month": "int_month",
                "lat_orig": "lat",
                "long_orig": "long",
            },
            inplace=True,
        )

        thresholds_df["quantile_str"] = (
            thresholds_df["quantile"].astype(str).str.replace("0.", "q")
        )
        thresholds_df.drop(columns="quantile", inplace=True)

        thresholds_wide_df = thresholds_df.pivot_table(
            index=["int_year", "int_month", "lat", "long"],
            columns="quantile_str",
            values="value",
        ).reset_index()
        del thresholds_df; gc.collect()

        thresholds_wide_df.rename(
            columns={
                "q75": "q75_monthly",
                "q8": "q80_monthly",
                "q85": "q85_monthly",
                "q9": "q90_monthly",
                "q95": "q95_monthly",
            },
            inplace=True,
        )

        part_path = rel_lookup_parts_dir / f"rel_lookup_{year}.parquet"
        pl.from_pandas(thresholds_wide_df).write_parquet(part_path)
        del thresholds_wide_df; gc.collect()

    del coords_pd; gc.collect()

    rel_lookup_lf = pl.concat(
        [
            pl.scan_parquet(part_path)
            for part_path in sorted(rel_lookup_parts_dir.glob("*.parquet"))
        ],
        how="diagonal_relaxed",
    )

    # Streaming merge relative thresholds using Polars
    rel_out = Path(output_path_version) / "data_monthly_expanded_rel_thresholds.parquet"
    (
        pl.scan_parquet(abs_out)
        .join(rel_lookup_lf, on=["int_year", "int_month", "lat", "long"], how="left")
        .filter(
            pl.col("consumption_pd").is_not_null()
            & pl.col("days_over_30C_monthly").is_not_null()
            & pl.col("total_precipitation_monthly").is_not_null()
        )
        .sink_parquet(rel_out)
    )
    del rel_lookup_lf; gc.collect()

    # before_rows = pl.scan_parquet(abs_out).select(pl.len()).collect().item()
    # after_rows = pl.scan_parquet(rel_out).select(pl.len()).collect().item()
    # logging.info(
    #     f"Dropped {before_rows - after_rows:,} rows with missing values in key merged variables (consumption_pd, days_over_30C_monthly) after merging monthly climate variables"
    # )

    # df_climate = pd.read_parquet(
    #     Path(output_path_version) / "data_monthly_expanded_rel_thresholds.parquet"
    # )


    # check:
    print(f"max age_month: {pl.scan_parquet(rel_out).select(pl.col('age_month').max()).collect().item()}")

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
        # "precipitation_days",
        "total_precipitation",
        # "mean_low_temperature",
        # "mean_high_temperature",
        # "relative_humidity",
        "elevation",
        "consumption",
        "consumption_pd",
        "days_over_24C_monthly",
        "days_over_25C_monthly",
        "days_over_26C_monthly",
        "days_over_27C_monthly",
        "days_over_28C_monthly",
        "days_over_29C_monthly",
        "days_over_30C_monthly",
        "days_over_31C_monthly",
        "days_over_32C_monthly",
        "mean_temperature_monthly",
        "total_precipitation_monthly",
        "q75_monthly",
        "q80_monthly",
        "q85_monthly",
        "q90_monthly",
        "q95_monthly",
    ]

    # Edit to preserve indv_id
    identity_vars = [
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
    ]

    group_by_vars_within_bin = identity_vars + [
        "age_1_m",
        "age_3_m",
        "age_6_m",
        "age_12_m",
        "age_24_m",
        "age_36_m",
        "age_48_m",
        "age_60_m",
    ]

    # Edit to preserve indv_id
    # Load only needed columns from rel_out into Polars (avoids full pandas load)
    needed_cols = list(
        dict.fromkeys(
            identity_vars + ["age_month"] + get_max_vars + get_avg_vars
        )
    )
    df_pl = pl.scan_parquet(rel_out).select(needed_cols).collect()

    # potentially no longer necessary: df_grouped_within_bin / data_within_bin.parquet
    # is never read by later code in this function
    """
    # Add bin columns
    for bin_name, bin_month in time_bin_dict.items():
        df_pl = df_pl.with_columns(
            ((pl.col("age_month") >= bin_month[0]) & (pl.col("age_month") < bin_month[1]))
            .cast(pl.Int32)
            .alias(bin_name)
        )

    # within bin
    df_grouped_within_bin = (
        df_pl.group_by(group_by_vars_within_bin)
        .agg(
            [pl.col(var).max() for var in get_max_vars]
            + [pl.col(var).mean() for var in get_avg_vars]
        )
        .to_pandas()
    )

    df_grouped_within_bin.to_parquet(
        Path(output_path_version) / "data_within_bin.parquet",
        index=False,
    )
    """
    # df_grouped_within_bin = pd.read_parquet(
    #     Path(output_path_version) / "data_within_bin.parquet"
    # )

    max_age_per_indv = df_pl.group_by("indv_id").agg(
        pl.col("age_month").max().alias("max_age_month")
    )

    cumulative_frames = []
    for bin_name, bin_month in time_bin_dict.items():
        lower, upper = bin_month
        # Only include individuals who have at least one observation within this bin,
        # meaning they did not exit the interview before reaching this age period
        eligible_ids = max_age_per_indv.filter(pl.col("max_age_month") >= lower).select("indv_id")

        # individuals_in_bin = (
        #     df_pl.filter((pl.col("age_month") >= lower) & (pl.col("age_month") < upper))
        #     .select(identity_vars)
        #     .unique()
        # )
        bin_frame = (
            df_pl.filter(pl.col("age_month") < upper)
            .join(eligible_ids, on="indv_id", how="inner")
            .group_by(identity_vars)
            .agg(
                [pl.col(var).max().alias(var) for var in get_max_vars]
                + [pl.col(var).mean().alias(f"{var}_cumul") for var in get_avg_vars]
            )
            .with_columns(pl.lit(bin_name).alias("bin_name"))
        )
        logging.info(
            f"Built cumulative frame for {bin_name}: {bin_frame.height:,} rows"
        )
        # del individuals_in_bin
        gc.collect()
        # Add the within-bin dummy columns so the output mirrors df_grouped_within_bin
        for other_bin in time_bin_dict:
            bin_frame = bin_frame.with_columns(
                pl.lit(1 if other_bin == bin_name else 0).alias(other_bin)
            )
        cumulative_frames.append(bin_frame)
        del bin_frame
        gc.collect()

    df_grouped_cumulative = pl.concat(cumulative_frames).to_pandas()

    # Perform quick fix. THe above loop resulted in extra
    # duplicate rows for children who either died or exited interview
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
        df_grouped_cumulative.loc[df_grouped_cumulative[ag] == 1, "age_group"] = ag

    bounds_df = pd.DataFrame(
        [
            (bin_name, bounds[0], bounds[1])
            for bin_name, bounds in time_bin_dict.items()
        ],
        columns=["bin_name", "lower_bound", "upper_bound"],
    )
    df_grouped_cumulative = df_grouped_cumulative.merge(
        bounds_df, on="bin_name", how="left"
    )

    df_grouped_cumulative = df_grouped_cumulative[
        df_grouped_cumulative["age_month"] >= df_grouped_cumulative["lower_bound"]
    ]

    # df_grouped_cumulative.to_parquet(
    #     Path(output_path_version) / "data_cumulative_bins.parquet",
    #     index=False,
    # )

    # potentially no longer necessary: "_constant" variables are not in the required output
    # Keep only 1 dummy in max age for cumulative age
    # vars. Equivalent to adding back on binned age vars

    # get max age per indv and add on
    # df_max_age = (
    #     df_grouped_cumulative.sort_values("age_month")
    #     .groupby("indv_id", as_index=False)
    #     .tail(1)
    #     .reset_index(drop=True)
    # )

    # # keep_cols
    # get_avg_vars_cumul = [f"{var}_cumul" for var in get_avg_vars]
    # keep_cols = ["indv_id"] + get_avg_vars_cumul
    # df_max_age_constant_vars = df_max_age[keep_cols]
    # for v in get_avg_vars:
    #     v_cumul = f"{v}_cumul"
    #     df_max_age_constant_vars.rename(
    #         columns={v_cumul: f"{v}_constant"}, inplace=True
    #     )

    # df_grouped_cumulative = pd.merge(
    #     df_grouped_cumulative, df_max_age_constant_vars, on="indv_id", how="left"
    # )

    print(len(df_grouped_cumulative))
    df_grouped_cumulative = df_grouped_cumulative.drop_duplicates()
    print(len(df_grouped_cumulative))

    # fix problematic rows
    # 1. Ensure that child_mortality is 1 only once, corresponding to the
    # age_month_original if child_alive = 0. 
    # For child_alive == 0, get max age_month

    # 2. Keep rows leading up to max age, then drop thereafter


    # Edit to preserve indv_id
    # df_grouped_cumulative.to_parquet(
    #     Path(output_path_version) / "data_constant_vars.parquet",
    #     index=False,
    # )

    # ---- post-processing, ported from
    # notebooks/child_mortality/child_mortality_data_checks_edits_10_07.ipynb and
    # child_mortality_data_checks_edits_10_14.ipynb --
    # Births further than this many months before the interview are dropped from
    # data.parquet but kept in data_no_10_yr_cutoff.parquet.  Inclusive.
    max_recall_months = 120
    # Age strata dropped from both saved versions.  The first month of life is the
    # neonatal_mortality model's, so child_mortality is fit from day 28 (5q28),
    # and 5q0 is rebuilt from the two downstream (strun fhs_scenarios).  Rows go,
    # the age_1_m dummy column stays, all zero, as in training data 2026_10_14.01.
    excluded_age_groups = ["age_1_m"]
    # Columns the model cannot tolerate nulls in.
    cols_to_check = [
        "child_mortality",
        "consumption_pd_cumul",
        "int_birth_year_diff_months",
        "birth_year",
        "days_over_30C_monthly_cumul",
        "total_precipitation_cumul",
        "lat",
        "long",
    ]

    # National median consumption from the RT income distributions rather than
    # the survey households: the reference scenario at the 50th population
    # percentile, divided to per-day so it sits on the same scale as
    # consumption_pd.  Uses the same LDI_VERSION as the household matching above,
    # so both read one income vintage.
    # Merged on ihme_loc_id + birth_year alone, unlike the neonatal equivalent:
    # this frame carries no location_id, so the merge brings one in alongside
    # consumption_median_RT.
    # Built from measure_root rather than reusing `cm_data`: that name is rebound
    # to DEFAULT_ROOT further up, so it no longer tracks this run's output_root.
    rt_consumption_df = ClimateMalnutritionData(measure_root).load_ldi_distributions(
        geospecificity="national", version=LDI_VERSION
    )
    rt_consumption_df = rt_consumption_df[
        (rt_consumption_df["scenario"] == "reference")
        & (rt_consumption_df["population_percentile"] == 0.5)
    ][["ihme_loc_id", "year_id", "location_id", "ldipc"]].rename(
        columns={"year_id": "birth_year", "ldipc": "consumption_median_RT"}
    )
    df_grouped_cumulative = df_grouped_cumulative.drop(
        columns="consumption_median_RT", errors="ignore"
    ).merge(rt_consumption_df, on=["ihme_loc_id", "birth_year"], how="left")
    n_missing_rt = int(df_grouped_cumulative["consumption_median_RT"].isna().sum())
    if n_missing_rt:
        msg = (
            f"{n_missing_rt:,} rows have no {LDI_VERSION} national median "
            "consumption after the merge; the LDI series is missing some "
            "ihme_loc_id/birth_year present in the survey data."
        )
        raise ValueError(msg)
    df_grouped_cumulative["consumption_median_RT"] /= 365

    n_rows = len(df_grouped_cumulative)
    df_grouped_cumulative = df_grouped_cumulative[
        ~df_grouped_cumulative["age_group"].isin(excluded_age_groups)
    ]
    print(
        f"Dropped {n_rows - len(df_grouped_cumulative):,} rows in age groups "
        f"{excluded_age_groups}"
    )

    # Every column the model needs must be complete.  Checked before the recall
    # cutoff, so it holds for both saved versions.
    missing_counts = {
        col: int(df_grouped_cumulative[col].isna().sum()) for col in cols_to_check
    }
    for col, n_missing in missing_counts.items():
        print(f"{col}: {n_missing} missing values")
    incomplete = {col: n for col, n in missing_counts.items() if n}
    if incomplete:
        msg = f"Missing values in required child_mortality columns: {incomplete}"
        raise ValueError(msg)

    # Save the uncut extract, then the recall-limited one the models train on.
    df_grouped_cumulative.to_parquet(
        Path(output_path_version) / "data_no_10_yr_cutoff.parquet", index=False
    )
    print(f"data_no_10_yr_cutoff.parquet saved ({len(df_grouped_cumulative):,} rows)")

    df_grouped_cumulative = df_grouped_cumulative[
        df_grouped_cumulative["int_birth_year_diff_months"] <= max_recall_months
    ]
    df_grouped_cumulative.to_parquet(
        Path(output_path_version) / "data.parquet", index=False
    )
    print(
        f"data.parquet saved ({len(df_grouped_cumulative):,} rows, "
        f"int_birth_year_diff_months <= {max_recall_months})"
    )
    # for measure in MEASURES_IN_SOURCE[data_source_type]:
    #     measure_df = df_climate[df_climate[measure].notna()].copy()
    #     measure_df["measure"] = measure
    #     measure_df["value"] = measure_df[measure]
    #     logging.info(
    #         f"Saving data for {measure} to {output_path_version} {len(measure_df)} rows"
    #     )
    #     for ldi_col in ["ldipc_weighted_no_match"]:  # ldi_cols:
    #         measure_df["ldi_pc_pd"] = measure_df[ldi_col] / 365
    #         logging.info(
    #             f"Saving data for {measure} to version {version} with {ldi_col} as LDI"
    #         )
    #         cm_data.save_training_data(measure_df, version)
    #         message = "Used " + ldi_col + " as LDI"
    #         # Save a small file with a record of which ldi column was used for this version
    #         with open(cm_data.training_data / version / "ldi_col.txt", "w") as f:
    #             f.write(message)


def resume_child_mortality_from_parquet(input_parquet_path: str | Path) -> None:
    """
    Resume run_training_data_prep_child_mortality_monthly processing from a saved
    intermediate parquet file, to avoid rerunning the full pipeline after a disconnect.
    Paste the remaining processing steps below, ending with a save to output_path_version.
    """
    output_root = DEFAULT_ROOT
    data_source_type = "child_mortality"
    module = "dem_br"
    measure_root = Path(output_root) / data_source_type
    version = "2026_08_24.01"
    output_path_version = Path(measure_root) / "training_data" / version
    os.makedirs(output_path_version, exist_ok=True, mode=0o777)

    # input_parquet_path = Path(input_parquet_path)

    # Paste subsequent processing steps here
    abs_out = input_parquet_path

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
        # "any_days_over_30C",
        "child_alive",
        "child_mortality",
    ]

    get_avg_vars = [
        # "mean_temperature",
        # "days_over_30C",
        # "precipitation_days",
        # "total_precipitation",
        # "mean_low_temperature",
        # "mean_high_temperature",
        # "relative_humidity",
        # "elevation",
        "consumption",
        "consumption_pd",
        "days_over_24C_monthly",
        "days_over_25C_monthly",
        "days_over_26C_monthly",
        "days_over_27C_monthly",
        "days_over_28C_monthly",
        "days_over_29C_monthly",
        "days_over_30C_monthly",
        "days_over_31C_monthly",
        "days_over_32C_monthly",
        "mean_temperature_monthly",
        "total_precipitation_monthly",
        # "q75_monthly",
        # "q80_monthly",
        # "q85_monthly",
        # "q90_monthly",
        # "q95_monthly",
    ]

    # Edit to preserve indv_id
    identity_vars = [
        "ihme_loc_id",
        "geospatial_id",
        "psu",
        "strata",
        "line_id",
        "hh_id",
        "lat",
        "long",
        # "lbd_admin2_id",
        "indv_id",
    ]

    group_by_vars_within_bin = identity_vars + [
        "age_1_m",
        "age_3_m",
        "age_6_m",
        "age_12_m",
        "age_24_m",
        "age_36_m",
        "age_48_m",
        "age_60_m",
    ]

    # Edit to preserve indv_id
    # Load only needed columns from rel_out into Polars (avoids full pandas load)
    needed_cols = list(
        dict.fromkeys(
            identity_vars + ["age_month"] + get_max_vars + get_avg_vars
        )
    )
    df_pl = pl.scan_parquet(abs_out).select(needed_cols).collect()

    max_age_per_indv = df_pl.group_by("indv_id").agg(
        pl.col("age_month").max().alias("max_age_month")
    )

    print("loopig over time bin frames")
    cumulative_frames = []
    for bin_name, bin_month in time_bin_dict.items():
        lower, upper = bin_month
        # Only include individuals who have at least one observation within this bin,
        # meaning they did not exit the interview before reaching this age period
        eligible_ids = max_age_per_indv.filter(pl.col("max_age_month") >= lower).select("indv_id")

        # individuals_in_bin = (
        #     df_pl.filter((pl.col("age_month") >= lower) & (pl.col("age_month") < upper))
        #     .select(identity_vars)
        #     .unique()
        # )
        bin_frame = (
            df_pl.filter(pl.col("age_month") < upper)
            .join(eligible_ids, on="indv_id", how="inner")
            .group_by(identity_vars)
            .agg(
                [pl.col(var).max().alias(var) for var in get_max_vars]
                + [pl.col(var).mean().alias(f"{var}_cumul") for var in get_avg_vars]
            )
            .with_columns(pl.lit(bin_name).alias("bin_name"))
        )
        # logging.info(
        #     f"Built cumulative frame for {bin_name}: {bin_frame.height:,} rows"
        # )
        # del individuals_in_bin
        gc.collect()
        # Add the within-bin dummy columns so the output mirrors df_grouped_within_bin
        for other_bin in time_bin_dict:
            bin_frame = bin_frame.with_columns(
                pl.lit(1 if other_bin == bin_name else 0).alias(other_bin)
            )
        cumulative_frames.append(bin_frame)
        del bin_frame
        gc.collect()

    print("concatenating dataframes")
    df_grouped_cumulative = pl.concat(cumulative_frames).to_pandas()

    # Perform quick fix. THe above loop resulted in extra
    # duplicate rows for children who either died or exited interview
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
        df_grouped_cumulative.loc[df_grouped_cumulative[ag] == 1, "age_group"] = ag

    bounds_df = pd.DataFrame(
        [
            (bin_name, bounds[0], bounds[1])
            for bin_name, bounds in time_bin_dict.items()
        ],
        columns=["bin_name", "lower_bound", "upper_bound"],
    )
    df_grouped_cumulative = df_grouped_cumulative.merge(
        bounds_df, on="bin_name", how="left"
    )

    df_grouped_cumulative = df_grouped_cumulative[
        df_grouped_cumulative["age_month"] >= df_grouped_cumulative["lower_bound"]
    ]

    print(len(df_grouped_cumulative))
    df_grouped_cumulative = df_grouped_cumulative.drop_duplicates()
    print(len(df_grouped_cumulative))

    print("saving data")
    # assert len(df_grouped_cumulative[df_grouped_cumulative["child_mortality"].isna()]) == 0, "Missing child_mortality values in final data"
    df_grouped_cumulative.to_parquet(
        Path(output_path_version) / "data.parquet",
        index=False,
    )
    print("data saved")

def run_training_data_prep_neonatal(
    output_root: str,
    data_source_type: str,
    module: str,
):
    """
    Overall structure of neonatal data prep:
    1. Load and format neonatal data from DEM_BR module
    2. Extracting and merging wealth dataset
    3. Extract and merging annual climate variables
    4. Extract monthly climate variables for previous months
    5. Merge in monthly climate variables for previous months
    6. Calculate averages over time periods analyzed for monthly climate variables
    7. Extract monthly relative climate thresholds for previous months
    8. Merge monthly relative climate thresholds with neonatal mortality data
    9. Calculate averages over time periods analyzed for monthly relative climate thresholds
    10. Add in temperature zones
    """

    # Set up logging and versioned output path
    measure_root = Path(output_root) / data_source_type
    os.makedirs(Path(measure_root) / "training_data", exist_ok=True, mode=0o777)
    cm_data = ClimateMalnutritionData(measure_root)
    version = cm_data.new_training_version()
    output_path_version = Path(measure_root) / "training_data" / version
    os.makedirs(
        output_path_version,
        exist_ok=True,
        mode=0o777,
    )

    dataprep_log_path = Path(output_path_version) / "data_prep_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(dataprep_log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Running training data prep for {data_source_type}...")

    logging.info(f"Creating new version stored under version: {version}")

    logging.info("Processing extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    ## 1. Load and format neonatal data from DEM_BR module
    data_raw = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_matched_2025_10_14.parquet"
    )

    logging.info(f"Total rows in concatenated raw data: {len(data_raw):,}")
    logging.info(
        f"Total unique NIDs in concatenated raw data: {data_raw['nid'].nunique():,}"
    )

    df = data_raw.copy()
    df = check_columns(df, module)

    # drop rows with missing key variables
    rows_before_na_drop = len(df)
    key_vars = [
        "nid",
        "psu",
        "birth_year",
        "birth_month",
        "age_month",
        "hh_id",
        "geospatial_id",
        "line_id",
        "lat",
        "long",
        "child_alive",
    ]
    df.dropna(subset=key_vars, inplace=True)
    na_rows_dropped = rows_before_na_drop - len(df)
    logging.info(
        f"Dropped {na_rows_dropped:,} rows with missing key variables: {key_vars}"
    )
    for var in key_vars:
        logging.info(f"- {var}: {data_raw[var].isna().sum():,} missing values")
    logging.info(f"NIDs with incomplete (some NAs) key variables: {key_vars}")
    for var in key_vars:
        logging.info(
            f"-{data_raw[data_raw[var].isna()]["nid"].nunique():,} contains missing {var} values"
        )

    logging.info(f"Total unique NIDs after dropped NA values: {df['nid'].nunique():,}")
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)

    # update variable data types besides hh_id
    int_cols = [
        "nid",
        "psu",
        "birth_year",
        "birth_month",
        "age_month",
        "child_alive",
        "geospatial_id",
    ]
    for col in int_cols:
        print(col)
        df[col] = df[col].astype("int")

    # Apply cleaning function to each group and update hh_id
    df["old_hh_id"] = df["hh_id"]
    df["hh_id"] = df.groupby(["nid", "ihme_loc_id", "psu"], group_keys=False).apply(
        clean_hh_id_subset
    )

    assert len(df[df["hh_id"].isna()]) == len(
        df[df["old_hh_id"].isna()]
    ), "NAs introduced by cleaning"

    df["hh_id"] = df["hh_id"].astype("int")
    df.drop(columns=["old_hh_id"], inplace=True)

    ## 2. Extracting and merging wealth dataset
    dhs_wealth_data_raw = get_DHS_wealth_dataset()
    dhs_wealth_data = dhs_wealth_data_raw.copy()


    # child_mortality use for both child mortality and neonatal.
    # variable name will be changed below
    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "child_mortality")
    # get_ldipc_from_asset_score uses year_start as year variable
    dhs_wealth_data_test = get_ldipc_from_asset_score(
        dhs_wealth_data,
        cm_data,
        asset_score_col="wealth_index_dhs",
        weights_col="hhweight",
        # plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "dhs_plots.pdf",
        ldi_version=LDI_VERSION,
    )

    wealth_nids = set(dhs_wealth_data_test.nid.unique())
    df_nids = set(df.nid.unique())
    common_nids = wealth_nids.intersection(df_nids)

    nid_with_wealth_pc = 100 * len(common_nids) / len(df_nids)
    logging.info(
        f"{nid_with_wealth_pc:.1f}% of df NIDs - {len(common_nids)} out of "
        f"{len(df_nids)} in wealth data NIDs"
    )

    dhs_wealth_data_test = dhs_wealth_data_test.query("nid in @df_nids")

    # Merge data
    df["ihme_loc_id"] = df["ihme_loc_id"].str.replace("KEN_.*", "KEN", regex=True)
    df[["year_start", "year_end"]] = df[["year_start", "year_end"]].astype(int)

    # Find out percent of anemia nids and hh_ids that can be matched in wealth data
    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]

    df_wealth = merge_left_without_inflating(
        df,
        dhs_wealth_data_test.drop(
            columns=["geospatial_id", "strata", "lat", "long", "hhweight"]
        ),
        on=merge_cols,
    )

    merged_percent = len(df_wealth[~df_wealth["ldipc_weighted_no_match"].isna()]) / len(
        df_wealth
    )
    print(f"Merged rows percent: {100*merged_percent:.1f}%")

    # Calculate proportion of NA and filter out nids with too much wealth missingness (bad merges)
    merged_na_props = (
        df_wealth.groupby(["nid"]).ldipc_weighted_no_match.count()
        / df_wealth.groupby(["nid"]).ldipc_weighted_no_match.size()
    )
    merged_nids = merged_na_props[merged_na_props > 0.95].index.to_list()
    df_merged = df_wealth.query("nid in @merged_nids").copy()
    dropped_too_missingness = len(df_wealth) - len(df_merged)
    logging.info(
        f"Dropped {dropped_too_missingness:,} rows from {len(df_wealth):,} due to excessive wealth missingness in NIDs"
    )
    logging.info(f"Total unique NIDs remaining: {df_merged['nid'].nunique():,}")

    # Add the birth-year version of LDI-PC alongside the survey-year one above.
    # It needs both frames: the raw extract to rank households against the whole
    # survey, and df_merged for each child's birth year. col_suffix keeps both
    # sets of columns so they can be compared.
    df_merged = get_ldipc_from_asset_score_by_birth_year(
        dhs_wealth_data,
        cm_data,
        df_merged,
        asset_score_col="wealth_index_dhs",
        weights_col="hhweight",
        ldi_version=LDI_VERSION,
        col_suffix="_birth_year",
    )

    # replace age_month with aod_months for rows with child_alive==0
    df_merged["age_month_original"] = df_merged["age_month"]  # keep copy of original
    df_merged.loc[df_merged.child_alive == 0, "age_month"] = df_merged.loc[
        df_merged.child_alive == 0, "aod_months"
    ]

    # drop data with no age_month
    before_rows = len(df_merged)
    df_merged = df_merged[df_merged["age_month"].notna()]
    logging.info(
        f"Dropped {before_rows - len(df_merged):,} rows with missing age_month or aod_months"
    )

    df_merged["age_month"] = df_merged["age_month"].astype(int)

    # Take out data with invalid lat and long
    before_rows = len(df_merged)
    df_merged = df_merged.dropna(subset=["lat", "long"])
    df_merged = df_merged.query("lat != 0 and long != 0")
    dropped_due_to_coords = before_rows - len(df_merged)
    logging.info(
        f"Dropped {dropped_due_to_coords:,} rows due to invalid lat and long values"
    )

    # NID 275090 is a very long survey in Peru, 2003-2008 that is coded as having
    # multiple year_starts. Removing it.
    # NID 411301 - updated: not in BR data extractions
    problematic_nids = [275090]
    before_rows = len(df_merged)
    df_merged = df_merged.query("nid not in @problematic_nids")
    dropped_problematic_nids = before_rows - len(df_merged)
    logging.info(
        f"Dropped {dropped_problematic_nids:,} rows due to problematic NIDs: {problematic_nids}"
    )

    # save temp files
    df_merged.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/neonatal_mortality_merged_wealth.parquet",
        index=False,
    )

    # read back in if required
    # df_merged = pd.read_parquet(
    #     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/neonatal_mortality_merged_wealth.parquet"
    # )

    ## 3. Extract and merging annual climate variables
    logging.info("Processing climate data...")
    # get climate variables based on birth_year rather than int_year
    climate_vars = get_climate_vars_for_dataframe(df_merged, year_col="birth_year")

    df_climate = merge_left_without_inflating(
        df_merged, climate_vars, on=["birth_year", "lat", "long"]
    )

    logging.info("Adding elevation data...")
    df_climate = get_elevation_for_dataframe(df_climate)
    df_climate = assign_lbd_admin2_location_id(df_climate)

    # save temp files
    df_climate.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/neonatal_mortality_with_climate.parquet",
        index=False,
    )

    # df_climate = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/neonatal_mortality_with_climate.parquet")

    # get unique invidiuals and clean variables
    df_climate["line_id"] = df_climate["line_id"].astype(int)

    df_climate["indv_id"] = (
        df_climate[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
    )
    logging.info(f"{df_climate['indv_id'].nunique():,} unique individuals in data")

    # flip child_alive so 1 = died, 0 = alive for easier interpretation
    df_climate["child_mortality"] = 1 - df_climate["child_alive"]

    df_climate["consumption"] = df_climate["ldipc_weighted_no_match_birth_year"]
    df_climate = df_climate[df_climate["consumption"].notnull()]

    # # collapse by average climate var exposure for each child
    # climate_vars = [
    #     "mean_temperature",
    #     "days_over_30C",
    #     "precipitation_days",
    #     "total_precipitation",
    #     "mean_low_temperature",
    #     "mean_high_temperature",
    #     "relative_humidity",
    #     "days_over_26C",
    #     "days_over_27C",
    #     "days_over_28C",
    #     "days_over_29C",
    #     "days_over_31C",
    #     "days_over_32C",
    #     "days_over_33C",
    #     "elevation",
    # ]

    # Neonatal
    # get the index of the row with the min age_month_at_year_end for each indv_id
    df_min_age = df_climate.copy()
    df_min_age = (
        df_climate.sort_values("age_month")
        .groupby("indv_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # for any child with age_month > 0, set their age_month to 0,
    # child_mortality to 0, and child_alive to 1. This should get true neonatal
    # mortality for all individuals.
    df_min_age["neonatal_mortality"] = df_min_age["child_mortality"]
    df_min_age.loc[df_min_age.age_month > 0, "child_alive"] = 1
    df_min_age.loc[df_min_age.age_month > 0, "neonatal_mortality"] = 0
    df_min_age.loc[df_min_age.age_month > 0, "age_month"] = 0


    # make version of consumption that is per day
    df_min_age["consumption_pd"] = df_min_age["consumption"] / 365

    # make any day over 30C variable binary
    df_min_age["any_days_over_30C"] = np.where(df_min_age["days_over_30C"] > 0, 1, 0)

    # save out neonatal data set
    df_min_age.to_parquet(
        Path(output_path_version) / "neonatal_data_old_climate_vars.parquet"
    )

    ## 4. Extract monthly climate variables for previous months
    # get previous monthly climate variables
    climate_vars_da = get_all_climate_vars_year_months_for_latlongs(df_min_age)

    # save temporary file
    climate_vars_da.to_netcdf(
        Path(output_path_version) / "climate_vars_for_locs_with_orig.nc"
    )
    print("climate vars extracted successfully")

    # read back in if necessary
    # climate_vars_da = xr.open_dataarray(Path(output_path_version) / "climate_vars_for_locs.nc")
    climate_vars_df = climate_vars_da.to_dataframe().reset_index()

    # save temporary file
    climate_vars_df.to_parquet(
        Path(output_path_version) / "climate_vars_for_locs.parquet"
    )

    # read back in if necessary
    # climate_vars_df = pd.read_parquet(
    #     Path(output_path_version) / "climate_vars_for_locs.parquet"
    # )
    climate_vars_df.drop(columns=["point"], inplace=True)

    climate_vars_df.rename(
        columns={
            "latitude": "lat_matched",
            "longitude": "long_matched",
            "lat_orig": "lat",
            "long_orig": "long",
            "year": "lookup_year",
            "month": "lookup_month",
        },
        inplace=True,
    )

    ## 5. Merge in monthly climate variables for previous months
    # Apply get_lookup_months to each row and concatenate the results into a new DataFrame
    df_min_age_unique_yr_mo = df_min_age[
        ["birth_year", "birth_month"]
    ].drop_duplicates()

    lookup_df = pd.concat(
        df_min_age_unique_yr_mo.apply(
            lambda row: get_lookup_months(row["birth_year"], row["birth_month"]), axis=1
        ).to_list(),
        ignore_index=True,
    )
    lookup_df.rename(
        columns={"year": "lookup_year", "month": "lookup_month"}, inplace=True
    )

    df_min_age_lookup = pd.merge(
        df_min_age, lookup_df, on=["birth_year", "birth_month"], how="left"
    )

    var_names = [
        "mean_temperature",
        # "precipitation_days",
        "total_precipitation",
        # "mean_low_temperature",
        # "mean_high_temperature",
        # "relative_humidity",
        "days_over_24C",
        "days_over_25C",
        "days_over_26C",
        "days_over_27C",
        "days_over_28C",
        "days_over_29C",
        "days_over_30C",
        "days_over_31C",
        "days_over_32C",
        # "days_over_33C",
    ]
    df_results = []
    for v in tqdm(var_names):
        climate_v = climate_vars_df.query("climate_var == @v").copy()

        df_merged_tmp = df_min_age_lookup.merge(
            climate_v,
            on=["lookup_year", "lookup_month", "lat", "long"],
            how="left",
        )
        df_merged_tmp = df_merged_tmp[
            [
                "birth_year",
                "birth_month",
                "lat",
                "long",
                "lookup_year",
                "lookup_month",
                "suffix",
                "climate_var",
                "long_matched",
                "lat_matched",
                "value",
            ]
        ]

        df_merged_tmp["climate_var_suffix"] = (
            df_merged_tmp["climate_var"] + "_" + df_merged_tmp["suffix"]
        )
        df_merged_tmp.drop(
            columns=["lookup_year", "lookup_month", "climate_var", "suffix"],
            inplace=True,
        )
        df_results.append(df_merged_tmp)

    df_results_concated = pd.concat(df_results, ignore_index=True)
    df_results_concated.to_parquet(
        Path(output_path_version) / "climate_vars_merged_long.parquet"
    )

    df_merged_tmp_wide = df_results_concated.pivot_table(
        index=[
            "birth_year",
            "birth_month",
            "lat",
            "long",
            "long_matched",
            "lat_matched",
        ],
        columns="climate_var_suffix",
        values="value",
    ).reset_index()
    df_merged_tmp_wide.to_parquet(
        Path(output_path_version) / "climate_vars_merged_wide.parquet"
    )

    df_min_age_updated = merge_left_without_inflating(
        df_min_age, df_merged_tmp_wide, on=["birth_year", "birth_month", "lat", "long"]
    )

    df_min_age_updated.to_parquet(
        Path(output_path_version) / "neonatal_merged_all_prev_months.parquet"
    )

    # read back in if necessary
    # df_min_age_updated = pd.read_parquet(
    #     Path(output_path_version) / "neonatal_merged_all_prev_months.parquet"
    # )

    ## 6. Calculate averages over time periods analyzed for monthly climate variables
    for v in tqdm(var_names):
        for i in [3, 6, 9]:
            prev_vars = get_prev_climate_var_months(v, i)
            df_min_age_updated[f"{v}_prev_{i}_mo_avg"] = df_min_age_updated[
                prev_vars
            ].mean(axis=1)

    df_min_age_updated.to_parquet(
        Path(output_path_version) / "neonatal_data_prev_month_vars.parquet"
    )
    # df_min_age_updated = pd.read_parquet(
    #     Path(output_path_version) / "neonatal_data_prev_month_vars.parquet"
    # )

    ## 7. Extract monthly relative climate thresholds for previous months

    # add index for easier merging
    df_min_age_updated = df_min_age_updated.reset_index()

    # get previous monthly climate variables, initially as xarray
    climate_vars = get_all_climate_thresholds_year_months_for_latlongs(
        df_min_age_updated
    )
    # save temp copy
    climate_vars.to_netcdf(Path(output_path_version) / "climate_thresholds.nc")

    # format climate_vars df
    climate_vars_df = climate_vars.to_dataframe().reset_index()
    climate_vars_df.drop(columns=["point"], inplace=True)

    climate_vars_df.rename(
        columns={
            "year": "lookup_year",
            "month": "lookup_month",
        },
        inplace=True,
    )
    # create quantile string variable that eventually become column names
    climate_vars_df["quantile_str"] = (
        climate_vars_df["quantile"].astype(str).str.replace("0.", "q")
    )
    climate_vars_df.drop(columns=["quantile"], inplace=True)

    climate_vars_df.rename(
        columns={
            "lat_orig": "lat",
            "long_orig": "long",
        },
        inplace=True,
    )
    # save temp copy
    climate_vars_df.to_parquet(
        Path(output_path_version) / "climate_thresholds_for_locs.parquet"
    )
    # read back in if necessary
    # climate_vars_df = pd.read_parquet(
    #     Path(output_path_version) / "climate_thresholds_for_locs.parquet"
    # )

    # Apply get_lookup_months to each row and concatenate the results into a new DataFrame
    df_min_age_unique_yr_mo = df_min_age_updated[
        ["birth_year", "birth_month"]
    ].drop_duplicates()

    lookup_df = pd.concat(
        df_min_age_unique_yr_mo.apply(
            lambda row: get_lookup_months(row["birth_year"], row["birth_month"]), axis=1
        ).to_list(),
        ignore_index=True,
    )
    lookup_df.rename(
        columns={"year": "lookup_year", "month": "lookup_month"}, inplace=True
    )

    # note: this is 10x as large as original df, as each row has 10 months of lookups
    df_min_age_lookup = pd.merge(
        df_min_age_updated, lookup_df, on=["birth_year", "birth_month"], how="left"
    )
    # save temp copy
    df_min_age_lookup.to_parquet(
        Path(output_path_version) / "df_min_age_lookup.parquet"
    )

    ## 8. Merge monthly relative climate thresholds with neonatal mortality data
    # reduce size
    climate_vars_df = climate_vars_df[
        climate_vars_df["quantile_str"].isin(["q75", "q8", "q85", "q9", "q95", "q99"])
    ]

    temp_save_loc = Path(output_path_version) / "merge_chunks"
    os.makedirs(temp_save_loc, exist_ok=True, mode=0o777)

    # loop over each month to reduce space complexity when pivoting wide
    for pre_suf in tqdm(df_min_age_lookup["suffix"].unique()):

        data_chunk = df_min_age_lookup[df_min_age_lookup["suffix"] == pre_suf].copy()

        df_chunk_merged = data_chunk.merge(
            climate_vars_df,
            on=["lookup_year", "lookup_month", "lat", "long"],
            how="left",
        )

        # pivot wide
        df_chunk_merged["quantile_str"] = (
            df_chunk_merged["quantile_str"] + "_" + df_chunk_merged["suffix"]
        )
        df_chunk_merged.drop(
            columns=["lookup_year", "lookup_month", "suffix", "longitude", "latitude"],
            inplace=True,
        )

        merge_cols = ["index", "birth_year", "birth_month", "lat", "long"]
        df_chunk_wide = (
            df_chunk_merged[merge_cols + ["quantile_str", "value"]]
            .pivot_table(
                index=merge_cols,
                columns="quantile_str",
                values="value",
            )
            .reset_index()
        )

        quantile_vars = (
            df_chunk_merged[~df_chunk_merged["quantile_str"].isna()]["quantile_str"]
            .unique()
            .tolist()
        )
        # merge onto original using index variable
        df_min_age_updated = df_min_age_updated.merge(
            df_chunk_wide[["index"] + quantile_vars], on="index", how="left"
        )

    # save temp copy
    df_min_age_updated.to_parquet(
        Path(output_path_version) / "neonatal_merged_all_thresholds.parquet"
    )

    ## 9. Calculate averages over time periods analyzed for monthly relative climate thresholds
    for v in tqdm([75, 8, 85, 9, 95, 99]):
        for i in [3, 6, 9]:
            prev_vars = get_prev_climate_threshold_months(v, i)
            df_min_age_updated[f"q{v}_prev_{i}_mo_avg"] = df_min_age_updated[
                prev_vars
            ].mean(axis=1)

    # save temp copy
    df_min_age_updated.to_parquet(
        Path(output_path_version) / "neonatal_threshold_averages.parquet"
    )

    ## 10. Add in temperature zones
    # (small enough not to require parallelization)
    lats = xr.DataArray(df_min_age_updated["lat"], dims="point")
    lons = xr.DataArray(df_min_age_updated["long"], dims="point")
    return_arrays = []
    zone_dir = "/mnt/share/erf/climate_downscale/results/annual/raw/historical/10_year_mean_temperature"

    for f in tqdm(os.listdir(zone_dir)):
        climate_da = xr.open_dataarray(
            f"{zone_dir}/{f}",
        )
        climate_da = climate_da.load()
        # Select nearest latitude and longitude
        climate_da = climate_da.sel(latitude=lats, longitude=lons, method="nearest")
        # Add original latitude and longitude as coordinates
        climate_da = climate_da.assign_coords(lat_orig=("point", lats.values))
        climate_da = climate_da.assign_coords(long_orig=("point", lons.values))

        # Drop the "point" dimension if not needed
        climate_da = climate_da.drop_vars("point")
        return_arrays.append(climate_da)

    # Concatenate all climate variables along the "climate_var" dimension
    zone_da = xr.concat(return_arrays, dim="last_year")
    zone_df = zone_da.to_dataframe().reset_index()
    zone_df.rename(
        columns={
            "value": "zone",
            "last_year": "birth_year",
            "lat_orig": "lat",
            "long_orig": "long",
        },
        inplace=True,
    )
    zone_df.drop(columns=["latitude", "longitude", "point"], inplace=True)
    zone_df["zone"] = zone_df["zone"].astype(int)
    zone_df = zone_df[(zone_df["zone"] > 5) & (zone_df["zone"] < 30)]
    zone_df = zone_df.drop_duplicates()
    df_min_age_final = df_min_age_updated.merge(
        zone_df, on=["birth_year", "lat", "long"], how="left"
    )
    # drop nas for any new variables
    new_vars = [
        "q9_prev_9_mo",
        "q95_prev_9_mo",
        "q9_prev_3_mo_avg",
        "q9_prev_6_mo_avg",
        "q9_prev_9_mo_avg",
        "q95_prev_3_mo_avg",
        "q95_prev_6_mo_avg",
        "q95_prev_9_mo_avg",
        "q99_prev_9_mo_avg",
        "q99_prev_3_mo_avg",
        "q99_prev_6_mo_avg",
        "q99_prev_9_mo_avg",
        # "zone",
    ]
    df_min_age_final = df_min_age_final.dropna(subset=new_vars)

    # drop rows with NAs for any other import vars
    no_na_vars = ["sex_id", "ihme_loc_id", "birth_year", "consumption_pd"]
    df_min_age_final = df_min_age_final.dropna(subset=no_na_vars)

    # Add variables required for validation against gbd: year_id and age_group_id
    df_min_age_final["year_id"] = df_min_age_final["birth_year"]
    df_min_age_final["age_group_id"] = 42

    df_min_age_final["int_birth_year_diff_months"] = df_min_age_final[
        "age_month_original"
    ]

    # ---- post-processing, ported from neonatal_data_checks_edits.ipynb ------
    # Births further than this many months before the interview are dropped
    # from data.parquet, but kept in data_no_10_yr_cutoff.parquet.  Inclusive.
    max_recall_months = 60
    # Columns the model cannot tolerate nulls in.
    cols_to_check = [
        "neonatal_mortality",
        "consumption_pd",
        "int_birth_year_diff_months",
        "birth_year",
        "days_over_30C_prev_0_mo",
        "total_precipitation_prev_0_mo",
    ]

    # National median consumption from the RT income distributions rather than
    # the survey households: the reference scenario at the 50th population
    # percentile, divided to per-day so it sits on the same scale as the
    # survey-derived consumption_pd.  Uses the same LDI_VERSION as the
    # household matching above, so both read one income vintage.
    # Built from measure_root rather than reusing `cm_data`: that name is
    # rebound to child_mortality under DEFAULT_ROOT further up, so it no longer
    # tracks this run's output_root.
    rt_consumption_df = ClimateMalnutritionData(measure_root).load_ldi_distributions(
        geospecificity="national", version=LDI_VERSION
    )
    rt_consumption_df = rt_consumption_df[
        (rt_consumption_df["scenario"] == "reference")
        & (rt_consumption_df["population_percentile"] == 0.5)
    ][["ihme_loc_id", "year_id", "location_id", "ldipc"]].rename(
        columns={"year_id": "birth_year", "ldipc": "consumption_median_RT"}
    )
    df_min_age_final = df_min_age_final.drop(
        columns="consumption_median_RT", errors="ignore"
    ).merge(
        rt_consumption_df,
        on=["ihme_loc_id", "location_id", "birth_year"],
        how="left",
    )
    n_missing_rt = int(df_min_age_final["consumption_median_RT"].isna().sum())
    if n_missing_rt:
        msg = (
            f"{n_missing_rt:,} rows have no {LDI_VERSION} national "
            "median consumption after the merge; the LDI series is missing some "
            "ihme_loc_id/location_id/birth_year present in the survey data."
        )
        raise ValueError(msg)
    df_min_age_final["consumption_median_RT"] /= 365

    # Every column the model needs must be complete.  Checked before the recall
    # cutoff, so it holds for both saved versions.
    missing_counts = {
        col: int(df_min_age_final[col].isna().sum())
        for col in cols_to_check
    }
    for col, n_missing in missing_counts.items():
        print(f"{col}: {n_missing} missing values")
    incomplete = {col: n for col, n in missing_counts.items() if n}
    if incomplete:
        msg = f"Missing values in required neonatal columns: {incomplete}"
        raise ValueError(msg)

    # Save the uncut extract, then the recall-limited one the models train on.
    # index=False on both: the frame has been filtered repeatedly, so its index
    # is non-contiguous and would otherwise be written as a column.
    df_min_age_final.to_parquet(
        Path(output_path_version) / "data_no_10_yr_cutoff.parquet", index=False
    )
    print(f"data_no_10_yr_cutoff.parquet saved ({len(df_min_age_final):,} rows)")

    df_min_age_final = df_min_age_final[
        df_min_age_final["int_birth_year_diff_months"] <= max_recall_months
    ]
    df_min_age_final.to_parquet(
        Path(output_path_version) / "data.parquet", index=False
    )
    print(
        f"data.parquet saved ({len(df_min_age_final):,} rows, "
        f"int_birth_year_diff_months <= {max_recall_months})"
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_source_type()
@click.option("--module", default=None, help="Specify the module (e.g., 'dem_br').")
def run_training_data_prep_task(
    output_root: str, source_type: str, module: str = None
) -> None:
    """Run training data prep for one source type (a single jobmon task)."""
    print(f"Running training data prep for {source_type}...")
    run_training_data_prep_main(output_root, source_type, module=module)


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_source_type(allow_all=True)
@click.option("--module", default=None, help="Specify the module (e.g., 'dem_br').")
@clio.with_queue()
def run_training_data_prep(
    output_root: str, source_type: str | list[str], queue: str, module: str = None
) -> None:
    """Launch training data prep on the cluster via jobmon."""
    source_types = [source_type] if isinstance(source_type, str) else list(source_type)

    # Only forward --module when set; jobmon builds a literal command string, so
    # a None here would be passed through as the string "None".
    task_args: dict[str, str] = {"output-root": str(output_root)}
    if module is not None:
        task_args["module"] = module

    jobmon.run_parallel(
        runner="sttask",
        task_name="data_prep",
        node_args={"source-type": source_types},
        task_args=task_args,
        task_resources={
            "queue": queue,
            # the prep functions fan out over mp.Pool(processes=25)
            "cores": 26,
            "memory": "900Gb",
            "runtime": "72h",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(output_root),
    )




def run_training_data_prep_main(  # noqa: PLR0915
    output_root: str | Path,
    data_source_type: str,
    module: str = None,
) -> None:
    if isinstance(data_source_type, list) and len(data_source_type) == 1:
        data_source_type = data_source_type[0]

    if data_source_type == "cgf":
        run_training_data_prep_cgf(output_root, data_source_type)
    elif data_source_type == "anemia":
        run_training_data_prep_anemia(output_root, data_source_type)
    elif data_source_type == "child_mortality":
        run_training_data_prep_child_mortality_monthly(
            output_root, data_source_type, module=module
        )
    elif data_source_type == "neonatal_mortality":
        run_training_data_prep_neonatal(output_root, data_source_type, module=module)
    # elif data_source_type == "neonatal_mortality":
    #     quick_fix_update_neonatal(
    #         output_root,
    #         data_source_type,
    #     )
    else:
        msg = f"Data source {data_source_type} not implemented yet."
        raise NotImplementedError(msg)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "resume_child_mortality_from_parquet":
        resume_child_mortality_from_parquet(sys.argv[2])
    else:
        run_training_data_prep()
