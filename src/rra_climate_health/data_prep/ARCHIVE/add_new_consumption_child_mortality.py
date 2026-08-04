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
) -> pd.DataFrame:
    """
    df = df_min_age.copy()
    lat_col = "lat"
    long_col= "long"
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

    min_year = df["birth_year"].min() - 1  # need prior year
    max_year = df["birth_year"].max()

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
) -> pd.DataFrame:
    """
    df = df_min_age.copy()
    lat_col = "lat"
    long_col= "long"
    """

    unique_coords = df[[lat_col, long_col]].drop_duplicates()

    min_year = df["birth_year"].min() - 1  # need prior year
    max_year = df["birth_year"].max()

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
    df["wealth_index_dhs"] = df["wealth_index_dhs"].astype(int)

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


## Set constants manually
output_root = DEFAULT_ROOT
data_source_type = "child_mortality"
module = "dem_br"


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

    data_raw = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_02_27.01/data.parquet"
    )

    # Prepping wealth dataset
    dhs_wealth_data_raw = get_DHS_wealth_dataset()
    dhs_wealth_data = dhs_wealth_data_raw.copy()

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

    # Merge data

    # fix df columns
    data_raw.drop(
        columns=[
            "unweighted_population_percentile",
            "cum_weight",
            "weighted_population_percentile",
            "ldipc_unweighted_no_match",
            "ldipc_weighted_no_match",
            "ldipc_unweighted_match",
            "ldipc_weighted_match",
        ],
        inplace=True,
    )

    df_wealth = merge_left_without_inflating(
        data_raw,
        dhs_wealth_data_test.drop(
            columns=["geospatial_id", "strata", "lat", "long", "hhweight"]
        ),
        on=merge_cols,
    )

    df_wealth.to_parquet(
        output_path_version / "child_mortality_with_wealth.parquet", index=False
    )

    # Update exploded data set
    data_exploded = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_03_16.01/data.parquet"
    )

    data_exploded.drop(
        columns=[
            "unweighted_population_percentile",
            "cum_weight",
            "weighted_population_percentile",
            "ldipc_unweighted_no_match",
            "ldipc_weighted_no_match",
            "ldipc_unweighted_match",
            "ldipc_weighted_match",
        ],
        inplace=True,
    )

    data_exploded_wealth = merge_left_without_inflating(
        data_exploded,
        dhs_wealth_data_test.drop(
            columns=["geospatial_id", "strata", "lat", "long", "hhweight"]
        ),
        on=merge_cols,
    )

    data_exploded_wealth.to_parquet(
        output_path_version / "child_mortality_exploded_updated_wealth.parquet",
        index=False,
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

    # save temp merged:
    df_merged.to_parquet(
        "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_tmp_wealth_merged.parquet",
        index=False,
    )

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
    df_exploded["age_month"] += 1

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

    # avg number of months per individual
    avg_months_per_indv = len(df_climate) / df_climate["indv_id"].nunique()
    print(f"Average number of months per individual: {avg_months_per_indv:.2f}")

    # flip child_alive so 1 = died, 0 = alive for easier interpretation
    df_climate["child_mortality"] = 1 - df_climate["child_alive"]

    df_climate["consumption"] = df_climate["ldipc_weighted_no_match"]

    # make version of consumption that is per day
    df_climate["consumption_pd"] = df_climate["consumption"] / 365

    # make any day over 30C variable binary
    df_climate["any_days_over_30C"] = np.where(df_climate["days_over_30C"] > 0, 1, 0)

    # Write to output
    df_climate.to_parquet(Path(output_path_version) / "data.parquet", index=False)

    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = df_climate[df_climate[measure].notna()].copy()
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
