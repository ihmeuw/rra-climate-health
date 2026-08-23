LDI_VERSION = "v8"
import multiprocessing as mp
import multiprocessing
import tqdm
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
import sys
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
from scipy.interpolate import PchipInterpolator


import rra_climate_health.cli_options as clio
from rra_climate_health import paths
from rra_climate_health.data_prep import upstream_paths
from rra_climate_health.data import (
    DEFAULT_ROOT,
    ClimateMalnutritionData,
)

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
    "lbw":  EXTRACTIONS_ROOT / "lbwsg" / "2025_10_30_imputed.parquet"
}

DATA_SOURCE_TYPE = {
    "stunting": "cgf",
    "wasting": "cgf",
    "underweight": "cgf",
    "low_adult_bmi": "bmi",
    "anemia": "anemia",
    "lbw": "lbw",
}
MEASURES_IN_SOURCE = {
    "cgf": ["stunting", "wasting", "underweight"],
    "bmi": ["low_adult_bmi"],
    "anemia": ["anemia"],
    "child_mortality": ["child_alive"],
    "lbw": ["low_birth_weight"],
}


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
            f"/mnt/share/erf/climate_downscale/results/_submission/annual/raw/historical/{climate_variable}/{yr}_era5.nc"
        )["value"]
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
        "days_over_30C",
        "precipitation_days",
        "total_precipitation",
        "mean_low_temperature",
        "mean_high_temperature",
        "relative_humidity",
        "days_over_26C",
        "days_over_27C",
        "days_over_28C",
        "days_over_29C",
        "days_over_31C",
        "days_over_32C",
        "days_over_33C",
    ]

    unique_coords = df[[lat_col, long_col, year_col]].drop_duplicates()

    df_splits = [year_df for _, year_df in unique_coords.groupby(year_col)]
    p = mp.Pool(processes=25)
    results_df = pd.concat(
        p.map(
            partial(get_climate_vars_for_year, climate_variables=var_names, year_col = year_col, lat_col = lat_col, long_col = long_col), df_splits
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
        elif 4.5 in ldi.scenario.unique():
            ldi = ldi.loc[ldi["scenario"] == 4.5].drop("scenario", axis=1)
        elif 'reference' in ldi.scenario.unique():
            ldi = ldi.loc[ldi["scenario"] == 'reference'].drop("scenario", axis=1)
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
    # Remove ".0" if present at the end
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned


def clean_hh_id_v2(row):
    hh_id = row["hh_id"]
    geo_str = str(row["geospatial_id"])

    if pd.isna(hh_id):
        return hh_id

    # Convert to string and trim leading/trailing whitespace
    hh_id = str(hh_id).strip()

    # Replace multiple spaces with a single space
    hh_id = re.sub(r"\s{2,}", " ", hh_id)

    # If the hh_id is already clean (no spaces or underscores), return it
    if (len(re.split(r"[_ ]", hh_id)) == 1) and not (hh_id.startswith("0")):
        return float(hh_id)

    # Handle cases with spaces or underscores
    if " " in hh_id:
        hh_id = hh_id.split(" ")[-1]
    elif "_" in hh_id:
        hh_id = hh_id.split("_")[-1]

    # Match and remove leading zeros followed by the geospatial_id
    pattern = r"^0+" + re.escape(geo_str)
    hh_id = re.sub(pattern, "", hh_id)

    # Strip remaining leading zeros and handle empty results
    hh_id = hh_id.lstrip("0") or "0"

    return float(hh_id)  # Return as float to handle NAs


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



output_root = DEFAULT_ROOT
data_source_type = "lbw"
using_source_as_wealth = True
geo = pd.read_parquet('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/lbwsg/tentative_geocoding.parquet')
imp_n = 1
# # Set up logging and versioned output path
# measure_root = Path(output_root) / data_source_type
# os.makedirs(Path(measure_root) / "training_data", exist_ok=True, mode=0o777)
# cm_data = ClimateMalnutritionData(measure_root)
# version = cm_data.new_training_version()
# output_path_version = Path(measure_root) / "training_data" / version
# os.makedirs(
#     output_path_version,
#     exist_ok=True,
#     mode=0o777,
# )

# Set up logging
# dataprep_log_path = Path(output_path_version) / "data_prep_log.txt"

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(message)s",
#     handlers=[
#         logging.FileHandler(dataprep_log_path, mode="w"),
#         logging.StreamHandler(),
#     ],
# )

survey_data_path = SURVEY_DATA_PATHS[data_source_type]
logging.info(f"Running training data prep for {data_source_type}...")

logging.info(f"Creating new version stored under version:")

logging.info("Processing extraction survey data...")
loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)


#lbw_data_raw = pd.read_parquet(survey_data_path)
imp_df = pd.read_csv('/mnt/share/mnch/lbwsg/data/processing/imputation/imputed/imputed_dataset1.csv')
imp_df['geospatial_id'] = imp_df['geospatial_id'].astype(int)
imp_df['ihme_loc_id_original'] = imp_df['ihme_loc_id']
imp_df['ihme_loc_id'] = imp_df['ihme_loc_id'].str[:3]
geo['geospatial_id'] = geo['geospatial_id'].astype(int)
merged_df_3 = imp_df.merge(geo.rename(columns={'iso3':'ihme_loc_id'}), on=['nid', 'geospatial_id', 'ihme_loc_id'], how='left', validate='m:1', indicator=True)
merged_df_3['lat_present'] = ~merged_df_3['lat'].isna()
merged_df_3['long_present'] = ~merged_df_3['long'].isna()
merged_df_3['both_present'] = merged_df_3['lat_present'] & merged_df_3['long_present']
good_geo_nids = merged_df_3.groupby('nid').both_present.mean().reset_index().query('both_present > 0.85').nid.unique()
merged_df_3 = merged_df_3[merged_df_3['nid'].isin(good_geo_nids)]
merged_df_3['psu'] = merged_df_3['psu'].astype(str)
merged_df_3['strata'] = merged_df_3['strata'].astype(str)
lbw_data_raw = merged_df_3.drop(columns=['latitude', 'longitude', 'point'])#.to_parquet('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/lbwsg/2025_10_30_imputed.parquet', index=False)


#measure_df.query("birth_weight.notnull()")[['birth_weight', 'lbw']].head()
# TODO implement cutoffs for too low and too high values of birth_weight
LOW_BIRTH_WEIGHT_CUTOFF = 2500  # grams
if lbw_data_raw.birth_weight.median() > 1000:
    print("birth_weight is in grams")
    lbw_data_raw['lbw'] = lbw_data_raw['birth_weight'] < LOW_BIRTH_WEIGHT_CUTOFF
else: # birth_weight is in kg
    print("birth_weight is in kg")
    lbw_data_raw['lbw'] = (lbw_data_raw['birth_weight']*1000) < LOW_BIRTH_WEIGHT_CUTOFF


if using_source_as_wealth:
    logging.info("Using source data as wealth data...")
    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "lbw")
    wealth_data = lbw_data_raw.groupby(['nid', 'ihme_loc_id', 'year_start', 'psu', 'strata', 'wealth_factor']).size().reset_index()
    wealth_data = wealth_data.rename(columns={'wealth_factor': 'wealth_index_dhs'})
    wealth_data['weights'] = 1
    wealth_data = get_ldipc_from_asset_score(
        wealth_data, cm_data, asset_score_col="wealth_index_dhs", weights_col="weights",
        ldi_version=LDI_VERSION, plot=False)

#Prepping wealth dataset
dhs_wealth_data_raw = get_DHS_wealth_dataset()
dhs_wealth_data = dhs_wealth_data_raw.copy()

cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "lbw")
dhs_wealth_data = get_ldipc_from_asset_score(
    dhs_wealth_data,
    cm_data,
    asset_score_col="wealth_index_dhs",
    weights_col="hhweight",
    ldi_version=LDI_VERSION,
)
dhs_wealth_data.to_parquet(f"/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/ldi/{LDI_VERSION}/wealth_distribution_matched.parquet", index=False)

# dhs_wealth_data = pd.read_parquet(f"/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/ldi/{LDI_VERSION}/wealth_distribution_matched.parquet")


#newer_lbw_data_raw = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/lbwsg/2025_10_20.parquet")[0:-1]
newer_lbw_data_raw = pd.read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/lbwsg/2025_11_25_NewerVictorExt.parquet")

# Prefer the imputed dataset where there are overlaps in NIDs
newer_lbw_data = newer_lbw_data_raw[~newer_lbw_data_raw['nid'].astype(int).isin(lbw_data_raw['nid'].unique())].copy()
for column in ['nid']:
    newer_lbw_data[column] = newer_lbw_data[column].astype(int)
#dhs_wealth_data = pd.read_parquet(f"/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/ldi/{LDI_VERSION}/wealth_distribution_matched.parquet")
newer_lbw_data = newer_lbw_data[newer_lbw_data['nid'] != 408226]  # remove MICS NID with no PSU info
# drop rows where birth_weight is null
newer_lbw_data = newer_lbw_data[newer_lbw_data['birth_weight'].notna()]
# take away unknowns and transform to kg
newer_lbw_data = newer_lbw_data[(newer_lbw_data['birth_weight'] <= 9000)]
newer_lbw_data['birth_weight'] = newer_lbw_data['birth_weight'] / 1000.0

newer_lbw_data.rename(columns={'wealth_index_dhs':'existing_wealth_index_dhs'}, inplace=True)
#lbw_df.query("wealth_index_dhs_x != wealth_index_dhs_y")
dhs_wealth_data.query("wealth_index_dhs.isnull()")
newer_lbw_data = newer_lbw_data.drop(columns=["psu_id", "strata_id" ], errors='ignore')
newer_lbw_data = newer_lbw_data.rename(columns=COLUMN_NAME_TRANSLATOR)

# Apply cleaning function to each group and update hh_id
newer_lbw_data["old_hh_id"] = newer_lbw_data["hh_id"]
newer_lbw_data["hh_id"] = newer_lbw_data["old_hh_id"].str.split(r"[_ ]").str[-1]
newer_lbw_data["hh_id"] = newer_lbw_data.apply(clean_hh_id, axis=1)


missing_hh_rows = newer_lbw_data[newer_lbw_data["hh_id"].isna()]
print(
    f"Dropping {len(missing_hh_rows)} rows from lbw data with missing hh_id"
)
newer_lbw_data = newer_lbw_data[newer_lbw_data["hh_id"].notna()]
newer_lbw_data["hh_id"] = newer_lbw_data["hh_id"].astype(int)

# Find out percent of lbw nids and hh_ids that can be matched in wealth data
merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]

dhs_wealth_data["hh_id"] = dhs_wealth_data["hh_id"].astype(int)
dhs_wealth_data["psu"] = dhs_wealth_data["psu"].astype(int)

wealth_nids = set(dhs_wealth_data.nid.unique())
lbw_nids = set(newer_lbw_data.nid.unique())
common_nids = wealth_nids.intersection(lbw_nids)

nid_with_wealth_pc = 100 * len(common_nids) / len(lbw_nids)
print(
    f"{nid_with_wealth_pc:.1f}% of lbw NIDs - {len(common_nids)} out of "
    f"{len(lbw_nids)} in wealth data NIDs"
)

dhs_wealth_data = dhs_wealth_data.query("nid in @lbw_nids")

newer_lbw_data = newer_lbw_data.drop(columns=["old_hh_id"])


# Merge data
lbw_data_wealth = newer_lbw_data.merge(dhs_wealth_data.drop(columns=["geospatial_id", "strata", "lat", "long"]), 
                     on=merge_cols, how='left', validate='m:1')

# Calculate proportion of NA and filter out nids with too much wealth missingness (bad merges)
merged_na_props = (
    lbw_data_wealth.groupby(["nid"]).ldipc_weighted_no_match.count()
    / lbw_data_wealth.groupby(["nid"]).ldipc_weighted_no_match.size()
)
merged_nids = merged_na_props[merged_na_props > 0.90].index.to_list()
newer_lbw_df = lbw_data_wealth.query("nid in @merged_nids").copy()
dropped_too_missingness = len(lbw_data_wealth) - len(newer_lbw_df)
print(
    f"Dropped {dropped_too_missingness:,} rows from {len(lbw_data_wealth):,} due to excessive missingness"
)

# drop other unmerged
unmergable_rows = newer_lbw_df[newer_lbw_df["wealth_index_dhs"].isna()]
newer_lbw_df = newer_lbw_df[newer_lbw_df["wealth_index_dhs"].notna()]

# Take out data with invalid lat and long
before_rows = len(newer_lbw_df)
newer_lbw_df = newer_lbw_df.dropna(subset=["lat", "long"])
newer_lbw_df = newer_lbw_df.query("lat != 0 and long != 0")
dropped_due_to_coords = before_rows - len(newer_lbw_df)
print(
    f"Dropped {dropped_due_to_coords:,} rows due to invalid lat and long values"
)
newer_lbw_df['ldi_pc_pd'] = newer_lbw_df['ldipc_weighted_no_match'] / 365


wealth_data = wealth_data.rename(columns={'wealth_index_dhs': 'wealth_factor'})
wealth_data['ldi_pc_pd'] = wealth_data['ldipc_unweighted_no_match'] / 365

lbw_data = lbw_data_raw.copy().drop(columns=['_merge'])

older_lbw_df = lbw_data.merge(wealth_data, on=['nid', 'ihme_loc_id', 'year_start', 'psu', 'strata', 'wealth_factor'], how='left', validate='many_to_one', indicator=True)


older_lbw_df['gbd_source'] = 'gbd20'
newer_lbw_df['gbd_source'] = 'gbd23'
lbw_df = older_lbw_df.copy() #pd.concat([older_lbw_df, newer_lbw_df])

# drop other unmerged
unmergable_rows = lbw_df[lbw_df["ldi_pc_pd"].isna()]
lbw_df = lbw_df[lbw_df["ldi_pc_pd"].notna()]

# Take out data with invalid lat and long
before_rows = len(lbw_df)
lbw_df = lbw_df.dropna(subset=["lat", "long"])
lbw_df = lbw_df.query("lat != 0 and long != 0")
dropped_due_to_coords = before_rows - len(lbw_df)
print(
    f"Dropped {dropped_due_to_coords:,} rows due to invalid lat and long values"
)

print(lbw_df.nid.nunique())
# show all columns in dataframe in notebook
# By nid, proportion of rows that have birth_weight missing
birth_weight_na_props = (
    lbw_df.groupby(["nid"]).birth_weight.count()
    / lbw_df.groupby(["nid"]).birth_weight.size()
)
birth_weight_na_props
# birth_weight_nids = birth_weight_na_props[birth_weight_na_props > 0.95].index.to_list()
# lbw_df = lbw_df.query("nid in @birth_weight_nids").copy()
# dropped_due_to_birth_weight = len(lbw_df) - len(lbw_df)
# print(
#     f"Dropped {dropped_due_to_birth_weight:,} rows due to excessive birth weight missingness"
# )

lbw_filtered = lbw_df.loc[lbw_df.birth_weight.notna()].copy()
#lbw_filtered['lbw'] = lbw_filtered['birth_weight'] < LOW_BIRTH_WEIGHT_CUTOFF #done at beginning
# deal with interview month missingness by imputing the mode for that NID
nids_with_missing_int_month = lbw_filtered[lbw_filtered['int_month'].isna()].nid.unique()
for nid in nids_with_missing_int_month:
    mode_month = lbw_filtered.query("nid == @nid and int_month.notnull()").int_month.mode()[0]
    lbw_filtered.loc[(lbw_filtered['nid'] == nid) & (lbw_filtered['int_month'].isna()), 'int_month'] = mode_month
lbw_filtered['birth_year'] = ((lbw_filtered['int_year'] * 12) + lbw_filtered['int_month'] - lbw_filtered['child_age'])  // 12
lbw_filtered['birth_month'] = ((lbw_filtered['int_year'] * 12) + lbw_filtered['int_month'] - lbw_filtered['child_age']) %12

for col in ['int_year', 'year_start', 'year_end', 'birth_year', 'birth_month']:
    lbw_filtered[col] = lbw_filtered[col].astype(int)
newer_lbw_df.columns
# Climate monthly
CLIMATE_MONTHLY_PATH = Path('/mnt/share/erf/climate_downscale/results/monthly/raw/historical')
MONTH_INTERVALS = [1, 3, 6, 9]

lbw_filtered = lbw_filtered.reset_index(drop=True)
lbw_filtered['row_id'] = lbw_filtered.index
df = lbw_filtered[['row_id', 'birth_year', 'birth_month', 'lat', 'long']].drop_duplicates().reset_index(drop=True)

# For exploding, create a "living_months" colum with a list of tuples (year, month) of the birth month and the 8 preceding months
df['living_months'] = df.apply(
    lambda row: [
        (
            (row['birth_year'] * 12 + row['birth_month'] - i - 1) // 12,
            1 + ((row['birth_year'] * 12 + row['birth_month'] - i - 1) % 12),
            1+i
        )
        for i in range(9)
    ],
    axis=1,
)
# Explode the living_months column
df_exploded = df.explode('living_months').reset_index(drop=True)
df_exploded[['climate_year', 'climate_month', 'months_since_birth']] = pd.DataFrame(
    df_exploded['living_months'].tolist(), index=df_exploded.index
)[[0, 1, 2]]
df_exploded = df_exploded.drop(columns=['living_months'])
for col in ['climate_year', 'climate_month', 'months_since_birth']:
    df_exploded[col] = df_exploded[col].astype(int)
unique_year_month_loc = df_exploded[['climate_year', 'climate_month', 'lat', 'long']].drop_duplicates().reset_index(drop=True)

def get_climate_vars_for_year(
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
        temp_df[climate_variable] = (
            climate_ds.sel(latitude=lats, longitude=lons, month=months, method="nearest")
            .to_numpy()
            .flatten()  # the flatten also wasn't there before
        )
    return temp_df

year_col = 'climate_year'

# climate_variables = ['days_over_28C', 'days_over_32C', 'mean_low_temperature', 'precipitation_days', 'total_precipitation',
# 'days_over_30C', 'mean_high_temperature', 'mean_temperature', 'relative_humidity']
climate_variables = ['days_over_24C', 'days_over_25C', 'days_over_26C','days_over_27C','days_over_28C',
                     'days_over_29C','days_over_30C','days_over_31C','days_over_32C', 'mean_temperature',
                     'total_precipitation']
n_threads = 25
df_splits =  [year_df for _, year_df in unique_year_month_loc.groupby(year_col)]

with multiprocessing.Pool(n_threads) as pool:
    results_df = pd.concat(list(tqdm.tqdm(pool.imap_unordered(partial(get_climate_vars_for_year, climate_variables=climate_variables), df_splits), total=len(df_splits))))

def get_days_over_threshold_for_year(
    year_df: pd.DataFrame,
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
    threshold_var = 'days_over_relative_threshold'
    threshold_ds = xr.open_dataset(
            f"/mnt/share/erf/climate_downscale/results/_submission/monthly/raw/historical/{threshold_var}/{yr}_era5.nc"
        )["value"].load()
    # Get all values of coordinate 'threshold' in the dataset
    thresholds = threshold_ds['quantile'].values
    for threshold in thresholds:
        temp_df[f"days_over_{threshold}_quant"] = (
            threshold_ds.sel(latitude=lats, longitude=lons, month=months, quantile=threshold, method="nearest")
            .to_numpy()
            .flatten()  # the flatten also wasn't there before
        )
    return temp_df

year_col = 'climate_year'

n_threads = 25
df_splits =  [year_df for _, year_df in unique_year_month_loc.groupby(year_col)]

with multiprocessing.Pool(n_threads) as pool:
    relative_thresh_df = pd.concat(list(tqdm.tqdm(pool.imap_unordered(get_days_over_threshold_for_year, df_splits), total=len(df_splits))))

relative_thresh_vars = [x for x in relative_thresh_df.columns if x.startswith('days_over_')]

results_df = results_df.merge(relative_thresh_df, on=['climate_year', 'climate_month', 'lat', 'long'], how='outer', validate='one_to_one', indicator=False)

results_df = df_exploded.merge(
    results_df,
    on=['climate_year', 'climate_month', 'lat', 'long'],
    how='left',
    validate='many_to_one'
)

for interval in MONTH_INTERVALS:
    interval_df = results_df[results_df['months_since_birth'] <= interval].copy()
    interval_agg = interval_df.groupby('row_id')[climate_variables + relative_thresh_vars].mean().reset_index()
    interval_agg = interval_agg.rename(columns={col: f"{col}_past_{interval}m" for col in climate_variables + relative_thresh_vars})
    if interval == MONTH_INTERVALS[0]:
        final_climate_df = interval_agg
    else:
        final_climate_df = final_climate_df.merge(interval_agg, on='row_id', how='left', validate='one_to_one')

    merged_df = lbw_filtered.merge(
    final_climate_df,
    on='row_id',
    how='left',
    validate='one_to_one'
)


# missing outcome variables
measure_columns = 'lbw'
rows_with_na_outcomes = merged_df[measure_columns].isna().any().sum()
rows_with_na_outcomes = int(rows_with_na_outcomes)

full_data_rows = len(merged_df) - rows_with_na_outcomes
print(
    f"Dropped {rows_with_na_outcomes:,} rows with missing outcome variables"
)
print(f"Data contains {full_data_rows:,} rows after cleaning")

merged_df = merged_df.dropna(subset=measure_columns)

print("Adding elevation data...")
merged_df = get_elevation_for_dataframe(merged_df)

#merged_df = assign_lbd_admin2_location_id(merged_df)


    # Write to output
measure = 'lbw'
measure_df = merged_df[merged_df[measure].notna()].copy()
measure_df["measure"] = measure
measure_df["value"] = measure_df[measure]


def add_temperature_zone_year(
    year_df: pd.DataFrame,
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
    climate_variable = '10_year_mean_temperature'
        # climate_ds = ClimateMalnutritionData(Path(DEFAULT_ROOT)/'stunting').load_climate_raster(climate_variable, 'ssp245', yr, 0)
        # Temporary workaround for climate data loading
    climate_ds = xr.open_dataset(
        f"/mnt/share/erf/climate_downscale/results/_submission/annual/raw/historical/{climate_variable}/{yr}.nc"
    )["value"]
    temp_df[climate_variable] = (
        climate_ds.sel(latitude=lats, longitude=lons, method="nearest")
        .to_numpy()
        .flatten()  # the flatten also wasn't there before
    )
    return temp_df


def add_temperature_zone(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
    year_col: str = "int_year",
) -> pd.DataFrame:
    
    unique_coords = df[[lat_col, long_col, year_col]].drop_duplicates()

    df_splits = [year_df for _, year_df in unique_coords.groupby(year_col)]
    p = mp.Pool(processes=25)
    results_df = pd.concat(
        p.map(
            partial(add_temperature_zone_year, year_col = year_col, lat_col = lat_col, long_col = long_col), df_splits
        )
    )
    p.close()
    p.join()
    results_df.rename(columns={'10_year_mean_temperature': 'temperature_zone'}, inplace=True)
    results_df['temperature_zone'] = results_df.temperature_zone.round(0).astype(int).clip(lower=6, upper=29)
    return results_df

tempzone_df = add_temperature_zone(measure_df, year_col='int_year')

print(len(measure_df))
final_df = measure_df.merge(
    tempzone_df[['lat', 'long', 'int_year', 'temperature_zone']],
    on=['lat', 'long', 'int_year'],
    how='left',
    validate='many_to_one'
)
print(len(measure_df))

print(len(measure_df))
final_df = final_df.merge(
    thresholds_df,
    on=['lat', 'long', 'int_year'],
    how='left',
    validate='many_to_one'
)
print(len(measure_df))


for ldi_col in ["ldipc_weighted_no_match"]:  # ldi_cols:
    measure_df["ldi_pc_pd"] = measure_df[ldi_col] / 365
measure_root = Path(output_root) / data_source_type
os.makedirs(Path(measure_root) / "training_data", exist_ok=True, mode=0o777)
cm_data = ClimateMalnutritionData(measure_root)
version = cm_data.new_training_version()
cm_data.save_training_data(measure_df.query("birth_weight < 9500"), version)
print(f"Saved to version {version}")
#measure_df.query("birth_weight < 9500").to_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/lbw/training_data/2025_10_28.01/data.parquet")
    # logging.info(
    #     f"Saving data for {measure} to version {version} with {ldi_col} as LDI"
    # )
    # cm_data.save_training_data(measure_df, version)
    # message = "Used " + ldi_col + " as LDI"
    # # Save a small file with a record of which ldi column was used for this version
    # with open(cm_data.training_data / version / "ldi_col.txt", "w") as f:
    #     f.write(message)

