LDI_VERSION = 'v5'
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
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
#/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/wasting_stunting_combined_2024-10-11.csv
SURVEY_DATA_PATHS = {
    "bmi": {"gbd": SURVEY_DATA_ROOT / "bmi" / "bmi_data_outliered_wealth_rex.csv"},
    "cgf": {
        "gbd": SURVEY_DATA_ROOT
        / "wasting_stunting" / "wasting_stunting_combined_2024-10-11.csv",
        "lsae": "/mnt/share/limited_use/LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/pre_collapse/cgf_lbw_2020_06_15.csv",
    },
    "wealth":  {
        "LSMS": WEALTH_DATA_ROOT / "LSMS_wealth.parquet",
        "DHS": WEALTH_DATA_ROOT / "DHS_wealth.parquet",
        "MICS": WEALTH_DATA_ROOT / "MICS_wealth.parquet",
    },
    "anemia": EXTRACTIONS_ROOT / "anemia" / "anemia_extracts_compiled_09_02_2025.csv",
    "child_mortality": {
        'dem_br': EXTRACTIONS_ROOT / "dem_br",
    }
}

DATA_SOURCE_TYPE = {"stunting": "cgf", "wasting": "cgf", "underweight":"cgf", "low_adult_bmi": "bmi","anemia":"anemia"}
MEASURES_IN_SOURCE = {"cgf": ["stunting", "wasting", "underweight"], "bmi": ["low_adult_bmi"], "anemia": ["anemia_anemic_brinda","anemia_mod_sev_brinda"],"child_mortality":["child_alive"]}

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
        #climate_ds = ClimateMalnutritionData(Path(DEFAULT_ROOT)/'stunting').load_climate_raster(climate_variable, 'ssp245', yr, 0)
        # Temporary workaround for climate data loading
        climate_ds = xr.open_dataset(f'/mnt/share/erf/climate_downscale/results/annual/raw/historical/{climate_variable}/{yr}_era5.nc')['value']
        temp_df[climate_variable] = (
            climate_ds.sel(latitude=lats, longitude=lons, method="nearest")
            .to_numpy().flatten() #the flatten also wasn't there before
        )
    return temp_df


def get_climate_vars_for_dataframe(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
    year_col: str = "int_year",
) -> pd.DataFrame:
    var_names = [
        'mean_temperature', 'days_over_30C', 'precipitation_days', 
        'total_precipitation', 'mean_low_temperature', 'mean_high_temperature',
        'relative_humidity', 'days_over_26C', 'days_over_27C', 'days_over_28C',
        'days_over_29C', 'days_over_31C', 'days_over_32C', 'days_over_33C',
    ]

    unique_coords = df[[lat_col, long_col, year_col]].drop_duplicates()

    df_splits = [year_df for _, year_df in unique_coords.groupby(year_col)]
    p = mp.Pool(processes=25)
    results_df = pd.concat(
        p.map(
            partial(get_climate_vars_for_year, climate_variables=var_names), df_splits
        )
    )
    p.close()
    p.join()
    return results_df

ELEVATION_FILEPATH = '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/elevation/GLOBE_DEM_MOSAIC_Y2016M02D09.TIF'
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
        nid_df["unweighted_population_percentile"] = nid_df[asset_score_col].rank(pct=True)
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
        #print(f"Using {grid_points} points for NID {nid}.")
        dynamic_grid = np.linspace(0, 1, grid_points)
        
        # Build the interpolator from the target (LDI) distribution.
        ldi_interpolator = PchipInterpolator(ldi_df["population_percentile"], ldi_df["ldipc"])
        # Evaluate reference LDI distribution on the dynamic grid.
        ldi_ref = ldi_interpolator(dynamic_grid)
        
        # Direct interpolation (no matching) for both unweighted and weighted.
        unweighted_no_match = ldi_interpolator(nid_df["unweighted_population_percentile"])
        weighted_no_match = ldi_interpolator(nid_df["weighted_population_percentile"])
        
        # Evaluate LDI distribution on the grid for matching purposes.
        interpolated_income_distrib = ldi_interpolator(dynamic_grid)

        #ldi_value_at_80th_percentile = ldi_interpolator(0.8)
        
        # Define the matching procedure.
        def match_distribution(pop_pct, asset_scores):
            """
            For a given set of population percentiles and asset scores,
            iterate over a range of clipping thresholds of the interpolated LDI
            distribution. The candidate curve that minimizes the absolute difference
            between its scaled version and the scaled asset scores is returned.
            """
            scaled_asset = (asset_scores - asset_scores.min()) / (asset_scores.max() - asset_scores.min())
            best_candidate = None
            minimum_difference = np.inf
            # Try thresholds from 25% to 100% in 76 steps.
            for threshold_quantile in np.linspace(0.8, 1, 21):
                threshold_value = np.quantile(interpolated_income_distrib, threshold_quantile)
                clipped_income = interpolated_income_distrib[interpolated_income_distrib <= threshold_value]
                if len(clipped_income) < 2:
                    continue
                clipped_interpolator = PchipInterpolator(
                    np.linspace(0, 1, len(clipped_income)), clipped_income
                )
                candidate = clipped_interpolator(pop_pct)
                scaled_candidate = (candidate - candidate.min()) / (candidate.max() - candidate.min())
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
            ax.plot(ldi_ref, dynamic_grid, label="Reference LDI", linestyle="--", color="black")
            
            # Plot the computed LDI-PC curves with flipped axes.
            sorted_unw = sorted(zip(unweighted_no_match, nid_df["unweighted_population_percentile"]))
            ax.plot(*zip(*sorted_unw), label="Unweighted No Match")
            
            sorted_w = sorted(zip(weighted_no_match, nid_df["weighted_population_percentile"]))
            ax.plot(*zip(*sorted_w), label="Weighted No Match")
            
            sorted_unw_match = sorted(zip(unweighted_match, nid_df["unweighted_population_percentile"]))
            ax.plot(*zip(*sorted_unw_match), label="Unweighted Match")
            
            sorted_w_match = sorted(zip(weighted_match, nid_df["weighted_population_percentile"]))
            ax.plot(*zip(*sorted_w_match), label="Weighted Match")
            
            # Compute scaling parameters for wealth index.
            wealth_min = nid_df[asset_score_col].min()
            wealth_max = nid_df[asset_score_col].max()
            ldi_min = ldi_ref.min()
            ldi_max = ldi_ref.max()
            
            # Define transformation functions for mapping wealth index to LDI range and vice versa.
            def wealth_to_ldi(w):
                return (w - wealth_min) / (wealth_max - wealth_min) * (ldi_max - ldi_min) + ldi_min

            def ldi_to_wealth(x):
                return (x - ldi_min) / (ldi_max - ldi_min) * (wealth_max - wealth_min) + wealth_min

            # Plot the original wealth index distribution, scaled to LDI range.
            sorted_wealth = sorted(zip(nid_df[asset_score_col], nid_df["unweighted_population_percentile"]))
            scaled_sorted_wealth = [(wealth_to_ldi(w), p) for w, p in sorted_wealth]
            ax.plot(*zip(*scaled_sorted_wealth), label="Wealth Index", linestyle=":", color="blue")

            sorted_wealth = sorted(zip(nid_df[asset_score_col], nid_df["weighted_population_percentile"]))
            scaled_sorted_wealth = [(wealth_to_ldi(w), p) for w, p in sorted_wealth]
            ax.plot(*zip(*scaled_sorted_wealth), label="Wealth Index (weighted)", linestyle=":", color="green")
            
            # Set labels and title.
            ax.set_xlabel("LDI-PC")
            ax.set_ylabel("Population Percentile")
            ax.set_title(f"NID {nid} | Year: {year} | ihme_loc_id: {ihme_loc_id}")
            ax.legend()
            ax.grid(True)
            
            # Add a secondary x-axis for Wealth Index.
            secax = ax.secondary_xaxis('top', functions=(ldi_to_wealth, wealth_to_ldi))
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
        "geospatial_id"]

def filter_point_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with point == 1."""
    return df[df["point"] == 1]

def validate_lat_long_uniqueness(df: pd.DataFrame) -> None:
    """Ensure that for each household (nid, psu, hh_id), lat and long are unique."""
    grouped = df.groupby(["nid", "psu", "hh_id"])
    if not (grouped["lat"].nunique().eq(1) & grouped["long"].nunique().eq(1)).all():
        raise ValueError("Multiple latitudes or longitudes for the same nid, psu, and hh_id.")

def validate_no_duplicate_entries(df: pd.DataFrame, group_cols: list, error_msg: str) -> None:
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
    validate_no_duplicate_entries(df, ["nid", "psu", "hh_id"],
                                  "Multiple entries for the same nid, psu, and hh_id.")
    
    if df.groupby(["nid", "hh_id", "year_start", "psu", "geospatial_id"]).size().max() != 1:
        raise ValueError("Multiple entries for the same nid, psu, hh_id, and year.")
    if df.groupby(["nid", "hh_id", "year_start", "psu"]).size().max() != 1:
        raise ValueError("Multiple entries for the same nid, psu, hh_id, and year.")
    
    # Load location metadata here (only used within these validations)
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)
    df = merge_left_without_inflating(df, loc_meta[["location_id", "ihme_loc_id"]], on="ihme_loc_id")
    if df.location_id.isna().any():
        raise RuntimeError("Null location IDs.")
    
    df["year_start"] = df["year_start"].astype(int)
    df["nid"] = df["nid"].astype(int)
    return df

def get_DHS_wealth_dataset() -> pd.DataFrame:
    wealth_raw = pd.read_parquet(SURVEY_DATA_PATHS["wealth"]["DHS"])
    df = filter_point_data(wealth_raw)
    
    # Specific renaming for DHS
    df = df.rename(columns={
        "wealth_score": "wealth_index_dhs",
        "iso3": "ihme_loc_id",
        "weight": "hhweight",
    })
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)
    
    # Use the global column order and remove duplicates
    df = df[WEALTH_DATASET_COMMON_COLUMNS + ['wealth_index_dhs']].drop_duplicates()
    
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
    df = df.rename(columns={
        "wealth_score": "wealth_index_dhs",
        "iso3": "ihme_loc_id",
    })
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)
    
    df = df[WEALTH_DATASET_COMMON_COLUMNS + ['wealth_index_dhs']].drop_duplicates()
    
    # For NID 7618, use geospatial_id as the hh_id
    df.loc[df.nid == 7618, "hh_id"] = df.loc[df.nid == 7618, "geospatial_id"]
    # Truncate ihme_loc_id to country-level (first 3 characters)
    df['ihme_loc_id'] = df['ihme_loc_id'].str[:3]
    
    df = common_validations(df)
    return df

def get_LSMS_wealth_dataset() -> pd.DataFrame:
    wealth_raw = pd.read_parquet(SURVEY_DATA_PATHS["wealth"]["LSMS"])
    df = filter_point_data(wealth_raw)
    
    # Exclude NIDs with no wealth measure available
    wealth_measure_availability = df.groupby('nid').apply(
        lambda x: x[['income', 'expenditure', 'consumption']].isna().all().all(),
        include_groups=False
    )
    nids_without_wealth = wealth_measure_availability[wealth_measure_availability].index.astype(int)
    df = df[~df.nid.isin(nids_without_wealth)].reset_index(drop=True)
    
    # Choose the best available wealth measure, preferring income, then consumption, then expenditure
    wealth_measure = df.groupby('nid').apply(
        lambda x: x[['income', 'expenditure', 'consumption']].isna().mean().idxmin(),
        include_groups=False
    ).reset_index()
    wealth_measure.columns = ['nid', 'wealth_measure']
    df = df.merge(wealth_measure, on='nid', how='left')
    df['wealth_measurement'] = df.apply(lambda row: row[row['wealth_measure']], axis=1)
    
    # Remove problematic NID with multiple measures
    df = df[~df.nid.isin([283013])].reset_index(drop=True)
    
    # Use hhweight if available; otherwise, use pweight
    df['hhweight'] = df['hhweight'].combine_first(df['pweight']).astype(float)
    
    # Validate that each household has a unique wealth measurement
    grouped = df.groupby(['nid', 'hh_id', 'psu', 'geospatial_id'])
    if not (grouped['wealth_measurement'].nunique().eq(1)).all():
        raise ValueError("Multiple wealth measurements for the same household.")
    
    df = df.rename(columns={
        "iso3": "ihme_loc_id",
    })
    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)
    
    df = df[WEALTH_DATASET_COMMON_COLUMNS + ['wealth_measurement', 'wealth_measure']].drop_duplicates()
    
    # Truncate ihme_loc_id to country-level (first 3 characters)
    df['ihme_loc_id'] = df['ihme_loc_id'].str[:3]
    
    df = common_validations(df)
    return df


def assign_age_group(df: pd.DataFrame,indicator="cgf") -> pd.DataFrame:
    age_group_spans = pd.read_parquet(paths.AGE_SPANS_FILEPATH)
    if indicator in ['cgf','child_mortality']:
        age_group_spans = age_group_spans.query("age_group_id in [388, 389, 238, 34]")
    df["age_group_id"] = np.nan
    if (indicator=='child_mortality') & ('age_year' not in df.columns):
        # aod_months should replace age_month for child_alive==0
        df['age_year'] = df['age_month'] / 12 # keep as float
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
        ClimateMalnutritionData._PROCESSED_DATA_ROOT / "ihme" / "lbd_admin2.parquet"  # noqa: SLF001
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

    result = merge_left_without_inflating(data, cgf_coords[[lat_col, long_col, "lbd_admin2_id"]], on=[lat_col, long_col])
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


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_source_type(allow_all=False)
def run_training_data_prep(output_root: str, source_type: str) -> None:
    """Run training data prep."""
    print(f"Running training data prep for {source_type}...")
    # for src in source_type:
    run_training_data_prep_main(output_root, source_type)

def clean_hh_id(row):
    if pd.isna(row['hh_id']):
        return row['hh_id']
    hh_id_str = str(row['hh_id'])
    geo_str = str(row['geospatial_id'])
    # Match one or more leading zeros followed by the geospatial_id at the start
    pattern = r'^0+' + re.escape(geo_str)
    # Remove the matched pattern (if found)
    cleaned = re.sub(pattern, '', hh_id_str)
    # Strip remaining leading zeros and handle empty results
    cleaned = cleaned.lstrip('0') or '0'
    return cleaned

def clean_hh_id_v2(row):
    """
    Function to clean household IDs (hh_id) for anemia and child mortality data.
    """
    hh_id = row['hh_id']
    geo_str = str(row['geospatial_id'])

    if pd.isna(hh_id):
        return hh_id

    # Convert to string and trim leading/trailing whitespace
    hh_id = str(hh_id).strip()

    # Replace multiple spaces with a single space
    hh_id = re.sub(r'\s{2,}', ' ', hh_id)

    # If the hh_id is already clean (no spaces or underscores), return it
    if (len(re.split(r"[_ ]", hh_id)) == 1) and not (hh_id.startswith('0')):
        return float(hh_id)

    # Handle cases with spaces or underscores
    if " " in hh_id:
        hh_id = hh_id.split(" ")[-1]
    elif "_" in hh_id:
        hh_id = hh_id.split("_")[-1]

    # Match and remove leading zeros followed by the geospatial_id
    pattern = r'^0+' + re.escape(geo_str)
    hh_id = re.sub(pattern, '', hh_id)

    # Strip remaining leading zeros and handle empty results
    hh_id = hh_id.lstrip('0') or '0'

    return float(hh_id)  # Return as float to handle NAs

output_root = DEFAULT_ROOT
data_source_type = "cgf"
def run_training_data_prep_main(  # noqa: PLR0915
    output_root: str | Path,
    data_source_type: str,
) -> None:
    if data_source_type == 'cgf':
        run_training_data_prep_cgf(output_root, data_source_type)
    elif data_source_type == "anemia":
        run_training_data_prep_anemia(output_root, data_source_type)
    else:
        msg = f"Data source {data_source_type} not implemented yet."
        raise NotImplementedError(msg)


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

    # Drop rows from GBD dataset with missing location information
    #gbd_cgf_data = gbd_cgf_data.dropna(subset=["lat", "long"], how="any")


    extra_nids = gbd_cgf_data_to_match.copy().drop(columns=['lat', 'long'])

    extra_nids["hh_id"] = extra_nids["hh_id"].str.split(r"[_ ]").str[-1]
    extra_nids["hh_id"] = extra_nids.apply(clean_hh_id, axis=1)
    extra_nids_nids = extra_nids.nid.unique()

    extra_nids_wealth = all_wealth_data.query("nid in @extra_nids_nids").copy()
    extra_nids_wealth["hh_id"] = extra_nids_wealth["hh_id"].str.split(r"[_ ]").str[-1]
    extra_nids_wealth["hh_id"] = extra_nids_wealth.apply(clean_hh_id, axis=1)
    extra_nids = merge_left_without_inflating(extra_nids, extra_nids_wealth, on=["nid", "ihme_loc_id", "hh_id", "psu", "year_start"])
    print(len(extra_nids))
    #extra_nids = extra_nids.dropna(subset=["ldipc"])

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



    cgf_consolidated = pd.concat(
        [lsae_merged, gbd_only], ignore_index=True
    ).reset_index(drop=True)

    cgf_consolidated = cgf_consolidated.drop(columns=["strata", "geospatial_id"])
    # selected_wealth_column = "ldi_pc_weighted_no_match"
    # cgf_consolidated["ldi_pc_pd"] = cgf_consolidated["ldipc"] / 365

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


    # Merge with climate data
    print("Processing climate data...")
    climate_df = get_climate_vars_for_dataframe(cgf_consolidated)
    cgf_consolidated = merge_left_without_inflating(cgf_consolidated, climate_df, on=["int_year", "lat", "long"])


    print("Adding elevation data...")
    cgf_consolidated = get_elevation_for_dataframe(cgf_consolidated)

    cgf_consolidated = assign_lbd_admin2_location_id(cgf_consolidated)
    #cgf_consolidated = assign_sdi(cgf_consolidated)


    #Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = cgf_consolidated[cgf_consolidated[measure].notna()].copy().drop(columns=[x for x in cgf_consolidated.columns if '_x' in x or '_y' in x])
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

    survey_data_path = SURVEY_DATA_PATHS[data_source_type]
    print(f"Running training data prep for {data_source_type}...")

    print("Processing extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    anemia_data_raw = pd.read_csv(
        survey_data_path
    )

    anemia_data = anemia_data_raw[
        [
            "nid",
            "ihme_loc_id",
            'survey_name',
            "year_start",
            "year_end",
            'urban',
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
            'anemia_anemic_brinda',
            'anemia_mod_sev_brinda',
            'latnum', 
            'longnum',
        ]
    ]

    anemia_data = anemia_data.rename(columns=COLUMN_NAME_TRANSLATOR)

    anemia_data["old_hh_id"] = anemia_data["hh_id"]

    anemia_data["hh_id"] = anemia_data.apply(clean_hh_id_v2, axis=1)

    assert len(anemia_data[anemia_data["hh_id"].isna()]) == len(anemia_data[anemia_data["old_hh_id"].isna()]), "NAs introduced by cleaning"

    # Prepping wealth dataset
    dhs_wealth_data_raw = get_DHS_wealth_dataset()
    dhs_wealth_data = dhs_wealth_data_raw.copy()

    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT)/'anemia')
    dhs_wealth_data = get_ldipc_from_asset_score(
            dhs_wealth_data, cm_data, asset_score_col="wealth_index_dhs", weights_col="hhweight",
            plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "dhs_plots.pdf",
            ldi_version = LDI_VERSION,
        )

    dhs_wealth_data["old_hh_id"] = dhs_wealth_data["hh_id"]
    dhs_wealth_data["hh_id"] = dhs_wealth_data.apply(clean_hh_id_v2, axis=1)

    assert len(dhs_wealth_data[dhs_wealth_data["hh_id"].isna()]) == len(dhs_wealth_data[dhs_wealth_data["old_hh_id"].isna()]), "NAs introduced by cleaning"

    missing_hh_rows = anemia_data[anemia_data['hh_id'].isna()]
    print(f"Dropping {len(missing_hh_rows)} rows from anemia data with missing hh_id")
    anemia_data = anemia_data[anemia_data["hh_id"].notna()]

    # Find out percent of anemia nids and hh_ids that can be matched in wealth data
    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]
    anemia_data['hh_id'] = anemia_data['hh_id'].astype(int)
    dhs_wealth_data['hh_id'] = dhs_wealth_data['hh_id'].astype(int)
    anemia_data['psu'] = anemia_data['psu'].astype(int)
    dhs_wealth_data['psu'] = dhs_wealth_data['psu'].astype(int)

    wealth_nids = set(dhs_wealth_data.nid.unique()) 
    anemia_nids = set(anemia_data.nid.unique()) 
    common_nids = wealth_nids.intersection(anemia_nids)

    nid_with_wealth_pc = 100*len(common_nids)/len(anemia_nids)
    print(f"{nid_with_wealth_pc:.1f}% of anemia NIDs - {len(common_nids)} out of "
          f"{len(anemia_nids)} in wealth data NIDs")

    dhs_wealth_data = dhs_wealth_data.query("nid in @anemia_nids")

    anemia_data.drop(columns=["old_hh_id"],inplace=True)
    dhs_wealth_data.drop(columns=["old_hh_id"],inplace=True)

    # Merge data
    anemia_data_wealth = merge_left_without_inflating(anemia_data, dhs_wealth_data.drop(columns=['geospatial_id', 'strata', 'lat', 'long']), on=merge_cols)

    #Calculate proportion of NA and filter out nids with too much wealth missingness (bad merges)
    merged_na_props = (anemia_data_wealth.groupby(['nid']).ldipc_weighted_no_match.count() / anemia_data_wealth.groupby(['nid']).ldipc_weighted_no_match.size())
    merged_nids = merged_na_props[merged_na_props > 0.95].index.to_list()
    anemia_df = anemia_data_wealth.query("nid in @merged_nids").copy()
    dropped_too_missingness = len(anemia_data_wealth) - len(anemia_df)

    # drop other unmerged
    unmergable_rows = anemia_df[anemia_df['wealth_index_dhs'].isna()]
    anemia_df = anemia_df[anemia_df['wealth_index_dhs'].notna()]

    # Assign age group
    before_rows = len(anemia_df)
    anemia_df = assign_age_group(anemia_df,indicator="anemia" )
    anemia_df = anemia_df.dropna(subset=["age_group_id"])
    dropped_due_to_age = before_rows - len(anemia_df)

    # Take out data with invalid lat and long
    before_rows = len(anemia_df)
    anemia_df = anemia_df.dropna(subset=["lat", "long"])
    anemia_df = anemia_df.query("lat != 0 and long != 0")
    dropped_due_to_coords = before_rows - len(anemia_df)

    # NID 275090 is a very long survey in Peru, 2003-2008 that is coded as having
    # multiple year_starts. Removing it.
    # NID 411301 is a Zambia survey in which the prevalences end up being 0
    # after removing data with invalid age columns, remove it.
    problematic_nids = [275090, 411301]
    before_rows = len(anemia_df)
    anemia_df = anemia_df.query("nid not in @problematic_nids")
    dropped_problematic_nids = before_rows - len(anemia_df)

    # missing outcome variables
    measure_columns = MEASURES_IN_SOURCE[data_source_type]
    rows_with_na_outcomes = anemia_df[measure_columns].isna().any(axis=1).sum()
    rows_with_na_outcomes = int(rows_with_na_outcomes)

    full_data_rows = len(anemia_df)-rows_with_na_outcomes
    print(f"Data contains {full_data_rows:,} "
          f"rows out of raw {len(anemia_data_raw):,}. "
          f"Dropped data includes:\n"
          f" - {len(missing_hh_rows):,} with missing hh_id in raw data\n"
          f" - {dropped_too_missingness:,} that were dropped due to excessive missingness\n"
          f' - {len(unmergable_rows):,} that further failed to merge on "nid", "ihme_loc_id", "hh_id", "psu", "year_start" variables\n'
          f" - {dropped_due_to_age:,} due to age groups not found among 388, 389, 238, 34\n"
          f" - {dropped_due_to_coords:,} due to invalid lat and long values\n"
          f" - {dropped_problematic_nids:,} due to problematic NIDs\n"
          f" - {rows_with_na_outcomes:,} with missing outcome variables ({measure_columns})\n"
          )

    # Merge with climate data
    print("Processing climate data...")
    climate_vars = get_climate_vars_for_dataframe(anemia_df)
    anemia_df = merge_left_without_inflating(anemia_df, climate_vars, on=["int_year", "lat", "long"])

    print("Adding elevation data...")
    anemia_df = get_elevation_for_dataframe(anemia_df)

    anemia_df = assign_lbd_admin2_location_id(anemia_df)

    #Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = anemia_df[anemia_df[measure].notna()].copy()
        measure_df["measure"] = measure
        measure_df["value"] = measure_df[measure]
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

def concat_valid_extractions(file_path:str) -> pd.DataFrame:
    # Concatenate all valid extraction files into a single DataFrame
    extraction_files = [f for f in os.listdir(file_path) if f.endswith(".csv")]
    dfs = []
    for f in extraction_files:
        df = pd.read_csv(os.path.join(file_path, f), low_memory=False)
        df['source_file'] = f  # Keep track of the source file
        # Perform any necessary validation on the DataFrame
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def check_columns(df: pd.DataFrame,module: str) -> pd.DataFrame:
    if module=="dem_br":
        # concatenated data contains 2 versions of hh_id, psu_id and strata_id
        df.drop(columns="hhid",inplace=True) # duplicate to hh_id
        
        # give preverance to psu over psu_id, which is sometimes not integerable.
        # e.g. psu_id= "0_15", psu = 6. psu_id values for which psu is na are 
        # also not clear enough to fill for missing values. 
        df.drop(columns="psu_id",inplace=True)

        # same with strata and strata_id
        df.drop(columns="strata_id",inplace=True)
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

def run_training_data_prep_child_mortality(  
    output_root: str | Path,
    data_source_type: str,
    module: str
) -> None:

    # Set up logging
    dataprep_log_path = Path(output_root) / data_source_type / module / "data_prep_log.txt"
    os.makedirs(dataprep_log_path.parent, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(dataprep_log_path, mode="w"),  
            logging.StreamHandler()  
        ]
    )


    # data_source_type = "child_mortality"
    # module = "dem_br"
    survey_data_path = SURVEY_DATA_PATHS[data_source_type][module]
    logging.info(f"Running training data prep for {data_source_type}...")

    logging.info("Processing extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    data_raw = concat_valid_extractions(
        survey_data_path
    )
    logging.info(f"Total rows in concatenated raw data: {len(data_raw):,}")
    df = data_raw.copy()
    df = check_columns(df,module)

    # drop rows with missing key variables
    rows_before_na_drop = len(df)
    key_vars = ['nid', 'psu', 'strata', 'hh_id', 'geospatial_id','latitude', 'longitude', 'child_alive']
    df.dropna(subset=key_vars, inplace=True)
    na_rows_dropped = rows_before_na_drop - len(df)
    logging.info(f"Dropped {na_rows_dropped:,} rows with missing key variables: {key_vars}")
    for var in key_vars:
        logging.info(f"- {var}: {data_raw[var].isna().sum():,} missing values")

    df = df.rename(columns=COLUMN_NAME_TRANSLATOR)

    df["old_hh_id"] = df["hh_id"]
    df["hh_id"] = df.apply(clean_hh_id_v2, axis=1)

    assert len(df[df["hh_id"].isna()]) == len(df[df["old_hh_id"].isna()]), "NAs introduced by cleaning"
    df.drop(columns=["old_hh_id"],inplace=True)

    df["nid"] = df["nid"].astype(int)
    df["psu"] = df["psu"].astype(int)
    df["hh_id"] = df["hh_id"].astype(int)
    df["strata"] = df["strata"].astype(int)
    df["geospatial_id"] = df["geospatial_id"].astype(int)

    # Prepping wealth dataset
    dhs_wealth_data_raw = get_DHS_wealth_dataset()
    dhs_wealth_data = dhs_wealth_data_raw.copy()

    cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT)/'anemia')
    dhs_wealth_data = get_ldipc_from_asset_score(
            dhs_wealth_data, cm_data, asset_score_col="wealth_index_dhs", weights_col="hhweight",
            # plot_pdf_path=Path(DEFAULT_ROOT) / "input"/ "ldi_plots"/ "dhs_plots.pdf",
            ldi_version = LDI_VERSION,
        )

    dhs_wealth_data["old_hh_id"] = dhs_wealth_data["hh_id"]
    dhs_wealth_data["hh_id"] = dhs_wealth_data.apply(clean_hh_id_v2, axis=1)

    assert len(dhs_wealth_data[dhs_wealth_data["hh_id"].isna()]) == len(dhs_wealth_data[dhs_wealth_data["old_hh_id"].isna()]), "NAs introduced by cleaning"

    # Find out percent of anemia nids and hh_ids that can be matched in wealth data
    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]
    dhs_wealth_data['hh_id'] = dhs_wealth_data['hh_id'].astype(int)
    dhs_wealth_data['psu'] = dhs_wealth_data['psu'].astype(int)

    wealth_nids = set(dhs_wealth_data.nid.unique()) 
    df_nids = set(df.nid.unique()) 
    common_nids = wealth_nids.intersection(df_nids)

    nid_with_wealth_pc = 100*len(common_nids)/len(df_nids)
    logging.info(f"{nid_with_wealth_pc:.1f}% of df NIDs - {len(common_nids)} out of "
          f"{len(df_nids)} in wealth data NIDs")

    dhs_wealth_data = dhs_wealth_data.query("nid in @df_nids")
    dhs_wealth_data.drop(columns=["old_hh_id"],inplace=True)

    # Merge data
    df_wealth = merge_left_without_inflating(df, dhs_wealth_data.drop(columns=['geospatial_id', 'strata', 'lat', 'long','hh_weight']), on=merge_cols)

    #Calculate proportion of NA and filter out nids with too much wealth missingness (bad merges)
    merged_na_props = (df_wealth.groupby(['nid']).ldipc_weighted_no_match.count() / df_wealth.groupby(['nid']).ldipc_weighted_no_match.size())
    merged_nids = merged_na_props[merged_na_props > 0.95].index.to_list()
    df_merged = df_wealth.query("nid in @merged_nids").copy()
    dropped_too_missingness = len(df_wealth) - len(df_merged)
    logging.info(f"Dropped {dropped_too_missingness:,} rows from {len(df_wealth):,} due to excessive wealth missingness in NIDs")

    # age_month is months since birth at time of interview.

    # drop other unmerged
    before_rows = len(df_merged)
    unmergable_rows = df_merged[df_merged['wealth_index_dhs'].isna()]
    df_merged = df_merged[df_merged['wealth_index_dhs'].notna()]
    logging.info(f"Dropped {before_rows - len(df_merged):,} rows that further failed to merge on 'nid', 'ihme_loc_id', 'hh_id', 'psu', 'year_start' variables")

    # Assign age group
    before_rows = len(df_merged)

    # replace age_month with aod_months for rows with child_alive==0
    df_merged.loc[df_merged.child_alive==0, 'age_month'] = df_merged.loc[df_merged.child_alive==0, 'aod_months']

    # drop data with no age_month
    df_merged = df_merged[df_merged['age_month'].notna()]
    logging.info(f"Dropped {before_rows - len(df_merged):,} rows with missing age_month or aod_months")

    # create list of years between birth year and year that the age_month lands on.
    df_merged["year_of_recorded_age"] = (df_merged["birth_year"]*12+df["birth_month"] + df_merged["age_month"]) // 12
    df_merged["year_of_recorded_age"] = df_merged["year_of_recorded_age"].astype(int)
    df_merged["birth_year"] = df_merged["birth_year"].astype(int)

    df_merged["years_to_expand"] = df_merged.apply(lambda x: [y for y in range(x["birth_year"],x["year_of_recorded_age"]+1)], axis=1)
    # explode data on years_to_expand
    df_exploded = df_merged.explode("years_to_expand") 
    df_exploded["age_month_at_year_end"] = df_exploded.apply(get_age_month_at_year_end, axis=1)
    # override age_month
    df_exploded["age_month"] = df_exploded["age_month_at_year_end"]
    logging.info(f"Exploded data to {len(df_exploded):,} rows by expanding on years between child birth and either age of death or age at interview")

    df_exploded = assign_age_group(df_exploded,indicator="child_mortality" )
    before_dropping_unused_age_groups = len(df_exploded)
    df_exploded = df_exploded.dropna(subset=["age_group_id"])
    dropped_due_to_age = before_dropping_unused_age_groups - len(df_exploded)
    logging.info(f"Dropped {dropped_due_to_age:,} rows due to age groups not found among 388, 389, 238, 34")

    df_exploded["age_group_id"] = df_exploded["age_group_id"].astype(int)
    df_exploded["age_group_id_agg"] = df_exploded["age_group_id_agg"].astype(int)
    df_exploded["age_year"] = df_exploded["age_year"].astype(int)
    df_exploded["age_month_at_year_end"] = df_exploded["age_month_at_year_end"].astype(int)
    # override int_year, used to get climate vars
    df_exploded.rename(columns={"years_to_expand": "int_year"}, inplace=True)

    # for rows with child_alive==0, replace with child_alive=1 if int_year < year_of_recorded_age
    df_exploded["child_alive"] = df_exploded["child_alive"].astype(int)
    df_exploded.loc[(df_exploded.child_alive==0) & (df_exploded.int_year < df_exploded.year_of_recorded_age), 'child_alive'] = 1


    # Take out data with invalid lat and long
    before_rows = len(df_exploded)
    df_exploded = df_exploded.dropna(subset=["lat", "long"])
    df_exploded = df_exploded.query("lat != 0 and long != 0")
    dropped_due_to_coords = before_rows - len(df_exploded)
    logging.info(f"Dropped {dropped_due_to_coords:,} rows due to invalid lat and long values")

    # NID 275090 is a very long survey in Peru, 2003-2008 that is coded as having
    # multiple year_starts. Removing it.
    # NID 411301 is a Zambia survey in which the prevalences end up being 0
    # after removing data with invalid age columns, remove it.
    problematic_nids = [275090, 411301]
    before_rows = len(df_exploded)
    df_exploded = df_exploded.query("nid not in @problematic_nids")
    dropped_problematic_nids = before_rows - len(df_exploded)
    logging.info(f"Dropped {dropped_problematic_nids:,} rows due to problematic NIDs: {problematic_nids}")

    # missing outcome variables
    measure_columns = MEASURES_IN_SOURCE[data_source_type]
    rows_with_na_outcomes = df_exploded[measure_columns].isna().any(axis=1).sum()
    rows_with_na_outcomes = int(rows_with_na_outcomes)
    logging.info(f"Dropped {rows_with_na_outcomes:,} rows with missing outcome variables ({measure_columns})")


    # Merge with climate data
    logging.info("Processing climate data...")
    climate_vars = get_climate_vars_for_dataframe(df_exploded)
    df_climate = merge_left_without_inflating(df_exploded, climate_vars, on=["int_year", "lat", "long"])

    logging.info("Adding elevation data...")
    df_climate = get_elevation_for_dataframe(df_climate)

    df_climate = assign_lbd_admin2_location_id(df_climate)

    #Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = df_climate[df_climate[measure].notna()].copy()
        measure_df["measure"] = measure
        measure_df["value"] = measure_df[measure]
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
