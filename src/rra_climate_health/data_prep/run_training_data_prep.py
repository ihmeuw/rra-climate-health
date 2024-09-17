import multiprocessing as mp
from functools import partial
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

import rra_climate_health.cli_options as clio
from rra_climate_health import paths
from rra_climate_health.data import (
    DEFAULT_ROOT,
    ClimateMalnutritionData,
)

SURVEY_DATA_ROOT = Path(
    "/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data"
)
LSAE_CGF_GEOSPATIAL_ROOT = Path(
    "/mnt/share/limited_use/LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/pre_collapse/"
)
CGF_FILEPATH_LSAE = Path(
    "/mnt/share/limited_use/LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/pre_collapse/cgf_lbw_2020_06_15.csv"
)


# TODO: Carve out the piece of the R script that makes this file,
#  move output someplace better
WEALTH_FILEPATH = Path(
    "/mnt/share/scratch/users/victorvt/cgfwealth_spatial/dhs_wealth_uncollapsed_again.parquet"
)

HISTORICAL_CLIMATE_ROOT = Path(
    "/mnt/share/erf/climate_downscale/results/annual/historical/"
)
ELEVATION_FILEPATH = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/srtm_elevation.tif"
)

LDIPC_NATIONAL_FILEPATH = Path(
    "/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/national_ldipc_estimates.csv"
)
LDIPC_SUBNATIONAL_FILEPATH = Path(
    "/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/admin2_ldipc_estimates.csv"
)
SDI_PATH = Path("/mnt/share/forecasting/data/7/past/sdi/20240531_gk24/sdi.nc")

SURVEY_DATA_PATHS = {
    "bmi": {"gbd": SURVEY_DATA_ROOT / "bmi" / "bmi_data_outliered_wealth_rex.csv"},
    "cgf": {
        "gbd": SURVEY_DATA_ROOT
        / "wasting_stunting"
        / "wasting_stunting_outliered_wealth_rex.csv",
        "lsae": CGF_FILEPATH_LSAE,
    },
    "wealth": {"lsae": WEALTH_FILEPATH},
}

DATA_SOURCE_TYPE = {"stunting": "cgf", "wasting": "cgf", "low_adult_bmi": "bmi"}
MEASURES_IN_SOURCE = {"cgf": ["stunting", "wasting"], "bmi": ["low_adult_bmi"]}

############################
# Wasting/Stunting columns #
############################

ihme_meta_columns = [
    "nid",
    "file_path",
]
survey_meta_columns = [
    "survey_name",
    "survey_module",
    "year_start",
    "year_end",
]
sample_meta_columns = [
    "psu",
    "psu_id",
    "pweight",
    "pweight_admin_1" "urban",
    "strata" "strata_id",
    "hh_id",
    "hhweight",
    "line_id",
    "int_year",
    "int_month",
    "int_day",
]
location_meta_columns = [
    "ihme_loc_id",
    "location_name",
    "super_region_name",
    "geospatial_id",
    "admin_1",
    "admin_1_id",
    "admin_1_mapped",
    "admin_1_urban_id",
    "admin_1_urban_mapped",
    "admin_2",
    "admin_2_id",
    "admin_2_mapped",
    "admin_3",
    "admin_4",
    "latnum",
    "longnum",
]
individual_meta_columns = [
    "individual_id",
    "sex_id",
    "age_year",
    "age_month",
    "age_day",
    "age_categorical",
]
value_columns = [
    "metab_height",
    "metab_height_unit",
    "metab_weight",
    "metab_weight_unit",
    "bmi",
    "overweight",
    "obese",
    "pregnant",
    "birth_weight",
    "birth_weight_unit",
    "birth_order",
    "mother_weight",
    "mother_height",
    "mother_age_month",
    "maternal_ed_yrs",
    "paternal_ed_yrs",
    "wealth_factor",
    "wealth_index_dhs",
    "suspicious.heights",
    "suspicious.weights",
    "HAZ",
    "HAZ_b1",
    "HAZ_b2",
    "HAZ_b3",
    "HAZ_b4",
    "WAZ",
    "WAZ_b1",
    "WAZ_b2",
    "WAZ_b3",
    "WAZ_b4",
    "WHZ",
    "WHZ_b1",
    "WHZ_b2",
    "WHZ_b3",
    "WHZ_b4",
]


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


def run_training_data_prep_main(  # noqa: PLR0915
    output_root: str | Path,
    data_source_type: str,
) -> None:
    if data_source_type != "cgf":
        msg = f"Data source {data_source_type} not implemented yet."
        raise NotImplementedError(msg)

    survey_data_path = SURVEY_DATA_PATHS[data_source_type]
    print(f"Running training data prep for {data_source_type}...")

    print("Processing gbd extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    lsae_cgf_data_raw = pd.read_csv(
        survey_data_path["lsae"],
        dtype={"hh_id": str, "year_start": int, "int_year": int, "year_end": int},
    )
    new_cgf_data_raw = pd.read_csv(
        survey_data_path["gbd"],
        dtype={"hh_id": str, "year_start": int, "int_year": int, "year_end": int},
    )

    # subset to columns of interest
    gbd_cgf_data = new_cgf_data_raw[
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
    nids_without_wealth = gbd_cgf_data[
        gbd_cgf_data.wealth_index_dhs.isna()
    ].nid.unique()
    gbd_cgf_data = gbd_cgf_data[~gbd_cgf_data.nid.isin(nids_without_wealth)]
    print(len(gbd_cgf_data))

    # We get wealth data to be merged with GBD CGF data and merge
    print("Processing wealth data...")
    gbd_data_wealth_distribution = (
        gbd_cgf_data.groupby(["nid", "ihme_loc_id", "year_start", "psu", "hh_id"])
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

    gbd_data_wealth_distribution = gbd_data_wealth_distribution.merge(
        loc_meta, how="left", on="ihme_loc_id"
    )

    gbd_data_wealth_distribution = get_ldipc_from_asset_score(
        gbd_data_wealth_distribution, asset_score_col="wealth_index_dhs"
    )
    gbd_data_wealth_distribution = gbd_data_wealth_distribution[
        ["nid", "ihme_loc_id", "location_id", "year_start", "psu", "hh_id", "ldipc"]
    ]
    check = gbd_cgf_data.merge(
        gbd_data_wealth_distribution,
        how="left",
        on=["nid", "ihme_loc_id", "year_start", "psu", "hh_id"],
    )

    if len(gbd_cgf_data) != len(check):
        msg = "Mismatch in length of GBD data and merged data."
        raise RuntimeError(msg)

    gbd_cgf_data = check

    # Now the old LSAE data
    # We get wealth data to be merged with CGF data and the lsae cgf data with standard column names
    raw_wealth_df = get_wealth_dataset()

    wealth_df = get_ldipc_from_asset_score(
        raw_wealth_df, asset_score_col="wealth_index_dhs"
    )
    lsae_cgf_data = lsae_cgf_data_raw[
        [
            "nid",
            "country",
            "year_start",
            "end_year",
            "geospatial_id",
            "psu",
            "pweight",
            "strata",
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

    # Get NIDs in common between wealth and LSAE extraction CGF data and subset to that
    common_nids = set(lsae_cgf_data["nid"].unique()).intersection(
        set(wealth_df["nid"].unique())
    )
    wealth_lsae_df = wealth_df[wealth_df["nid"].isin(common_nids)][
        [
            "nid",
            "ihme_loc_id",
            "location_id",
            "year_start",
            "psu",
            "hh_id",
            "geospatial_id",
            "lat",
            "long",
            "wealth_index_dhs",
            "ldipc",
        ]
    ]

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
    print(len(lsae_cgf_data))

    # Some NIDs need extra cleaning so that hh_id can be merged.
    # Take those out and merge LSAE CGF data with wealth
    print(len(lsae_cgf_data))

    maybe_fixable_df = lsae_cgf_data[
        lsae_cgf_data["nid"].isin([157057, 286780, 341838])
    ].copy()
    lsae_cgf_data = lsae_cgf_data[~lsae_cgf_data["nid"].isin([157057, 286780, 341838])]
    merge_cols = ["nid", "ihme_loc_id", "hh_id", "psu", "year_start"]

    lsae_merged = lsae_cgf_data.drop(columns=["old_hh_id"]).merge(
        wealth_lsae_df[
            [
                "ihme_loc_id",
                "location_id",
                "nid",
                "psu",
                "hh_id",
                "year_start",
                "wealth_index_dhs",
                "ldipc",
                "lat",
                "long",
            ]
        ],
        on=merge_cols,
        how="left",
    )
    print(len(lsae_cgf_data))
    print(len(lsae_merged))

    print(len(maybe_fixable_df))
    bad_nids = [157057, 286780, 341838]
    for nid in bad_nids:
        maybe_fixable_df.loc[maybe_fixable_df.nid == nid, "hh_id"] = (
            maybe_fixable_df.loc[maybe_fixable_df.nid == nid]
            .psu.astype(int)
            .astype(str)
            .str.zfill(4)
            + maybe_fixable_df.loc[maybe_fixable_df.nid == nid]
            .old_hh_id.astype(int)
            .astype(str)
            .str.zfill(3)
        )

    merged_fixed_nids_df = maybe_fixable_df.drop(columns=["old_hh_id"]).merge(
        wealth_lsae_df[
            [
                "ihme_loc_id",
                "location_id",
                "nid",
                "psu",
                "hh_id",
                "year_start",
                "ldipc",
                "lat",
                "long",
            ]
        ],
        on=merge_cols,
        how="left",
    )
    merged_fixed_nids_df = merged_fixed_nids_df.dropna(
        subset=["location_id", "lat", "long", "ldipc"], how="any"
    )
    merged_fixed_nids_df = merged_fixed_nids_df.dropna(
        subset=["stunting", "wasting", "underweight"], how="all"
    )
    print(len(merged_fixed_nids_df))

    # Only interested in rows have have either stunting, wasting or underweight information, with both wealth and location
    lsae_merged = lsae_merged.dropna(
        subset=["stunting", "wasting", "underweight"], how="all"
    )
    lsae_merged = lsae_merged.dropna(subset=["ldipc", "lat", "long"], how="any")
    lsae_merged.loc[lsae_merged.sex_id == 0, "sex_id"] = 2
    print(len(lsae_merged))

    # Drop rows from GBD dataset with missing location information
    gbd_cgf_data = gbd_cgf_data.dropna(subset=["lat", "long"], how="any")

    # For the GBD NIDs that need wealth information, attempt to get it
    extra_nids = new_cgf_data_raw.query("nid in @gbd_nids_need_wealth")[
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
            "WAZ_b2",
            "WHZ_b2",
            "latnum",
            "longnum",
        ]
    ]
    extra_nids = extra_nids.rename(columns=COLUMN_NAME_TRANSLATOR)
    extra_nids["hh_id"] = extra_nids["hh_id"].str.split(r"[_ ]").str[-1]
    extra_nids = extra_nids.merge(
        wealth_df[
            [
                "ihme_loc_id",
                "location_id",
                "nid",
                "psu",
                "hh_id",
                "year_start",
                "wealth_index_dhs",
                "ldipc",
            ]
        ],
        on=["nid", "ihme_loc_id", "year_start", "psu", "hh_id"],
        how="left",
    )
    extra_nids = extra_nids.dropna(subset=["ldipc"])

    # Bring the two datasets (LSAE and GBD) together, giving preference to the new extractions
    new_extraction_nids = gbd_cgf_data.nid.unique()
    lsae_only = lsae_merged.loc[~lsae_merged.nid.isin(new_extraction_nids)]

    # Merge the two datasets and the NIDs that needed wealth information
    cgf_consolidated = pd.concat(
        [gbd_cgf_data, lsae_only, extra_nids], ignore_index=True
    ).reset_index(drop=True)

    cgf_consolidated = cgf_consolidated.drop(columns=["strata", "geospatial_id"])
    cgf_consolidated["ldi_pc_pd"] = cgf_consolidated["ldipc"] / 365

    # Assign age group
    cgf_consolidated = assign_age_group(cgf_consolidated)
    cgf_consolidated = cgf_consolidated.dropna(subset=["age_group_id"])

    # Take out data with invalid lat and long
    cgf_consolidated = cgf_consolidated.dropna(subset=["lat", "long"])
    cgf_consolidated = cgf_consolidated.query("lat != 0 and long != 0")

    # Merge with climate data
    print("Processing climate data...")
    climate_df = get_climate_vars_for_dataframe(cgf_consolidated)
    cgf_consolidated = cgf_consolidated.merge(
        climate_df, on=["int_year", "lat", "long"], how="left"
    )

    print("Adding elevation data...")
    cgf_consolidated = get_elevation_for_dataframe(cgf_consolidated)

    cgf_consolidated = assign_lbd_admin2_location_id(cgf_consolidated)
    cgf_consolidated = assign_sdi(cgf_consolidated)

    # Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = cgf_consolidated[cgf_consolidated[measure].notna()].copy()
        measure_df["cgf_measure"] = measure
        measure_df["cgf_value"] = measure_df[measure]
        measure_root = Path(output_root) / measure
        cm_data = ClimateMalnutritionData(measure_root)
        version = cm_data.new_training_version()
        print(f"Saving data for {measure} to version {version}...")

        cm_data.save_training_data(measure_df, version)


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
        climate_ds = xr.open_dataset(
            HISTORICAL_CLIMATE_ROOT / climate_variable / f"{yr}.nc"
        )
        temp_df[climate_variable] = (
            climate_ds["value"]
            .sel(latitude=lats, longitude=lons, year=years, method="nearest")
            .to_numpy()
        )
    return temp_df


def get_climate_vars_for_dataframe(
    df: pd.DataFrame,
    lat_col: str = "lat",
    long_col: str = "long",
    year_col: str = "int_year",
) -> pd.DataFrame:
    var_names = [
        child.name for child in HISTORICAL_CLIMATE_ROOT.iterdir() if child.is_dir()
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


def get_elevation_for_dataframe(
    df: pd.DataFrame, lat_col: str = "lat", long_col: str = "long"
) -> pd.DataFrame:
    unique_coords = df[[lat_col, long_col]].drop_duplicates()
    elevation_ds: xr.Dataset = rioxarray.open_rasterio(ELEVATION_FILEPATH)  # type: ignore[assignment]
    unique_coords["elevation"] = elevation_ds.sel(
        x=xr.DataArray(unique_coords.long, dims="z"),
        y=xr.DataArray(unique_coords.lat, dims="z"),
        band=1,
        method="nearest",
    ).to_numpy()

    unique_coords["elevation"] = unique_coords["elevation"].astype(int)
    nodata_val = -32768
    unique_coords.loc[unique_coords.elevation == nodata_val, "elevation"] = 0

    results_df = df.merge(unique_coords, on=[lat_col, long_col], how="left")
    if results_df["elevation"].isna().any():
        msg = "Null elevation values."
        raise RuntimeError(msg)
    return results_df


def get_ldipc_from_asset_score(
    asset_df: pd.DataFrame,
    asset_score_col: str = "wealth_index_dhs",
    year_df_col: str = "year_start",
    *,
    use_weights: bool = False,
    match_distributions: bool = False,
) -> pd.DataFrame:
    from scipy.interpolate import PchipInterpolator

    ldi = pd.read_csv(LDIPC_NATIONAL_FILEPATH)
    print(
        f"Converting income from asset score {'not' if not use_weights else ''} using weights, {'not' if not match_distributions else ''} matching distributions"
    )
    dfs = []
    for nid in asset_df.nid.unique():
        nid_df = (
            asset_df.loc[asset_df.nid == nid]
            .copy()
            .sort_values(["nid", "ihme_loc_id", year_df_col, asset_score_col])
        )
        if nid_df.year_start.nunique() > 1:
            print(f"Multiple years for NID {nid}: {nid_df.year_start.unique()}")
            year = nid_df.year_start.min()
        else:
            year = nid_df.year_start.iloc[0]
        if nid_df.ihme_loc_id.nunique() > 1:
            msg = f"Multiple locations for NID {nid}"
            raise ValueError(msg)
        ihme_loc_id = nid_df.ihme_loc_id.iloc[0]
        nid_df["unweighted_population_percentile"] = nid_df[asset_score_col].rank(
            pct=True
        )

        nid_df["cum_weight"] = nid_df["pweight"].cumsum()
        total_weight = nid_df["pweight"].sum()
        nid_df["weighted_population_percentile"] = nid_df["cum_weight"] / total_weight
        nid_df["population_percentile"] = (
            nid_df.weighted_population_percentile
            if use_weights
            else nid_df.unweighted_population_percentile
        )

        ldi_df = ldi.loc[
            (ldi.ihme_loc_id == ihme_loc_id) & (ldi.year_id == year)
        ].sort_values(["population_percentile"])
        if ldi_df.empty:
            print(f"No LDI data for NID {nid} in year {year}")
            raise ValueError
        population_percentiles = np.linspace(0, 1, 1000001)
        interpolator = PchipInterpolator(
            ldi_df["population_percentile"], ldi_df["ldipc"]
        )

        if not match_distributions:
            nid_df["ldipc"] = interpolator(nid_df["population_percentile"])
            dfs.append(nid_df)
            continue
        # Take away values below threshold
        # Get the distribution of incomes, subset to below threshold and create new interpolator from them
        interpolated_income_distrib = interpolator(population_percentiles)
        minimum_difference = np.inf
        for threshold_quantile in np.linspace(0.25, 1, 76):
            threshold_value = np.quantile(
                interpolated_income_distrib, threshold_quantile
            )
            interpolated_incomes_clipped = interpolated_income_distrib[
                interpolated_income_distrib <= threshold_value
            ]

            clipped_interpolator = PchipInterpolator(
                np.linspace(0, 1, len(interpolated_incomes_clipped)),
                interpolated_incomes_clipped,
            )
            interp_survey_income = clipped_interpolator(nid_df["population_percentile"])
            scaled_interp_survey_income = (
                interp_survey_income - interp_survey_income.min()
            ) / (interp_survey_income.max() - interp_survey_income.min())
            scaled_wealth_index = (
                nid_df["wealth_index_dhs"] - nid_df["wealth_index_dhs"].min()
            ) / (nid_df["wealth_index_dhs"].max() - nid_df["wealth_index_dhs"].min())
            diff = np.abs((scaled_interp_survey_income - scaled_wealth_index).sum())

            if diff < minimum_difference:
                minimum_difference = diff
                nid_df["ldipc"] = interp_survey_income
        dfs.append(nid_df)

    asset_df_ldipc = pd.concat(dfs)
    if asset_df_ldipc["ldipc"].isna().sum() != 0:
        msg = "Null LDI-PC values."
        raise RuntimeError(msg)

    if len(asset_df_ldipc) != len(asset_df):
        msg = "Mismatch in length of asset data and LDI-PC data."
        raise RuntimeError(msg)

    return asset_df_ldipc  # type: ignore[no-any-return]


def get_wealth_dataset() -> pd.DataFrame:
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)
    wealth_raw = pd.read_parquet(SURVEY_DATA_PATHS["wealth"]["lsae"])
    wealth_df = wealth_raw[wealth_raw["point"] == 1]
    wealth_df = wealth_df.rename(
        columns={
            "wealth_score": "wealth_index_dhs",
            "iso3": "ihme_loc_id",
            "weight": "pweight",
        }
    )
    wealth_df = wealth_df.rename(columns=COLUMN_NAME_TRANSLATOR)
    wealth_df = wealth_df[
        [
            "nid",
            "ihme_loc_id",
            "year_start",
            "year_end",
            "geospatial_id",
            "psu",
            "strata",
            "hh_id",
            "pweight",
            "wealth_index_dhs",
            "lat",
            "long",
        ]
    ]
    wealth_df = wealth_df[
        [
            "ihme_loc_id",
            "nid",
            "psu",
            "hh_id",
            "pweight",
            "lat",
            "long",
            "year_start",
            "wealth_index_dhs",
            "geospatial_id",
        ]
    ].drop_duplicates()

    # In the wealth team's dataset, sometimes there are multiple asset scores for a given household id.
    # Take away those NIDs

    # Make sure that by nid, psu and hh_id they all have the same lat and long
    grouped = wealth_df.groupby(["nid", "psu", "hh_id"])

    if not (grouped["lat"].nunique().eq(1) & grouped["long"].nunique().eq(1)).all():
        msg = "Multiple latitudes or longitudes for the same nid, psu, and hh_id."
        raise ValueError(msg)

    if grouped.size().max() != 1:
        msg = "Multiple entries for the same nid, psu, and hh_id."
        raise ValueError(msg)

    # Sometimes an nid has more than a year
    if (
        wealth_df.groupby(["nid", "hh_id", "year_start", "psu", "geospatial_id"])
        .size()
        .sort_values()  # type: ignore[call-overload]
        .max()
        != 1
    ):
        msg = "Multiple entries for the same nid, psu, hh_id, and year."
        raise ValueError(msg)

    bad_wealth = wealth_df.groupby(
        [
            "nid",
            "hh_id",
            "year_start",
            "psu",
        ]
    ).size()
    bad_nid_wealth = list(bad_wealth[bad_wealth.gt(1)].reset_index().nid.unique())  # type: ignore[index]
    bad_nid_wealth = [*bad_nid_wealth, 20315, 20301, 20537]
    wealth_df = wealth_df[~wealth_df.nid.isin(bad_nid_wealth)]
    # Sometimes an nid has more than a year
    if (
        wealth_df.groupby(["nid", "hh_id", "year_start", "psu"])
        .size()
        .sort_values()  # type: ignore[call-overload]
        .max()
        != 1
    ):
        msg = "Multiple entries for the same nid, psu, hh_id, and year."
        raise ValueError(msg)

    wealth_df = wealth_df.merge(
        loc_meta[["location_id", "ihme_loc_id"]], on="ihme_loc_id", how="left"
    )
    if wealth_df.location_id.isna().any():
        msg = "Null location IDs."
        raise RuntimeError(msg)

    wealth_df["year_start"] = wealth_df["year_start"].astype(int)
    wealth_df["nid"] = wealth_df["nid"].astype(int)
    return wealth_df


def assign_age_group(df: pd.DataFrame) -> pd.DataFrame:
    age_group_spans = pd.read_parquet(paths.AGE_SPANS_FILEPATH)
    age_group_spans = age_group_spans.query("age_group_id in [388, 389, 238, 34]")
    df["age_group_id"] = np.nan
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

    # TODO: Delete when (or if) we get population at the more granular age group level
    age_id_map = {
        388: 4,
        389: 4,
        238: 5,
        34: 5,
    }
    df["age_group_id"] = df["age_group_id"].map(age_id_map)
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

    lencheck = len(data)
    result = data.merge(
        cgf_coords[[lat_col, long_col, "lbd_admin2_id"]],
        on=[lat_col, long_col],
        how="left",
    )
    if len(result) != lencheck:
        msg = "Mismatch in length of data and merged data."
        raise RuntimeError(msg)
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
