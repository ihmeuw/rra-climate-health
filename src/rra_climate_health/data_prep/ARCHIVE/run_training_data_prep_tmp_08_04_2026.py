"""
Temporary resume script for run_training_data_prep.py
Picks up from line 3093 of the original script, loading saved artifacts
from version 2026_08_03.02.
"""

import gc
import multiprocessing as mp
from functools import partial
from pathlib import Path

import logging
import numpy as np
import os
import pandas as pd
import polars as pl
from tqdm import tqdm
import sys
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[2]))

from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData

# ── Helper functions needed from original script ──


def merge_left_without_inflating(df_left, df_right, **kwargs):
    """Merge left without inflating the left dataframe."""
    df = df_left.merge(df_right, how="left", **kwargs)
    if len(df) != len(df_left):
        msg = "Mismatch in length of data and merged data."
        raise RuntimeError(msg)
    return df


def get_climate_thresholds_all_locs(
    year_df: pd.DataFrame,
    year_col: str = "lookup_year",
    lat_col: str = "lat",
    long_col: str = "long",
) -> pd.DataFrame:
    temp_df = year_df.copy()
    lats = xr.DataArray(temp_df[lat_col], dims="point")
    lons = xr.DataArray(temp_df[long_col], dims="point")
    lookup_yr = temp_df[year_col].iloc[0]

    try:
        climate_da = xr.open_dataarray(
            f"/mnt/share/erf/climate_downscale/results/monthly/raw/historical/days_over_relative_threshold/{lookup_yr}_era5.nc"
        )
        climate_da = climate_da.load()
        climate_da = climate_da.sel(latitude=lats, longitude=lons, method="nearest")
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
    unique_coords = df[[lat_col, long_col]].drop_duplicates()

    min_year = int(df[year_var].min()) - 1
    max_year = int(df[year_var].max())

    df_splits = []
    for year in range(min_year, max_year + 1):
        df_split = unique_coords.copy()
        df_split["lookup_year"] = year
        df_splits.append(df_split)

    p = mp.Pool(processes=4)
    results_xarrays = list(
        tqdm(
            p.imap(
                partial(
                    get_climate_thresholds_all_locs,
                ),
                df_splits,
            ),
            total=len(df_splits),
            desc="Processing climate variables",
        )
    )
    p.close()
    p.join()

    results_xarrays = [da for da in results_xarrays if da is not None]
    if len(results_xarrays) == 0:
        return None
    results_da = xr.concat(results_xarrays, dim="year")
    return results_da


# ── Setup variables to resume from saved state ──

output_root = DEFAULT_ROOT
data_source_type = "child_mortality"
module = "dem_br"
measure_root = Path(output_root) / data_source_type
version = "2026_08_03.02"
output_path_version = Path(measure_root) / "training_data" / version

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(Path(output_path_version) / "data_prep_log.txt", mode="a"),
        logging.StreamHandler(),
    ],
)

# ── Load saved artifacts ──
# Step 1: Build abs threshold lookup table (small) WITHOUT loading data.parquet
print("Loading saved abs_month_climate_vars.nc...")
climate_vars_da = xr.open_dataarray(output_path_version / "abs_month_climate_vars.nc")

climate_vars_df = climate_vars_da.to_dataframe().reset_index()
del climate_vars_da
gc.collect()

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
climate_vars_wide_df = climate_vars_df.pivot_table(
    index=["int_year", "int_month", "lat", "long"],
    columns="climate_var",
    values="value",
).reset_index()
del climate_vars_df
gc.collect()

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

# Convert lookup to Polars and save for reuse
abs_lookup_pl = pl.from_pandas(climate_vars_wide_df)
del climate_vars_wide_df
gc.collect()

# Step 2: Streaming merge abs thresholds onto data.parquet using Polars
abs_out = Path(output_path_version) / "data_monthly_expanded_abs_thresholds.parquet"
if not abs_out.exists():
    print("Merging abs thresholds via Polars streaming...")
    (
        pl.scan_parquet(output_path_version / "data.parquet")
        .join(
            abs_lookup_pl.lazy(),
            on=["int_year", "int_month", "lat", "long"],
            how="left",
        )
        .sink_parquet(abs_out)
    )
    del abs_lookup_pl
    gc.collect()
    print(f"Saved abs thresholds to {abs_out}")
else:
    print(f"Reusing existing abs thresholds parquet at {abs_out}")

# Step 3: Build relative threshold lookup in year-sized chunks to keep memory low.
print("Building relative threshold lookup...")
rel_lookup_parts_dir = output_path_version / "rel_lookup_parts"
rel_lookup_parts_dir.mkdir(parents=True, exist_ok=True)

coords_df = (
    pl.scan_parquet(abs_out)
    .select(["lat", "long", "int_year", "int_month"])
    .unique()
    .collect()
)
coords_pd = coords_df.to_pandas()
del coords_df
gc.collect()

years = sorted(coords_pd["int_year"].dropna().unique().tolist())
for year in years:
    year_part_path = rel_lookup_parts_dir / f"rel_lookup_{year}.parquet"
    if year_part_path.exists():
        continue

    year_coords = (
        coords_pd.loc[coords_pd["int_year"] == year, ["lat", "long"]]
        .drop_duplicates()
        .copy()
    )
    year_coords["lookup_year"] = year

    year_da = get_climate_thresholds_all_locs(
        year_coords,
        year_col="lookup_year",
        lat_col="lat",
        long_col="long",
    )
    if year_da is None:
        del year_coords
        gc.collect()
        continue

    year_df = year_da.to_dataframe().reset_index()
    del year_da
    gc.collect()

    year_df.drop(columns=["point", "longitude", "latitude"], inplace=True)
    year_df.rename(
        columns={
            "year": "int_year",
            "month": "int_month",
            "lat_orig": "lat",
            "long_orig": "long",
        },
        inplace=True,
    )
    year_df["quantile_str"] = year_df["quantile"].astype(str).str.replace("0.", "q")
    year_df.drop(columns="quantile", inplace=True)

    year_wide_df = year_df.pivot_table(
        index=["int_year", "int_month", "lat", "long"],
        columns="quantile_str",
        values="value",
    ).reset_index()
    del year_df
    gc.collect()

    year_wide_df.rename(
        columns={
            "q75": "q75_monthly",
            "q8": "q80_monthly",
            "q85": "q85_monthly",
            "q9": "q90_monthly",
            "q95": "q95_monthly",
        },
        inplace=True,
    )

    pl.from_pandas(year_wide_df).write_parquet(year_part_path)
    del year_wide_df
    gc.collect()

del coords_pd
gc.collect()

# Step 4: Streaming merge relative thresholds, year by year to avoid one giant relational join.
rel_out = Path(output_path_version) / "data_monthly_expanded_rel_thresholds.parquet"
if rel_out.exists():
    print(f"Reusing existing relative-threshold parquet at {rel_out}")
else:
    print("Merging rel thresholds via Polars streaming...")
    rel_year_outs = []
    for part_path in sorted(rel_lookup_parts_dir.glob("*.parquet")):
        year = part_path.stem.split("_")[-1]
        rel_year_out = (
            Path(output_path_version)
            / f"data_monthly_expanded_rel_thresholds_{year}.parquet"
        )
        (
            pl.scan_parquet(abs_out)
            .filter(pl.col("int_year") == int(year))
            .join(
                pl.scan_parquet(part_path).lazy(),
                on=["int_year", "int_month", "lat", "long"],
                how="left",
            )
            .filter(
                pl.col("consumption_pd").is_not_null()
                & pl.col("days_over_30C_monthly").is_not_null()
                & pl.col("total_precipitation_monthly").is_not_null()
            )
            .sink_parquet(rel_year_out)
        )
        rel_year_outs.append(rel_year_out)

    if rel_year_outs:
        pl.concat(
            [pl.scan_parquet(path) for path in rel_year_outs], how="diagonal_relaxed"
        ).sink_parquet(rel_out)
        for path in rel_year_outs:
            path.unlink(missing_ok=True)

before_rows = pl.scan_parquet(abs_out).select(pl.len()).collect().item()
after_rows = pl.scan_parquet(rel_out).select(pl.len()).collect().item()
logging.info(
    f"Dropped {before_rows - after_rows:,} rows with missing values in key merged variables (consumption_pd, days_over_30C_monthly) after merging monthly climate variables"
)
print(f"Saved rel thresholds to {rel_out}")

# Step 5: Time-bin collapsing — all in Polars lazy, reading from rel_out
print("Starting time-bin collapsing...")
print(
    f"max age_month: {pl.scan_parquet(rel_out).select(pl.col('age_month').max()).collect().item()}"
)

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

get_max_vars = [
    "year_start",
    "year_end",
    "nid",
    "int_year",
    "int_month",
    "sex_id",
    "age_month",
    "pweight",
    "birth_year",
    "birth_month",
    "int_birth_year_diff_months",
    "age_month_original",
    "int_year_original",
    "int_month_original",
    "any_days_over_30C",
    "child_alive",
    "child_mortality",
]

get_avg_vars = [
    "mean_temperature",
    "days_over_30C",
    "total_precipitation",
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

# Read only the columns needed for binning into Polars (lazy scan → collect subset)
needed_cols = list(
    dict.fromkeys(identity_vars + ["age_month"] + get_max_vars + get_avg_vars)
)
print(f"Loading {len(needed_cols)} columns from rel thresholds parquet into Polars...")
df_pl = pl.scan_parquet(rel_out).select(needed_cols).collect()
print(f"Loaded {df_pl.height:,} rows into Polars")

# Add bin columns
for bin_name, bin_month in time_bin_dict.items():
    df_pl = df_pl.with_columns(
        ((pl.col("age_month") >= bin_month[0]) & (pl.col("age_month") < bin_month[1]))
        .cast(pl.Int32)
        .alias(bin_name)
    )

group_by_vars_within_bin = identity_vars + list(time_bin_dict.keys())

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

cumulative_frames = []
for bin_name, bin_month in time_bin_dict.items():
    lower = bin_month[0]
    upper = bin_month[1]
    # Only include individuals who have at least one observation within this bin,
    # meaning they did not exit the interview before reaching this age period
    individuals_in_bin = (
        df_pl.filter((pl.col("age_month") >= lower) & (pl.col("age_month") < upper))
        .select(identity_vars)
        .unique()
    )
    bin_frame = (
        df_pl.filter(pl.col("age_month") < upper)
        .join(individuals_in_bin, on=identity_vars, how="inner")
        .group_by(identity_vars)
        .agg(
            [pl.col(var).max().alias(var) for var in get_max_vars]
            + [pl.col(var).mean().alias(f"{var}_cumul") for var in get_avg_vars]
        )
        .with_columns(pl.lit(bin_name).alias("bin_name"))
    )
    logging.info(f"Built cumulative frame for {bin_name}: {bin_frame.height:,} rows")
    del individuals_in_bin
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
    [(bin_name, bounds[0], bounds[1]) for bin_name, bounds in time_bin_dict.items()],
    columns=["bin_name", "lower_bound", "upper_bound"],
)
df_grouped_cumulative = df_grouped_cumulative.merge(
    bounds_df, on="bin_name", how="left"
)

df_grouped_cumulative = df_grouped_cumulative[
    df_grouped_cumulative["age_month"] >= df_grouped_cumulative["lower_bound"]
]

df_grouped_cumulative.to_parquet(
    Path(output_path_version) / "data_cumulative_bins.parquet",
    index=False,
)

# Keep only 1 dummy in max age for cumulative age
# vars. Equivalent to adding back on binned age vars

# get max age per indv and add on
df_max_age = (
    df_grouped_cumulative.sort_values("age_month")
    .groupby("indv_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

# keep_cols
get_avg_vars_cumul = [f"{var}_cumul" for var in get_avg_vars]
keep_cols = ["indv_id"] + get_avg_vars_cumul
df_max_age_constant_vars = df_max_age[keep_cols]
for v in get_avg_vars:
    v_cumul = f"{v}_cumul"
    df_max_age_constant_vars.rename(columns={v_cumul: f"{v}_constant"}, inplace=True)

df_grouped_cumulative = pd.merge(
    df_grouped_cumulative, df_max_age_constant_vars, on="indv_id", how="left"
)

print(len(df_grouped_cumulative))
df_grouped_cumulative = df_grouped_cumulative.drop_duplicates()
print(len(df_grouped_cumulative))
df_grouped_cumulative.to_parquet(
    Path(output_path_version) / "data_constant_vars.parquet",
    index=False,
)

assert (
    len(df_grouped_cumulative[df_grouped_cumulative["child_mortality"].isna()]) == 0
), "Missing child_mortality values in final data"
df_grouped_cumulative.to_parquet(
    Path(output_path_version) / "data.parquet",
    index=False,
)

print("Done! Final data.parquet saved.")
