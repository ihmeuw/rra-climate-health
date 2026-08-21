"""Data loading and hierarchy aggregation helpers for the residual step.

Ported from ``malnutrition_fhs/src/data_utils.py``.  Everything here reads from
the pipeline's own output tree via a :class:`ClimateMalnutritionData` so that
the paths stay in one place.

Nothing in this module talks to the IHME databases.  The cached GBD inputs it
reads are created by ``data_prep/save_gbd_inputs.py``, which is run separately
under an IHME environment; if one is missing, the loaders below say so and name
that script rather than silently reaching for ``get_draws``/``db_queries``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from rra_climate_health import paths
from rra_climate_health.data import ClimateMalnutritionData

IDX_COLS = ["location_id", "year_id", "age_group_id", "sex_id"]

# Draw counts that the GBD input script saves files for.
SAVED_DRAW_COUNT = 100

_MISSING_INPUT_HINT = (
    "Create it by running data_prep/save_gbd_inputs.py with an IHME "
    "environment, e.g.\n"
    "    /path/to/ihme/python src/rra_climate_health/data_prep/save_gbd_inputs.py "
    "--output-root {output_root}"
)


def _missing_input_message(filepath: Path, cm_data: ClimateMalnutritionData) -> str:
    output_root = cm_data.root.parent
    return f"Could not find {filepath}.\n" + _MISSING_INPUT_HINT.format(
        output_root=output_root
    )


##############
# GBD inputs #
##############


def get_gbd_data(
    cm_data: ClimateMalnutritionData,
    measure: str,
    n_draws: int,
    *,
    draws: bool = False,
    metric: str = "prevalence",
) -> pd.DataFrame:
    """Load the cached GBD estimates for a measure."""
    root_path = cm_data.gbd_inputs
    if draws and measure != "neonatal_mortality":
        filepath = root_path / f"gbd_draws_{measure}_{metric}_{n_draws}.parquet"
    else:
        filepath = root_path / f"gbd_mean_{measure}_{metric}.parquet"
    if not filepath.exists():
        if draws and n_draws < SAVED_DRAW_COUNT:
            # Only the 100-draw file is saved, so resample down from it.
            return resample_like_fhs(
                get_gbd_data(
                    cm_data, measure, SAVED_DRAW_COUNT, draws=True, metric=metric
                ),
                n_draws,
            )
        raise FileNotFoundError(_missing_input_message(filepath, cm_data))
    if measure == "neonatal_mortality" and draws:
        # For neonatal mortality we only have mean prevalence, so we make fake
        # draws by repeating the mean value.
        mean_df = pd.read_parquet(filepath)
        draw_cols = [f"draw_{i}" for i in range(n_draws)]
        res_df = pd.concat([mean_df["gbd_mean_prevalence"]] * n_draws, axis=1)
        res_df.columns = draw_cols
        return res_df
    return pd.read_parquet(filepath)


def get_prev_to_sev_table(
    cm_data: ClimateMalnutritionData, measure: str
) -> pd.DataFrame:
    """Load the historical prevalence/SEV pairs the conversion is fit on."""
    filepath = cm_data.gbd_inputs / f"prev_to_sev_{measure}.parquet"
    print(f"Loading prev_to_sev table from {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(_missing_input_message(filepath, cm_data))
    return pd.read_parquet(filepath)



def get_sdi() -> pd.DataFrame:
    """Load past and future SDI, averaged over draws."""
    past_sdi = xr.open_dataset(paths.PAST_SDI_FILEPATH)
    past_sdi_df = past_sdi.to_dataframe().reset_index()
    past_sdi_df_agg = (
        past_sdi_df.groupby(["location_id", "year_id"])
        .agg({"draws": "mean"})
        .reset_index()
        .rename(columns={"draws": "sdi"})
    )
    future_sdi = xr.open_dataset(paths.FUTURE_SDI_FILEPATH)
    future_sdi_df = future_sdi.to_dataframe().reset_index()
    future_sdi_df = future_sdi_df[
        future_sdi_df["scenario"] == paths.FUTURE_SDI_SCENARIO
    ]
    future_sdi_df = future_sdi_df.drop(columns=["scenario"])
    future_sdi_df_agg = (
        future_sdi_df.groupby(["location_id", "year_id"])
        .agg({"draws": "mean"})
        .reset_index()
        .rename(columns={"draws": "sdi"})
    )
    sdi_df = pd.concat([past_sdi_df_agg, future_sdi_df_agg])
    return sdi_df


def resample_like_fhs(draws_df: pd.DataFrame, n_draws: int) -> pd.DataFrame:
    """Resample a draw set to ``n_draws`` columns.

    Adapted from
    https://stash.ihme.washington.edu/projects/FHSENG/repos/fhs-lib-data-transformation/browse/src/fhs_lib_data_transformation/lib/resample.py
    """
    if not all(col.startswith("draw_") for col in draws_df.columns):
        message = "All columns must start with 'draw_'"
        raise ValueError(message)
    common_draw_numbers = (100, 500, 1000)

    num_of_draws = n_draws
    draw_indices_available = [
        int(col.replace("draw_", ""))
        for col in draws_df.columns
        if col.startswith("draw")
    ]
    draw_indices_available.sort()
    if len(draw_indices_available) not in common_draw_numbers:
        print(f"Weird: this data array has {len(draw_indices_available)} draws!")
    num_of_draws_available = len(draw_indices_available)
    num_of_full_sets = num_of_draws // num_of_draws_available
    draw_indices = draw_indices_available * num_of_full_sets
    remainder = num_of_draws % num_of_draws_available
    if remainder:
        step = int(np.ceil(num_of_draws_available / remainder))
        if 1 + step * (remainder - 1) > num_of_draws_available:
            # This is to ensure that we have enough draws available.
            step = 1
        draw_indices += draw_indices_available[0 : step * remainder : step]
    draw_indices = [f"draw_{draw}" for draw in draw_indices]

    draws_df = draws_df[draw_indices]
    draws_df.columns = ["draw_" + str(i) for i in range(num_of_draws)]
    return draws_df


###########################
# Hierarchy aggregation   #
###########################


def aggregate_forecast_hierarchy(
    forecast: pd.DataFrame,
    population: pd.DataFrame,
    hierarchy: pd.DataFrame,
    *,
    counts: bool = False,
    detailed_demographics: bool = True,
) -> pd.DataFrame:
    """Aggregate forecast data for every location in the hierarchy.

    Assumes forecast data is for the most detailed locations.
    Expects hierarchy to include a 'path_to_top_parent' column where
    each value is a comma-separated string of ancestor location_ids.
    Accepts draws or means. Can be used for counts or rates.

    Args:
        forecast: Forecast data in draws or means, indexed by loc/age/etc.
        population: Population draws.
        hierarchy: Hierarchy data with location_id and path_to_top_parent columns.
        counts: If True, returns counts. If False, returns rates.
        detailed_demographics: If False, collapse age_group and sex dimensions too.
    """
    hierarchy_detailed = hierarchy[hierarchy["most_detailed"] == True][  # noqa: E712
        ["location_id", "path_to_top_parent"]
    ].set_index("location_id")
    locs = hierarchy_detailed.index.unique()
    forecast_idx = forecast.index.names
    population_idx = population.index.names
    sexes = forecast.index.get_level_values("sex_id").unique()
    age_groups = forecast.index.get_level_values("age_group_id").unique()
    years = forecast.index.get_level_values("year_id").unique()

    forecast_hierarchy = forecast.query("location_id in @locs")
    population_hierarchy = population.query(
        "location_id in @locs and sex_id in @sexes and "
        "age_group_id in @age_groups and year_id in @years"
    )

    # Check if the datasets are draws or not
    if all(forecast_hierarchy.columns.str.contains("draw")):
        draws = True
        if set(forecast_hierarchy.columns) != set(population_hierarchy.columns):
            message = "Forecast and population draw columns do not match."
            raise ValueError(message)
    else:
        # Operating on means, not draws, so take population mean
        population_hierarchy = population_hierarchy.mean(axis=1).to_frame(
            name="population"
        )
        draws = False

    # Merge forecast with the detailed hierarchy to bring in the ancestry path
    forecast_hierarchy = forecast_hierarchy.join(
        hierarchy_detailed, on="location_id", how="left"
    )
    population_hierarchy = population_hierarchy.join(
        hierarchy_detailed, on="location_id", how="left"
    )

    # Convert the path_to_top_parent string to a list of ints for each row
    forecast_hierarchy["agg_ids"] = forecast_hierarchy["path_to_top_parent"].apply(
        lambda x: [int(i) for i in x.split(",")] if isinstance(x, str) else []
    )
    population_hierarchy["agg_ids"] = population_hierarchy["path_to_top_parent"].apply(
        lambda x: [int(i) for i in x.split(",")] if isinstance(x, str) else []
    )

    # Explode the list so each forecast row appears once per aggregated location id
    forecast_exploded = forecast_hierarchy.explode("agg_ids")
    forecast_exploded = (
        forecast_exploded.reset_index()
        .drop(columns=["path_to_top_parent"])
        .rename(columns={"agg_ids": "location_id", "location_id": "old_location_id"})
        .set_index(forecast_idx + ["old_location_id"])
    )

    population_exploded = population_hierarchy.explode("agg_ids")
    population_exploded = (
        population_exploded.reset_index()
        .drop(columns=["path_to_top_parent"])
        .rename(columns={"agg_ids": "location_id", "location_id": "old_location_id"})
        .set_index(population_idx + ["old_location_id"])
    )

    if draws:
        forecast_exploded = forecast_exploded.mul(population_exploded)
    else:
        forecast_exploded = forecast_exploded.mul(
            population_exploded.population, axis="index"
        )

    result_idx = (
        forecast_idx
        if detailed_demographics
        else [x for x in forecast_idx if x not in ["age_group_id", "sex_id"]]
    )
    cols_to_drop = (
        ["old_location_id"]
        if detailed_demographics
        else ["age_group_id", "sex_id", "old_location_id"]
    )
    forecast_agg = (
        forecast_exploded.reset_index()
        .drop(columns=cols_to_drop)
        .groupby(result_idx, dropna=False)
        .sum(min_count=1)
    )
    if not counts:
        agg_pop_idx = (
            population_idx
            if detailed_demographics
            else [x for x in population_idx if x not in ["age_group_id", "sex_id"]]
        )
        population_agg = (
            population_exploded.reset_index()
            .drop(columns=cols_to_drop)
            .groupby(agg_pop_idx)
            .sum()
        )
        if draws:
            forecast_agg = forecast_agg.div(population_agg)
        else:
            forecast_agg = forecast_agg.div(population_agg.population, axis="index")
    return forecast_agg


def copy_values_to_children_locations(
    forecast: pd.DataFrame, hierarchy: pd.DataFrame
) -> pd.DataFrame:
    """Copy values from parent locations to missing most-detailed children.

    Assumes forecast data is for some level 3 and some level 4 locations.
    Expects hierarchy to include a 'parent_id' column where each value is the
    location_id of the parent location.
    """
    locs_needed = hierarchy.query("level == 3 or level == 4")["location_id"].unique()
    locs_available = forecast.index.get_level_values("location_id").unique()
    locs_to_copy = set(locs_needed) - set(locs_available)
    if not locs_to_copy:
        return forecast
    # Filter hierarchy to only include the locations we need to copy
    hierarchy_to_copy = (
        hierarchy.query("location_id in @locs_to_copy")[["location_id", "parent_id"]]
        .copy()
        .rename(
            columns={"parent_id": "location_id", "location_id": "child_location_id"}
        )
    )

    child_rows = forecast.reset_index().merge(
        hierarchy_to_copy, how="right", on="location_id"
    )
    child_rows = child_rows.drop(columns=["location_id"]).rename(
        columns={"child_location_id": "location_id"}
    )
    child_rows = child_rows.set_index(forecast.index.names)
    result = pd.concat([forecast, child_rows], axis=0)
    return result


def aggregate_age_and_sex(
    forecast: pd.DataFrame,
    population: pd.DataFrame,
    *,
    counts: bool = False,
) -> pd.DataFrame:
    """Collapse the age and sex dimensions of a forecast.

    Accepts draws or means. Can be used for counts or rates.

    Args:
        forecast: Forecast data in draws or means, indexed by loc/age/etc.
        population: Population draws.
        counts: If True, returns counts. If False, returns rates.
    """
    locs = forecast.index.get_level_values("location_id").unique()
    forecast_idx = forecast.index.names
    population_idx = population.index.names
    sexes = forecast.index.get_level_values("sex_id").unique()
    age_groups = forecast.index.get_level_values("age_group_id").unique()
    years = forecast.index.get_level_values("year_id").unique()

    population_filtered = population.query(
        "location_id in @locs and sex_id in @sexes and "
        "age_group_id in @age_groups and year_id in @years"
    )
    if not set(locs).issubset(
        set(population_filtered.index.get_level_values("location_id").unique())
    ):
        message = (
            "Not all location_ids in forecast are in population. "
            "Check the hierarchy and population data."
        )
        raise ValueError(message)

    # Check if the datasets are draws or not
    if all(forecast.columns.str.contains("draw")):
        draws = True
        if set(forecast.columns) != set(population_filtered.columns):
            message = "Forecast and population draw columns do not match."
            raise ValueError(message)
    else:
        # Operating on means, not draws, so take population mean
        population_filtered = population_filtered.mean(axis=1).to_frame(
            name="population"
        )
        draws = False

    if draws:
        forecast_counts = forecast.mul(population_filtered)
    else:
        forecast_counts = forecast.mul(population_filtered.population, axis="index")

    result_idx = [x for x in forecast_idx if x not in ["age_group_id", "sex_id"]]
    cols_to_drop = ["age_group_id", "sex_id"]
    forecast_agg = (
        forecast_counts.reset_index()
        .drop(columns=cols_to_drop)
        .groupby(result_idx, dropna=False)
        .sum(min_count=1)
    )
    if not counts:
        agg_pop_idx = [
            x for x in population_idx if x not in ["age_group_id", "sex_id"]
        ]
        population_agg = (
            population_filtered.reset_index()
            .drop(columns=cols_to_drop)
            .groupby(agg_pop_idx)
            .sum()
        )
        if draws:
            forecast_agg = forecast_agg.div(population_agg)
        else:
            forecast_agg = forecast_agg.div(population_agg.population, axis="index")
    return forecast_agg
