"""Create the cached GBD inputs that the residual step reads.

**Run this with an IHME environment, not the project's pixi environment.**
Make yourself a clone of the official IHME GBD environment (``gbdenv``) and use
its interpreter::

    /path/to/your/gbdenv/bin/python \\
        src/rra_climate_health/data_prep/save_gbd_inputs.py \\
        --output-root /mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition

It is the only thing in this repository that touches the IHME database
libraries, and those pull in a large internal dependency tree that we
deliberately keep out of ``pixi.lock``.  For that reason this module is
intentionally standalone: it imports nothing from ``rra_climate_health`` (the
package ``__init__`` imports ``rpy2``, which an IHME environment will not have)
and uses only ``argparse``, ``pandas`` and ``numpy`` besides the IHME libraries
themselves.  Do not add it to the ``strun``/``sttask`` CLI.

Ported from ``malnutrition_fhs/src/2026_07_07_SaveGBDNew.ipynb``.

Steps
-----

* ``prevalence`` -- ``gbd_mean_{measure}_prevalence.parquet`` and
  ``gbd_draws_{measure}_prevalence_{n_draws}.parquet``, pulled with
  ``ihme_cc_get_estimates.get_model_estimates`` over the GBD hierarchy and
  aggregated up to any FHS location the GBD hierarchy does not cover.
* ``mortality`` -- ``gbd_{mean,draws}_{neonatal,child}_mortality_prevalence*``
  from the life table (``life_table_parameter_id=3``, age groups 42 and 1).
  The draws are the mean repeated, since the life table has no draws here.
* ``5q28`` -- ``gbd_{mean,draws}_5q28_prevalence*``, the probability of dying
  between 28 days and age 5 given survival to 28 days, derived from the two
  files the ``mortality`` step writes. Needs no database access.
* ``age-metadata`` -- ``age_group_metadata.parquet``, which the residual
  diagnostics read for age group names.
* ``sev`` -- ``gbd_{mean,draws}_{measure}_sev*``, for the three CGF measures.
* ``prev-to-sev`` -- ``prev_to_sev_{measure}.parquet``, the historical
  prevalence/SEV pairs the conversion is fit on.

.. note::

   ``sev`` and ``prev-to-sev`` still need updating for the more recent changes
   to the shared IHME functions, so they are not in the default step list.
   They are carried over as they last worked -- ``get_draws(source="sev", ...)``
   and ``get_outputs``/``get_model_results`` -- which is what produced the
   copies currently on disk, but those entry points are not available in a
   current GBD environment.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- Releases and location sets -------------------------------------------
GBD_RELEASE_ID = 16  # GBD23
LBW_RELEASE_ID = 34  # GBD25, used for low birth weight only
FHS_LOCATION_SET_ID = 39
FHS_LOCATION_RELEASE_ID = 38
GBD_LOCATION_SET_ID = 35
AGE_GROUP_SET_ID = 25

DEFAULT_N_DRAWS = 100

# Modelable entity IDs used to pull GBD prevalence.  Duplicated from
# rra_climate_health.constants on purpose -- see the module docstring: this
# script must run where the package is not importable.
ME_ID_DICT = {
    "stunting": 10556,
    "wasting": 10558,
    "underweight": 10560,
    "anemia": 10507,
    "lbw": 24450,
}

# Risk-exposure IDs, for the measures that have SEVs.
REI_ID_DICT = {
    "stunting": 241,
    "wasting": 240,
    "underweight": 94,
}

# Age groups and years the prevalence-to-SEV conversion is fit on.
PREV_TO_SEV_AGE_GROUP_IDS = [388, 389, 238, 34]
PREV_TO_SEV_SEX_IDS = [1, 2]
PREV_TO_SEV_YEAR_IDS = range(1990, 2023)

# Life-table parameter 3 is the probability of death; age group 42 is the
# neonatal aggregate and 1 is under-5.
LIFE_TABLE_PARAMETER_ID = 3
NEONATAL_AGE_GROUP_ID = 42
CHILD_AGE_GROUP_ID = 1
# 28 days to 5 years is the union of GBD aggregates 4 ([388, 389]) and 5
# ([238, 34]) and has no age_group_id of its own.
#
# It is nonetheless written as 1, the under-5 id, because the residual step
# joins model output to GBD on age_group_id and the forecast step stamps its
# aggregate as 1 regardless of which strata went into it
# (``inference.run_inference.forecast_scenarios``). A distinct id here would
# make that join return nothing. The cost is that a 5q28 series and a 5q0
# series are indistinguishable by age id -- they are told apart by the model
# version behind the results version, not by the data.
FIVE_Q_28_AGE_GROUP_ID = 1

IDX_COLS = ["location_id", "year_id", "age_group_id", "sex_id"]

ALL_STEPS = ["prevalence", "sev", "prev-to-sev", "mortality", "5q28", "age-metadata"]
# The steps that run against a current GBD environment.
DEFAULT_STEPS = ["prevalence", "mortality", "5q28", "age-metadata"]


def release_for_measure(measure: str) -> int:
    """LBW is modeled in a later release than the rest."""
    return LBW_RELEASE_ID if measure == "lbw" else GBD_RELEASE_ID


def resample_like_fhs(draws_df: pd.DataFrame, n_draws: int) -> pd.DataFrame:
    """Resample a draw set to ``n_draws`` columns.

    Adapted from
    https://stash.ihme.washington.edu/projects/FHSENG/repos/fhs-lib-data-transformation/browse/src/fhs_lib_data_transformation/lib/resample.py

    Kept in sync with ``rra_climate_health.residual.residual_data``.
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


def aggregate_forecast_hierarchy(
    forecast: pd.DataFrame,
    population: pd.DataFrame,
    hierarchy: pd.DataFrame,
    *,
    counts: bool = False,
    detailed_demographics: bool = True,
) -> pd.DataFrame:
    """Population-weight estimates up a location hierarchy.

    A copy of ``rra_climate_health.residual.residual_data`` version, which this
    script cannot import.  Keep the two in sync.
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

    forecast_hierarchy = forecast_hierarchy.join(
        hierarchy_detailed, on="location_id", how="left"
    )
    population_hierarchy = population_hierarchy.join(
        hierarchy_detailed, on="location_id", how="left"
    )

    forecast_hierarchy["agg_ids"] = forecast_hierarchy["path_to_top_parent"].apply(
        lambda x: [int(i) for i in x.split(",")] if isinstance(x, str) else []
    )
    population_hierarchy["agg_ids"] = population_hierarchy["path_to_top_parent"].apply(
        lambda x: [int(i) for i in x.split(",")] if isinstance(x, str) else []
    )

    forecast_exploded = forecast_hierarchy.explode("agg_ids")
    forecast_exploded = (
        forecast_exploded.reset_index()
        .drop(columns=["path_to_top_parent"])
        .rename(columns={"agg_ids": "location_id", "location_id": "old_location_id"})
        .set_index(list(forecast_idx) + ["old_location_id"])
    )

    population_exploded = population_hierarchy.explode("agg_ids")
    population_exploded = (
        population_exploded.reset_index()
        .drop(columns=["path_to_top_parent"])
        .rename(columns={"agg_ids": "location_id", "location_id": "old_location_id"})
        .set_index(list(population_idx) + ["old_location_id"])
    )

    if draws:
        forecast_exploded = forecast_exploded.mul(population_exploded)
    else:
        forecast_exploded = forecast_exploded.mul(
            population_exploded.population, axis="index"
        )

    result_idx = (
        list(forecast_idx)
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
            list(population_idx)
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


###############
# Prevalence  #
###############


def get_gbd_prevalence_for_fhs_hierarchy(
    measure: str, *, draws: bool = True
) -> pd.DataFrame:
    """Pull GBD prevalence and reshape it onto the FHS most-detailed locations.

    GBD does not estimate every FHS most-detailed location, so anything missing
    is population-weighted up the GBD hierarchy first.
    """
    from db_queries import (  # type: ignore[import-not-found]
        get_location_metadata,
        get_population,
    )
    from ihme_cc_get_estimates import (  # type: ignore[import-not-found]
        get_model_estimates,
    )

    release_id = release_for_measure(measure)
    fhs_locs = (
        get_location_metadata(
            location_set_id=FHS_LOCATION_SET_ID, release_id=FHS_LOCATION_RELEASE_ID
        )
        .query("most_detailed == 1")
        .location_id.unique()
    )
    gbd_loc_meta = get_location_metadata(
        location_set_id=GBD_LOCATION_SET_ID, release_id=release_id
    )

    gbd_draws = get_model_estimates(
        modelable_entity_id=ME_ID_DICT[measure],
        estimates="draws" if draws else "PE",
        release_id=release_id,
        location_id=gbd_loc_meta.location_id.unique(),
    )

    gbd_pop = get_population(
        release_id=release_id,
        sex_id=[1, 2],
        location_id="all",
        age_group_id="all",
        year_id="all",
    )
    gbd_pop = gbd_pop.drop(columns=["run_id"], errors="ignore").set_index(IDX_COLS)

    gbd_draws = gbd_draws.set_index(IDX_COLS)
    if gbd_draws.measure_id.nunique() != 1:
        message = "Expected only one measure_id in gbd_draws"
        raise ValueError(message)
    if gbd_draws.metric_id.nunique() != 1:
        message = "Expected only one metric_id in gbd_draws"
        raise ValueError(message)

    if draws:
        gbd_draws = gbd_draws.drop(
            columns=[c for c in gbd_draws.columns if not c.startswith("draw_")],
            errors="ignore",
        )
        # Give the population fake draws so it lines up with the estimate draws.
        n_draws = len([c for c in gbd_draws.columns if c.startswith("draw_")])
        gbd_pop_draws = pd.concat([gbd_pop] * n_draws, axis=1)
        gbd_pop_draws.columns = [f"draw_{i}" for i in range(n_draws)]
    else:
        gbd_pop_draws = gbd_pop
        gbd_draws = gbd_draws.drop(
            columns=[
                c for c in gbd_draws.columns if not c.startswith("point_estimate")
            ]
        )
        gbd_draws = gbd_draws.rename(
            columns={"point_estimate": "gbd_mean_prevalence"}
        )

    missing_locs = set(fhs_locs) - set(gbd_draws.index.get_level_values("location_id"))
    if missing_locs:
        print(
            f"Missing GBD estimates for {len(missing_locs)} FHS locations, "
            "running aggregation"
        )
        aggregated_gbd = aggregate_forecast_hierarchy(
            gbd_draws, gbd_pop_draws, gbd_loc_meta
        )
        return aggregated_gbd.query("location_id in @fhs_locs")
    return gbd_draws.query("location_id in @fhs_locs")


########
# SEV  #
########


def get_gbd_sev_draws(measure: str) -> pd.DataFrame:
    """Pull GBD SEV draws.

    Uses ``get_draws``: SEVs are keyed by ``rei_id``, which
    ``ihme_cc_get_estimates.get_model_estimates`` cannot express.  See the
    module docstring -- this needs an environment with ``get_draws``.
    """
    from get_draws.api import get_draws  # type: ignore[import-not-found]

    gbd_draws = get_draws(
        source="sev",
        gbd_id_type="rei_id",
        gbd_id=REI_ID_DICT[measure],
        release_id=GBD_RELEASE_ID,
        downsample=True,
        sex_id=[1, 2],
        num_workers=10,
    )
    gbd_draws = gbd_draws.drop(
        columns=["measure_id", "metric_id", "rei_id", "version_id"],
        errors="ignore",
    )
    return gbd_draws.set_index(IDX_COLS)


def get_gbd_data(
    measure: str,
    n_draws: int,
    *,
    draws: bool = False,
    metric: str = "prevalence",
) -> pd.DataFrame:
    """Pull one measure/metric, as draws or as a point estimate."""
    if metric == "prevalence":
        gbd_data = get_gbd_prevalence_for_fhs_hierarchy(measure, draws=draws)
        if not draws:
            # The PE pull already comes back as a single gbd_mean_prevalence
            # column, so there is nothing to average.
            return gbd_data
    elif metric == "sev":
        gbd_data = get_gbd_sev_draws(measure)
        if not draws:
            return gbd_data.mean(axis=1).to_frame(name="gbd_mean_sev")
    else:
        message = f"Unknown metric {metric}, expected 'prevalence' or 'sev'."
        raise ValueError(message)

    draw_cols = [col for col in gbd_data.columns if col.startswith("draw_")]
    if len(draw_cols) != n_draws:
        print(
            f"Number of draws available ({len(draw_cols)}) does not match "
            f"requested ({n_draws}), resampling."
        )
        return resample_like_fhs(gbd_data, n_draws)
    return gbd_data


def get_prev_to_sev_table(measure: str) -> pd.DataFrame:
    """Pull the historical prevalence/SEV pairs for one measure.

    Needs ``db_queries.get_outputs``/``get_model_results``; see the module
    docstring.
    """
    from db_queries import (  # type: ignore[import-not-found]
        get_model_results,
        get_outputs,
    )

    year_ids = list(PREV_TO_SEV_YEAR_IDS)
    sevs = get_outputs(
        topic="rei",
        release_id=GBD_RELEASE_ID,
        rei_id=REI_ID_DICT[measure],
        age_group_id=PREV_TO_SEV_AGE_GROUP_IDS,
        sex_id=PREV_TO_SEV_SEX_IDS,
        year_id=year_ids,
        location_id="all",
        measure_id=29,
        metric_id=3,
    )[
        ["location_id", "year_id", "age_group_id", "sex_id", "val", "upper", "lower"]
    ].rename(
        columns={"val": "sev_val", "upper": "sev_upper", "lower": "sev_lower"}
    )
    # get_model_results' year_id argument apparently doesn't work, so we pull
    # every year and filter afterwards.
    prevs = (
        get_model_results(
            gbd_team="epi",
            gbd_id=ME_ID_DICT[measure],
            release_id=GBD_RELEASE_ID,
            measure_id=5,
            age_group_id=PREV_TO_SEV_AGE_GROUP_IDS,
            sex_id=PREV_TO_SEV_SEX_IDS,
        )[
            [
                "location_id",
                "year_id",
                "age_group_id",
                "sex_id",
                "mean",
                "lower",
                "upper",
            ]
        ]
        .rename(
            columns={"mean": "prev_val", "lower": "prev_lower", "upper": "prev_upper"}
        )
        .query("year_id in @year_ids")
    )
    return prevs.merge(
        sevs,
        how="outer",
        on=["location_id", "year_id", "age_group_id", "sex_id"],
        suffixes=("_prev", "_sev"),
    )


###########
# Savers  #
###########


def save_prevalence(gbd_root: Path, measures: list[str], n_draws: int) -> None:
    for measure in measures:
        for draws in [False, True]:
            gbd_data = get_gbd_data(
                measure, n_draws=n_draws, draws=draws, metric="prevalence"
            )
            if draws:
                filepath = (
                    gbd_root / f"gbd_draws_{measure}_prevalence_{n_draws}.parquet"
                )
            else:
                filepath = gbd_root / f"gbd_mean_{measure}_prevalence.parquet"
            gbd_data.to_parquet(filepath, index=True)
            print(f"Saved {filepath}")


def save_sev(gbd_root: Path, measures: list[str], n_draws: int) -> None:
    for measure in measures:
        if measure not in REI_ID_DICT:
            print(f"Skipping SEV for {measure}: no rei_id")
            continue
        for draws in [False, True]:
            gbd_data = get_gbd_data(measure, n_draws=n_draws, draws=draws, metric="sev")
            if draws:
                filepath = gbd_root / f"gbd_draws_{measure}_sev_{n_draws}.parquet"
            else:
                filepath = gbd_root / f"gbd_mean_{measure}_sev.parquet"
            gbd_data.to_parquet(filepath, index=True)
            print(f"Saved {filepath}")


def save_prev_to_sev_tables(gbd_root: Path, measures: list[str]) -> None:
    for measure in measures:
        if measure not in REI_ID_DICT:
            print(f"Skipping prev-to-sev for {measure}: no rei_id")
            continue
        prev_to_sev = get_prev_to_sev_table(measure)
        filepath = gbd_root / f"prev_to_sev_{measure}.parquet"
        prev_to_sev.to_parquet(filepath, index=False)
        print(f"Saved {filepath}")


def save_mortality(gbd_root: Path, n_draws: int) -> None:
    """Write neonatal and child mortality from the GBD life table.

    The life table has no draws here, so the draw files repeat the mean.
    """
    from db_queries import get_life_table  # type: ignore[import-not-found]

    life_df = get_life_table(
        release_id=GBD_RELEASE_ID,
        with_ui=False,
        age_group_id=[NEONATAL_AGE_GROUP_ID, CHILD_AGE_GROUP_ID],
        sex_id=[1, 2],
        life_table_parameter_id=LIFE_TABLE_PARAMETER_ID,
    )
    life_df = life_df.drop(
        columns=["life_table_parameter_id", "run_id"], errors="ignore"
    ).rename(columns={"mean": "gbd_mean_prevalence"})
    life_df = life_df.set_index(IDX_COLS)

    for name, age_group_id in [
        ("neonatal_mortality", NEONATAL_AGE_GROUP_ID),
        ("child_mortality", CHILD_AGE_GROUP_ID),
    ]:
        filepath = gbd_root / f"gbd_mean_{name}_prevalence.parquet"
        life_df.query("age_group_id == @age_group_id").to_parquet(
            filepath, index=True
        )
        print(f"Saved {filepath}")

    draws_df = life_df.copy()
    for i in range(n_draws):
        draws_df[f"draw_{i}"] = draws_df["gbd_mean_prevalence"]
    draws_df = draws_df.drop(columns=["gbd_mean_prevalence"])

    for name, age_group_id in [
        ("neonatal_mortality", NEONATAL_AGE_GROUP_ID),
        ("child_mortality", CHILD_AGE_GROUP_ID),
    ]:
        filepath = gbd_root / f"gbd_draws_{name}_prevalence_{n_draws}.parquet"
        draws_df.query("age_group_id == @age_group_id").to_parquet(
            filepath, index=True
        )
        print(f"Saved {filepath}")


def derive_5q28(
    child_df: pd.DataFrame,
    neonatal_df: pd.DataFrame,
    age_group_id: int = FIVE_Q_28_AGE_GROUP_ID,
) -> pd.DataFrame:
    """Probability of dying between 28 days and age 5, given survival to 28 days.

    Both inputs are qx measured from birth (``life_table_parameter_id=3``):
    ``child_df`` is 5q0 (age group 1) and ``neonatal_df`` is 28q0 (age group 42).
    Probabilities sharing an origin do not subtract into a qx; they compose
    through survivorship::

        1 - 5q0 = (1 - 28q0) * (1 - 5q28)

    so

        5q28 = 1 - (1 - 5q0) / (1 - 28q0)

    Note this is NOT ``5q0 - 28q0``. That difference is the unconditional
    probability, from birth, of dying in the window -- a decomposition of
    deaths, not a hazard. It runs about 1.9% below this value at the median and
    further below it wherever neonatal mortality is high, which is exactly where
    the indicator matters most.

    The boundary is day 28, not day 30: GBD age group 42 is [0, 28) days, so
    these two components decompose cleanly there and nowhere else.

    Both frames are indexed by ``IDX_COLS``; ``age_group_id`` is dropped for the
    join and the result is relabelled with ``age_group_id``.
    """
    join_cols = [c for c in IDX_COLS if c != "age_group_id"]

    def _one(df: pd.DataFrame, expected_age: int, name: str) -> pd.Series:
        d = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
        ages = set(d["age_group_id"].unique())
        if ages != {expected_age}:
            msg = f"{name} should hold only age_group_id {expected_age}, found {sorted(ages)}"
            raise ValueError(msg)
        return d.set_index(join_cols)["gbd_mean_prevalence"].rename(name)

    q_u5 = _one(child_df, CHILD_AGE_GROUP_ID, "q_u5")
    q_nn = _one(neonatal_df, NEONATAL_AGE_GROUP_ID, "q_nn")

    joined = pd.concat([q_u5, q_nn], axis=1, join="outer")
    if joined.isna().any().any():
        counts = joined.isna().sum().to_dict()
        msg = f"5q0 and 28q0 do not cover the same {join_cols}; unmatched: {counts}"
        raise ValueError(msg)

    # Guard the two ways the identity can fail. Neither occurs in GBD 2023 --
    # checked over all 138,750 cells -- but a silent negative probability or a
    # divide-by-zero here would propagate into the residual step unnoticed.
    if (joined["q_nn"] > joined["q_u5"]).any():
        n = int((joined["q_nn"] > joined["q_u5"]).sum())
        msg = f"28q0 exceeds 5q0 in {n} cells; 5q28 would be negative"
        raise ValueError(msg)
    if (joined["q_nn"] >= 1).any():
        n = int((joined["q_nn"] >= 1).sum())
        msg = f"28q0 is >= 1 in {n} cells; 5q28 is undefined there"
        raise ValueError(msg)

    out = (1.0 - (1.0 - joined["q_u5"]) / (1.0 - joined["q_nn"])).to_frame(
        name="gbd_mean_prevalence"
    )
    out["age_group_id"] = age_group_id
    return out.reset_index().set_index(IDX_COLS).sort_index()


def save_5q28(
    gbd_root: Path,
    n_draws: int,
    age_group_id: int = FIVE_Q_28_AGE_GROUP_ID,
) -> None:
    """Write 5q28 from the saved 5q0 and 28q0 files, in their own format.

    Derived from files already on disk, so unlike the other steps this one needs
    no IHME database access and runs in any environment with pandas.
    """
    child_path = gbd_root / f"gbd_mean_child_mortality_prevalence.parquet"
    neonatal_path = gbd_root / f"gbd_mean_neonatal_mortality_prevalence.parquet"
    for path in (child_path, neonatal_path):
        if not path.exists():
            msg = f"{path} not found; run the 'mortality' step first."
            raise FileNotFoundError(msg)

    mean_df = derive_5q28(
        pd.read_parquet(child_path), pd.read_parquet(neonatal_path), age_group_id
    )
    filepath = gbd_root / "gbd_mean_5q28_prevalence.parquet"
    mean_df.to_parquet(filepath, index=True)
    print(f"Saved {filepath}")

    # Draws repeat the mean, as they do for its two components: the life table
    # has no draws at life_table_parameter_id=3, so there is no uncertainty to
    # propagate through the identity.
    draws_df = mean_df.copy()
    for i in range(n_draws):
        draws_df[f"draw_{i}"] = draws_df["gbd_mean_prevalence"]
    draws_df = draws_df.drop(columns=["gbd_mean_prevalence"])
    filepath = gbd_root / f"gbd_draws_5q28_prevalence_{n_draws}.parquet"
    draws_df.to_parquet(filepath, index=True)
    print(f"Saved {filepath}")


def save_age_group_metadata(gbd_root: Path) -> None:
    """Write the age metadata the residual diagnostics read for age names."""
    from db_queries import get_age_metadata  # type: ignore[import-not-found]

    filepath = gbd_root / "age_group_metadata.parquet"
    get_age_metadata(
        release_id=GBD_RELEASE_ID, age_group_set_id=AGE_GROUP_SET_ID
    ).to_parquet(filepath)
    print(f"Saved {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the cached GBD inputs the residual step reads. "
            "Run with a clone of the official IHME GBD environment."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition",
        help="Pipeline output root; files land in {output_root}/input/gbd_prevalence.",
    )
    parser.add_argument(
        "--measure",
        dest="measures",
        action="append",
        choices=sorted(ME_ID_DICT),
        help="Measure to pull; repeatable. Defaults to all of them.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=DEFAULT_STEPS,
        choices=["all", *ALL_STEPS],
        help=(
            "Which outputs to create. 'sev' and 'prev-to-sev' are excluded by "
            "default: they still need updating for the more recent changes to "
            "the shared IHME functions (see the module docstring)."
        ),
    )
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measures = args.measures or sorted(ME_ID_DICT)
    steps = set(ALL_STEPS) if "all" in args.steps else set(args.steps)

    gbd_root = Path(args.output_root) / "input" / "gbd_prevalence"
    if not gbd_root.exists():
        message = f"{gbd_root} does not exist; check --output-root."
        raise FileNotFoundError(message)
    print(f"Writing GBD inputs to {gbd_root}")
    print(f"  measures: {measures}")
    print(f"  steps:    {sorted(steps)}")

    if "prevalence" in steps:
        save_prevalence(gbd_root, measures, args.n_draws)
    if "sev" in steps:
        save_sev(gbd_root, measures, args.n_draws)
    if "prev-to-sev" in steps:
        save_prev_to_sev_tables(gbd_root, measures)
    if "mortality" in steps:
        save_mortality(gbd_root, args.n_draws)
    if "5q28" in steps:
        save_5q28(gbd_root, args.n_draws)
    if "age-metadata" in steps:
        save_age_group_metadata(gbd_root)


if __name__ == "__main__":
    main()
