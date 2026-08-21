"""Create the cached GBD inputs that the residual step reads.

**Run this with an IHME environment, not the project's pixi environment.**

It is the only thing in this repository that touches ``get_draws`` and
``db_queries``, and those pull in a large IHME-internal dependency tree that we
deliberately keep out of ``pixi.lock``.  For that reason this module is
intentionally standalone: it imports nothing from ``rra_climate_health`` (the
package ``__init__`` imports ``rpy2``, which an IHME environment will not have)
and uses only ``argparse``, ``pandas`` and ``numpy`` besides the IHME libraries
themselves.  Do not add it to the ``strun``/``sttask`` CLI.

Usage::

    /path/to/ihme/python src/rra_climate_health/data_prep/save_gbd_inputs.py \\
        --output-root /mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition

Outputs, all written to ``{output_root}/input/gbd_prevalence/``:

* ``gbd_mean_{measure}_{metric}.parquet`` -- draw means
* ``gbd_draws_{measure}_{metric}_{n_draws}.parquet`` -- draws
* ``prev_to_sev_{measure}.parquet`` -- historical prevalence/SEV pairs, used to
  fit the prevalence-to-SEV conversion

for ``metric`` in {prevalence, sev} and each requested measure.

Ported from ``malnutrition_fhs/src/2025_05_01_SaveGBD.ipynb``.  The age
metadata files in that directory are not produced here.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Duplicated from rra_climate_health.constants on purpose -- see the module
# docstring: this script must run in an environment where the package is not
# importable.  Keep the two in sync if the IDs ever change.
ME_ID_DICT = {
    "stunting": 10556,
    "wasting": 10558,
    "underweight": 10560,
}

REI_ID_DICT = {
    "stunting": 241,
    "wasting": 240,
    "underweight": 94,
}

# The release the prevalence/SEV estimates are pulled from.
DEFAULT_RELEASE_ID = 16
DEFAULT_N_DRAWS = 100

# Age groups and years the prevalence-to-SEV conversion is fit on.
PREV_TO_SEV_AGE_GROUP_IDS = [388, 389, 238, 34]
PREV_TO_SEV_SEX_IDS = [1, 2]
PREV_TO_SEV_YEAR_IDS = range(1990, 2023)

IDX_COLS = ["location_id", "year_id", "age_group_id", "sex_id"]


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


def get_gbd_data(
    measure: str,
    n_draws: int,
    *,
    draws: bool = False,
    metric: str = "prevalence",
    release_id: int = DEFAULT_RELEASE_ID,
) -> pd.DataFrame:
    """Pull GBD prevalence or SEV for one measure."""
    from get_draws.api import get_draws  # type: ignore[import-not-found]

    if metric == "prevalence":
        gbd_draws = get_draws(
            source="epi",
            gbd_id_type="modelable_entity_id",
            gbd_id=ME_ID_DICT[measure],
            release_id=release_id,
            downsample=True,
            sex_id=[1, 2],
            num_workers=10,
        )
        gbd_draws = gbd_draws.drop(
            columns=[
                "measure_id",
                "metric_id",
                "modelable_entity_id",
                "model_version_id",
            ],
        )
    elif metric == "sev":
        gbd_draws = get_draws(
            source="sev",
            gbd_id_type="rei_id",
            gbd_id=REI_ID_DICT[measure],
            release_id=release_id,
            downsample=True,
            sex_id=[1, 2],
            num_workers=10,
        )
        gbd_draws = gbd_draws.drop(
            columns=["measure_id", "metric_id", "rei_id", "version_id"],
        )
    else:
        message = f"Unknown metric {metric}, expected 'prevalence' or 'sev'."
        raise ValueError(message)

    gbd_draws = gbd_draws.set_index(IDX_COLS)
    if draws:
        draw_cols = [col for col in gbd_draws.columns if col.startswith("draw_")]
        if len(draw_cols) != n_draws:
            print(
                f"Number of draws available ({len(draw_cols)}) does not match "
                f"requested ({n_draws}), resampling."
            )
            return resample_like_fhs(gbd_draws, n_draws)
        return gbd_draws
    else:
        return gbd_draws.mean(axis=1).to_frame(name=f"gbd_mean_{metric}")


def get_prev_to_sev_table(
    measure: str, release_id: int = DEFAULT_RELEASE_ID
) -> pd.DataFrame:
    """Pull the historical prevalence/SEV pairs for one measure."""
    from db_queries import (  # type: ignore[import-not-found]
        get_model_results,
        get_outputs,
    )

    year_ids = list(PREV_TO_SEV_YEAR_IDS)
    sevs = get_outputs(
        topic="rei",
        release_id=release_id,
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
            release_id=release_id,
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


def save_gbd_estimates(
    gbd_root: Path,
    measures: list[str],
    n_draws: int,
    release_id: int,
) -> None:
    for measure in measures:
        for metric in ["prevalence", "sev"]:
            for draws in [False, True]:
                gbd_data = get_gbd_data(
                    measure,
                    n_draws=n_draws,
                    draws=draws,
                    metric=metric,
                    release_id=release_id,
                )
                if draws:
                    filepath = (
                        gbd_root / f"gbd_draws_{measure}_{metric}_{n_draws}.parquet"
                    )
                else:
                    filepath = gbd_root / f"gbd_mean_{measure}_{metric}.parquet"
                gbd_data.to_parquet(filepath, index=True)
                print(f"Saved {filepath}")


def save_prev_to_sev_tables(
    gbd_root: Path, measures: list[str], release_id: int
) -> None:
    for measure in measures:
        prev_to_sev = get_prev_to_sev_table(measure, release_id=release_id)
        filepath = gbd_root / f"prev_to_sev_{measure}.parquet"
        prev_to_sev.to_parquet(filepath, index=False)
        print(f"Saved {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the cached GBD inputs the residual step reads. "
            "Run with an IHME environment."
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
        default=["all"],
        choices=["all", "gbd", "prev-to-sev"],
        help="Which outputs to create.",
    )
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument("--release-id", type=int, default=DEFAULT_RELEASE_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measures = args.measures or sorted(ME_ID_DICT)
    steps = set(args.steps)
    run_all = "all" in steps

    gbd_root = Path(args.output_root) / "input" / "gbd_prevalence"
    if not gbd_root.exists():
        message = f"{gbd_root} does not exist; check --output-root."
        raise FileNotFoundError(message)
    print(f"Writing GBD inputs for {measures} to {gbd_root}")

    if run_all or "gbd" in steps:
        save_gbd_estimates(gbd_root, measures, args.n_draws, args.release_id)
    if run_all or "prev-to-sev" in steps:
        save_prev_to_sev_tables(gbd_root, measures, args.release_id)


if __name__ == "__main__":
    main()
