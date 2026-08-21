"""Prepare residual-step SEV draws for the FHS ``save_results`` upload.

This is **not** a pipeline step.  It is run on demand, after the residual step
has finished, to reshape ``adjusted_sev_draws.parquet`` into the per-scenario
HDF5 files that the FHS upload process expects as its input.  It is
deliberately not registered on the ``strun``/``sttask`` CLI; run it directly::

    python -m rra_climate_health.residual.prepare_submission \\
        --measure stunting \\
        --results-version 2026_07_13.01 \\
        --submission-dir /ihme/scratch/users/<user>/save_results_fhs

Files are written to
``{submission-dir}/{today}/{measure}/{fhs_scenario_id}/sev/sev_data_{fhs_scenario_id}.h5``
with HDF key ``data``, one per scenario.

Ported from ``malnutrition_fhs/src/PrepareSubmission2026.ipynb``, where the
output directory was hardcoded to a personal scratch path.

A note on size: the upload wants a full demographic grid, so the draws are
reindexed onto *every* age group in ``age_group_metadata.parquet`` (152 of
them) with zeros for the age groups the measure does not model.  For the child
growth failure measures that inflates ~0.9M rows to ~34M rows, or roughly
27 GB across the three scenario files.  Use ``--dry-run`` to see the shape and
the destination paths without writing anything.
"""

from pathlib import Path

import click
import pandas as pd

from rra_climate_health import cli_options as clio
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from rra_climate_health.inference.run_inference import load_population_timeseries
from rra_climate_health.residual.residual_data import (
    aggregate_forecast_hierarchy,
    resample_like_fhs,
)

# Maps our scenario names onto the FHS scenario IDs the upload expects.
# The wider mapping the notebook carried, for reference:
#   {'ssp119': 53, 'ssp126': 66, 'ssp245': 0, 'reference': 0,
#    'ssp585': 54, 'constant_climate': 55}
SCENARIO_FHS_TRANSLATION = {
    "ssp126": 160,
    "ssp245": 159,
    "ssp585": 161,
}

# Only years from here on are uploaded.
YEAR_UPLOAD_START = 2024

# Risk-exposure IDs, needed as a column on the SEV submission.
REI_ID_DICT = {
    "stunting": 241,
    "wasting": 240,
    "underweight": 94,
}

SUBMISSION_COLUMNS = {
    "prevalence": {
        "measure_id": 5,
        "metric_id": 3,
        "cause_id": 391,
        "release_id": 38,
    },
    "sev": {
        "measure_id": 29,
        "metric_id": 3,
        # rei_id is measure-specific and filled in below.
        "release_id": 38,
    },
}


def add_submission_columns(
    df: pd.DataFrame, measure: str, metric: str
) -> pd.DataFrame:
    """Stamp the constant identifier columns the upload expects."""
    extra_columns = dict(SUBMISSION_COLUMNS[metric])
    if metric == "sev":
        extra_columns["rei_id"] = REI_ID_DICT[measure]
    for col, val in extra_columns.items():
        df[col] = val
    return df


def build_sev_submission(
    cm_data: ClimateMalnutritionData,
    measure: str,
    results_version: str,
    year_upload_start: int = YEAR_UPLOAD_START,
) -> pd.DataFrame:
    """Aggregate, inflate and label the SEV draws for upload."""
    fhs_loc_meta = cm_data.load_fhs_hierarchy()
    age_meta = cm_data.load_age_group_metadata()

    sev_draws = cm_data.load_adjusted_sev_draws(results_version)

    necessary_locs = fhs_loc_meta.query("level == 3 or level == 4")[
        "location_id"
    ].unique()
    necessary_age_groups = age_meta.age_group_id.unique()

    sev_draws = sev_draws.query("year_id >= @year_upload_start")
    n_draws = len([x for x in sev_draws.columns if x.startswith("draw_")])
    print(f"Loaded {len(sev_draws):,} rows of SEV draws ({n_draws} draws)")

    population = load_population_timeseries(
        None, sev_draws.index.get_level_values("age_group_id").unique()
    )
    future_population = resample_like_fhs(population, n_draws)

    print("Aggregating up the location hierarchy")
    sev_all_locs = aggregate_forecast_hierarchy(
        sev_draws, future_population, fhs_loc_meta
    )

    sev_submission = sev_all_locs.query("location_id in @necessary_locs")

    # The upload wants a complete demographic grid, so inflate to every age
    # group, filling the ones this measure does not model with zeros.
    loc_year_sex_scenarios = [
        sev_submission.index.get_level_values(name).unique()
        for name in ["location_id", "year_id", "sex_id", "scenario"]
    ]
    full_index = pd.MultiIndex.from_product(
        loc_year_sex_scenarios + [necessary_age_groups],
        names=["location_id", "year_id", "sex_id", "scenario", "age_group_id"],
    ).reorder_levels(sev_submission.index.names)

    print(
        f"Reindexing {len(sev_submission):,} rows onto "
        f"{len(necessary_age_groups)} age groups -> {len(full_index):,} rows"
    )
    sev_submission = sev_submission.reindex(full_index, fill_value=0)

    sev_submission = add_submission_columns(
        sev_submission.reset_index(), measure, "sev"
    )
    sev_submission["scenario"] = sev_submission["scenario"].map(
        SCENARIO_FHS_TRANSLATION
    )

    unmapped = sev_submission["scenario"].isna().sum()
    if unmapped:
        message = (
            f"{unmapped} rows have a scenario with no FHS ID. Known scenarios: "
            f"{sorted(SCENARIO_FHS_TRANSLATION)}."
        )
        raise ValueError(message)
    if sev_submission.isna().to_numpy().any():
        na_cols = sev_submission.columns[sev_submission.isna().any()].tolist()
        message = f"Submission frame has missing values in {na_cols}."
        raise ValueError(message)

    return sev_submission


def write_sev_submission(
    sev_submission: pd.DataFrame,
    submission_dir: Path,
    measure: str,
    *,
    dry_run: bool = False,
) -> None:
    """Write one HDF5 per scenario under ``{submission_dir}/{today}/{measure}``."""
    today = pd.Timestamp.today().strftime("%Y%m%d")
    out_root = Path(submission_dir) / today / measure

    for sc in sev_submission.scenario.unique():
        sc_path = out_root / f"{sc}" / "sev"
        out_path = sc_path / f"sev_data_{sc}.h5"
        scenario_rows = sev_submission.loc[sev_submission.scenario == sc]
        if dry_run:
            print(f"[dry run] would write {len(scenario_rows):,} rows to {out_path}")
            continue
        sc_path.mkdir(parents=True, exist_ok=True)
        scenario_rows.to_hdf(out_path, key="data")
        print(f"Wrote {len(scenario_rows):,} rows to {out_path}")


def prepare_submission_main(
    output_dir: Path,
    measure: str,
    results_version: str,
    submission_dir: Path,
    year_upload_start: int = YEAR_UPLOAD_START,
    *,
    dry_run: bool = False,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir / measure)
    sev_submission = build_sev_submission(
        cm_data, measure, results_version, year_upload_start
    )
    write_sev_submission(
        sev_submission, submission_dir, measure, dry_run=dry_run
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@click.option(
    "--submission-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help=(
        "Directory to write the submission files to. A "
        "{today}/{measure}/{scenario}/sev/ tree is created underneath it."
    ),
)
@click.option(
    "--year-upload-start",
    type=int,
    default=YEAR_UPLOAD_START,
    show_default=True,
    help="First year to include in the upload.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Build the submission frame and report it, but write nothing.",
)
def prepare_submission(
    output_root: str,
    measure: str,
    results_version: str,
    submission_dir: Path,
    year_upload_start: int,
    dry_run: bool,
) -> None:
    """Reshape residual SEV draws into FHS save_results upload files."""
    prepare_submission_main(
        Path(output_root),
        measure,
        results_version,
        submission_dir,
        year_upload_start,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    prepare_submission()
