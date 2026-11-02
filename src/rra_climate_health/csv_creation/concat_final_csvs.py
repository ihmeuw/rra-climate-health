"""Concatenate final results CSVs into one.

Takes the Goalkeepers-format CSVs written by ``strun csv_creation`` (and the
mortality CSVs from apply_relative_scenario_differences_to_FHS.py, which share
the format) and stacks them, in the order given, into one CSV. Each run writes
to a new versioned directory under ``<output root>/final_csvs``, alongside a
list of the source files, so nothing is overwritten.

Every input must be an absolute path with exactly the Goalkeepers columns,
and no row may appear in more than one input (e.g. the same indicator from
two results versions).
"""

from pathlib import Path

import click
import pandas as pd

from rra_climate_health.csv_creation.run_csv_creation import OUTPUT_COLUMNS
from rra_climate_health.data import DEFAULT_ROOT, get_run_directory

OUTPUT_DIR = "final_csvs"
OUTPUT_FILE = "combined_mean_estimates.csv"
SOURCES_FILE = "sources.txt"

# One row per these; a duplicate across inputs means the same results twice.
ROW_KEYS = [
    "location_id",
    "Sex",
    "Scenario type",
    "Year",
    "Indicator",
    "Age",
    "Indicator units",
]


def load_final_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != OUTPUT_COLUMNS:
        message = (
            f"{path} does not have the expected columns.\n"
            f"  expected: {OUTPUT_COLUMNS}\n  found:    {list(df.columns)}"
        )
        raise click.ClickException(message)
    return df


def concat_final_csvs_main(csv_paths: list[Path], output_root: Path) -> Path:
    frames = []
    for path in csv_paths:
        df = load_final_csv(path)
        indicators = ", ".join(df["Indicator"].unique())
        print(f"  {len(df):>9,} rows  {indicators}  <- {path}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    duplicated = combined.duplicated(ROW_KEYS, keep=False)
    if duplicated.any():
        repeated = combined.loc[duplicated, "Indicator"].unique()
        message = (
            f"{duplicated.sum():,} rows appear in more than one input, for "
            f"indicators {list(repeated)}. Pass each results CSV only once."
        )
        raise click.ClickException(message)

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = get_run_directory(output_root)
    run_dir.mkdir()
    out_path = run_dir / OUTPUT_FILE
    combined.to_csv(out_path, index=False)
    (run_dir / SOURCES_FILE).write_text("".join(f"{p}\n" for p in csv_paths))
    return out_path


@click.command()  # type: ignore[arg-type]
@click.option(
    "--csv-path",
    "-l",
    "csv_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
    help="Absolute path to a final results CSV. Repeat for each CSV.",
)
@click.option(
    "--output-root",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(DEFAULT_ROOT) / OUTPUT_DIR,
    show_default=True,
    help="Directory under which a new versioned output directory is made.",
)
def concat_final_csvs(csv_paths: tuple[Path, ...], output_root: Path) -> None:
    """Concatenate final results CSVs into one CSV."""
    if relative := [str(p) for p in csv_paths if not p.is_absolute()]:
        message = f"CSV paths must be absolute: {relative}"
        raise click.ClickException(message)
    if len(set(csv_paths)) < len(csv_paths):
        message = "The same CSV path was passed more than once."
        raise click.ClickException(message)

    print(f"Concatenating {len(csv_paths)} CSVs:")
    out_path = concat_final_csvs_main(list(csv_paths), output_root)
    print(f"Combined CSV written to {out_path}")
