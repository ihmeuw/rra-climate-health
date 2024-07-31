from typing import ParamSpec, TypeVar, Callable, Any
from pathlib import Path

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
    with_choice,
    with_debugger,
    with_input_directory,
    with_num_cores,
    with_progress_bar,
    with_queue,
    with_verbose,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")


VALID_MEASURES = ["wasting", "stunting"]


def get_choice_callback(
    allow_all: bool,
    choices: list[str],
) -> Callable[[Any, Any, Any], list[str] | str]:
    if allow_all:
        return lambda ctx, param, value: choices if value == RUN_ALL else [value]
    else:
        return lambda ctx, param, value: value


def with_measure(
    *,
    choices: list[str] = VALID_MEASURES,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "measure",
        "m",
        allow_all=allow_all,
        choices=choices,
        help="The nutrition measure to run.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_SOURCE_TYPES = [
    'cgf',
]


def with_source_type(
    *,
    choices: list[str] = VALID_SOURCE_TYPES,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "source_type",
        allow_all=allow_all,
        choices=choices,
        help="The source type of data to prep (cgf vs bmi).",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_CMIP6_SCENARIOS = [
    "ssp119",
    # "ssp126",
    "ssp245",
    # "ssp370",
     "ssp585",
     "constant_climate",
]


def with_cmip6_scenario(
    *,
    choices: list[str] = VALID_CMIP6_SCENARIOS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "cmip6-scenario",
        "c",
        allow_all=allow_all,
        choices=choices,
        help="The CMIP6 scenario to run.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_SEX_IDS = ["1", "2"]


def with_sex_id(
    *,
    choices: list[str] = VALID_SEX_IDS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "sex-id",
        "s",
        allow_all=allow_all,
        choices=choices,
        help="The sex ID to run. 1 is Male and 2 is Female.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_AGE_GROUP_IDS = ["4", "5"]


def with_age_group_id(
    *,
    choices: list[str] = VALID_AGE_GROUP_IDS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "age-group-id",
        "a",
        allow_all=allow_all,
        choices=choices,
        help="The age group ID to run.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_PREDICTION_YEARS = [str(year) for year in range(2000, 2101)]


def with_year(
    *,
    choices: list[str] = VALID_PREDICTION_YEARS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "year",
        "y",
        allow_all=allow_all,
        choices=choices,
        help="The year to run.",
        callback=get_choice_callback(allow_all, choices),
    )


def with_overwrite() -> ClickOption[_P, _T]:
    return click.option(
        "--overwrite",
        help="Overwrite existing files.",
        is_flag=True,
    )


def with_output_root(default: str | Path) -> ClickOption[_P, _T]:
    return click.option(
        "--output-root",
        "-o",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        default=default,
        show_default=True,
        help="Root directory where outputs will be saved.",
    )


def with_results_version() -> ClickOption[_P, _T]:
    return click.option(
        "--results-version",
        "-r",
        type=str,
        required=True,
        help="The results version to run.",
    )


def with_model_version() -> ClickOption[_P, _T]:
    return click.option(
        "--model-version",
        "-t",
        type=str,
        required=True,
        help="The model version to run.",
    )
