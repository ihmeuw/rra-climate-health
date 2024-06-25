from typing import ParamSpec, TypeVar

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
    with_choice,
    with_debugger,
    with_input_directory,
    with_num_cores,
    with_output_directory,
    with_progress_bar,
    with_queue,
    with_verbose,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")


VALID_MEASURES = ["wasting", "stunting"]


def with_measure(
    *,
    choices: list[str] = VALID_MEASURES,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "measure",
        "-m",
        allow_all=allow_all,
        choices=choices,
        help="The nutrition measure to run.",
    )


def with_model_id() -> ClickOption[_P, _T]:
    return click.option(
        "--model-id",
        help="A string that identifies the particular model to run.",
        type=str,
    )


VALID_FHS_LOCATION_IDS = ["179"]


def with_location_id(
    *,
    choices: list[str] = VALID_FHS_LOCATION_IDS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "location-id",
        "-l",
        allow_all=allow_all,
        choices=choices,
        help="The location ID to run.",
    )


VALID_CMIP6_SCENARIOS = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]


def with_cmip6_scenario(
    *,
    choices: list[str] = VALID_CMIP6_SCENARIOS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "cmip6-scenario",
        "-c",
        allow_all=allow_all,
        choices=choices,
        help="The CMIP6 scenario to run.",
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
    )


VALID_PREDICTION_YEARS = [str(year) for year in range(2021, 2101)]


def with_year(
    *,
    choices: list[str] = VALID_PREDICTION_YEARS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "year",
        "-y",
        allow_all=allow_all,
        choices=choices,
        help="The year to run.",
    )
