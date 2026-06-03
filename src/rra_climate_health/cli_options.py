from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
    with_choice,
    with_queue,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")


VALID_MEASURES = ["wasting", "stunting", "underweight", "anemia", "lbw", "neonatal_mortality"]


def get_choice_callback(
    allow_all: bool,
    choices: list[str],
) -> Callable[[Any, Any, Any], list[str] | str]:
    if allow_all:
        return lambda ctx, param, value: choices if value == RUN_ALL else [value]  # noqa: ARG005
    else:
        return lambda ctx, param, value: value  # noqa: ARG005


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
    "cgf",
    "anemia",
    "child_mortality",
    "lbw"
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
    #"ssp119",
    "ssp126",
    "ssp245",
    # "ssp370",
    "ssp585",
    #"constant_climate",
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

AGE_GROUP_IDS_BY_MEASURE: dict[str, list[str]] = {
    "stunting":           ["238", "388", "389", "34"],
    "wasting":            ["238", "388", "389", "34"],
    "underweight":        ["238", "388", "389", "34"],
    "anemia":             ["34", "8", "9", "10", "11", "12", "13", "14", "238", "389", "15"],
    "lbw":                ["164"],
    "neonatal_mortality": ["42"],
}

VALID_AGE_GROUP_IDS = sorted(
    {a for ages in AGE_GROUP_IDS_BY_MEASURE.values() for a in ages},
    key=int,
)

def resolve_age_group_ids_for_measure(
    measure: str,
    age_group_ids: list[str],
) -> list[str]:
    """Validate explicit IDs against the measure; expand 'all' to the measure's set.

    The age-group callback returns the full VALID_AGE_GROUP_IDS list when the
    user passes `all`. That is indistinguishable from an explicit listing of
    every union member, so we treat 'received the full union' as 'expand to
    the measure-specific set'. A hard error is raised on any explicit ID that
    is not valid for the chosen measure.
    """
    allowed = AGE_GROUP_IDS_BY_MEASURE[measure]
    if set(age_group_ids) == set(VALID_AGE_GROUP_IDS):
        return allowed
    invalid = [a for a in age_group_ids if a not in allowed]
    if invalid:
        raise click.BadParameter(
            f"age-group-id(s) {invalid} not valid for measure '{measure}'. "
            f"Allowed for {measure}: {allowed}",
        )
    return age_group_ids

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

def with_wealth_version() -> ClickOption[_P, _T]:
    return click.option(
        "--wealth-version",
        "-w",
        type=str,
        required=True,
        help="The version of wealth (income/consumption) to use.",
    )

def with_n_draws() -> ClickOption[_P, _T]:
    return click.option(
        "--draws",
        "-d",
        type=int,
        default=1,
        required=True,
        help="The number of draws to run.",
    )

def with_draw() -> ClickOption[_P, _T]:
    return click.option(
        "--draw",
        "-d",
        type=int,
        required=True,
        help="The draw to run.",
    )

__all__ = [
    "VALID_MEASURES",
    "with_measure",
    "VALID_SOURCE_TYPES",
    "with_source_type",
    "VALID_CMIP6_SCENARIOS",
    "with_cmip6_scenario",
    "VALID_SEX_IDS",
    "with_sex_id",
    "VALID_AGE_GROUP_IDS",
    "AGE_GROUP_IDS_BY_MEASURE",
    "resolve_age_group_ids_for_measure",
    "with_age_group_id",
    "VALID_PREDICTION_YEARS",
    "with_year",
    "with_overwrite",
    "with_output_root",
    "with_results_version",
    "with_model_version",
    "with_queue",
    "with_wealth_version"
]
