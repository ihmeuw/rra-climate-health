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


VALID_MEASURES = ["wasting", "stunting", "underweight", "anemia", "lbw", "neonatal_mortality", "child_mortality"]


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
    "lbw",
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
    #"anemia":             ["34", "8", "9", "10", "11", "12", "13", "14", "238", "389", "15"],
    "anemia":             ["8", "9", "10", "11", "12", "13", "14"],
    "lbw":                ["2"],
    "neonatal_mortality": ["42"],
    #"child_mortality":    ["42", "388", "888", "389", "238", "50", "51", "52"],
    "child_mortality": ['age_1_m', 'age_3_m', 'age_6_m', 'age_12_m', 'age_24_m', 'age_36_m', 'age_48_m', 'age_60_m',]
}

# Measures whose "age groups" are not GBD age group IDs at all.  ``child_mortality`` is
# fit on named discrete-time survival intervals (``age_1_m`` is [0, 1) months,
# ``age_3_m`` is [1, 3), ... ``age_60_m`` is [48, 60)), which tile 0-60 months and do not
# line up with any GBD age group.  They exist only as a modeling stratum: the forecast
# step collapses them into the single reported age group 1 (Under 5) via
# ``inference.run_inference.aggregate_mortality_over_ages``.  Every other measure's age
# groups are GBD IDs and must be ints by the time they reach a dataframe, because the
# population and GBD inputs are keyed on int64 ``age_group_id``.
NAMED_AGE_STRATA_MEASURES: set[str] = {"child_mortality"}


def normalize_age_group_id(measure: str, age_group_id: str | int) -> int | str:
    """Convert a CLI age-group value to the type the rest of the pipeline expects.

    ``AGE_GROUP_IDS_BY_MEASURE`` holds strings because click choices are strings.  For
    every measure except those in ``NAMED_AGE_STRATA_MEASURES`` those strings are numeric
    GBD IDs and must become ints here -- this is the single conversion point between the
    CLI's string world and the int64 world of the population and GBD inputs.  Named
    strata are passed through untouched.
    """
    if measure in NAMED_AGE_STRATA_MEASURES:
        return age_group_id
    return int(age_group_id)


def normalize_age_group_ids(
    measure: str,
    age_group_ids: list[str] | list[int] | list[str | int],
) -> list[int | str]:
    """List form of :func:`normalize_age_group_id`."""
    return [normalize_age_group_id(measure, a) for a in age_group_ids]


def _age_group_sort_key(age_group_id: str) -> tuple[int, int, str]:
    """Sort numeric age group IDs numerically, then named strata by their month bound.

    Keeps ``age_1_m, age_3_m, ... age_60_m`` in interval order rather than the
    lexical order that would put ``age_12_m`` before ``age_1_m``.
    """
    if age_group_id.isdigit():
        return (0, int(age_group_id), "")
    digits = "".join(c for c in age_group_id if c.isdigit())
    return (1, int(digits) if digits else 0, age_group_id)


# A deterministically ordered list, not a set: ``rra_tools.with_choice`` takes
# ``choices[-1]`` as the default when ``allow_all`` is False, and renders the choices in
# order in ``--help``.  A set would make both vary from process to process.
VALID_AGE_GROUP_IDS = sorted(
    {a for ages in AGE_GROUP_IDS_BY_MEASURE.values() for a in ages},
    key=_age_group_sort_key,
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


def with_save_rasters() -> ClickOption[_P, _T]:
    return click.option(
        "--save-rasters",
        help=(
            "Save the prevalence rasters for the last GBD year and the last "
            "forecast year, coalesce the forecast-year draws to a mean raster, "
            "and produce raster diff diagnostics."
        ),
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
    "NAMED_AGE_STRATA_MEASURES",
    "normalize_age_group_id",
    "normalize_age_group_ids",
    "resolve_age_group_ids_for_measure",
    "with_age_group_id",
    "VALID_PREDICTION_YEARS",
    "with_year",
    "with_overwrite",
    "with_save_rasters",
    "with_output_root",
    "with_results_version",
    "with_model_version",
    "with_queue",
    "with_wealth_version"
]
