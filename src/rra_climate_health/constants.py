"""Constants shared across pipeline steps.

Ported from the ``malnutrition_fhs`` repository (``src/constants.py``) when the
residual modeling step was brought into this pipeline.  See
``docs/residual_duplication_notes.md`` for the constants that are currently
defined in more than one place.
"""

# The offset used when logit-transforming prevalence draws, and the floor
# applied to prevalence before the transform.
MIN_PREV_VALUE = DEFAULT_OFFSET = 1e-10

# Last year for which GBD estimates are available.  Forecast draws are shifted
# so that they line up with GBD in this year.
LAST_GBD_YEAR = 2023

# First year of the forecast.  Years before this are "past" years and are only
# estimated for the reference scenario.
FIRST_FORECAST_YEAR = 2024

# Last year of the forecast.  Together with LAST_GBD_YEAR, this is one of the
# two years whose prediction rasters are written out when inference is run
# with --save-rasters.
LAST_FORECAST_YEAR = 2100

REFERENCE_SCENARIO = "ssp245"

# The GBD modelable-entity and risk-exposure IDs live in
# ``data_prep/save_gbd_inputs.py``, which is the only thing that uses them.  It
# is deliberately standalone so it can run under an IHME environment, so it
# cannot import from here.

# The age groups that *results* are reported for, by measure.  These are the
# age groups present in the forecast draws, and are not necessarily the age
# groups the models were fit on (see
# ``cli_options.AGE_GROUP_IDS_BY_MEASURE`` for those).
VALID_AGE_GROUPS_FOR_MEASURE = {
    "anemia": [34, 8, 9, 10, 11, 12, 13, 14, 238, 389, 15],
    "neonatal_mortality": [42],
    "child_mortality": [1],
    "stunting": [238, 388, 389, 34],
    "wasting": [238, 388, 389, 34],
    "underweight": [238, 388, 389, 34],
    "lbw": [2],
}

# Aggregate age groups and the detailed age groups they are made of.  Used when
# pulling population for an aggregate age group.
AGE_GROUP_AGGREGATES: dict[int, list[int]] = {
    4: [388, 389],
    5: [238, 34],
    42: [2, 3],
    1: [2, 3, 388, 389, 238, 34],
}

# Measures for which a prevalence-to-SEV conversion is available.
MEASURES_WITH_SEV = ["stunting", "wasting", "underweight"]

# The one (age_group_id, sex_id) whose rasters are used for the raster diff
# diagnostics, per measure.  Age and sex are just categorical terms in the
# model, so a single stratum is representative; there are no population
# forecast rasters to aggregate over.  Values must be *modeled* strata (see
# ``cli_options.AGE_GROUP_IDS_BY_MEASURE``), because the rasters are written
# per modeled stratum.  A missing measure, or an entry not present in the
# results specification, falls back to the first age group / first sex in the
# results spec.  child_mortality is intentionally omitted: its rasters are per
# named survival stratum (age_1_m ... age_60_m), so the fallback applies.
RASTER_DIAGNOSTIC_AGE_SEX: dict[str, tuple[int | str, int]] = {
    "anemia": (8, 2),
    "neonatal_mortality": (42, 2),
    "stunting": (388, 2),
    "wasting": (388, 2),
    "underweight": (388, 2),
    "lbw": (2, 2),
}

# Colorbar label for the raster diff diagnostics.  Plotted values are
# multiplied by 1000, so the labels are phrased per 1000.
RASTER_DIAGNOSTIC_LABELS: dict[str, str] = {
    "anemia": "Anemia Prevalence\n(per 1000)",
    "stunting": "Stunting Prevalence\n(per 1000)",
    "wasting": "Wasting Prevalence\n(per 1000)",
    "underweight": "Underweight Prevalence\n(per 1000)",
    "lbw": "Low Birthweight Prevalence\n(per 1000)",
    "neonatal_mortality": "Neonatal Mortality Rate\nper 1000 births",
    "child_mortality": "Child Mortality Rate\nper 1000 births",
}
