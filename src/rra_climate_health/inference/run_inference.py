import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from functools import partial

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterra as rt
import xarray as xr
from rasterio.features import rasterize
from rra_tools import jobmon

from copy import deepcopy
from rra_climate_health import cli_options as clio
from rra_climate_health import utils
# Re-exported at module level for backwards compatibility with scripts that
# import these from here; the values live in rra_climate_health.constants.
from rra_climate_health.constants import (
    AGE_GROUP_AGGREGATES,
    FIRST_FORECAST_YEAR,
    LAST_FORECAST_YEAR,
    LAST_GBD_YEAR,
    REFERENCE_SCENARIO,
)
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from rra_climate_health.inference.inference_diagnostics import (
    create_inference_diagnostics_report,
)
from rra_climate_health.model_specification import (
    ModelSpecification,
    PredictorSpecification,
)

#from memory_profiler import profile
import gc

#FORECASTED_POPULATIONS_FILEPATH = '/mnt/share/forecasting/data/9/future/population/20250219_draining_fix_old_pop_v5/population.nc'
#HISTORICAL_POPULATIONS_FILEPATH = '/mnt/share/forecasting/data/9/past/population/20231002_etl_run_id_359/population.nc'

FORECASTED_POPULATIONS_FILEPATH = '/mnt/share/forecasting/data/32/future/population/future_population_s130v41/population.nc'
HISTORICAL_POPULATIONS_FILEPATH = '/mnt/share/forecasting/data/16/past/population/20250603_etl_run_id_417/population.nc'

CMIP_LDI_SCENARIO_MAP = {
    #"ssp119": "1",
    "ssp126": "better",
    "ssp245": "reference",
    # "ssp370",
    "ssp585": "worse",
    "constant_climate": "better",
}

# Income predictors are resolved by exact name rather than by prefix. A prefix
# test cannot tell the per-decile income variable apart from a country-level
# summary of the same variable (consumption_pd_national_median starts with
# "consumption_pd"), and `next()` would silently return whichever the spec
# happened to list first.
#
# DECILE_INCOME_PREDICTORS  - varies across the income distribution, so the
#     prevalence integration walks it decile by decile.
# NATIONAL_INCOME_PREDICTORS - one value per country-year, read off the national
#     LDI distribution at NATIONAL_INCOME_PERCENTILE. Constant across deciles, so
#     it is added to z_accum once, before the decile loop.
#
# Anything not named here falls through to get_transformed_variable_raster,
# which looks for a climate NetCDF under
# .../climate_downscale/results/annual/<scenario>/<name>/<draw>.nc. A national
# income predictor that is missing from this tuple therefore fails with a
# FileNotFoundError naming a climate path, which is misleading - the variable
# is not climate data at all. Every training-time alias for the national
# median belongs here; they differ only in how the column was built during
# training, and all resolve to the same quantity at inference.
#
# All of them must be trained in dollars per DAY, which is what
# load_national_ldi_percentile returns. A column left in the annual units
# ldipc is stored in does not raise; under a log transform it silently shifts
# the linear predictor by beta * log(365).
# consumption_pd_cumul is the child_mortality analogue of consumption_pd: built
# at training time as a cumulative mean over the child's exposure months, but
# resolved at inference exactly like consumption_pd, off the per-decile LDI
# distribution. Training already treats it that way - get_points_for_spline_effect
# tests startswith("consumption_pd"), so the decile-mapped spline effects file is
# written for it - and this list is the matching entry on the inference side.
DECILE_INCOME_PREDICTORS = ("ldi_pc_pd", "consumption_pd", "consumption_pd_cumul")
NATIONAL_INCOME_PREDICTORS = (
    "consumption_pd_national_median",
    "consumption_median_RT",
)
NATIONAL_INCOME_PERCENTILE = 0.5


def get_national_income_z(
    predictor,
    cm_data: ClimateMalnutritionData,
    ldi_scenario: str,
    ldi_version: str,
    year: int,
    raster_template: rt.RasterArray,
    coefficients,
    var_info,
) -> rt.RasterArray:
    """beta * transform(national LDI percentile), broadcast over a country's pixels.

    The predictor is a NATIONAL quantity, so the value is broadcast downwards:
    one number per country, copied to every admin2 unit in it and then to every
    pixel of that admin2. Nothing is aggregated up from admin2. admin2 only
    appears because the raster stores admin2 location_ids, so it is the join key
    between a pixel and its country.
    """
    national = cm_data.load_national_ldi_percentile(
        ldi_version, ldi_scenario, year, NATIONAL_INCOME_PERCENTILE
    )
    # admin2 location_id -> its country's ihme_loc_id -> that country's value
    national_parent = cm_data.load_national_parent_of_admin2(
        ldi_version, ldi_scenario, year
    )
    by_admin2 = national_parent.map(national).astype(np.float32)

    admin2_raster = cm_data.load_admin2_raster()
    loc_flat = pd.Series(admin2_raster.to_numpy().ravel())
    mapped = loc_flat.map(by_admin2)
    values = mapped.to_numpy().reshape(raster_template.shape)

    # Only pixels that carry a real admin2 id count as missing. Ocean and other
    # off-land pixels hold the raster's no_data_value and are NaN by design, so
    # counting every NaN in `values` would report the whole ocean as a failure.
    on_land = loc_flat.notna()
    no_data = admin2_raster.no_data_value
    if no_data is not None and not (isinstance(no_data, float) and np.isnan(no_data)):
        on_land &= loc_flat != no_data
    unmapped = on_land & mapped.isna()
    if unmapped.any():
        missing_ids = sorted(loc_flat[unmapped].unique())
        print(
            f"{predictor.name}: {int(unmapped.sum()):,} land pixels across "
            f"{len(missing_ids)} admin2 unit(s) have no national LDI value and "
            f"stay NaN (first few: {missing_ids[:10]})"
        )

    beta = coefficients.loc[predictor.name].astype(np.float32)
    transform_func = var_info[predictor.name]["transformer"]
    return rt.RasterArray(
        beta * np.array(transform_func(values)).astype(np.float32),
        transform=raster_template.transform,
        crs=raster_template.crs,
        no_data_value=np.nan,
    ).astype(np.float32)


def categorical_values_match(label: Any, value: Any) -> bool:
    """Does a coefficient's category label refer to the value being predicted?

    The two sides never share a type. The label is parsed out of a formula
    string such as "C(sex_id)2", so it is always text, while the value comes
    from the task arguments and is typically an int.

    Casting both through the training dtype does not reconcile them when that
    dtype is `object`: numpy's object_ is a no-op, so '2' == 2 stays False. The
    failure is silent rather than loud - every level misses, the function falls
    through to its reference-level branch and returns 0.0, and the term simply
    vanishes from the linear predictor. That is what made both sex_id tasks
    write byte-identical rasters at the male (reference) level.

    Numeric equality is tried first so 2, '2' and 2.0 agree, with a trimmed
    string comparison behind it for genuinely non-numeric levels.
    """
    if label == value:
        return True
    try:
        return float(label) == float(value)
    except (TypeError, ValueError):
        return str(label).strip() == str(value).strip()


def get_categorical_coefficient(
    coefs: "pd.Series[float]",
    variable_name: str,
    variable_value: Any,
    training_data_types: "pd.Series[Any]",  # noqa: ARG001  kept for call compatibility
) -> float:
    # Regex to extract the value from the categorical predictor coefficients
    pattern = re.compile(rf"C\({variable_name}\)(.+)")

    # Look for match
    for idx in coefs.index:
        match = pattern.match(idx)
        if match and categorical_values_match(match.group(1), variable_value):
            print(
                f"Variable '{variable_name}' found with value '{variable_value}' in coefficients."
            )
            return coefs.loc[idx]  # type: ignore[no-any-return]

    # Check if the base variable is in the predictors as a categorical, if so, assume the provided value is the reference
    base_var_index = f"C({variable_name})"
    matching_indices = [idx for idx in coefs.index if idx.startswith(base_var_index)]

    if matching_indices:
        print(
            f"Variable '{variable_name}' found, but the value '{variable_value}' is not in coefficients, assuming it's the reference."
        )
        return 0.0

    # If not categorical, just return the coefficient
    if variable_name in coefs.index:
        return coefs[variable_name]

    error_message = f"Variable '{variable_name}' not found in the series."
    raise KeyError(error_message)


def get_intercept_raster(
    pred_spec: PredictorSpecification,
    coefs: pd.DataFrame,
    ranefs: pd.DataFrame,
    fhs_shapes: gpd.GeoDataFrame,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    icept = coefs.loc["(Intercept)"]
    if pred_spec.random_effect == "ihme_loc_id":
        shapes = list(
            ranefs["X.Intercept."]
            .reset_index()
            .merge(fhs_shapes, left_on="index", right_on="ihme_lc_id", how="left")
            .loc[:, ["geometry", "X.Intercept."]]
            .itertuples(index=False, name=None)
        )
        icept_arr = rasterize(
            shapes,
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        icept_raster = rt.RasterArray(
            icept + icept_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan,
        )
    elif pred_spec.random_effect == "lbd_admin2_id":
        shapes = list(
            ranefs["X.Intercept."]
            .reset_index()
            .merge(fhs_shapes, left_on="index", right_on="loc_id", how="left")
            .loc[:, ["geometry", "X.Intercept."]]
            .itertuples(index=False, name=None)
        )
        icept_arr = rasterize(
            shapes,
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        icept_raster = rt.RasterArray(
            icept + icept_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan,
        )
    elif not pred_spec.random_effect:
        icept_raster = raster_template + icept
    else:
        msg = "Only location random intercepts are supported"
        raise NotImplementedError(msg)
    return icept_raster

#@profile
def get_nonspatial_predictor_raster(
    pred_spec: PredictorSpecification,
    pred_value: float,
    coefs: pd.DataFrame,
    ranefs: pd.DataFrame,
    fhs_shapes: gpd.GeoDataFrame,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    var_fixef = coefs.loc[pred_spec.name]
    if pred_spec.random_effect == "ihme_loc_id":
        shapes = list(
            ranefs[pred_spec.name]
            .reset_index()
            .merge(fhs_shapes, left_on="index", right_on="ihme_lc_id", how="left")
            .loc[:, ["geometry", pred_spec.name]]
            .itertuples(index=False, name=None)
        )
        var_arr = rasterize(
            shapes,
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        var_raster = rt.RasterArray(
            var_fixef + var_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan,
        ).astype(np.float32)
    elif not pred_spec.random_effect:
        var_raster = raster_template + var_fixef
    else:
        msg = "Only location random intercepts are supported"
        raise NotImplementedError(msg)
    return var_raster * pred_value  # type: ignore[no-any-return]

# Inference reads the ANNUAL climate rasters, but a `_prev_N_mo` training
# column is looked up straight out of the MONTHLY ERA5 file with no arithmetic
# (see get_climate_vars_for_prev_months in run_training_data_prep). The two
# sources are not in the same units, so the annual raster has to be rescaled to
# whatever the monthly file holds before the fitted coefficient is applied.
#
# Verified against the sources for 2010 by averaging all twelve monthly slices
# and comparing with the annual value on the same grid:
#
#   mean over the 12 monthly slices :    2.7234
#   annual value                    :  893.69
#   annual/12                          mean 74.4743   ratio to monthly 27.346
#   annual/365                         mean  2.4485   ratio to monthly  0.899
#
# so total_precipitation is a per-DAY rate monthly and a per-YEAR total
# annually: the divisor is 365, not 12. Dividing by 12 inflated the term about
# 27-fold, which at beta = -0.003976 subtracted 0.13 to 0.46 from the linear
# predictor and pushed predictions well below both GBD and the survey data.
#
# days_over_* is a per-month day count in the monthly file and a per-year day
# count annually, so 12 is right there. mean_temperature is an average rather
# than an accumulation and needs no rescaling at all.
ANNUAL_TO_TRAINING_DIVISOR = {
    "total_precipitation": 365.0,
    "mean_temperature": 1.0,
}
DEFAULT_ANNUAL_TO_TRAINING_DIVISOR = 12.0


def get_annual_to_training_divisor(variable: str) -> float:
    """Rescale an annual climate raster into the units the model was fitted on."""
    return ANNUAL_TO_TRAINING_DIVISOR.get(
        variable, DEFAULT_ANNUAL_TO_TRAINING_DIVISOR
    )


#@profile
def get_transformed_variable_raster(predictor, cm_data, cmip6_scenario, year, raster_template, coefficients, var_info, draw, spline_effect_loaders, apply_coefficient=True):
    print(f"Getting raster for {predictor.name} {year} {cmip6_scenario} {draw} {apply_coefficient}")
    if predictor.random_effect:
        msg = "Random slopes not implemented"
        raise NotImplementedError(msg)
    
    transform_func = var_info[predictor.name]["transformer"]
    beta = coefficients.loc[predictor.name].astype(np.float32) if apply_coefficient and predictor.spline is None else 1.0
    transform_meta = predictor.transform

    if predictor.name == "elevation":
        v = cm_data.load_elevation().resample_to(raster_template).astype(np.float32)
        return rt.RasterArray(
                beta * np.array(transform_func(v)).astype(np.float32),
                transform=v.transform,
                crs=v.crs,
                no_data_value=np.nan,
        ).astype(np.float32)
    else:
        variable = (
            transform_meta.from_column
            if hasattr(transform_meta, "from_column")
            else predictor.name
        )
        # TODO: temporary
        divisor = 1.0
        if re.match(r"(days_over_[^_]+|precipitation_days|total_precipitation|mean_temperature)_.+", variable):
            variable = re.sub(r"^(days_over_[^_]+|precipitation_days|total_precipitation|mean_temperature)_.+$", r"\1", variable)
            divisor = get_annual_to_training_divisor(variable)
            print(f"Using variable {variable} for predictor {predictor.name}")
        ds = cm_data.load_climate_raster(variable, cmip6_scenario, year, draw).astype(np.float32)
        if divisor != 1.0:
            print(f"Rescaling annual {variable} to training units (/{divisor:g})")
            ds = ds / divisor
    rasterize = lambda x: utils.xarray_to_raster(x, nodata=np.nan).resample_to(raster_template).astype(np.float32)

    if predictor.spline is not None:
        # Apply spline transformation to the variable raster
        spline_effects = spline_effect_loaders[predictor.name]()
        spline_effects_ds = xr.DataArray(spline_effects.set_index("value").sort_index().effect.values, coords=[spline_effects.value.sort_values()], dims=["value"])
        ds = spline_effects_ds.sel(value=ds, method='nearest').drop_vars('value')
        return rt.RasterArray(
            np.array(rasterize(ds).astype(np.float32)),
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan,
            ).astype(np.float32)

    return rt.RasterArray(
        beta * np.array(transform_func(rasterize(ds).astype(np.float32))),
        transform=raster_template.transform,
        crs=raster_template.crs,
        no_data_value=np.nan,
        ).astype(np.float32)


def get_spline_effect_loaders(spec: ModelSpecification, cm_data: ClimateMalnutritionData):
    spline_effects = {}
    for predictor in spec.predictors:
        if predictor.spline is not None:
            # partial(function, arg1, arg2) creates a callable with those args pre-filled
            spline_effects[predictor.name] = partial(
                cm_data.load_spline_effects, 
                spec.version.model, 
                predictor.name
            )
            print(f"Loaded spline effects for predictor {predictor.name}")
    return spline_effects

#@profile
def get_model_prevalence(  # noqa: C901 PLR0912
    spec: ModelSpecification,
    cmip6_scenario: str,
    year: int,
    age_group_id: int | str,
    sex_id: int,
    cm_data: ClimateMalnutritionData,
    raster_template: rt.RasterArray,
    training_data_version: str,
    draw: int,
) -> rt.RasterArray:
    coefs, ranefs = cm_data.load_submodel_coefficients(spec.version.model, [("age_group_id", age_group_id), ("sex_id", sex_id)])
    var_info = cm_data.load_model_variable_info(spec.version.model)
    training_data_types = cm_data.load_training_data_types(training_data_version)
    spline_effect_loaders = get_spline_effect_loaders(spec, cm_data)

    # The decile integration below needs these, and so does the national-income
    # branch inside the loop, so resolve them up front.
    ldi_scenario = CMIP_LDI_SCENARIO_MAP[cmip6_scenario]
    income_predictor = next(
        (x for x in spec.predictors if x.name in DECILE_INCOME_PREDICTORS), None
    )
    if income_predictor is None:
        error_message = (
            "No per-decile income predictor in the specification; expected one of "
            f"{DECILE_INCOME_PREDICTORS}."
        )
        raise ValueError(error_message)
    ldi_version = income_predictor.version

    # partial_estimates = {}
    z_accum = np.zeros_like(raster_template) #raster_template.copy()
    for predictor in spec.predictors:
        print(predictor.name)
        if predictor.name in DECILE_INCOME_PREDICTORS:
            continue  # deal with after

        if predictor.name in NATIONAL_INCOME_PREDICTORS:
            # Constant within a country-year, so it belongs in z_accum rather
            # than inside the decile loop.
            z_accum = z_accum + get_national_income_z(
                predictor,
                cm_data,
                ldi_scenario,
                predictor.version or ldi_version,
                year,
                raster_template,
                coefs,
                var_info,
            )
            continue

        if predictor.name == "intercept":
            z_accum = z_accum + cm_data.load_rasterized_intercept(spec.version.model)

        elif predictor.name in ["year_start", "year", "year_id", "birth_year"]:
            z_accum = z_accum + coefs.loc[predictor.name].astype(np.float32) * var_info[predictor.name]["transformer"](np.array([[var_info[predictor.name]['max']]])).astype(np.float32)[0][0]

        elif predictor.name == "sdi":
            sdi = cm_data.load_rasterized_variable(predictor.name, year)
            z_accum = z_accum + rt.RasterArray(
                coefs.loc["sdi"] * np.array(var_info["sdi"]["transformer"](sdi)),
                transform=sdi.transform,
                crs=sdi.crs,
                no_data_value=np.nan,
            )
        elif predictor.name in {"sex_id", "age_group_id", "age_sex"}:
            if predictor.transform.type != "categorical":
                error_message = (
                    f"Only categorical predictors are allowed for {predictor.name}"
                )
                raise ValueError(error_message)
            category_coef = get_categorical_coefficient(
                coefs,
                predictor.name,
                age_group_id if predictor.name == "age_group_id" else sex_id if predictor.name == "sex_id" else f"{age_group_id}_{sex_id}",
                training_data_types, 
            )
            # Not a raster, but the coefficient applies to the whole raster and can be added to the sum
            z_accum = z_accum + category_coef # type: ignore[assignment]
        elif (
            spec.measure in clio.NAMED_AGE_STRATA_MEASURES
            and predictor.transform.type == "categorical"
            and predictor.name in clio.AGE_GROUP_IDS_BY_MEASURE[spec.measure]
        ):
            # One-hot age-interval dummy (e.g. age_1_m): 1 for the stratum this
            # task is computing, 0 (the reference level) for every other stratum.
            category_coef = get_categorical_coefficient(
                coefs,
                predictor.name,
                1 if predictor.name == age_group_id else 0,
                training_data_types,
            )
            z_accum = z_accum + category_coef # type: ignore[assignment]
        else:
            z_accum = z_accum + get_transformed_variable_raster(predictor, cm_data, cmip6_scenario, year, raster_template, coefs, var_info, draw, spline_effect_loaders)

    threshold_flag_varname = next(
        (x.name for x in spec.predictors if x.name.startswith("any")), None
    )
    threshold_predictor = next(
        (x for x in spec.predictors if x.name == threshold_flag_varname), None
    )
    # income_predictor, ldi_scenario and ldi_version are resolved before the
    # predictor loop above, because the national-income branch needs them too.
    prevalence = 0
    
    if not threshold_flag_varname and income_predictor.spline is None:
        beta_ldi = coefs.loc[income_predictor.name].astype(np.float32)
        for i in range(1, 11):
            gc.collect()
            print(i)
            prevalence += 0.1 * 1 / (1 + np.exp(-(z_accum + 
                get_ldi_z_component(ldi_scenario, year, ldi_version, beta_ldi, 1, var_info, i / 10., cm_data, raster_template))) +
                0)
    elif threshold_flag_varname and income_predictor.spline is None:
        if spec.extra_terms != [f"{threshold_flag_varname} * {income_predictor.name}"] and spec.extra_terms != [f"{threshold_flag_varname} : {income_predictor.name}"]:
            msg = "Only threshold variable binary flag and LDI interaction is supported"
            raise NotImplementedError(msg)

        beta_interaction = (
            coefs.loc[f"{income_predictor.name}:{threshold_flag_varname}"]
            / coefs.loc[threshold_flag_varname]
        )
        beta_ldi = (
            beta_interaction * get_transformed_variable_raster(threshold_predictor, cm_data, cmip6_scenario, year, raster_template, coefs, var_info, draw, spline_effect_loaders)#partial_estimates[threshold_flag_varname]
            + coefs.loc[income_predictor.name]
        ).to_numpy().astype(np.float32)
        for i in range(1, 11):
            gc.collect()
            print(i)
            prevalence += 0.1 * 1 / (1 + np.exp(-(z_accum + 
                get_ldi_z_component(ldi_scenario, year, ldi_version, beta_ldi, 1, var_info, i / 10., cm_data, raster_template))) +
                0)
    elif income_predictor.spline is not None and threshold_flag_varname is None:
        loc_raster_flat = cm_data.load_admin2_raster().to_numpy().ravel()
        for i in range(1, 11):
            gc.collect()
            income_decile = i / 10.
            income_effect_mapping = spline_effect_loaders[income_predictor.name](filters=[("year_id", "==", year), 
                ("scenario", "==", ldi_scenario), ("population_percentile", "==", income_decile)]).set_index('location_id')['effect']

            prevalence += 0.1 * 1 / (1 + np.exp(-(z_accum + 
                pd.Series(loc_raster_flat).map(income_effect_mapping).to_numpy().reshape(raster_template.shape)
                )))
            
    elif income_predictor.spline is not None and threshold_flag_varname is not None:
        print("Threshold binary {threshold_flag_varname} and income splines")
        loc_raster_flat = cm_data.load_admin2_raster().to_numpy().ravel()
        if spec.extra_terms != [f"{threshold_flag_varname} * {income_predictor.name}"] and spec.extra_terms != [f"{threshold_flag_varname} : {income_predictor.name}"]:
            msg = "Only threshold variable binary flag and LDI interaction is supported"
            raise NotImplementedError(msg)

        beta_ldi = (
            coefs.loc[f"{threshold_flag_varname}:{income_predictor.name}"] * get_transformed_variable_raster(threshold_predictor, cm_data, cmip6_scenario, year, raster_template, coefs, var_info, draw, spline_effect_loaders)
        ).to_numpy().astype(np.float32)
        for i in range(1, 11):
            gc.collect()
            income_decile = i / 10.
            income_effect_mapping = spline_effect_loaders[income_predictor.name](filters=[("year_id", "==", year), 
                ("scenario", "==", ldi_scenario), ("population_percentile", "==", income_decile)]).set_index('location_id')['effect']

            prevalence += 0.1 * 1 / (1 + np.exp(-(z_accum + 
                pd.Series(loc_raster_flat).map(income_effect_mapping).to_numpy().reshape(raster_template.shape) +
                get_ldi_z_component(ldi_scenario, year, ldi_version, beta_ldi, 1, var_info, i / 10., cm_data, raster_template)
                )))

    return prevalence.astype(np.float32)  # type: ignore[attr-defined,no-any-return]

# def get_ldi_betas(predictors, extra_terms, coefs, cm_data, cmip6_scenario, year, raster_template, var_info, draw):
#     ldi_present = any(x.name == "ldi_pc_pd" for x in predictors)
#     if not ldi_present:
#         return None
    
#     beta_dict = {}
#     current_degree = 1
#     while(True):
#         ldi_pred_name = "ldi_pc_pd" if current_degree == 1 else f"I(ldi_pc_pd^{current_degree})"
#         if ldi_pred_name in coefs.index:
#             beta_dict[current_degree] = coefs.loc[ldi_pred_name].astype(np.float32)
#             interacting_terms = [x for x in coefs.index if ldi_pred_name in x and x != ldi_pred_name]
#             for interacting_term in interacting_terms:
#                 interacting_predictor = [x for x in predictors if x.name == interacting_term.replace(f"{ldi_pred_name}:", "")][0]
#                 beta_dict[current_degree] += coefs.loc[interacting_term] * get_transformed_variable_raster(interacting_predictor, cm_data, cmip6_scenario, year, raster_template, coefs, var_info, draw, apply_coefficient=False).astype(np.float32)
#             current_degree += 1
#         else:
#             break
#     return beta_dict

#@profile
def get_ldi_z_component(ldi_scenario:str, year:int, ldi_version:str, beta_ldi, degree, var_info, decile, cm_data: ClimateMalnutritionData, raster_template):
    # Get the per-decile consumption/income variable name. Matched exactly, so a
    # country-level summary such as consumption_pd_national_median cannot be
    # picked up here and have its transform applied to the decile raster.
    for var in DECILE_INCOME_PREDICTORS:
        if var in var_info:
            ldi_varname = var
            break
    else:
        raise ValueError(
            "No per-decile LDI or consumption variable found in variable info "
            f"object; expected one of {DECILE_INCOME_PREDICTORS}."
        )
    
    transform_func = var_info[ldi_varname]["transformer"]
    dec_str = f"{decile:.1f}"

    z_ldi = rt.RasterArray(
        beta_ldi * np.power(np.array(transform_func(cm_data.load_ldi_raster(ldi_scenario, year, dec_str, ldi_version)).astype(np.float32)), degree),
        transform=raster_template.transform,
        crs=raster_template.crs,
        no_data_value=np.nan,
    ).astype(np.float32)
    return z_ldi

#@profile
def model_inference_main(
    output_dir: Path,
    measure: str,
    results_version: str,
    model_version: str,
    cmip6_scenario: str,
    year: int,
    sex_id: int,
    age_group_id: int | str,
    draw: int,
    save_rasters: bool = False,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir / measure)
    spec = cm_data.load_model_specification(model_version)
    training_data_version = spec.version.training_data
    print("loading raster template and shapes")
    raster_template = cm_data.load_raster_template()

    # We need to produce raster results for each age, sex, year, scenario
    # Year and scenario are paralellized for, so we deal with age and sex here
    #training_data = cm_data.load_training_data(spec.version.training_data)
    # TODO: clean up comments and extra code here
    #for age_group_id in training_data.age_group_id.unique():
    #    for sex_id in training_data.sex_id.unique():
    print(f"Computing prevalence for age {age_group_id} and sex {sex_id}")

    model_prevalence = get_model_prevalence(
        spec,
        cmip6_scenario,
        year,
        age_group_id,
        sex_id,
        cm_data,
        # model_location_effect_shapes,
        raster_template,
        training_data_version,
        draw,
    )
    if save_rasters and year in (LAST_GBD_YEAR, LAST_FORECAST_YEAR):
        cm_data.save_raster_results(
            model_prevalence,
            results_version,
            cmip6_scenario,
            year,
            age_group_id,
            sex_id,
            draw,
        )

    print("Computing zonal statistics")
    result_shapes = cm_data.load_fhs_shapes(most_detailed_only=False)
    if "lbd_admin2_id" in spec.random_effects:
        model_location_effect_shapes = cm_data.load_lbd_admin2_shapes()
    else:
        r = result_shapes
    print("loading population")
    pop_raster = cm_data.load_population_raster().set_no_data_value(np.nan)

    shape_map = (
        result_shapes[result_shapes.most_detailed == 1]
        .set_index("loc_id")
        .geometry.to_dict()
    )
    out = []
    count = model_prevalence * pop_raster
    for shape_id, shape in shape_map.items():
        numerator = np.nansum(count.clip(shape).mask(shape))
        denominator = np.nansum(pop_raster.clip(shape).mask(shape))
        out.append(
            (
                shape_id,
                age_group_id,
                sex_id,
                numerator / denominator,
            )
        )

    print("saving results table")
    df = pd.DataFrame(out, columns=["location_id", "age_group_id", "sex_id", "value"])
    df["year_id"] = year
    df["draw"] = draw
    df["scenario"] = cmip6_scenario
    cm_data.save_results_table(df, results_version, cmip6_scenario, year, age_group_id, sex_id, draw)


def load_population_timeseries(
    locs_of_interest: Sequence[int] | None,
    age_group_ids: Sequence[int],
) -> pd.DataFrame:
    """Load past and future population draws.

    Pass ``locs_of_interest=None`` to load every location in the population
    files, which is what the residual step's hierarchy aggregation wants.
    """
    loc_selector = (
        {} if locs_of_interest is None else {"location_id": list(locs_of_interest)}
    )
    age_group_ids = list(age_group_ids)

    non_integer = [a for a in age_group_ids if not isinstance(a, (int, np.integer))]
    if non_integer:
        msg = (
            "load_population_timeseries needs integer GBD age group IDs, got "
            f"{non_integer!r}. The population and GBD inputs are keyed on int64 "
            "age_group_id, so named model strata (child_mortality's age_1_m ... "
            "age_60_m) must be collapsed to a reported age group before this point."
        )
        raise TypeError(msg)

    requested_aggregates = [a for a in age_group_ids if a in AGE_GROUP_AGGREGATES]
    requested_detailed = [a for a in age_group_ids if a not in AGE_GROUP_AGGREGATES]
    detailed_to_load = sorted(
        set(requested_detailed).union(
            *(AGE_GROUP_AGGREGATES[a] for a in requested_aggregates)
        )
    )

    def _aggregate(pop_df: pd.DataFrame) -> pd.DataFrame:
        idx_names = list(pop_df.index.names)
        age_level = pop_df.index.get_level_values("age_group_id")
        parts: list[pd.DataFrame] = []
        if requested_detailed:
            parts.append(pop_df.loc[age_level.isin(requested_detailed)])
        for agg in requested_aggregates:
            sub = pop_df.loc[age_level.isin(AGE_GROUP_AGGREGATES[agg])].reset_index()
            sub["age_group_id"] = agg
            parts.append(sub.groupby(idx_names).sum())
        return pd.concat(parts).sort_index()

    forecast_pop = (
        xr.open_dataset(FORECASTED_POPULATIONS_FILEPATH)
        .sel(
            age_group_id=detailed_to_load,
            year_id=range(FIRST_FORECAST_YEAR, LAST_FORECAST_YEAR + 1),
            scenario=130,
            **loc_selector,
        )
        .to_dataframe().drop(columns = ["scenario"])
        .pivot_table(
            index=["location_id", "year_id", "age_group_id", "sex_id"],
            columns="draw",
            values="draws",
        )
    )
    forecast_pop.columns = [f"draw_{i}" for i in forecast_pop.columns]
    idx_cols = ['location_id', 'year_id', 'age_group_id', 'sex_id']
    forecast_pop = forecast_pop.reset_index().set_index(idx_cols)

    forecast_pop = _aggregate(forecast_pop)

    forecast_sex_ids = forecast_pop.index.get_level_values("sex_id").unique().tolist()
    historical_pop = (
        xr.open_dataset(HISTORICAL_POPULATIONS_FILEPATH)
        .sel(
            age_group_id=detailed_to_load,
            sex_id=forecast_sex_ids,
            **loc_selector,
        )
        .to_dataframe()
        .reorder_levels(forecast_pop.index.names)
    )
    historical_pop = _aggregate(historical_pop)
    historical_pop = historical_pop.loc[
        historical_pop.index.get_level_values("year_id")
        < forecast_pop.index.get_level_values("year_id").min()
    ]
    historical_pop = historical_pop[['population']*250]
    historical_pop.columns = forecast_pop.columns
    pop = pd.concat([historical_pop, forecast_pop], axis=0).sort_index()
    return pop

def aggregate_mortality_over_ages(df, resulting_age_group_id = 1, age_column = 'age_group_id'):
    original_idx = list(df.index.names)
    
    # Convert to probability of surviving the age, getting the product, 
    # then re-transforming to the aggregate mortality rate
    temp_df = 1.0 - df
    temp_df = temp_df.reset_index().drop(columns = [age_column])
    temp_df = 1.0 - (temp_df.groupby([x for x in original_idx if x != age_column]).prod())
    temp_df = temp_df.reset_index()
    temp_df[age_column] = resulting_age_group_id
    temp_df = temp_df.set_index(original_idx)
    return temp_df

def forecast_scenarios(
    output_dir: Path,
    measure: str,
    results_version: str,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir / measure)
    results_spec = cm_data.load_results_specification(results_version)
    years = results_spec.years
    scenarios = results_spec.scenarios
    age_group_ids = results_spec.age_groups
    draws = results_spec.draws
    sex_ids = results_spec.sex_ids

    locs_of_interest = (
        cm_data.load_fhs_hierarchy()
        .query("most_detailed == 1")
        .location_id.unique()
        .tolist()
    )

    loc_region_map = cm_data.load_fhs_hierarchy()[
        ["location_id", "location_name", "region_name", "super_region_name"]
    ]
    idx_cols = ["location_id", "year_id",  "sex_id", "age_group_id", "scenario"]

    results_path = Path(DEFAULT_ROOT) / measure / "results" / results_version
    dfs = []
    ingested_files = []
    for scenario in scenarios:
        scenario_dfs = []
        scenario_years = [yr for yr in years if yr >= FIRST_FORECAST_YEAR or scenario == REFERENCE_SCENARIO]
        for year in years:
            for age_group_id in age_group_ids:
                for sex_id in sex_ids:
                    for draw in range(0,draws):
                        sc = scenario if year >= FIRST_FORECAST_YEAR else REFERENCE_SCENARIO
                        d = draw if year >= FIRST_FORECAST_YEAR else 0
                        fp = results_path / f"{year}_{sc}_{age_group_id}_{sex_id}_{d}.parquet"
                        df = pd.read_parquet(fp)
                        df.loc[:, "scenario"] = scenario
                        df.loc[:, "draw"] = draw
                        scenario_dfs.append(df)
                        ingested_files.append(fp)
        scenario_df = pd.concat(scenario_dfs)
        scenario_df = scenario_df.pivot(index=idx_cols, columns='draw', values='value')
        scenario_df.columns = [f'draw_{col}' for col in scenario_df.columns]
        scenario_df = scenario_df.reset_index().set_index(idx_cols).sort_index()
        if measure == 'child_mortality':
            scenario_df = aggregate_mortality_over_ages(scenario_df, resulting_age_group_id = 1)
        scenario_df.to_parquet(
            results_path / f"{scenario}.parquet", index=True)
        dfs.append(scenario_df)


    combined = pd.concat(dfs)
    draw_cols = combined.columns
    combined['prevalence'] = combined.mean(axis=1)
    combined = combined.drop(columns=draw_cols)
    pop = load_population_timeseries(
        locs_of_interest, age_group_ids=combined.index.get_level_values('age_group_id').unique()
    ).mean(axis=1).rename("population").to_frame()

    merged = combined.merge(pop, left_index=True, right_index=True, how="left")
    merged["affected"] = merged["prevalence"] * merged["population"]

    if REFERENCE_SCENARIO in scenarios:
        reference = combined.query(
            "scenario == @REFERENCE_SCENARIO"
        ).droplevel("scenario").rename(
            columns={"prevalence": "ref_prev", "affected": "ref_affected"}
        )
        merged = merged.join(reference, how="left")

        merged["delta"] = (merged["ref_prev"] * merged["population"]) - merged[
            "affected"
        ]
    else:
        merged["delta"] = np.nan

    merged = merged.merge(
        loc_region_map.set_index("location_id"), left_index=True, right_index=True
    )
    cm_data.save_forecast(merged, results_version)
    #delete ingested files
    # for file in ingested_files:
    #     file.unlink()


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
def forecast_scenarios_task(
    output_root: str,
    measure: str,
    results_version: str,
) -> None:
    """Run forecasting applying the inference results, and output diagnostics."""
    forecast_scenarios(
        Path(output_root),
        measure,
        results_version,
    )
    model_version = ClimateMalnutritionData(Path(output_root) / measure).load_results_specification(results_version).version.model
    create_inference_diagnostics_report(
        Path(output_root), measure, results_version, model_version
    )


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@clio.with_model_version()
@clio.with_cmip6_scenario()
@clio.with_year()
@clio.with_sex_id()
@clio.with_age_group_id()
@clio.with_draw()
@clio.with_save_rasters()
def model_inference_task(
    output_root: str,
    measure: str,
    results_version: str,
    model_version: str,
    cmip6_scenario: str,
    year: str,
    sex_id: str,
    age_group_id: str,
    draw: str,
    save_rasters: bool,
) -> None:
    """Run model inference."""
    [resolved_age_group_id] = clio.resolve_age_group_ids_for_measure(measure, [age_group_id])
    model_inference_main(
        Path(output_root),
        measure,
        results_version,
        model_version,
        cmip6_scenario,
        int(year),
        int(sex_id),
        clio.normalize_age_group_id(measure, resolved_age_group_id),
        int(draw),
        save_rasters=save_rasters,
    )

@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_model_version()
@clio.with_measure()
@clio.with_cmip6_scenario(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_sex_id(allow_all=True)
@clio.with_age_group_id(allow_all=True)
@clio.with_queue()
@clio.with_n_draws()
@clio.with_save_rasters()
def model_inference(
    output_root: str,
    model_version: str,
    measure: str,
    cmip6_scenario: list[str],
    year: list[str],
    sex_id: list[int],
    age_group_id: list[int | str],
    draws: int,
    queue: str,
    save_rasters: bool,
) -> None:
    """Run model inference."""
    asked_for_all = set(age_group_id) == set(clio.VALID_AGE_GROUP_IDS)
    age_group_id = clio.resolve_age_group_ids_for_measure(measure, age_group_id)
    cm_data = ClimateMalnutritionData(Path(output_root) / measure)
    # Which strata exist is a property of the measure; which ones this model can
    # speak to is a property of the fit. Narrow to the latter, because a stratum
    # the model never saw is silently returned as a copy of the reference cell
    # rather than raising -- see ClimateMalnutritionData.model_age_strata.
    supported = cm_data.model_age_strata(model_version)
    if supported is not None:
        unsupported = [a for a in age_group_id if a not in supported]
        if unsupported and not asked_for_all:
            msg = (
                f"age-group-id(s) {unsupported} are not covered by model "
                f"{model_version}, which was fit on {supported}."
            )
            raise click.BadParameter(msg)
        if unsupported:
            print(
                f"Model {model_version} covers {len(supported)} of the "
                f"{len(age_group_id)} strata valid for {measure}; "
                f"dropping {unsupported}."
            )
            age_group_id = [a for a in age_group_id if a in supported]
    # The spec records the types the forecast and residual steps will work in; the task
    # fan-out below keeps the raw CLI strings, which each task re-normalizes itself.
    spec_age_group_ids = clio.normalize_age_group_ids(measure, age_group_id)
    results_version = cm_data.new_results_version(model_version, spec_age_group_ids,
        sex_id, year, cmip6_scenario, draws, save_rasters=save_rasters)
    draw_range = list(range(0, draws))
    print(
        f"Running inference for {measure} using {model_version}, making {draws} draws. \n \
        Results version: {results_version}"
    )
    historical_years = [yr for yr in year if int(yr) < FIRST_FORECAST_YEAR]
    forecast_years = [yr for yr in year if int(yr) >= FIRST_FORECAST_YEAR]

    arg_names = (
        "measure",
        "cmip6-scenario",
        "year",
        "sex-id",
        "age-group-id",
        "draw",
    )
    task_tuples: list[tuple] = []

    if len(historical_years) > 0:
        if len(cmip6_scenario) == 1:
            historical_scenarios = [cmip6_scenario[0]]
        elif REFERENCE_SCENARIO in cmip6_scenario:
            historical_scenarios = [REFERENCE_SCENARIO]
        else:
            historical_scenarios = cmip6_scenario
        task_tuples.extend(itertools.product(
            [measure], historical_scenarios, historical_years,
            sex_id, age_group_id, [0],
        ))

    if len(forecast_years) > 0:
        task_tuples.extend(itertools.product(
            [measure], cmip6_scenario, forecast_years,
            sex_id, age_group_id, draw_range,
        ))

    inference_task_args: dict[str, Any] = {
        "output-root": output_root,
        "model-version": model_version,
        "results-version": results_version,
    }
    if save_rasters:
        # A None value renders as a bare --save-rasters in the task command.
        inference_task_args["save-rasters"] = None

    if task_tuples:
        print(
            f"Submitting {len(task_tuples)} inference tasks "
            f"(historical years: {historical_years}, forecast years: {forecast_years})"
        )
        jobmon.run_parallel(
            runner="sttask",
            task_name="inference",
            flat_node_args=(arg_names, task_tuples),
            task_args=inference_task_args,
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": "55Gb",
                "runtime": "80m",
                "project": "proj_rapidresponse",
            },
            max_attempts=2,
            concurrency_limit = 1000,
            log_root=str(cm_data.results / results_version),
        )
    
    jobmon.run_parallel(
        runner="sttask",
        task_name="forecast",
        node_args={
            "measure": [measure],
        },
        task_args={
            "output-root": output_root,
            "results-version": results_version,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "30Gb",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        max_attempts=2,
        log_root=str(cm_data.results / results_version),
    )

    # The rasters only exist for LAST_GBD_YEAR and LAST_FORECAST_YEAR, and only
    # the forecast year has multiple draws to coalesce.  These run after the
    # forecast so a raster problem never blocks the main deliverable; each
    # run_parallel call blocks, so all draw rasters exist before coalescing and
    # all mean rasters exist before the diff diagnostics.
    if save_rasters and str(LAST_FORECAST_YEAR) in [str(yr) for yr in year]:
        jobmon.run_parallel(
            runner="sttask",
            task_name="coalesce_rasters",
            node_args={
                "measure": [measure],
                "cmip6-scenario": cmip6_scenario,
                "age-group-id": age_group_id,
                "sex-id": sex_id,
            },
            task_args={
                "output-root": output_root,
                "results-version": results_version,
            },
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": "20Gb",
                "runtime": "120m",
                "project": "proj_rapidresponse",
            },
            max_attempts=2,
            log_root=str(cm_data.results / results_version),
        )
        jobmon.run_parallel(
            runner="sttask",
            task_name="raster_diagnostics",
            node_args={
                "measure": [measure],
            },
            task_args={
                "output-root": output_root,
                "results-version": results_version,
            },
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": "60Gb",
                "runtime": "30m",
                "project": "proj_rapidresponse",
            },
            max_attempts=2,
            log_root=str(cm_data.results / results_version),
        )

    print(
        f"Inference complete, results can be found at {cm_data.results / results_version}"
    )

