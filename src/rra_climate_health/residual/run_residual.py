"""Residual modeling step.

This runs after forecasting.  It takes the forecast draws produced by
``strun inference`` / ``sttask forecast`` and

1. fits an SDI-based model of the residual between GBD prevalence and the
   climate model's reference-scenario prevalence, and adds the predicted
   residual back onto the model draws,
2. intercept-shifts the result so that it matches GBD in the last GBD year,
   carrying the scenario deltas over from the unshifted draws,
3. for the child growth failure measures, converts the shifted prevalence to
   SEVs and shifts those to match GBD SEVs in the last GBD year,
4. writes a table of means and produces the diagnostic plots.

Ported from ``malnutrition_fhs/src/residual.py``.
"""

import re
from pathlib import Path

import click
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from rra_tools import jobmon
from scipy.special import expit, logit

from rra_climate_health import cli_options as clio
from rra_climate_health.constants import (
    LAST_GBD_YEAR,
    MEASURES_WITH_SEV,
    MIN_PREV_VALUE,
    REFERENCE_SCENARIO,
)
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from rra_climate_health.residual.residual_data import (
    get_gbd_data,
    get_prev_to_sev_table,
    get_sdi,
)
from rra_climate_health.residual.residual_diagnostics import make_results_plots

IDX_COLS = ["location_id", "year_id", "age_group_id", "sex_id"]

# Whether the SDI slope of the residual model is constrained to be
# non-positive for every super region.
ENFORCE_NON_POSITIVE_SDI_SLOPES = True


def get_residual_prediction(
    cm_data: ClimateMalnutritionData,
    measure: str,
    results_version: str,
    location_hierarchy: pd.DataFrame,
    last_gbd_year: int = LAST_GBD_YEAR,
    gbd_measure: str | None = None,
) -> pd.DataFrame:
    """Fit the GBD-vs-model residual on SDI and predict it over the forecast.

    The residual is modeled as a function of SDI with a super-region specific
    slope and a location/age/sex specific intercept.
    """
    gbd_df = get_gbd_data(
        cm_data, measure, 0, draws=False, metric="prevalence",
        gbd_measure=gbd_measure,
    ).reset_index()
    sdi_df = get_sdi()

    model_df = cm_data.load_forecast(results_version)
    model_df = model_df.reset_index()
    age_group_ids = model_df.age_group_id.unique()
    sex_ids = model_df.sex_id.unique()

    model_ref_df = model_df[model_df["scenario"] == REFERENCE_SCENARIO]
    model_ref_df = model_ref_df[
        ["location_id", "year_id", "sex_id", "age_group_id", "prevalence"]
    ]
    model_ref_df = model_ref_df.rename(columns={"prevalence": "model_value"})

    gbd_df = gbd_df[gbd_df["year_id"] <= last_gbd_year]
    gbd_df = gbd_df[gbd_df.age_group_id.isin(model_ref_df.age_group_id.unique())] 
    gbd_df = gbd_df.rename(columns={"gbd_mean_prevalence": "gbd_value"})

    gbd_model_df = pd.merge(
        gbd_df,
        model_ref_df,
        on=["location_id", "year_id", "sex_id", "age_group_id"],
        how="inner",
    )

    gbd_model_df["residual_value"] = (
        gbd_model_df["gbd_value"] - gbd_model_df["model_value"]
    )
    gbd_model_df = pd.merge(
        gbd_model_df, sdi_df, on=["location_id", "year_id"], how="left"
    )

    gbd_model_df["location_age_sex"] = (
        gbd_model_df.location_id.astype(str)
        + "_"
        + gbd_model_df.age_group_id.astype(str)
        + "_"
        + gbd_model_df.sex_id.astype(str)
    )
    gbd_model_df["location_age"] = (
        gbd_model_df.location_id.astype(str)
        + "_"
        + gbd_model_df.age_group_id.astype(str)
    )
    gbd_model_df["location_sex"] = (
        gbd_model_df.location_id.astype(str) + "_" + gbd_model_df.sex_id.astype(str)
    )
    gbd_model_df = gbd_model_df.merge(
        location_hierarchy[["location_id", "region_name", "super_region_name"]],
        on="location_id",
        how="left",
    )

    residual_prediction_df = sdi_df[
        sdi_df["location_id"].isin(gbd_model_df["location_id"].unique())
    ]
    residual_prediction_df = residual_prediction_df.merge(
        pd.MultiIndex.from_product(
            [age_group_ids, sex_ids], names=["age_group_id", "sex_id"]
        ).to_frame(index=False),
        how="cross",
    )

    residual_prediction_df = residual_prediction_df.merge(
        location_hierarchy[["location_id", "region_name", "super_region_name"]],
        on="location_id",
        how="left",
    )
    residual_prediction_df["location_age_sex"] = (
        residual_prediction_df.location_id.astype(str)
        + "_"
        + residual_prediction_df.age_group_id.astype(str)
        + "_"
        + residual_prediction_df.sex_id.astype(str)
    )

    # `sdi:C(...)` rather than `sdi + sdi*C(...)`: a pure interaction with no
    # main sdi term makes patsy use full dummy coding, so EVERY super region
    # gets its own standalone slope coefficient and none is a reference level.
    #
    # The old formula gave each region `sdi + sdi:C(...)[T.region]`, a slope
    # shared with the reference region plus an offset. That shared term is what
    # made the non-positive-slope adjustment move every region at once (see
    # adjust_sdi_slopes_non_positive). It also expanded out C(super_region_name)
    # main effects that the location_age_sex dummies already span, which is what
    # made the design rank-deficient.
    residual_model_sr = smf.ols(
        "residual_value ~ 0 + sdi:C(super_region_name) + C(location_age_sex)",
        data=gbd_model_df,
    ).fit()
    print(residual_model_sr.summary())

    if ENFORCE_NON_POSITIVE_SDI_SLOPES:
        adjusted_coeffs = adjust_sdi_slopes_non_positive(
            residual_model_sr.params,
            residual_model_sr.model.exog_names,
            gbd_model_df["super_region_name"].unique(),
        )

        # Predict using the adjusted coefficients with the custom function
        predicted_values = custom_predict_with_coeffs(
            residual_prediction_df,
            adjusted_coeffs,
            residual_model_sr.model.data.design_info,
        )
        residual_prediction_df["super_region_adjusted_slopes_predicted_residual"] = (
            predicted_values
        )
        residual_prediction_df["predicted_residual"] = residual_prediction_df[
            "super_region_adjusted_slopes_predicted_residual"
        ]
    else:
        predicted_values = residual_model_sr.predict(residual_prediction_df)
        residual_prediction_df["super_region_slopes_predicted_residual"] = (
            predicted_values
        )
        residual_prediction_df["predicted_residual"] = residual_prediction_df[
            "super_region_slopes_predicted_residual"
        ]
    return residual_prediction_df


def convert_prev_to_sev(
    cm_data: ClimateMalnutritionData,
    prev_df: pd.DataFrame,
    measure: str,
    historical_prev_sev_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert prevalence draws to SEV draws.

    Fits one mixed model of SEV on prevalence per age group, with a location
    random intercept, on the historical GBD prevalence/SEV pairs.
    """
    if historical_prev_sev_df is None:
        historical_prev_sev_df = get_prev_to_sev_table(cm_data, measure)

    # Check if all locations and ages in prev_df are also in historical_prev_sev_df
    missing_locs = set(prev_df.index.get_level_values("location_id")) - set(
        historical_prev_sev_df["location_id"].unique()
    )
    if missing_locs:
        message = (
            f"Locations {sorted(missing_locs)} are missing from the "
            f"prevalence-to-SEV table for {measure}."
        )
        raise ValueError(message)
    missing_ages = set(prev_df.index.get_level_values("age_group_id")) - set(
        historical_prev_sev_df["age_group_id"].unique()
    )
    if missing_ages:
        message = (
            f"Age groups {sorted(missing_ages)} are missing from the "
            f"prevalence-to-SEV table for {measure}."
        )
        raise ValueError(message)

    age_group_ids = historical_prev_sev_df["age_group_id"].unique()
    prev_to_sev_models = {}
    for age_group_id in age_group_ids:
        df_filtered = historical_prev_sev_df[
            (historical_prev_sev_df["age_group_id"] == age_group_id)
        ].dropna()
        model = smf.mixedlm(
            "sev_val ~ prev_val",
            df_filtered,
            groups=df_filtered["location_id"],
            re_formula="1",
        )
        model_fit = model.fit()
        prev_to_sev_models[age_group_id] = model_fit

    sev_draws = prev_df.clip(MIN_PREV_VALUE, 1)
    draw_cols = sev_draws.columns
    sev_draws["location_id"] = sev_draws.index.get_level_values("location_id")

    for col in draw_cols:
        if not col.startswith("draw_"):
            message = f"Expected only draw columns, got {col}."
            raise ValueError(message)
        temp_col = []
        for age_group_id in age_group_ids:
            temp_df = sev_draws.query("age_group_id == @age_group_id")[
                ["location_id", col]
            ].rename(columns={col: "prev_val"})
            temp_df["sev_val"] = prev_to_sev_models[age_group_id].predict(temp_df)
            temp_col.append(temp_df["sev_val"])
        sev_draws[col] = pd.concat(temp_col)
    sev_draws = sev_draws.drop(columns=["location_id"])
    return sev_draws


def shift_sev_draws(
    cm_data: ClimateMalnutritionData,
    sev_draws: pd.DataFrame,
    gbd_sev_draws: pd.DataFrame,
    results_version: str,
    last_gbd_year: int = LAST_GBD_YEAR,
    reference_scenario: str = REFERENCE_SCENARIO,
) -> pd.DataFrame:
    """Logit-shift the SEV draws so they match GBD SEVs in the last GBD year."""
    print(f"Shifting SEV draws to align with GBD {last_gbd_year} values")
    sev_adjustment_draws = logit(
        gbd_sev_draws.query("year_id == @last_gbd_year")
    ) - logit(
        sev_draws.query(
            "year_id == @last_gbd_year and scenario == @reference_scenario"
        )
        .droplevel("scenario")
        .clip(MIN_PREV_VALUE, 1)
    )
    sev_adjustment_draws = sev_adjustment_draws.droplevel("year_id")

    adjusted_sev_draws = expit(
        logit(sev_draws.clip(MIN_PREV_VALUE, 1)) + sev_adjustment_draws
    )
    cm_data.save_adjusted_sev_draws(adjusted_sev_draws, results_version)
    return adjusted_sev_draws


def custom_predict_with_coeffs(
    new_data_df: pd.DataFrame,
    custom_coeffs_series: "pd.Series[float]",
    original_model_design_info: patsy.design_info.DesignInfo,
) -> "pd.Series[float]":
    """
    Predicts values using new data, custom coefficients,
    and the design_info from the original model.

    Args:
        new_data_df (pd.DataFrame): The new data for which to make predictions.
                                    Must contain all columns referenced in the formula.
        custom_coeffs_series (pd.Series): A Pandas Series where the index contains
                                          the term names (matching original model's exog_names)
                                          and values are the (potentially modified) coefficients.
        original_model_design_info (patsy.DesignInfo): The `design_info` object from the
                                    fitted statsmodels model's data attribute
                                    (e.g., `residual_model_sr.model.data.design_info`).

    Returns:
        pd.Series: The predicted values.
    """
    if not isinstance(custom_coeffs_series, pd.Series):
        message = "custom_coeffs_series must be a Pandas Series."
        raise TypeError(message)
    if not isinstance(original_model_design_info, patsy.design_info.DesignInfo):
        message = "original_model_design_info must be a patsy.DesignInfo object."
        raise TypeError(message)

    # Create the design matrix from the new data using the original model's design_info.
    # This ensures consistent handling of categorical variables, interactions, etc.
    # build_design_matrices returns a list of matrices (one for LHS, one for RHS if
    # formula had LHS). We only need the RHS matrix (X matrix).
    # If the original formula had an explicit outcome variable (e.g., "y ~ x"),
    # original_model_design_info might describe both. We are interested in the
    # predictors' matrix.
    # In statsmodels, model.data.design_info usually refers to the RHS.
    try:
        # The first argument to build_design_matrices is a list of DesignInfo objects
        design_matrices_new = patsy.build_design_matrices(
            [original_model_design_info], new_data_df, return_type="dataframe"
        )
        # design_matrices_new will be a list containing one design matrix (the RHS)
        # if original_model_design_info was for the RHS.
        if not design_matrices_new or len(design_matrices_new) == 0:
            message = (
                "patsy.build_design_matrices returned an empty list. "
                "Check data and design_info."
            )
            raise ValueError(message)
        design_matrix_new_rhs = design_matrices_new[0]

    except patsy.PatsyError as e:
        print(f"PatsyError during design matrix creation: {e}")
        print(
            "Ensure all necessary columns are in new_data_df and data types "
            "are consistent."
        )
        raise
    except Exception as e:
        print(f"An unexpected error occurred with patsy.build_design_matrices: {e}")
        raise

    # Align the columns of the new design matrix with the coefficient names.
    aligned_design_matrix = design_matrix_new_rhs.reindex(
        columns=custom_coeffs_series.index, fill_value=0
    )

    # Perform the prediction: X * beta
    predictions = aligned_design_matrix.dot(custom_coeffs_series)
    return predictions


def adjust_sdi_slopes_non_positive(
    coefficients_series: "pd.Series[float]",
    model_exog_names: list[str],
    all_super_region_levels: list[str],
) -> "pd.Series[float]":
    """Clip any positive SDI slope to zero, one super region at a time.

    The model is fitted with `sdi:C(super_region_name)` and no main sdi term,
    so each super region owns a standalone slope coefficient and none of them
    is shared. Clipping one therefore cannot move any other.

    This used to be far more involved, because the old formula gave each region
    a shared `sdi` term plus an offset. Fixing a positive REFERENCE region meant
    zeroing that shared term, which silently re-based every other region's
    slope by the same amount - Sub-Saharan Africa went from -2.844 to -6.565
    without anything in its own data changing, and because the intercepts were
    left alone the predictions stopped averaging to the observed residuals.

    Args:
        coefficients_series: the fitted coefficients.
        model_exog_names: the fitted model's exog_names, used to find the slope
            terms.
        all_super_region_levels: every super region level present in the data,
            used only to warn when one has no slope term.

    Returns:
        A copy of the coefficients with every positive SDI slope set to zero.
    """
    modified_coeffs = coefficients_series.copy()

    # Terms look like 'sdi:C(super_region_name)[Sub-Saharan Africa]'. The
    # optional 'T.' is tolerated so a reference-coded fit still parses.
    pattern = re.compile(r"^sdi:C\(super_region_name\)\[(?:T\.)?(.*)\]$")
    slope_terms = {}
    for term_name in model_exog_names:
        match = pattern.match(term_name)
        if match:
            slope_terms[match.group(1)] = term_name

    if not slope_terms:
        print(
            "Warning: no 'sdi:C(super_region_name)[...]' slope terms found; "
            "leaving coefficients unchanged. Check the model formula."
        )
        return modified_coeffs

    missing = [lvl for lvl in all_super_region_levels if lvl not in slope_terms]
    if missing:
        print(
            f"Warning: no slope term for super region(s) {missing}. With "
            "`sdi:C(super_region_name)` every level should have one; a level "
            "without a term is being treated as a reference and cannot be "
            "clipped."
        )
    if "sdi" in modified_coeffs.index:
        print(
            "Warning: a bare 'sdi' coefficient is present, so slopes are "
            "shared across super regions and clipping one will move the "
            "others. Expected the formula to use `sdi:C(super_region_name)`."
        )

    n_clipped = 0
    for level, term_name in sorted(slope_terms.items()):
        slope = float(modified_coeffs[term_name])
        if slope > 0:
            print(f"  {level}: slope {slope:+.4f} is positive -> 0.0")
            modified_coeffs[term_name] = 0.0
            n_clipped += 1
        else:
            print(f"  {level}: slope {slope:+.4f} is not positive, left as fitted")
    print(
        f"Clipped {n_clipped} of {len(slope_terms)} super-region SDI slopes; "
        f"the other {len(slope_terms) - n_clipped} are unchanged."
    )
    return modified_coeffs


def residual_main(  # noqa: PLR0915
    output_dir: Path,
    measure: str,
    results_version: str,
    gbd_measure: str | None = None,
) -> None:
    """Run the residual model, the intercept shift and the SEV conversion."""
    cm_data = ClimateMalnutritionData(output_dir / measure)

    print(
        f"Running residual model and SEV conversion for {measure} with results "
        f"version {results_version}"
    )
    if gbd_measure:
        print(f"  scoring against the GBD files for '{gbd_measure}', not '{measure}'")

    results_spec = cm_data.load_results_specification(results_version)
    scenarios = results_spec.scenarios
    n_draws = results_spec.draws
    # Bound locally so it can be referenced from `DataFrame.query` expressions.
    reference_scenario = REFERENCE_SCENARIO

    fhs_loc_meta = cm_data.load_fhs_hierarchy()

    model_draws = cm_data.load_scenario_draws(results_version, REFERENCE_SCENARIO)
    age_group_ids = (
        results_spec.age_groups
        if measure != "child_mortality"
        else model_draws.index.get_level_values("age_group_id").unique()
    )

    model_draws = model_draws.droplevel("scenario")
    gbd_draws = get_gbd_data(cm_data, measure, n_draws, draws=True,
                             gbd_measure=gbd_measure)
    locs = model_draws.index.get_level_values("location_id").unique()
    gbd_draws = gbd_draws.query(
        "age_group_id in @age_group_ids and location_id in @locs"
    )

    residual_prediction_df = get_residual_prediction(
        cm_data, measure, results_version, fhs_loc_meta, gbd_measure=gbd_measure
    )
    model_plus_predicted_residual = model_draws.add(
        residual_prediction_df.query("year_id >= 2000").set_index(IDX_COLS)[
            "predicted_residual"
        ],
        axis=0,
    )

    # Shifting prevalence to align to GBD last year
    last_gbd_year = LAST_GBD_YEAR
    shift_amount = gbd_draws.query(
        "year_id == @last_gbd_year"
    ) - model_plus_predicted_residual.query("year_id == @last_gbd_year")
    shift_amount = shift_amount.droplevel("year_id")

    shifted_draws = model_plus_predicted_residual.add(shift_amount)

    shifted_allscenarios = []
    for scenario in scenarios:
        if scenario == REFERENCE_SCENARIO:
            shifted_scenario = shifted_draws.copy()
        else:
            diff_scenario = cm_data.load_scenario_draws(
                results_version, scenario
            ).droplevel("scenario") - cm_data.load_scenario_draws(
                results_version, REFERENCE_SCENARIO
            ).droplevel("scenario")
            shifted_scenario = shifted_draws.add(diff_scenario)
        # A prevalence cannot be negative. The residual model is an additive
        # extrapolation in prevalence space with nothing bounding it, so it can
        # and does push the shifted draws below zero far from the anchor year.
        # Clip here, at the point the draws are written, so the saved files and
        # everything downstream of them agree with the diagnostic plots - which
        # were already clipping at display time only.
        shifted_scenario = shifted_scenario.clip(lower=0.0)
        shifted_scenario["scenario"] = scenario
        shifted_scenario = shifted_scenario.set_index("scenario", append=True)
        cm_data.save_shifted_scenario_draws(
            shifted_scenario, results_version, scenario
        )
        shifted_allscenarios.append(shifted_scenario)
    shifted_allscenarios = pd.concat(shifted_allscenarios)
    cm_data.save_shifted_prevalence(shifted_allscenarios, results_version)

    if measure in MEASURES_WITH_SEV:
        sev_draws = convert_prev_to_sev(cm_data, shifted_allscenarios, measure)
        gbd_sev_draws = get_gbd_data(cm_data, measure, n_draws, draws=True, metric="sev")
        gbd_sev_draws = gbd_sev_draws.query(
            "age_group_id in @age_group_ids and location_id in @locs"
        )
        adjusted_sev_draws = shift_sev_draws(
            cm_data,
            sev_draws,
            gbd_sev_draws,
            results_version,
            last_gbd_year,
            reference_scenario=REFERENCE_SCENARIO,
        )

        draws_mean_df = (
            adjusted_sev_draws.mean(axis=1)
            .rename("adjusted_sev_val")
            .to_frame()
            .query("scenario == @reference_scenario")
            .droplevel("scenario")
        )
        draws_mean_df["predicted_sev_val"] = (
            sev_draws.query("scenario == @reference_scenario")
            .mean(axis=1)
            .droplevel("scenario")
        )
        draws_mean_df["gbd_sev"] = gbd_sev_draws.reorder_levels(
            draws_mean_df.index.names
        ).mean(axis=1)

        for scenario in scenarios:
            if scenario != REFERENCE_SCENARIO:
                draws_mean_df[f"adjusted_sev_{scenario}"] = (
                    adjusted_sev_draws.query("scenario == @scenario")
                    .mean(axis=1)
                    .droplevel("scenario")
                )
    else:
        draws_mean_df = pd.DataFrame()

    for scenario in scenarios:
        if scenario != REFERENCE_SCENARIO:
            draws_mean_df[f"adjusted_{scenario}"] = (
                shifted_allscenarios.query("scenario == @scenario")
                .mean(axis=1)
                .droplevel("scenario")
            )
    draws_mean_df["adjusted_model_value"] = shifted_draws.clip(lower=0.0).mean(axis=1)
    draws_mean_df["residual_model_value"] = model_plus_predicted_residual.reorder_levels(
        draws_mean_df.index.names
    ).mean(axis=1)
    draws_mean_df["model_value"] = model_draws.reorder_levels(
        draws_mean_df.index.names
    ).mean(axis=1)
    draws_mean_df["gbd_value"] = gbd_draws.reorder_levels(
        draws_mean_df.index.names
    ).mean(axis=1)
    draws_mean_df["adjusted_model_value_"] = draws_mean_df["adjusted_model_value"].clip(
        MIN_PREV_VALUE, 1
    )

    cm_data.save_sev_means(draws_mean_df, results_version)


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@clio.with_gbd_measure()
def residual_task(
    output_root: str,
    measure: str,
    results_version: str,
    gbd_measure: str | None,
) -> None:
    """Run the residual model on a set of forecast results, and plot diagnostics."""
    residual_main(
        Path(output_root),
        measure,
        results_version,
        gbd_measure=gbd_measure,
    )
    make_results_plots(Path(output_root), measure, results_version)


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@clio.with_queue()
@clio.with_gbd_measure()
def run_residual(
    output_root: str,
    measure: str,
    results_version: str,
    queue: str,
    gbd_measure: str | None,
) -> None:
    """Run the residual step for a results version produced by inference."""
    cm_data = ClimateMalnutritionData(Path(output_root) / measure)
    # Fail fast with a clear message if forecasting hasn't been run yet.
    forecast_path = cm_data.results / results_version / "forecast.parquet"
    if not forecast_path.exists():
        message = (
            f"No forecast found at {forecast_path}. Run the forecast step for "
            f"{measure} results version {results_version} first."
        )
        raise click.ClickException(message)

    if gbd_measure:
        # Fail before the fan-out rather than inside the task.
        for _p in (
            cm_data.gbd_inputs / f"gbd_mean_{gbd_measure}_prevalence.parquet",
            cm_data.gbd_inputs / f"gbd_draws_{gbd_measure}_prevalence_100.parquet",
        ):
            if not _p.exists():
                message = f"--gbd-measure {gbd_measure} given, but {_p} is missing."
                raise click.ClickException(message)
        print(f"Scoring against the GBD files for '{gbd_measure}', not '{measure}'")

    print(f"Running residual step for {measure}, results version {results_version}")
    jobmon.run_parallel(
        runner="sttask",
        task_name="residual",
        node_args={
            "measure": [measure],
        },
        task_args={
            "output-root": output_root,
            "results-version": results_version,
            **({"gbd-measure": gbd_measure} if gbd_measure else {}),
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "150Gb",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(cm_data.results / results_version),
    )
    print(
        f"Residual step complete, results can be found at "
        f"{cm_data.results / results_version}"
    )
