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
) -> pd.DataFrame:
    """Fit the GBD-vs-model residual on SDI and predict it over the forecast.

    The residual is modeled as a function of SDI with a super-region specific
    slope and a location/age/sex specific intercept.
    """
    gbd_df = get_gbd_data(
        cm_data, measure, 0, draws=False, metric="prevalence"
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
    gbd_df = gbd_df[~gbd_df.age_group_id.isin([2, 3])]  # filter out 2/3
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

    residual_model_sr = smf.ols(
        "residual_value ~ 0 + sdi + sdi*C(super_region_name) + C(location_age_sex)",
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


def adjust_sdi_slopes_non_positive(  # noqa: C901, PLR0912
    coefficients_series: "pd.Series[float]",
    model_exog_names: list[str],
    all_super_region_levels: list[str],
) -> "pd.Series[float]":
    """
    Adjusts SDI-related coefficients in a copy of the coefficients series
    so that the effective slope of 'sdi' for any super_region_name is never positive.

    The adjustment prioritizes changing the main 'sdi' coefficient if the
    reference region's slope is positive, and then adjusts interaction terms.

    Args:
        coefficients_series (pd.Series): The original coefficient series from the model.
        model_exog_names (list): List of all exogenous variable names from the
                                 fitted model (e.g., residual_model_sr.model.exog_names).
                                 Used to identify interaction terms correctly.
        all_super_region_levels (list): A list of all unique super_region_name
                                        category levels (e.g., from
                                        gbd_model_df['super_region_name'].cat.categories.tolist()).

    Returns:
        pd.Series: A new pandas Series with the modified coefficients.
    """
    modified_coeffs = coefficients_series.copy()

    sdi_main_coeff_name = "sdi"  # Standard name for the main sdi coefficient

    if sdi_main_coeff_name not in modified_coeffs:
        print(
            f"Warning: Main SDI coefficient '{sdi_main_coeff_name}' not found in "
            "coefficients. Cannot perform slope adjustments for SDI."
        )
        return modified_coeffs

    current_beta_sdi = modified_coeffs[sdi_main_coeff_name]
    print(f"Initial main '{sdi_main_coeff_name}' coefficient: {current_beta_sdi:.4f}")

    # Discover sdi:C(super_region_name) interaction terms and their associated levels
    sdi_interaction_term_map = {}  # Maps super_region_level -> interaction_term_name
    found_interaction_levels = []

    # Regex to capture level from terms like 'sdi:C(super_region_name)[T.SR2]' or
    # 'sdi:C(super_region_name)[SR_North]'
    # This pattern tries to be flexible for common patsy naming conventions.
    interaction_pattern = re.compile(r"sdi:C\(super_region_name\)\[(T\.)?(.*?)\]$")

    for term_name in model_exog_names:
        match = interaction_pattern.fullmatch(term_name)
        if match:
            # group(2) is the actual level name, e.g., "SR2" or "SR_North"
            level_name = match.group(2)
            sdi_interaction_term_map[level_name] = term_name
            found_interaction_levels.append(level_name)

    # Determine reference levels (those in all_super_region_levels but not having an
    # explicit interaction term)
    reference_super_region_levels = [
        level for level in all_super_region_levels if level not in found_interaction_levels
    ]

    if not reference_super_region_levels and not sdi_interaction_term_map:
        print(
            "Warning: No SDI interaction terms found and no reference levels "
            "identified. Check model formula and `super_region_name` levels if "
            "this is unexpected."
        )
    elif not reference_super_region_levels and len(found_interaction_levels) == len(
        all_super_region_levels
    ):
        print(
            "Note: All super_region_name levels appear to have explicit sdi "
            "interaction terms. The main 'sdi' coefficient's role might differ from "
            "a typical reference slope. Proceeding by checking its value first."
        )

    # Step 1: Adjust for reference super_region_name level(s)
    # The effective slope for a reference level is simply the main 'sdi' coefficient.
    if reference_super_region_levels:  # Only if there are clearly defined reference levels
        print(
            "Identified reference super_region_name level(s) for sdi interaction: "
            f"{reference_super_region_levels}"
        )
        if current_beta_sdi > 0:
            print(
                f"  Slope for reference region(s) ({current_beta_sdi:.4f}) is positive. "
                f"Adjusting main '{sdi_main_coeff_name}' coefficient: "
                f"{current_beta_sdi:.4f} -> 0.0"
            )
            modified_coeffs[sdi_main_coeff_name] = 0.0
            current_beta_sdi = 0.0  # Update for subsequent calculations
        else:
            print(
                f"  Slope for reference region(s) ({current_beta_sdi:.4f}) is not "
                f"positive. No change to main '{sdi_main_coeff_name}'."
            )
    elif not sdi_interaction_term_map:  # No interactions at all, only main sdi effect
        if current_beta_sdi > 0:
            print(
                f"  Only main '{sdi_main_coeff_name}' effect found. Its value "
                f"({current_beta_sdi:.4f}) is positive. Adjusting main "
                f"'{sdi_main_coeff_name}' coefficient: {current_beta_sdi:.4f} -> 0.0"
            )
            modified_coeffs[sdi_main_coeff_name] = 0.0
            current_beta_sdi = 0.0
    else:
        # All levels have interaction terms, or some other structure.
        # Check main sdi independently.
        if current_beta_sdi > 0 and not reference_super_region_levels:
            # If main SDI term represents a general intercept for slope before
            # interactions
            print(
                f"  Main '{sdi_main_coeff_name}' coefficient ({current_beta_sdi:.4f}) "
                "is positive. Given all regions have interactions or no clear "
                "reference, setting it to 0.0 to ensure it doesn't contribute "
                "positively before interactions are considered."
            )
            # Decision: Do we always zero out main sdi if positive and all regions
            # have interactions? Or do we let each region's interaction fully define
            # its offset from potentially positive main sdi? Forcing overall slope to
            # be <=0, it is safer to adjust main sdi if it is positive.
            print(
                f"  Adjusting main '{sdi_main_coeff_name}' coefficient: "
                f"{current_beta_sdi:.4f} -> 0.0"
            )
            modified_coeffs[sdi_main_coeff_name] = 0.0
            current_beta_sdi = 0.0

    # Step 2: Adjust for non-reference super_region_name levels (those with explicit
    # interaction terms)
    print("\nProcessing non-reference regions (or all regions if explicit terms exist):")
    for level, interaction_term_name in sdi_interaction_term_map.items():
        beta_sdi_interaction = modified_coeffs[interaction_term_name]
        # Effective slope for this level = (potentially modified) main_sdi_coeff +
        # interaction_coeff
        effective_slope = current_beta_sdi + beta_sdi_interaction

        print(f"  Region '{level}' (term: '{interaction_term_name}'):")
        print(
            f"    Current main_sdi_coeff: {current_beta_sdi:.4f}, "
            f"Interaction_coeff: {beta_sdi_interaction:.4f}"
        )
        print(f"    Calculated effective_slope: {effective_slope:.4f}")

        if effective_slope > 0:
            # We want: current_beta_sdi + new_beta_sdi_interaction = 0
            # So, new_beta_sdi_interaction = -current_beta_sdi
            new_interaction_coeff_value = -current_beta_sdi
            print(
                f"    Effective slope is positive. Adjusting interaction "
                f"'{interaction_term_name}': {beta_sdi_interaction:.4f} -> "
                f"{new_interaction_coeff_value:.4f}"
            )
            modified_coeffs[interaction_term_name] = new_interaction_coeff_value
        else:
            print(
                f"    Effective slope is not positive. No change to "
                f"'{interaction_term_name}'."
            )

    return modified_coeffs


def residual_main(  # noqa: PLR0915
    output_dir: Path,
    measure: str,
    results_version: str,
) -> None:
    """Run the residual model, the intercept shift and the SEV conversion."""
    cm_data = ClimateMalnutritionData(output_dir / measure)

    print(
        f"Running residual model and SEV conversion for {measure} with results "
        f"version {results_version}"
    )

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
    gbd_draws = get_gbd_data(cm_data, measure, n_draws, draws=True)
    locs = model_draws.index.get_level_values("location_id").unique()
    gbd_draws = gbd_draws.query(
        "age_group_id in @age_group_ids and location_id in @locs"
    )

    residual_prediction_df = get_residual_prediction(
        cm_data, measure, results_version, fhs_loc_meta
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
    draws_mean_df["adjusted_model_value"] = shifted_draws.mean(axis=1)
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
def residual_task(
    output_root: str,
    measure: str,
    results_version: str,
) -> None:
    """Run the residual model on a set of forecast results, and plot diagnostics."""
    residual_main(
        Path(output_root),
        measure,
        results_version,
    )
    make_results_plots(Path(output_root), measure, results_version)


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@clio.with_queue()
def run_residual(
    output_root: str,
    measure: str,
    results_version: str,
    queue: str,
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
