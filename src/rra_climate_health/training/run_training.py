import itertools
import os
from pathlib import Path
from typing import Any

import click
import pandas as pd
import numpy as np
import rasterra as rt
from pymer4.models.Lmer import Lmer

from rpy2 import robjects
from rpy2.robjects import pandas2ri, packages, ListVector, FloatVector
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

from rra_tools import jobmon

from rra_climate_health import cli_options as clio
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from rra_climate_health.model_specification import (
    ModelSpecification,
)
from rra_climate_health.transforms import transform_column
from rra_climate_health import utils

from rra_climate_health.training import training_validation, training_diagnostics
from rra_climate_health.training.training_validation import get_knot_values
from rra_climate_health.model_specification import ModelType
import re
import pickle


def create_spline_lookup(
    model_obj,
    model_spec,
    lookup_var: str,
    full_df: pd.DataFrame,
    num_points: int = 1000,
) -> pd.DataFrame:
    """
    Helper function to create a lookup table for the spline contribution of a
    variable from a fitted scam model.
    """

    # Create lookup table for variable
    df = full_df.copy()

    # get the outcome variable from the model specification
    outcome_tuple = [i for i in model_spec if "measure" in i][0]
    outcome_var = outcome_tuple[1].value

    # Get list of variables that are not outcome, lookup variable, or intercept
    non_lookup_vars = [
        col for col in df.columns if not col in (lookup_var, "intercept", outcome_var)
    ]

    intervals = (df[lookup_var].max() - df[lookup_var].min()) / (num_points - 1)
    grid = [df[lookup_var].min() + i * intervals for i in range(num_points)]

    # create constant prediction dataset for all variables lookup variable
    # Use median for numeric variables, mode for categorical variables, and the
    # grid for the variable of interest
    constant_df = pd.DataFrame({lookup_var: grid})
    for var in non_lookup_vars:

        if pd.api.types.is_numeric_dtype(df[var]):
            constant_df[var] = df[var].median()
        elif pd.api.types.is_categorical_dtype(df[var]) or pd.api.types.is_object_dtype(
            df[var]
        ):
            constant_df[var] = pd.Categorical(
                [df[var].mode()[0]] * len(constant_df),
                categories=df[var].unique(),
            )
        else:
            raise ValueError(f"Unimplemented variable type for {var}")

    # Convert the constant DataFrame to an R data frame
    with localconverter(default_converter + pandas2ri.converter):
        r_constant_df = pandas2ri.py2rpy(constant_df)

    # Predict spline contributions
    pred_terms = stats.predict(model_obj, newdata=r_constant_df, type="terms")

    marginal_contribution_varname = f"s({lookup_var})"
    pred_terms_var = pred_terms.rx(
        True, pred_terms.colnames.index(marginal_contribution_varname) + 1
    )

    # Extract the spline contribution for variable
    with localconverter(default_converter + pandas2ri.converter):
        pred_terms_df = pandas2ri.rpy2py(pred_terms_var)

    pred_terms_df = pd.DataFrame(pred_terms_df, columns=[marginal_contribution_varname])
    climate_lookup_df = pd.DataFrame(
        {
            lookup_var: grid,
            "smooth_contribution": pred_terms_df,
        }
    )

    return climate_lookup_df


def model_training_main(
    output_root: Path,
    measure: str,
    model_version: str,
    submodel: list[tuple[str, str]] | None = None,
) -> None:

    output_dir = os.path.join(output_root, measure, "inference", model_version)
    cm_data = ClimateMalnutritionData(output_root / measure)
    model_spec = cm_data.load_model_specification(model_version)

    # Load training data
    full_training_data = cm_data.load_training_data(model_spec.version.training_data)
    # TODO: Prep leaves a bad index
    full_training_data = full_training_data.reset_index(drop=True)
    full_training_data["intercept"] = 1.0

    subset_mask = pd.Series(True, index=full_training_data.index)  # noqa: FBT003
    if submodel:
        for var, value in submodel:
            # Convert value to the type of the column in the training data and build subset mask
            retyped_value = full_training_data[var].dtype.type(value)
            subset_mask = (full_training_data[var] == retyped_value) & subset_mask

    year_variables = [
        v for v in model_spec.raw_variables if re.search("year", v, re.IGNORECASE)
    ]
    if len(year_variables) == 1:
        year_variable = year_variables[0]
    else:
        year_variable = utils.get_year_variable(full_training_data)
    columns_to_keep = model_spec.raw_variables
    if year_variable not in columns_to_keep:
        columns_to_keep.append(year_variable)

    raw_df = full_training_data.loc[:, columns_to_keep]
    null_mask = raw_df.isna().any(axis=1)
    if null_mask.sum() > 0:
        msg = f"Null values found in raw data for {null_mask.sum()} rows"
        print(msg)

    df, var_info = cm_data.prepare_model_data(raw_df, model_spec)

    raw_df = raw_df.loc[subset_mask].reset_index(drop=True)
    df = df.loc[subset_mask].reset_index(drop=True)
    # # Print descriptions of both raw and processed data to illustrate transformations
    # for col in df.columns:
    #     print(f"Column: {col}")
    #     print("  Raw data:")
    #     print(raw_df[col].describe())
    #     print("  Processed data:")
    #     print(df[col].describe())
    # TODO: Test/train split
    print(
        f"Training {model_spec.lmer_formula} for {measure} {model_version} "
        f"submodel {submodel} cols {df.columns}"
        f" with {len(df)} rows"
    )

    model_type = model_spec.model_type
    if model_type == ModelType.LINEAR_MIXED_EFFECTS:
        model = Lmer(model_spec.lmer_formula, data=df, family="binomial")
        model.fit()
        if len(model.warnings) > 0:
            # TODO: save these to a file
            print(model.warnings)
            msg = f"Model {model_spec} did not fit."
            raise ValueError(msg)
        raw_df["fits"] = model.fits
        df["fits"] = model.fits
        no_re_pred = model.predict(
            model.design_matrix, use_rfx=False, verify_predictions=False
        )
        raw_df["no_re_fits"] = no_re_pred
        df["no_re_fits"] = no_re_pred
    elif model_type == ModelType.SPLINE_MIXED_EFFECTS:
        scam_lib = packages.importr("scam")
        base = packages.importr("base")
        stats = packages.importr("stats")
        # pandas2ri.activate()
        with localconverter(default_converter + pandas2ri.converter):
            r_df = pandas2ri.py2rpy(df)

        knots_dict = {}
        for predictor in model_spec.predictors:
            if (
                predictor.spline is not None
                and predictor.spline.knot_strategy is not None
            ):
                knots = get_knot_values(df, predictor.name, predictor.spline, var_info)
                print(f"Knots for {predictor.name}: {knots}")
                knots_dict[predictor.name] = FloatVector(list(knots))
        knots = ListVector(knots_dict) if len(knots_dict) > 0 else None
        if knots is not None:
            model = scam_lib.scam(
                stats.as_formula(model_spec.lmer_formula),
                data=r_df,
                family=stats.binomial(link="logit"),
                knots=knots,
            )
        else:
            model = scam_lib.scam(
                stats.as_formula(model_spec.lmer_formula),
                data=r_df,
                family=stats.binomial(link="logit"),
            )
        print(base.summary(model))
        fits = np.array(model.rx2("fitted.values"))
        raw_df["fits"] = fits
        df["fits"] = fits

        # Predict no_re_pred over average
        # previous method:
        # no_re_pred = scam_lib.predict_scam(
        #     model, newdata=r_df, type="response", exclude="s(ihme_loc_id)"
        # )
        # no_re_pred = np.array(no_re_pred)
        # raw_df["no_re_fits"] = no_re_pred
        # df["no_re_fits"] = no_re_pred

        # current method:
        # Work directly in R to avoid pandas<->R type conversion issues
        r_df_avg = base.data_frame(r_df)
        robjects.globalenv["r_df_avg"] = r_df_avg

        # Average numeric covariates — replace in R to ensure type change
        if "total_precipitation_prev_0_mo" in df.columns:
            mean_val = float(df["total_precipitation_prev_0_mo"].mean())
            robjects.r(
                f"r_df_avg$total_precipitation_prev_0_mo <- rep({mean_val}, nrow(r_df_avg))"
            )

        if "birth_year" in df.columns:
            mean_val = float(df["birth_year"].astype(float).mean())
            robjects.r(f"r_df_avg$birth_year <- rep({mean_val}, nrow(r_df_avg))")

        r_df_avg = robjects.globalenv["r_df_avg"]

        # sex_id: predict per factor level, weight by observed proportion
        sex_levels = df["sex_id"].cat.categories
        no_re_pred = np.zeros(len(df))
        orig_sex_levels = base.levels(r_df.rx2("sex_id"))

        for sex_val in sex_levels:
            robjects.globalenv["rdf_temp"] = base.data_frame(r_df_avg)
            # Build the levels string for R
            levels_str = ", ".join(f'"{str(lv)}"' for lv in orig_sex_levels)
            robjects.r(
                f'rdf_temp$sex_id <- factor(rep("{sex_val}", nrow(rdf_temp)), '
                f"levels = c({levels_str}))"
            )
            r_df_temp = robjects.globalenv["rdf_temp"]

            pred = scam_lib.predict_scam(
                model, newdata=r_df_temp, type="response", exclude="s(ihme_loc_id)"
            )
            weight = (df["sex_id"] == sex_val).mean()
            no_re_pred += np.array(pred) * weight

        raw_df["no_re_fits"] = no_re_pred
        df["no_re_fits"] = no_re_pred

    model.var_info = var_info
    model.raw_data = raw_df
    model.submodel = submodel

    cm_data.save_model(model, model_version, model_spec, submodel)

    # Validation
    target_measure = model_spec.measure.value
    if year_variable not in df.columns:
        df[year_variable] = raw_df[year_variable]
    summary = training_validation.validate_model(
        df, model_spec, target_measure, year_variable, var_info
    )
    summary.to_csv(
        cm_data.models / model_version / "validation_results.csv", index=False
    )
    training_validation.update_results_file(
        summary, cm_data.models / "validation_results.csv", model_version, submodel
    )

    training_diagnostics.run_training_diagnostics(
        model, df, model_spec, cm_data, model_version, submodel, raw_df, var_info
    )

    if not submodel and model_type != ModelType.SPLINE_MIXED_EFFECTS:  # TODO Temporary
        # Only save intercept raster for full model
        icept_raster = utils.get_intercept_raster(
            model_spec, model.coefs, model.ranef, cm_data
        )
        cm_data.save_rasterized_intercept(model_version, icept_raster, predictor=1)
    cm_data.save_model(model, output_dir, submodel)

    # Create lookup tables for spline variables if applicable
    if model_type == ModelType.SPLINE_MIXED_EFFECTS:
        predictor_vars = [i for i in model_spec if "predictors" in i][0][1]
        predictor_specs_with_spline = [
            spec for spec in predictor_vars if spec.spline is not None
        ]
        # remove consumption_pd
        predictor_specs_with_spline = [
            spec
            for spec in predictor_specs_with_spline
            if not ("consumption" in spec.name)
        ]
        if len(predictor_specs_with_spline) > 0:
            for predictor_spec in predictor_specs_with_spline:
                lookup_var = predictor_spec.name
                climate_lookup_df = create_spline_lookup(
                    model, model_spec, lookup_var, full_df=full_training_data
                )
                cm_data.save_climate_lookup_table(climate_lookup_df, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_model_version()
@click.option(
    "--submodel",
    "-s",
    multiple=True,
    type=(str, str),
    help="Submodel specification.",
)
def model_training_task(
    output_root: str,
    measure: str,
    model_version: str,
    submodel: list[tuple[str, str]],
) -> None:
    """Run model training."""
    model_training_main(
        Path(output_root),
        measure,
        model_version,
        submodel,
    )


@click.command()  # type: ignore[arg-type]
@click.argument(
    "model_specification_path",
    type=click.Path(exists=True),
)
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_queue()
def model_training(
    model_specification_path: str,
    output_root: str,
    queue: str,
) -> None:
    """Run model training."""
    model_spec = ModelSpecification.from_yaml(model_specification_path)
    measure = model_spec.measure
    measure_root = Path(output_root) / measure
    cm_data = ClimateMalnutritionData(measure_root)
    model_version = cm_data.new_model_version()
    version_root = cm_data.models / model_version
    model_spec.version.model = model_version
    cm_data.save_model_specification(model_spec, model_version)
    training_data = cm_data.load_training_data(model_spec.version.training_data)

    # Deal with submodels
    submodel_vars = [var.name for var in (model_spec.submodel_vars or [])]
    print("Training submodels by:", ", ".join(submodel_vars))
    submodel_var_values = [training_data[var].unique() for var in submodel_vars]
    # cross product of all submodel_var_values lists
    submodel_specs = [
        list(zip(submodel_vars, values, strict=False))
        for values in itertools.product(*submodel_var_values)
    ]
    submodel_specs_strs = [
        " --submodel ".join([f"{var} {val}" for var, val in spec])
        for spec in submodel_specs
    ]
    node_args = {"submodel": submodel_specs_strs} if submodel_vars else {}
    node_args["measure"] = [measure]

    print("Running model training for model version", model_version)

    jobmon.run_parallel(
        runner="sttask",
        task_name="training",
        node_args=node_args,  # type: ignore[arg-type]
        task_args={
            "output-root": output_root,
            "model-version": model_version,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "600Gb",
            "runtime": "120h",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(version_root),
    )

    print("Model training complete. Results can be found at", version_root)
