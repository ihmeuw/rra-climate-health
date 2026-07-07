import itertools
from pathlib import Path
from typing import Any

import click
import pandas as pd
import rasterra as rt
from pymer4.models.Lmer import Lmer
from rpy2.robjects import pandas2ri, packages,  ListVector, FloatVector

from rra_tools import jobmon

from rra_climate_health import cli_options as clio
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData, extract_fixed_effects_from_scam, extract_random_effects_from_scam

from rra_climate_health.model_specification import (
    ModelSpecification,
)
from rra_climate_health.transforms import transform_column
from rra_climate_health import utils
from rra_climate_health.training import training_validation, training_diagnostics
from rra_climate_health.training.training_validation import get_knot_values
from rra_climate_health.model_specification import ModelType

def model_training_main(
    output_root: Path,
    measure: str,
    model_version: str,
    submodel: list[tuple[str, str]] | None = None,
) -> None:
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
        raw_df['fits'] = model.fits
        df['fits'] = model.fits
        no_re_pred = model.predict(model.design_matrix, use_rfx=False, verify_predictions=False)
        raw_df['no_re_fits'] = no_re_pred
        df['no_re_fits'] = no_re_pred
        coefs = model.coefs
        ranefs = model.ranef
    elif model_type == ModelType.SPLINE_MIXED_EFFECTS:
        pandas2ri.activate()
        scam_lib = packages.importr('scam')
        base = packages.importr('base')
        stats = packages.importr('stats')
        
        knots_dict = {}
        for predictor in model_spec.predictors:
            if predictor.spline is not None and predictor.spline.knot_strategy is not None:
                knots = get_knot_values(df, predictor.name, predictor.spline, var_info)
                print(f"Knots for {predictor.name}: {knots}")
                knots_dict[predictor.name] = FloatVector(knots)
        knots = ListVector(knots_dict) if len(knots_dict) > 0 else None
        if knots is not None:
            model = scam_lib.scam(stats.as_formula(model_spec.lmer_formula), data=df, 
                                  family = stats.binomial(link = "logit"), knots = knots )
        else:
            model = scam_lib.scam(stats.as_formula(model_spec.lmer_formula), data=df, family = stats.binomial(link = "logit") )
        print(base.summary(model))
        raw_df['fits'] = model.rx2('fitted.values')
        df['fits'] = model.rx2('fitted.values')
        no_re_pred = scam_lib.predict_scam(model, newdata=df, type="response", exclude="s(ihme_loc_id)")
        raw_df['no_re_fits'] = no_re_pred
        df['no_re_fits'] = no_re_pred
        raw_df.to_parquet(cm_data.models / model_version / "raw_with_predictions.parquet")
        coefs = extract_fixed_effects_from_scam(model)
        ranefs = extract_random_effects_from_scam(model, 'ihme_loc_id')

    model.var_info = var_info
    model.raw_data = raw_df
    model.submodel = submodel

    cm_data.save_model(model, model_version, model_spec, df, submodel)

    # Validation
    target_measure = model_spec.measure.value
    if year_variable not in df.columns:
        df[year_variable] = raw_df[year_variable]

    # summary = training_validation.validate_model(df, model_spec, target_measure, year_variable, var_info)
    # summary.to_csv(cm_data.models / model_version / "validation_results.csv", index=False)
    # training_validation.update_results_file(summary, cm_data.models / "validation_results.csv", 
    #                                         model_version, submodel)
    
    # training_diagnostics.run_training_diagnostics(model, df, model_spec, cm_data, model_version, submodel, raw_df, var_info)

    if not submodel:
        # Only save intercept raster for full model
        icept_raster = utils.get_intercept_raster(model_spec, coefs, ranefs, cm_data)
        cm_data.save_rasterized_intercept(model_version, icept_raster, predictor = 1)


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
            "memory": "250Gb",
            "runtime": "6h",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(version_root),
    )

    print("Model training complete. Results can be found at", version_root)
