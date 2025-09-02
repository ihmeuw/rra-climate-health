import itertools
from pathlib import Path
from typing import Any

import click
import pandas as pd
import rasterra as rt
from pymer4.models.Lmer import Lmer
from rra_tools import jobmon

from rra_climate_health import cli_options as clio
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from rra_climate_health.model_specification import (
    ModelSpecification,
)
from rra_climate_health.transforms import transform_column
from rra_climate_health import utils



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

    raw_df = full_training_data.loc[:, model_spec.raw_variables]
    null_mask = raw_df.isna().any(axis=1)
    if null_mask.sum() > 0:
        msg = f"Null values found in raw data for {null_mask.sum()} rows"
        print(msg)

    df, var_info = cm_data.prepare_model_data(raw_df, model_spec)

    raw_df = raw_df.loc[subset_mask].reset_index(drop=True)
    df = df.loc[subset_mask].reset_index(drop=True)

    # TODO: Test/train split
    print(
        f"Training {model_spec.lmer_formula} for {measure} {model_version} "
        f"submodel {submodel} cols {df.columns}"
        f" with {len(df)} rows"
    )
    model = Lmer(model_spec.lmer_formula, data=df, family="binomial")
    model.fit()
    if len(model.warnings) > 0:
        # TODO: save these to a file
        print(model.warnings)
        msg = f"Model {model_spec} did not fit."
        raise ValueError(msg)
    model.var_info = var_info
    model.raw_data = raw_df
    model.submodel = submodel

    cm_data.save_model(model, model_version, submodel)
    icept_raster = utils.get_intercept_raster(model_spec, model.coefs, model.ranef, cm_data)
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
            "memory": "60Gb",
            "runtime": "4h",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(version_root),
    )

    print("Model training complete. Results can be found at", version_root)

