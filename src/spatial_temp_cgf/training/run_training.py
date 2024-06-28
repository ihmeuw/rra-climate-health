from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import click
from rra_tools import jobmon
from pymer4.models.Lmer import Lmer

from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf import binning, scaling
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData
from spatial_temp_cgf.model_specification import (
    ModelSpecification,
)


def prepare_model_data(
    raw_model_data: pd.DataFrame,
    model_spec: ModelSpecification,
) -> tuple[pd.DataFrame, dict]:
    # Do all required data transformations
    transformed_data = {}
    var_info = {}
    for var, transform in model_spec.transform_map.items():
        if transform.type == 'binning':
            transformed, info = binning.bin_column(raw_model_data, var, transform)
        elif transform.type == 'scaling':
            transformed, info = scaling.scale_column(raw_model_data, var, transform)
        else:
            raise ValueError(f"Unknown transformation type {transform.type}")
        transformed_data[var] = transformed
        var_info[var] = info

    df = pd.DataFrame(transformed_data)

    if model_spec.grid_predictors:
        grid_spec = model_spec.grid_predictors.grid_spec
        grid_vars = model_spec.grid_predictors.raw_variables
        df['grid_cell'] = df[grid_vars].astype(str).apply('_'.join, axis=1)

        grid_spec['grid_definition_categorical'] = (
            df[grid_vars + ['grid_cell']]
            .drop_duplicates()
            .sort_values(grid_vars)
        )
        grid_spec['grid_definition'] = (
            grid_spec['grid_definition_categorical'].astype(str)
        )
        var_info['grid_cell'] = grid_spec

    for random_effect in model_spec.random_effects:
        if random_effect not in df:
            df[random_effect] = raw_model_data[random_effect]

    return df, var_info


def model_training_main(
    output_root: Path,  
    measure: str,
    model_version: str,
    age_group_id: int,
    sex_id: int,
):
    cm_data = ClimateMalnutritionData(output_root / measure)
    model_spec = cm_data.load_model_specification(model_version)

    # Load training data
    full_training_data = cm_data.load_training_data(model_spec.version.training_data)
    raw_df = full_training_data.loc[:, model_spec.raw_variables]

    # TODO: Test/train split

    df, var_info = prepare_model_data(raw_df, model_spec)

    subset_mask = (df.sex_id == sex_id) & (df.age_group_id == age_group_id)
    df = df.loc[subset_mask].copy()

    model = Lmer(model_spec.lmer_formula, data=df, family='binomial')
    model.fit()
    if not model.fitted:
        raise ValueError(f"Model {model_spec} did not fit.")

    model.var_info = var_info
    model.raw_data = raw_df
    cm_data.save_model(model, model_version, age_group_id, sex_id)


@click.command()
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@click.option(
    '--model-version',
    '-v',
    type=str,
    required=True,
)
@clio.with_age_group_id()
@clio.with_sex_id()
def model_training_task(
    output_root: str,
    measure: str,
    model_version: str,
    age_group_id: str,
    sex_id: str,
) -> None:
    """Run model training."""
    model_training_main(
        Path(output_root),
        measure,
        model_version,
        int(sex_id),
        int(age_group_id),
    )


@click.command()
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
    model_spec.version.model = model_version
    cm_data.save_model_specification(model_spec, model_version)

    jobmon.run_parallel(
        runner="sttask",
        task_name="training",
        node_args={
            "age-group-id": clio.VALID_AGE_GROUP_IDS,
            "sex-id": clio.VALID_SEX_IDS,
        },
        task_args={
            "output-root": output_root,
            "measure": measure,
            "model-version": model_version,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "20Gb",
            "runtime": "96h",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        max_attempts=1,
    )




