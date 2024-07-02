from pathlib import Path

import pandas as pd
import click
from rra_tools import jobmon
from pymer4.models.Lmer import Lmer

from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.transforms import transform_column
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
    for var, transform_spec in model_spec.transform_map.items():
        transformed, transformer = transform_column(
            raw_model_data, var, transform_spec
        )
        transformed_data[var] = transformed
        var_info[var] = {
            "transformer": transformer,
            "transform_spec": transform_spec,
        }

    df = pd.DataFrame(transformed_data)

    if model_spec.grid_predictors:
        grid_spec = model_spec.grid_predictors.grid_spec
        grid_vars = grid_spec['grid_order']
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

    df[model_spec.measure] = raw_model_data[model_spec.measure]

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
    # FIXME: Prep leaves a bad index
    full_training_data = full_training_data.reset_index(drop=True)    
    full_training_data['intercept'] = 1.

    subset_mask = (
        (full_training_data.sex_id == sex_id) 
        & (full_training_data.age_group_id == age_group_id)
    )
    
    raw_df = full_training_data.loc[:, model_spec.raw_variables]   
    null_mask = raw_df.isnull().any(axis=1)
    assert null_mask.sum() == 0
    
    df, var_info = prepare_model_data(raw_df, model_spec)

    raw_df = raw_df.loc[subset_mask].reset_index(drop=True)
    df = df.loc[subset_mask].reset_index(drop=True)

    # TODO: Test/train split
    print(
        f"Training {model_spec.lmer_formula} for {measure} {model_version} "
        f"age {age_group_id} sex {sex_id} cols {df.columns}"
    )
    model = Lmer(model_spec.lmer_formula, data=df, family='binomial')
    model.fit()
    if len(model.warnings) > 0:
        # TODO save these to a file
        print(model.warnings)
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
        int(age_group_id),
        int(sex_id),
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
    version_root = cm_data.models / model_version
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
            "runtime": "1h",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(version_root),
    )


def extract_coefficients_from_model(model, model_spec):
    model_non_grid_predictors = [v.name for v in model_spec.predictors]
    model_grid_predictors = [v.name for v in [model_spec.grid_predictors.x, model_spec.grid_predictors.y]] if model_spec.grid_predictors else []
    all_model_predictors = set(model_non_grid_predictors + model_grid_predictors)

    binned_non_grid_predictors = [v.name for v in model_spec.predictors if v.transform.type == 'binning']
    all_binned_variables = binned_non_grid_predictors.copy() + model_grid_predictors #grid variables are always binned, or rather, categorical
    
    extra_terms = [v for v in model_spec.extra_terms]
    
    # model_vars_without_location = [v for v in model_vars if v != location_var]
    print("Random effects on intercept")
    print(model.ranef)
    categorical_coefs = dict()
    if type(model.ranef) == pd.DataFrame:
        varname = model.ranef_var.index[0]
        categorical_coefs[varname] = model.ranef.reset_index().rename(columns={'index':varname, 'X.Intercept.':varname + '_coef'})
    elif type(model.ranef) == list:
        for i in range(len(model.ranef_var)):
            varname = model.ranef_var.index[i]
            categorical_coefs[model.ranef_var.index[i]] = model.ranef[i].reset_index().rename(columns={'index':varname, 'X.Intercept.':varname + '_coef'})

    # Categorical fixed effects
    print(model.coefs)
    for var in all_binned_variables:
        if var in categorical_coefs.keys():
            continue
        betas = model.coefs[model.coefs.index.str.startswith(var)][['Estimate']].reset_index()
        betas['index'] = betas['index'].str.replace(var, '')
        betas = betas.rename(columns={'index':var, 'Estimate':var + '_coef'})
        continuous_coefs[var] = betas

    # Continuous fixed effects
    continuous_coefs = dict()
    non_binned_non_extra_terms_non_intercept_vars = [v for v in all_model_predictors if (v not in all_binned_variables) and (v.lower() != 'intercept')]
    for var in non_binned_non_extra_terms_non_intercept_vars:
        print(f"var |{var}| in {model.coefs.index} ?")
        assert var not in categorical_coefs.keys()
        assert var in list(model.coefs.index)
        continuous_coefs[var] = pd.DataFrame({var+'_coef' : [model.coefs.loc[var, 'Estimate']]})
    
    # Extra terms (e.g. interaction terms) (assumed here to be continuous, not binned)
    for var in extra_terms:
        equivalent_coef_name = var.replace('*', ':').replace(' ','')
        if equivalent_coef_name not in model.coefs.index:
            raise ValueError(f"Variable {var} not found in model {model.coefs.index}")
        continuous_coefs[var] = pd.DataFrame({equivalent_coef_name+'_coef' : [model.coefs.loc[equivalent_coef_name, 'Estimate']]})

    continuous_coefs['intercept'] = {'coef': model.coefs.loc['(Intercept)', 'Estimate']}
    return categorical_coefs, continuous_coefs