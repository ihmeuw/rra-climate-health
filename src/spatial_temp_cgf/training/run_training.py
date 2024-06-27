import re
import sys
import logging
from pathlib import Path

import pandas as pd
import sklearn
import click
from rra_tools import jobmon

from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf import cgf_utils
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData
from spatial_temp_cgf.model_specification import (
    BinningCategory,
    BinningStrategy,
    BinningSpecification,
    ModelSpecification,
)


STANDARD_BINNING_SPECS = {
    "ldi_pc_pd": BinningSpecification(
        category=BinningCategory.HOUSEHOLD,
        strategy=BinningStrategy.QUANTILES,
        nbins=10,
    ),
    "over_30": BinningSpecification(
        category=BinningCategory.LOCATION,
        strategy=BinningStrategy.CUSTOM_DAYSOVER,
        nbins=10,
    ),
    "temp": BinningSpecification(
        category=BinningCategory.LOCATION,
        strategy=BinningStrategy.QUANTILES,
        nbins=10,
    ),
}

def get_modeling_input_data(measure: str) -> pd.DataFrame:
    df = pd.read_parquet(f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/new_{measure}.parquet')
    return df


def make_model(
    cgf_measure: str,
    model_spec: str,
    grid_list: list[str] | None = None,
    binning_spec=None,
    sex_id=None,
    age_group_id=None,
    location_var='ihme_loc_id',
) -> None:
    df = get_modeling_input_data(cgf_measure)

    model_vars = re.findall(r'[a-zA-Z][a-zA-Z0-9_]+', model_spec)
    outcome_var = model_vars[0]
    model_vars = model_vars[1:]
    binned_vars = [v for v in model_vars if v.endswith('_bin')]
    vars_to_bin = [v.replace('_bin','') for v in binned_vars] + (grid_list if grid_list is not None else [])
    binned_vars = [v + '_bin' for v in vars_to_bin]

    grid_present = False

    if 'grid_cell' in model_spec and grid_list is None:
        raise ValueError("Model specification includes grid_cell but no grid_list is provided")
    if 'grid_cell' not in model_spec and grid_list is not None:
        raise ValueError("Model specification does not include grid_cell but grid_list is provided")
    if grid_list is not None:
        grid_present = True
        grid_spec = {'grid_order': grid_list}
    if binning_spec is None:
        binning_spec = dict()
        for var in vars_to_bin:
            binning_spec[var] = STANDARD_BINNING_SPECS[var]
    else:
        raise NotImplementedError("Custom binning specs not yet implemented")

    binned_df = df.copy()
    var_info = dict()

    for var in vars_to_bin:
        var_info[var] = dict()
        var_info[var]['var_bin'] = var + '_bin'
        var_info[var]['bin_edges'], binned_df = cgf_utils.group_and_bin_column_definition(
            binned_df,
            binning_spec[var],
        )
        var_info[var]['bins_categorical'] = pd.DataFrame({var_info[var]['var_bin']:binned_df[var_info[var]['var_bin']].unique().sort_values()})
        var_info[var]['bins'] = var_info[var]['bins_categorical'].astype(str)

    if grid_present:
        var_info['grid_cell'] = dict()
        grid_components_binned = [v + '_bin' for v in grid_list]
        binned_df['grid_cell'] = binned_df[grid_components_binned].astype(str).apply('_'.join, axis=1)
        grid_spec['grid_definition_categorical'] = binned_df[grid_components_binned + ['grid_cell']].drop_duplicates().sort_values(grid_components_binned)
        grid_spec['grid_definition'] = grid_spec['grid_definition_categorical'].astype(str)
    cols_that_should_be_scaled = ['temp', 'precip', 'over_30']
    cols_to_scale = [v for v in model_vars if v in cols_that_should_be_scaled] # 'over30_avgperyear', 'precip', 'precip_cumavg_5' ,'temp', 'temp_cumavg_5', 'income_per_day']
    scaler = sklearn.preprocessing.MinMaxScaler()
    if cols_to_scale:
        scaled_cols = [f'sc_{col}' for col in cols_to_scale]
        binned_df[scaled_cols] = scaler.fit_transform(binned_df[cols_to_scale])

    for v in cols_to_scale:
        model_spec = model_spec.replace(v, 'sc_'+v)

    filter = None
    if filter: # We bin before we filter
        binned_df = binned_df.query(filter).copy()
    if sex_id:
        binned_df = binned_df.loc[binned_df.sex_id == sex_id].copy()
    if age_group_id:
        binned_df = binned_df.loc[binned_df.age_group_id == age_group_id].copy()

    vars_in_model_definition = re.findall(r'[a-zA-Z][a-zA-Z0-9_]+', model_spec)
    model = Lmer(model_spec, data=binned_df[vars_in_model_definition], family='binomial')
    model.fit()
    if not model.fitted:
        raise ValueError(f"Model {model_spec} did not fit.")

    model_vars_without_location = [v for v in model_vars if v != location_var]
    # weight_grid = binned_df[model_vars_without_location + binned_vars].drop_duplicates().merge(pd.DataFrame({location_var:binned_df[location_var].unique()}), how = 'cross')
    # weight_grid['prediction_weight'] = model.predict(weight_grid, verify_predictions=False, skip_data_checks=True)
    # modelled_locations = binned_df[location_var].unique()

    # nocountry_grid = binned_df[model_vars_without_location + binned_vars].drop_duplicates()
    # nocountry_grid[location_var] = np.nan
    # nocountry_grid['prediction_weight'] = model.predict(nocountry_grid, verify_predictions=False, skip_data_checks=True)

    randomeffects = dict()
    if type(model.ranef) == pd.DataFrame:
        varname = model.ranef_var.index[0]
        varname = varname if varname != location_var else location_var
        varname_bin = varname + ('_bin' if varname not in [location_var, 'grid_cell'] else '')
        randomeffects[varname] = model.ranef.reset_index().rename(columns={'index':varname_bin, 'X.Intercept.':varname + '_coef'})
    elif type(model.ranef) == list:
        for i in range(len(model.ranef_var)):
            varname = model.ranef_var.index[i]
            varname_bin = varname + ('_bin' if varname not in [location_var, 'grid_cell'] else '')
            varname = varname if varname != location_var else location_var
            randomeffects[model.ranef_var.index[i]] = model.ranef[i].reset_index().rename(columns={'index':varname_bin, 'X.Intercept.':varname + '_coef'})

    fixedeffects = dict()
    for var in [x.replace('_bin', '') for x in model_vars]:
        if var in randomeffects.keys():
            # This is a random effect
            pass #for now
        elif var in model.coefs.index:
            # This is a continuous variable
            fixedeffects[var] = pd.DataFrame({var+'_coef' : [model.coefs.loc[var, 'Estimate']]})
        elif 'sc_' + var in model.coefs.index:
            # This is a continuous variable
            fixedeffects[var] = pd.DataFrame({var+'_coef' : [model.coefs.loc['sc_' + var, 'Estimate']]})
        elif var+'_bin' in model_vars:
            # This is a categorical variable
            varname_bin = var+'_bin'
            betas = model.coefs[model.coefs.index.str.startswith(varname_bin)][['Estimate']].reset_index()
            betas['index'] = betas['index'].str.replace(varname_bin, '')
            betas = betas.rename(columns={'index':varname_bin, 'Estimate':var + '_coef'})
            fixedeffects[var] = betas
        else:
            raise ValueError(f'Variable {var} not found in model')

    for var in [x.replace('_bin', '') for x in model_vars]:
        if var in randomeffects.keys():
            if var not in var_info.keys():
                var_info[var] = dict()
            var_info[var]['coefs'] = randomeffects[var]
        elif var in fixedeffects.keys():
            if var not in var_info.keys():
                var_info[var] = dict()
            var_info[var]['coefs'] = fixedeffects[var]
        else:
            raise ValueError(f'Variable {var} not found in model')

    var_info['intercept'] = {'coef': model.coefs.loc['(Intercept)', 'Estimate']}

    model.scaling_bounds = {
        col: (binned_df[col].min(), binned_df[col].max())
        for col in cols_to_scale
    }
    model.var_info = var_info
    model.model_vars = model_vars
    model.available_locations = list(binned_df[location_var].unique())
    model.has_grid = grid_present
    model.grid_spec = grid_spec if grid_present else None
    model.binned_vars = binned_vars
    model.vars_to_bin = vars_to_bin
    model.scaled_vars = cols_to_scale
    return model




def run_model_and_save(measure, model_identifier, sex_id, age_group_id, model_spec, grid_vars):
    model = make_model(measure, model_spec, grid_list=grid_vars, sex_id=sex_id, age_group_id = age_group_id)
    model.model_identifier = model_identifier
    model_filepath = paths.MODELS / model_identifier / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
    model_filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
    df = cm_data.load_training_data(model_spec.version.training_data)

    # Build model
    model = ...

    # Fit model

    # Save model
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




