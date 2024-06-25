import re
import sys
import logging
from pathlib import Path

import pandas as pd
import sklearn
import click



from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf import cgf_utils
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData


def get_modeling_input_data(measure: str) -> pd.DataFrame:
    df = pd.read_parquet(f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/new_{measure}.parquet')
    return df


STANDARD_BINNING_SPECS = {
    'ldi_pc_pd': {'bin_category': 'household', 'bin_strategy': 'quantiles', 'nbins': 10,
                  'type': 'income'},
    'over_30': {'bin_category': 'location', 'bin_strategy': 'custom_daysover',
                'nbins': 10, 'type': 'climate'},
    'temp': {'bin_category': 'location', 'bin_strategy': 'quantiles', 'nbins': 10,
             'type': 'climate'}
}


def make_model(
    cgf_measure: str,
    model_spec: str,
    grid_list: list[str] | None = None,
    binning_spec: None,
    sex_id = None, age_group_id = None, filter = None, location_var = 'ihme_loc_id'):

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
        grid_spec = {'grid_order' : grid_list}
    if binning_spec is None:
        binning_spec = dict()
        for var in vars_to_bin:
            binning_spec[var] = STANDARD_BINNING_SPECS[var]
    else:
        #not doing this yet, really
        raise NotImplementedError("Custom binning specs not yet implemented")
    #model_spec = f'{cgf_measure} ~ (1 | {location_var}) + (1 | grid_cell)'
    # grid_spec = {'grid_order' : ['ldi_pc_pd', 'over_30'],
    #     'ldi_pc_pd': {'bin_category': 'household', 'bin_strategy' : 'quantiles', 'nbins': 10, 'type': 'income'},
    #     'over_30': {'bin_category': 'location', 'bin_strategy' : 'custom_daysover', 'nbins': 10, 'type':'climate'}}

    binned_df = df.copy()
    var_info = dict()

    for var in vars_to_bin:
        var_info[var] = dict()
        var_info[var]['var_bin'] = var + '_bin'
        var_info[var]['bin_edges'], binned_df = cgf_utils.group_and_bin_column_definition(binned_df, var,
            binning_spec[var]['bin_category'], binning_spec[var]['nbins'], bin_strategy = binning_spec[var]['bin_strategy'], retbins = True)
        var_info[var]['bins_categorical'] = pd.DataFrame({var_info[var]['var_bin']:binned_df[var_info[var]['var_bin']].unique().sort_values()}) #binned_df[var_info[var]['var_bin']].unique().sort_values()
        var_info[var]['bins'] = var_info[var]['bins_categorical'].astype(str)

    if grid_present:
        var_info['grid_cell'] = dict()
        grid_components_binned = [v + '_bin' for v in grid_list]
        binned_df['grid_cell'] = binned_df[grid_components_binned].astype(str).apply('_'.join, axis=1)
        grid_spec['grid_definition_categorical'] = binned_df[grid_components_binned + ['grid_cell']].drop_duplicates().sort_values(grid_components_binned)
        grid_spec['grid_definition'] = grid_spec['grid_definition_categorical'].astype(str)
    cols_that_should_be_scaled = ['temp', 'precip', 'over_30']
    cols_to_scale = [v for v in model_vars if v in cols_that_should_be_scaled]#'over30_avgperyear', 'precip', 'precip_cumavg_5' ,'temp', 'temp_cumavg_5', 'income_per_day']
    scaler = sklearn.preprocessing.MinMaxScaler()
    if cols_to_scale:
        scaled_cols = [f'sc_{col}' for col in cols_to_scale]
        binned_df[scaled_cols] = scaler.fit_transform(binned_df[cols_to_scale])

    for v in cols_to_scale:
        model_spec = model_spec.replace(v, 'sc_'+v)

    filter = None
    if filter: #We bin before we filter
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

    scaling_bounds = None
    if len(cols_to_scale) > 0:
        scaling_bounds = dict()
        for a in cols_to_scale:
            scaling_bounds[a] = (binned_df[a].min(), binned_df[a].max())
        model.scaling_bounds = scaling_bounds

    model.var_info = var_info
    model.model_vars = model_vars
    #model.nocountry_grid = nocountry_grid
    #model.weight_grid = weight_grid
    model.available_locations = list(binned_df[location_var].unique())
    model.has_grid = grid_present
    model.grid_spec = grid_spec if grid_present else None
    model.binned_vars = binned_vars
    model.vars_to_bin = vars_to_bin
    model.scaled_vars = cols_to_scale
    #return model
    return model




def run_model_and_save(measure, model_identifier, sex_id, age_group_id, model_spec, grid_vars):
    model = make_model(measure, model_spec, grid_list=grid_vars, sex_id=sex_id, age_group_id = age_group_id)
    model.model_identifier = model_identifier
    model_filepath = paths.MODELS / model_identifier / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
    model_filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def model_training_main(
    output_dir: Path,
    model_id: str,
    model_spec: str,
    grid_vars: str,
    measure: str,
    sex_id: int,
    age_group_id: int,
):
    pass


@click.command()
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_model_id()
@click.option(
    "--model-spec",
    help='model spec like "stunting ~ ihme_loc_id + precip"',
    type=str,
)
@click.option(
    "--grid-vars",
    help='list of vars to be binned and added to the grid in order',
    type=str,
)
@clio.with_measure()
@clio.with_sex_id()
@clio.with_age_group_id()
def model_training_task(
    output_dir: str,
    model_id: str,
    model_spec: str,
    grid_vars: str,
    measure: str,
    sex_id: str,
    age_group_id: str,
) -> None:
    """Run model training."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    model_training_main(
        Path(output_dir),
        model_id,
        model_spec,
        grid_vars,
        measure,
        int(sex_id),
        int(age_group_id),
    )



