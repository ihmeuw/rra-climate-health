import sys
import logging
from pathlib import Path

import click

from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData


def fit_and_predict_LMER(data:pd.DataFrame, model_spec:str):
    from pymer4 import Lmer
    import re
    model_vars = re.findall(r'[a-zA-Z][a-zA-Z0-9_]+', model_spec)
    model = Lmer(model_spec, data=data[model_vars], family='binomial')
    model.fit()
    pred_df = data.copy()
    pred_df['ihme_loc_id'] = np.nan
    pred_df['model_fit'] = model.fits
    pred_df['model_residual'] = pred_df.cgf_value - pred_df.model_fit
    pred_df['model_fit_nocountry'] = model.predict(pred_df)
    pred_df['model_fit_nocountry_res'] = pred_df['model_fit_nocountry'] + pred_df['model_residual']
    return pred_df, model

def get_modeling_input_data(measure):
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


def make_model(cgf_measure, model_spec, grid_list=None, grid_spec=None,
               sex_id=None, age_group_id=None, filter=None, location_var='ihme_loc_id'):
    import re
    import utils
    import sklearn
    df = get_modeling_input_data(cgf_measure)

    if 'grid_cell' in model_spec and grid_list is None and grid_spec is None:
        raise ValueError(
            "Model specification includes grid_cell but no grid_list or grid_spec is provided")
    if grid_list is not None:
        grid_spec = {'grid_order': grid_list}
        for var in grid_list:
            grid_spec[var] = STANDARD_BINNING_SPECS[var]
    # model_spec = f'{cgf_measure} ~ (1 | {location_var}) + (1 | grid_cell)'
    # grid_spec = {'grid_order' : ['ldi_pc_pd', 'over_30'],
    #     'ldi_pc_pd': {'bin_category': 'household', 'bin_strategy' : 'quantiles', 'nbins': 10, 'type': 'income'},
    #     'over_30': {'bin_category': 'location', 'bin_strategy' : 'custom_daysover', 'nbins': 10, 'type':'climate'}}

    binned_df = df.copy()

    for var in grid_spec['grid_order']:
        grid_spec[var]['var_bin'] = var + '_bin'
        grid_spec[var]['bins'], binned_df = cgf_utils.group_and_bin_column_definition(
            binned_df, var,
            grid_spec[var]['bin_category'], grid_spec[var]['nbins'],
            bin_strategy=grid_spec[var]['bin_strategy'], retbins=True)

    binned_vars = [grid_spec[var]['var_bin'] for var in grid_spec['grid_order']]
    binned_df['grid_cell'] = binned_df[binned_vars].astype(str).apply('_'.join, axis=1)

    cols_to_scale = []  # 'over30_avgperyear', 'precip', 'precip_cumavg_5' ,'temp', 'temp_cumavg_5', 'income_per_day']
    scaler = sklearn.preprocessing.MinMaxScaler()
    if cols_to_scale:
        scaled_cols = [f'sc_{col}' for col in cols_to_scale]
        binned_df[scaled_cols] = scaler.fit_transform(binned_df[cols_to_scale])

    if filter:
        binned_df = binned_df.query(filter).copy()
    if sex_id:
        binned_df = binned_df.loc[binned_df.sex_id == sex_id].copy()
    if age_group_id:
        binned_df = binned_df.loc[binned_df.age_group_id == age_group_id].copy()

    pred_df, model = fit_and_predict_LMER(binned_df, model_spec)

    model_vars = re.findall(r'[a-zA-Z][a-zA-Z0-9_]+', model_spec)

    weight_grid = binned_df[model_vars[2:] + binned_vars].drop_duplicates().merge(
        pd.DataFrame({location_var: binned_df[location_var].unique()}), how='cross')
    weight_grid['prediction_weight'] = model.predict(weight_grid,
                                                     verify_predictions=False)
    modelled_locations = binned_df[location_var].unique()

    nocountry_grid = binned_df[model_vars[2:] + binned_vars].drop_duplicates()
    nocountry_grid[location_var] = np.nan
    nocountry_grid['prediction_weight'] = model.predict(nocountry_grid,
                                                        verify_predictions=False)

    model.model_vars = model_vars
    model.nocountry_grid = nocountry_grid
    model.weight_grid = weight_grid
    model.available_locations = modelled_locations
    model.grid_spec = grid_spec
    return model


def run_model_and_save(measure, model_identifier, sex_id, age_group_id, model_spec, grid_vars):
    model = make_model(measure, model_spec, grid_list=grid_vars, sex_id=sex_id, age_group_id = age_group_id)
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



