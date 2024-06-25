import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterra as rt
from rra_tools import jobmon
from scipy.special import expit

from spatial_temp_cgf import income_funcs, paths
from spatial_temp_cgf.training import mf
from spatial_temp_cgf.data_prep.location_mapping import load_fhs_lsae_mapping
from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData


def model_inference_main(
    output_dir: Path,
    model_id: str,
    measure: str,
    fhs_location_id: int,
    cmip6_scenario: str,
    sex_id: int,
    age_group_id: int,
    year: int,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir)
    model = cm_data.load_model(model_id, measure, age_group_id, sex_id)

    loc_mapping = load_fhs_lsae_mapping(fhs_location_id)
    fhs_shapefile = loc_mapping.iloc[0].fhs_shape
    income_df = income_funcs.load_binned_income_distribution_proportions(
        fhs_location_id=fhs_location_id, measure=measure, year_id=year
    )

    fhs_pop_raster = rt.load_raster(paths.GLOBAL_POPULATION_FILEPATH, fhs_shapefile.bounds).set_no_data_value(np.nan)
    fhs_pop_raster = fhs_pop_raster.clip(fhs_shapefile)

    possible_climate_variables = ['temp', 'precip', 'over_30']
    climate_vars_to_match_bins = [v for v in model.vars_to_bin if v in possible_climate_variables]
    continuous_climate_vars = [v for v in ['temp', 'over_30'] if (v in possible_climate_variables) and (v not in climate_vars_to_match_bins) and (v in model.model_vars)]
    climate_vars = [x for x in possible_climate_variables if x in set(model.model_vars + model.vars_to_bin)]

    model.var_info['ihme_loc_id']['coefs'].rename(columns={'ihme_loc_id': 'worldpop_iso3'}, inplace=True)

    climate_rasters = {}
    for var in climate_vars:
        climate_rasters[var] = mf.get_climate_variable_raster(cmip6_scenario, year, var, None, None, untreated=True)
        climate_rasters[var] = climate_rasters[var].resample_to(fhs_pop_raster).clip(fhs_shapefile)

    admin_dfs = []
    for _, admin2_row in loc_mapping.iterrows():
        lsae_location_id = admin2_row.lsae_location_id
        lsae_shapefile = admin2_row.lsae_shape

        pop_raster = fhs_pop_raster.clip(lsae_shapefile).mask(lsae_shapefile)
        pop_array = pop_raster.set_no_data_value(np.nan).to_numpy()

        rasters = {'population': pop_array.flatten()}

        for var in climate_vars_to_match_bins:
            climate_raster = climate_rasters[var].clip(lsae_shapefile).mask(lsae_shapefile)
            climate_array = climate_raster.to_numpy()
            assert pop_array.shape == climate_array.shape
            binned_climate_array = np.digitize(climate_array, model.var_info[var]['bin_edges'], right=False) - 1
            rasters[var+'_bin_idx'] = binned_climate_array.flatten()

        for var in continuous_climate_vars:
            climate_raster = climate_rasters[var].clip(lsae_shapefile).mask(lsae_shapefile)
            climate_array = climate_raster.to_numpy()
            assert pop_array.shape == climate_array.shape
            rasters[var] = climate_array.flatten()

        # Alternative approach is to group pixels to lsae
        #temp_df = pd.DataFrame({'pop': pop_array.flatten(), 'climate_bin_idx': binned_climate_array.flatten()}).groupby('climate_bin_idx', as_index=False).pop.sum()
        # Keeping it as pixels
        temp_df = pd.DataFrame(rasters)
        temp_df['lsae_pop'] = np.nansum(pop_array)
        temp_df['lsae_location_id'] = lsae_location_id
        temp_df['worldpop_iso3'] = admin2_row.worldpop_iso3

        local_income_df = income_df.query('lsae_location_id == @lsae_location_id')
        for var in climate_vars_to_match_bins:
            temp_df = temp_df.merge(model.var_info[var]['bins'], left_on=var+'_bin_idx', right_index=True, how='inner')
        temp_df = temp_df.merge(local_income_df, on='lsae_location_id', how='left')

        # The lines from now on have coefficients and so are age_group and sex_id - specific
        # Parallelizing by them could happen here - These can be precomputed for a model
        if model.has_grid:
            temp_df = temp_df.merge(model.grid_spec['grid_definition'], how='left')
            temp_df = temp_df.merge(model.var_info['grid_cell']['coefs'], how='left')

        for var in continuous_climate_vars:
            temp_df = temp_df.merge(model.var_info[var]['coefs'], how='left')

        temp_df = temp_df.merge(model.var_info['ihme_loc_id']['coefs'], how='left', on='worldpop_iso3') 
        temp_df.ihme_loc_id_coef = temp_df.ihme_loc_id_coef.fillna(0)
        
        #build the logistic input one variable at a time
        temp_df['logistic_input'] = model.var_info['intercept']['coef']
        temp_df['logistic_input'] += temp_df['ihme_loc_id_coef']
        for var in climate_vars_to_match_bins:
            if var in model.grid_spec['grid_order']: continue
            temp_df['logistic_input'] += temp_df[var+'_bin_coef']
        for var in continuous_climate_vars:
            temp_df['logistic_input'] += temp_df[var+'_coef']
        if model.has_grid:
            temp_df['logistic_input'] += temp_df['grid_cell_coef']
        temp_df['prediction'] = expit(temp_df['logistic_input'])

        admin_dfs.append(temp_df)

    fhs_df = pd.concat(admin_dfs)
    #24s to 92s

    fhs_df['population_at_income'] = fhs_df['population'] * fhs_df['proportion_at_income']
    fhs_df['population_proportion_at_income'] = fhs_df['population_at_income'] / fhs_df['lsae_pop'] / len(loc_mapping)
    fhs_df['affected_proportion'] = fhs_df['population_proportion_at_income'] * fhs_df['prediction']

    result = pd.DataFrame({'fhs_location_id' : [fhs_location_id], 'year_id': [year], 'age_group_id': [age_group_id], 'scenario' : [scenario], 'sex_id':[sex_id],
        'cgf_measure':[measure], 'prevalence': [fhs_df.affected_proportion.sum()]})

    cm_data.save_results(
        result,
        model_id=model_id,
        measure=measure,
        scenario=cmip6_scenario,
        year=year,
        age_group_id=age_group_id,
        sex_id=sex_id
    )


@click.command()
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_model_id()
@clio.with_measure()
@clio.with_location_id()
@clio.with_cmip6_scenario()
@clio.with_sex_id()
@clio.with_age_group_id()
@clio.with_year()
def model_inference_task(
    output_dir: str,
    model_id: str,
    measure: str,
    location_id: str,
    cmip6_scenario: str,
    sex_id: str,
    age_group_id: str,
    year: str,
):
    """Run model inference."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    model_inference_main(
        Path(output_dir),
        model_id,
        measure,
        int(location_id),
        cmip6_scenario,
        int(sex_id),
        int(age_group_id),
        int(year),
    )


@click.command()
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_model_id()
@clio.with_measure(allow_all=True)
@clio.with_location_id(allow_all=True)
@clio.with_cmip6_scenario(allow_all=True)
@clio.with_sex_id(allow_all=True)
@clio.with_age_group_id(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_queue()
def model_inference(
    output_dir: str,
    model_id: str,
    measure: list[str],
    location_id: list[str],
    cmip6_scenario: list[str],
    sex_id: list[str],
    age_group_id: list[str],
    year: list[str],
    queue: str,
) -> None:
    """Run model inference."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    jobmon.run_parallel(
        runner="sttask",
        task_name="model_inference",
        node_args={
            "measure": measure,
            "location-id": location_id,
            "cmip6-scenario": cmip6_scenario,
            "sex-id": sex_id,
            "age-group-id": age_group_id,
            "year": year,
        },
        task_args={
            "output-dir": output_dir,
            "model-id": model_id,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "15Gb",
            "runtime": "30m",
            "project": "proj_rapidresponse",
            "constraints": "archive",

        },
        max_attempts=1,
    )
