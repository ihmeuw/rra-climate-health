
import xarray as xr
import numpy as np
import pandas as pd
import rasterio as rio
import rasterra as rt
from pathlib import Path
import geopandas as gpd
from pymer4 import Lmer
import glob
import logging
import pickle
import sys
from scipy.special import expit

from location_mapping import load_fhs_lsae_mapping
import cgf_utils
import mf
import paths
import income_funcs
#from income_funcs import load_binned_income_distribution_proportions


def get_predictions(measure, fhs_location_id, scenario, year_id, model_identifier):
    GLOBAL_POPULATION_FILEPATH = '/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/global-human-settlement-layer/2020/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif'

    loc_mapping = load_fhs_lsae_mapping(fhs_location_id)
    fhs_shapefile = loc_mapping.iloc[0].fhs_shape
    location_iso3 = loc_mapping.iloc[0].worldpop_iso3
    simple_loc_mapping = loc_mapping[['fhs_location_id', 'lsae_location_id']]
    income_df = income_funcs.load_binned_income_distribution_proportions(fhs_location_id=fhs_location_id, measure= measure, year_id = year_id) #and year

    models = get_model_family(model_identifier, measure)
    model = models[0]['model'] #getting the first model to base bins and variable info off of

    fhs_pop_raster = rt.load_raster(GLOBAL_POPULATION_FILEPATH, fhs_shapefile.bounds).set_no_data_value(np.nan)
    fhs_pop_raster = fhs_pop_raster.clip(fhs_shapefile)

    possible_climate_variables = ['temp', 'precip', 'over_30']
    climate_vars_to_match_bins = [v for v in model.vars_to_bin if v in possible_climate_variables]
    continuous_climate_vars = [v for v in ['temp', 'over_30'] if (v in possible_climate_variables) and (v not in climate_vars_to_match_bins) and (v in model.model_vars)]
    climate_vars = [x for x in possible_climate_variables if x in set(model.model_vars + model.vars_to_bin)]

    climate_rasters = {}
    for var in climate_vars:
        climate_rasters[var] = mf.get_climate_variable_raster(scenario, year_id, var, None, None, untreated=True)
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

        #climate_raster = big_climate_raster.clip(lsae_shapefile).mask(lsae_shapefile)#.resample_to(pop_raster)
        for var in continuous_climate_vars:
            climate_raster = climate_rasters[var].clip(lsae_shapefile).mask(lsae_shapefile)
            climate_array = climate_raster.to_numpy()
            assert pop_array.shape == climate_array.shape
            if var in model.scaled_vars:
                varmin, varmax = model.scaling_bounds[var]
                climate_array = scale_like_input_data(climate_array, varmin, varmax)
            rasters[var] = climate_array.flatten()

        # Alternative approach is to group pixels to lsae
        #temp_df = pd.DataFrame({'pop': pop_array.flatten(), 'climate_bin_idx': binned_climate_array.flatten()}).groupby('climate_bin_idx', as_index=False).pop.sum()
        # Keeping it as pixels
        pixels_df = pd.DataFrame(rasters)
        pixels_df = pixels_df.dropna(subset=['population']) # TODO: Reconsider this, pretty sure it's right
        pixels_df['lsae_pop'] = np.nansum(pop_array)
        pixels_df['lsae_location_id'] = lsae_location_id
        pixels_df['worldpop_iso3'] = admin2_row.worldpop_iso3

        local_income_df = income_df.query('lsae_location_id == @lsae_location_id')
        for var in climate_vars_to_match_bins:
            pixels_df = pixels_df.merge(model.var_info[var]['bins'], left_on=var+'_bin_idx', right_index=True, how='inner')
        pixels_df = pixels_df.merge(local_income_df, on='lsae_location_id', how='left')

        # The lines from now on have coefficients and so are age_group and sex_id - specific
        # So model-specific. Parallelizing by them
        for m in models:
            model = m['model']
            px_for_model_df = pixels_df.copy()
            model.var_info['ihme_loc_id']['coefs'].rename(columns={'ihme_loc_id': 'worldpop_iso3'}, inplace=True)
            
            if model.has_grid:
                px_for_model_df = px_for_model_df.merge(model.grid_spec['grid_definition'], how='left')
                px_for_model_df = px_for_model_df.merge(model.var_info['grid_cell']['coefs'], how='left')

            for var in continuous_climate_vars:
                assert(len(model.var_info[var]['coefs'].columns) == 1)
                colname = model.var_info[var]['coefs'].columns[0]
                px_for_model_df[colname] = model.var_info[var]['coefs'][colname].item()

            px_for_model_df = px_for_model_df.merge(model.var_info['ihme_loc_id']['coefs'], how='left', on='worldpop_iso3') 
            px_for_model_df.ihme_loc_id_coef = px_for_model_df.ihme_loc_id_coef.fillna(0)
            
            #build the logistic input one variable at a time
            px_for_model_df['logistic_input'] = model.var_info['intercept']['coef']
            px_for_model_df['logistic_input'] += px_for_model_df['ihme_loc_id_coef']
            for var in climate_vars_to_match_bins:
                if var in model.grid_spec['grid_order']: continue
                px_for_model_df['logistic_input'] += px_for_model_df[var+'_bin_coef']
            for var in continuous_climate_vars:
                px_for_model_df['logistic_input'] += px_for_model_df[var+'_coef']
            if model.has_grid:
                px_for_model_df['logistic_input'] += px_for_model_df['grid_cell_coef']
            px_for_model_df['prediction'] = expit(px_for_model_df['logistic_input'])
            px_for_model_df['age_group_id'] = m['age_group_id']
            px_for_model_df['sex_id'] = m['sex_id']
            admin_dfs.append(px_for_model_df)

    fhs_df = pd.concat(admin_dfs)
    #24s to 92s
    #2:27 to 
    #24s to 92s
    fhs_df['population_at_income'] = fhs_df['population'] * fhs_df['proportion_at_income']
    fhs_df['population_proportion_at_income'] = fhs_df['population_at_income'] / fhs_df['lsae_pop'] / len(loc_mapping)
    fhs_df['affected_proportion'] = fhs_df['population_proportion_at_income'] * fhs_df['prediction']

    result_df = fhs_df.groupby(['fhs_location_id', 'age_group_id', 'sex_id']).agg(
        #susceptible_proportion = ('population_proportion_at_income', 'sum'),  #For testing, HAS to be == 1
        affected_proportion = ('affected_proportion', 'sum'))
    result_df['year_id'] = year_id
    result_df['scenario'] = scenario
    result_df['measure'] = measure

    return result_df

def predict_on_model(measure, model_identifier, fhs_location_id, scenario, year_id, ):
    pred_df = get_predictions(measure, fhs_location_id, scenario, year_id, model_identifier)
    out_filepath = paths.MODEL_ROOTS / model_identifier / 'predictions' / measure / scenario / str(year_id) / \
        f'mp_{fhs_location_id}_{scenario}_{year_id}.parquet'
    out_filepath.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_filepath)

def get_model_family(model_identifier, measure):
    import glob
    import re
    folder_path = paths.MODEL_ROOTS / model_identifier 
    models = []

    # Pattern to match the files and capture X and Y values
    pattern = f"model_{measure}_([0-9]+)_([0-9]+).pkl"

    # Iterate through matching files in the folder
    for file in folder_path.glob(f"model_{measure}_*.pkl"):
        match = re.match(pattern, file.name)
        if match:
            age_group_id, sex_id = match.groups()
            with open(file, 'rb') as f:
                models.append({'model': pickle.load(f), 'age_group_id': age_group_id, 'sex_id': sex_id})
        else:
            raise ValueError(f"Filename {file} does not match the pattern {pattern.pattern}")
    return models


def scale_like_input_data(to_scale, input_min, input_max):
    return (to_scale - input_min) / (input_max - input_min)

def reverse_scaling(X_scaled, original_min, original_max, scaled_min, scaled_max):
    X_std = (X_scaled - scaled_min) / (scaled_max - scaled_min)
    X_original = X_std * (original_max - original_min) + original_min
    return X_original
