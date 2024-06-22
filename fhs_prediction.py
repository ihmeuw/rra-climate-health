
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


def get_predictions(measure, fhs_location_id, scenario, year, sex_id, age_group_id, model):
    GLOBAL_POPULATION_FILEPATH = '/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/global-human-settlement-layer/2020/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif'

    loc_mapping = load_fhs_lsae_mapping(fhs_location_id)
    fhs_shapefile = loc_mapping.iloc[0].fhs_shape
    location_iso3 = loc_mapping.iloc[0].worldpop_iso3
    simple_loc_mapping = loc_mapping[['fhs_location_id', 'lsae_location_id']]
    income_df = income_funcs.load_binned_income_distribution_proportions(fhs_location_id=fhs_location_id, measure= measure, year_id = year_id) #and year

    fhs_pop_raster = rt.load_raster(GLOBAL_POPULATION_FILEPATH, fhs_shapefile.bounds).set_no_data_value(np.nan)
    fhs_pop_raster = fhs_pop_raster.clip(fhs_shapefile)

    possible_climate_variables = ['temp', 'precip', 'over_30']
    climate_vars_to_match_bins = [v for v in model.vars_to_bin if v in possible_climate_variables]
    continuous_climate_vars = [v for v in ['temp', 'over_30'] if (v in possible_climate_variables) and (v not in climate_vars_to_match_bins) and (v in model.model_vars)]
    climate_vars = [x for x in possible_climate_variables if x in set(model.model_vars + model.vars_to_bin)]

    model.var_info['ihme_loc_id']['coefs'].rename(columns={'ihme_loc_id': 'worldpop_iso3'}, inplace=True)

    climate_rasters = {}
    for var in climate_vars:
        climate_rasters[var] = mf.get_climate_variable_raster(scenario, year_id, var, None, None, untreated=True)
        climate_rasters[var] = climate_rasters[var].resample_to(fhs_pop_raster).clip(fhs_shapefile)

    #def get_climate_variable_raster(location_iso3, scenario, year, climate_var, shapefile, reference_raster, nodata = np.nan, untreated=False):

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
            rasters[var] = climate_array.flatten()

        # Alternative approach is to group pixels to lsae
        #temp_df = pd.DataFrame({'pop': pop_array.flatten(), 'climate_bin_idx': binned_climate_array.flatten()}).groupby('climate_bin_idx', as_index=False).pop.sum()
        # Keeping it as pixels
        temp_df = pd.DataFrame(rasters)
        temp = temp_df.dropna(subset=['population'])
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

    result = pd.DataFrame({'fhs_location_id' : [fhs_location_id], 'year_id': [year_id], 'age_group_id': [age_group_id], 'scenario' : [scenario], 'sex_id':[sex_id],
        'cgf_measure':[measure], 'prevalence': [fhs_df.affected_proportion.sum()]})

    return result

def predict_on_model(measure, model_identifier, fhs_location_id, scenario, year, sex_id, age_group_id):
    MODEL_ROOTS = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/output/models/')
    model_filepath = paths.MODEL_ROOTS / model_identifier / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    pred_df = get_predictions(measure, fhs_location_id, scenario, year, sex_id, age_group_id, model)
    out_filepath = MODEL_ROOTS / model_identifier / 'predictions' / measure / scenario / str(year) / \
        f'mp_{fhs_location_id}_{scenario}_{year}_{age_group_id}_{sex_id}.parquet'
    out_filepath.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_filepath)
