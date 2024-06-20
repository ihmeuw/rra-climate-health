#from ctypes import cdll
#cdll.LoadLibrary('/mnt/share/homes/victorvt/envs/cgf_temperature/lib/libstdc++.so.6')
import xarray as xr
import numpy as np
import pandas as pd
import rasterio as rio
import rasterra as rt
from pathlib import Path
import geopandas as gpd
import glob
import logging
import pickle
import argparse
import sys
import logging
import paths


CHELSA_CRS = rio.crs.CRS.from_string('WGS84')
MOLLEWEIDE_CRS = rio.crs.CRS.from_string('ESRI:54009')

# 241	Child stunting
# 240	Child wasting
# 500	Low body-mass index
# 94	Child underweight

def get_location_pair(identifier):
    from db_queries import get_location_metadata
    loc_meta = get_location_metadata(location_set_id = 39, release_id = 9)[['ihme_loc_id', 'location_id']]
    if isinstance(identifier, int):
        row = loc_meta.loc[loc_meta.location_id == identifier]
    elif isinstance(identifier, str):
        row = loc_meta.loc[loc_meta.ihme_loc_id == identifier]
    assert(len(row) == 1)
    return row.location_id.iloc[0], row.ihme_loc_id.iloc[0]

def get_worldpop_population_raster(location_iso3, return_year = False):
    import glob
    OTHER_GRIDDED_PROJECTS_ROOTPATH = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/other-gridded-pop-projects/')#worldpop-bespoke/NGA/2019.tif')
    OTHER_GRIDDED_PROJECTS_RAW_ROOTPATH = Path('/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/')#worldpop-bespoke/NGA/2019.tif')

    #Try to get a bespoke one
    # 
    worldpop_products = ['worldpop-bespoke', 'worldpop-constrained']
    worldpop_product = None
    for prod in worldpop_products:
        if (OTHER_GRIDDED_PROJECTS_ROOTPATH / prod / location_iso3).exists():
            worldpop_product = prod
            break
    if not worldpop_product:
        raise FileNotFoundError(f'No worldpop for {location_iso3}')
    
    file_paths = glob.glob(str(OTHER_GRIDDED_PROJECTS_ROOTPATH / 'worldpop-constrained' / location_iso3 / f"*.tif"), recursive=True)
    if len(file_paths) == 0:
        file_paths = glob.glob(str(OTHER_GRIDDED_PROJECTS_RAW_ROOTPATH / 'worldpop-constrained' / location_iso3 / f"*.tif"), recursive=True)
        WORLDPOP_FILEPATH = OTHER_GRIDDED_PROJECTS_RAW_ROOTPATH / 'worldpop-constrained' / location_iso3 / f"{location_iso3.lower()}_ppp_2020_constrained.tif"
        logging.info(f"Using WorldPop raw {worldpop_product} for {location_iso3}")
        pop_raster = rt.load_raster(WORLDPOP_FILEPATH).to_crs(MOLLEWEIDE_CRS)
    else:
        year_versions = [int(Path(x).stem) for x in file_paths]
        year_version = max(year_versions)
        WORLDPOP_FILEPATH = OTHER_GRIDDED_PROJECTS_ROOTPATH / 'worldpop-constrained' / location_iso3 / f"{year_version}.tif"
        logging.info(f"Using WorldPop {worldpop_product} year {year_version} for {location_iso3}")
        pop_raster = rt.load_raster(WORLDPOP_FILEPATH)
    if return_year:
        return year_version, pop_raster
    return pop_raster


def get_rra_shapefile(location_iso3):
    SHAPEFILE_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/shapefiles/GLOBAL/2022/admin0.parquet')
    #shapefile = Path('/mnt/team/rapidresponse/pub/population/data/01-raw-data/shapefiles/NGA_adm/NGA_adm0.shp')
    shapefile = gpd.read_parquet(SHAPEFILE_FILEPATH)
    shapefile = shapefile.loc[shapefile.shape_id == location_iso3]
    assert(len(shapefile) == 1)
    return shapefile

def get_LSAE_shapefile(lsae_location_id):
    shapefile = gpd.read_file(paths.LSAE_SHAPEFILE_LOCATION)
    shapefile = shapefile.loc[shapefile.loc_id == lsae_location_id]
    assert(len(shapefile) == 1)
    return shapefile

def get_FHS_shapefile(fhs_location_id):
    shapefile = gpd.read_file(paths.GBD_SHAPEFILE_LOCATION)
    shapefile = shapefile.loc[shapefile.loc_id == fhs_location_id]
    assert(len(shapefile) == 1)
    return shapefile


def xarray_to_raster_old(ds: xr.DataArray, nodata: float | int) -> rt.RasterArray:
    from affine import Affine
    """Convert an xarray DataArray to a RasterArray."""
    lat, lon = ds["lat"].data, ds["lon"].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-dlat,
        f=lat[-1],
    )
    raster = rt.RasterArray(
        data=ds.data[::-1],
        transform=transform,
        crs="EPSG:4326",
        no_data_value=nodata,
    )
    return raster

def xarray_to_raster(ds: xr.DataArray, nodata: float | int) -> rt.RasterArray:
    from affine import Affine
    """Convert an xarray DataArray to a RasterArray."""
    lat, lon = ds["latitude"].data, ds["longitude"].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-dlat,
        f=lat[-1],
    )
    raster = rt.RasterArray(
        data=ds.data[::-1],
        transform=transform,
        crs="EPSG:4326",
        no_data_value=nodata,
    )
    return raster

def get_CHELSA_projection_old(location_iso3, threshold, scenario, year):
    CLIMATE_ROOT = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/human-niche/chelsa-downscaled-projections')
    OVER30_FSTR = "days_above_threshold_{threshold}_{scenario}.nc"
    projection_filepath = CLIMATE_ROOT / location_iso3 / OVER30_FSTR.format(threshold=threshold, scenario=scenario)
    #over30_refscenario_filepath = CLIMATE_ROOT / location_iso3 / OVER30_FSTR.format(threshold="30-0", scenario='ssp245')
    #over30_testscenario_filepath = CLIMATE_ROOT / location_iso3 / OVER30_FSTR.format(threshold="30-0", scenario='ssp126')
    projection_ds = xr.open_dataset(projection_filepath)
    # TODO: Check if year is there
    projection_ds =projection_ds.loc[dict(year=year)]
    return projection_ds
    
def get_CHELSA_projection_raster_old(location_iso3, threshold, scenario, year, shapefile, reference_raster, untreated=False):
    NODATA_CLIMATE = -99
    projection_ds = get_CHELSA_projection_old(location_iso3, threshold, scenario, year)
    # TODO check that variable 'tas' is there
    result_raster = xarray_to_raster_old(projection_ds.tas, NODATA_CLIMATE) 
    if untreated:
        return result_raster
    result_raster = result_raster.to_crs(reference_raster.crs)\
            .clip(shapefile)\
            .mask(shapefile)\
            .resample_to(reference_raster)
    return result_raster

def get_climate_variable_dataset(scenario, year, climate_var):
    import xarray as xr
    climate_variables = {'temp' : 'mean_temperature.nc',
                        'precip' : 'total_precipitation.nc'}
    for threshold in range(30,31):
        climate_variables[f'over_{threshold}'] = f'days_over_{threshold}C/{year}.nc'

    if climate_var not in climate_variables.keys():
        raise ValueError(f"Climate variable {climate_var} not recognized")
    filename = climate_variables[climate_var] if not climate_var.startswith('over') else climate_var + '.nc'
    CLIMATE_FILEPATH = paths.CLIMATE_PROJECTIONS_ROOT / scenario / climate_variables[climate_var]
    
    projection_ds = xr.open_dataset(CLIMATE_FILEPATH)
    projection_ds =projection_ds.loc[dict(year=year)]
    return projection_ds
    
def get_climate_variable_raster(location_iso3, scenario, year, climate_var, shapefile, reference_raster, nodata = np.nan, untreated=False):
    # HACK FOR NOW
    if False: #climate_var.startswith('over') or climate_var.startswith('days'):
        import re
        threshold = str(float(re.search(r'\d+', climate_var).group())).replace(".", "-")# int(climate_var.split('_')[-1][:-1])
        projection_da = get_CHELSA_projection_old(location_iso3, threshold, scenario, year).tas
    else:
        projection_da = get_climate_variable_dataset(scenario, year, climate_var).value
    result_raster = xarray_to_raster(projection_da, nodata) 
    if untreated:
        return result_raster
    result_raster = result_raster.to_crs(reference_raster.crs)\
            .clip(shapefile)\
            .mask(shapefile)\
            .resample_to(reference_raster)
    return result_raster

def get_population_distributions_for_climate_threshold_quantities(climate_array, population_array):
    assert(climate_array.shape == population_array.shape)
    result_dict = dict()
    total_count = np.nansum(population_array)
    ongoing_count = 0
    for condition_value in range(0, 365):
        mask = climate_array == condition_value
        pop_at_climate = int(np.nansum(population_array[mask]))
        result_dict[condition_value] = pop_at_climate
        ongoing_count += pop_at_climate
    return result_dict, total_count

def get_income_bin_proportions(location_id, year, model):
    INCOME_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/income_distributions.parquet')

    income_df_raw = pd.read_parquet(INCOME_FILEPATH)
    income_bins = model.binned_vars[income_var]

    income_df = income_df_raw.query("location_id == @location_id and year_id == @year")[['pop_percent', 'cdf']]
    income_df = pd.concat([income_df, pd.DataFrame({'pop_percent': np.nan, 'cdf':income_bins})]).sort_values(['cdf']).set_index('cdf').interpolate(method='slinear', limit_area='inside').reset_index()
    income_df.loc[income_df.cdf == income_bins[-1] , 'pop_percent'] = 1
    income_df.loc[income_df.cdf == income_bins[0] , 'pop_percent'] = 0
    income_df = income_df.loc[income_df.cdf.isin(income_bins)]

    income_df['cum_pop_percent'] = income_df['pop_percent']
    income_df['cum_pop_percent_upper'] = income_df['pop_percent'].shift(-1)
    income_df['upper_bound'] = income_df.cdf.shift(-1)
    income_df['pop_percent'] = income_df.diff().shift(-1).pop_percent
    income_df = income_df.rename(columns={'cdf':'lower_bound'})
    income_df = income_df[income_df.upper_bound.notna()]
    income_df['income_per_day_bin'] = model.nocountry_grid['income_per_day_bin'].sort_values().unique()
    return income_df

def get_ldipc_bin_proportions(location_id, year, model):
    LDIPC_FILEPATH = Path('/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/national_ldipc_estimates.csv')
    LDIPC_SUBNAT_FILEPATH = Path('/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/admin2_ldipc_estimates.csv')

    income_df_raw = pd.read_csv(LDIPC_SUBNAT_FILEPATH)
    income_df_raw['ldi_pc_pd'] = income_df_raw['ldipc'] / 365
    #income_df_raw = income_df_raw[['population_percentile', 'ldi_pc_pd', 'location_id', 'year_id']]

    income_bins = model.grid_spec['ldi_pc_pd']['bins']

    income_df = income_df_raw.query("location_id == @location_id and year_id == @year")
    #There are nan values, check for them
    if income_df.ldi_pc_pd.isna().any():
        logging.warning(f"Found NaN values in income_df for {location_id}")
        national_ihme_loc_id = income_df.national_ihme_loc_id.unique().item()
        # Use the national instead
        income_df = pd.read_csv(LDIPC_FILEPATH)
        income_df['ldi_pc_pd'] = income_df['ldipc'] / 365
        # I've checked that this national dataset only has a ihme loc id per location id
        income_df = income_df.query("ihme_loc_id == @national_ihme_loc_id and year_id == @year")[['population_percentile', 'ldi_pc_pd']]

    income_df = pd.concat([income_df[['population_percentile', 'ldi_pc_pd']], pd.DataFrame({'population_percentile': np.nan, 'ldi_pc_pd':income_bins})]).sort_values(['ldi_pc_pd']).set_index('ldi_pc_pd').interpolate(method='slinear', limit_area='inside').reset_index()
    income_df.loc[income_df.ldi_pc_pd == income_bins[-1] , 'population_percentile'] = 1
    income_df.loc[income_df.ldi_pc_pd == income_bins[0] , 'population_percentile'] = 0
    income_df = income_df.loc[income_df.ldi_pc_pd.isin(income_bins)]

    income_df['cum_pop_percent'] = income_df['population_percentile']
    income_df['cum_pop_percent_upper'] = income_df['population_percentile'].shift(-1)
    income_df['upper_bound'] = income_df.ldi_pc_pd.shift(-1)
    income_df['population_percentile'] = income_df.diff().shift(-1).population_percentile
    income_df = income_df.rename(columns={'ldi_pc_pd':'lower_bound'})
    income_df = income_df[income_df.upper_bound.notna()]
    income_df['ldi_pc_pd_bin'] = model.nocountry_grid['ldi_pc_pd_bin'].sort_values().unique()
    income_df = income_df.rename(columns={'population_percentile':'proportion_at_income'})
    return income_df



def get_climate_threshold_proportions(model, threshold_counts, threshold_col = 'over_30', threshold_col_bin = 'over_30_bin', bins_left_closed = True, debug=False):
    climate_var_bins = model.grid_spec[threshold_col]['bins']
    over_threshold_bins_df = pd.DataFrame({'lower_bound':climate_var_bins[:-1],
        'upper_bound':climate_var_bins[1:], 
        threshold_col_bin:model.nocountry_grid[threshold_col_bin].sort_values().unique(),
        })

    counts_df = pd.DataFrame.from_dict(threshold_counts, orient='index').reset_index().rename(columns={'index':'days_above_threshold', 0:'count'})
    if not bins_left_closed:
        raise NotImplementedError()

    assign_df = over_threshold_bins_df.merge(counts_df, how='cross')
    assign_df['belong_left_closed'] = (assign_df['days_above_threshold'].ge(assign_df.lower_bound)) & \
        (assign_df['days_above_threshold'].lt(assign_df.upper_bound))
    over_threshold_df = assign_df.loc[assign_df.belong_left_closed].groupby(threshold_col_bin).agg(threshold_pop = ('count', 'sum'))
    counts_df_total_pop = counts_df['count'].sum()
    over_threshold_df['over_threshold_proportion'] = over_threshold_df['threshold_pop'] / counts_df_total_pop
    over_threshold_df = over_threshold_df.reset_index()
    if debug:
        return over_threshold_df, assign_df, counts_df, over_threshold_bins_df
    return over_threshold_df

def get_forecasted_population(location_id, year, sex_id, age_group_id):
    FORECASTED_POPULATIONS_FILEPATH = '/mnt/share/forecasting/data/7/future/population/20240529_500d_2100_lhc_ref_squeeze_hiv_shocks_covid_all_gbd_7_shifted/population.nc'
    future_pop_x = xr.open_dataset(FORECASTED_POPULATIONS_FILEPATH)
    filtered_pop_x = future_pop_x.loc[dict(year_id=year, location_id = location_id, sex_id = sex_id, age_group_id=age_group_id)]
    mean_population = filtered_pop_x.mean(dim='draw').population.to_numpy()[0]
    return mean_population

def get_forecasted_population_full():
    FORECASTED_POPULATIONS_FILEPATH = '/mnt/share/forecasting/data/7/future/population/20240529_500d_2100_lhc_ref_squeeze_hiv_shocks_covid_all_gbd_7_shifted/population.nc'
    future_pop_ds = xr.open_dataset(FORECASTED_POPULATIONS_FILEPATH)
    mean_population = filtered_pop_x.mean(dim='draw').population.to_numpy()[0]
    return mean_population

def get_model(measure, threshold, age_group_id, sex_id):
    MODEL_PICKLE_FILEPATH = f'/ihme/scratch/users/victorvt/threshold_inv/{measure}/{threshold}/model_gdp.pkl'
    with open(MODEL_PICKLE_FILEPATH, 'rb') as f:
        model = pickle.load(f)
    return model

def get_predictions(measure, lsae_location_id, scenario, year, sex_id, age_group_id, model):
    #measure, location_identifier, scenario, year, threshold = 'stunting', 'NGA', 'ssp245', 2050, 30
    #location_identifier = 'NGA'
    #scenario = 'ssp245'
    #year = 2050
    income_var = 'ldi_pc_pd'
    climate_var = 'over_30'

    location_var = 'ihme_loc_id'
    
    income_var_bin = income_var + '_bin'
    climate_var_bin = climate_var + '_bin'
    location_var = 'ihme_loc_id'
    # sex_id = 1
    # age_group_id = 5
    #measure = 'stunting'

    assert(scenario in ['ssp245', 'ssp119'])

    #location_id, location_iso3 = get_location_pair(location_identifier)
    loc_mapping = get_fhs_lsae_location_mapping(short=True).query("lsae_location_id == @lsae_location_id")
    assert(len(loc_mapping) == 1)
    loc_mapping = loc_mapping.iloc[0]
    lsae_location_id = loc_mapping['lsae_location_id']
    location_iso3 = loc_mapping['worldpop_iso3']
    
    pop_raster = get_worldpop_population_raster(location_iso3)
    logging.debug(f"WorldPop population raster CRS {pop_raster.crs}" )

    #for later use in determining the proportion of population between LSAE and FHS areas
    fhs_location_id = loc_mapping['fhs_location_id']
    fhs_shapefile = get_FHS_shapefile(fhs_location_id).to_crs(MOLLEWEIDE_CRS)
    fhs_pop_raster = pop_raster.clip(fhs_shapefile).mask(fhs_shapefile)

    shapefile = get_LSAE_shapefile(lsae_location_id)
    shapefile = shapefile.to_crs(MOLLEWEIDE_CRS)#(pop_raster.crs)
    pop_raster = pop_raster.clip(shapefile).mask(shapefile)

    climate_threshold_raster = get_climate_variable_raster(location_iso3, scenario, year, climate_var, shapefile, pop_raster, nodata=-99)

    pop_array = pop_raster.to_numpy()
    climate_threshold_array = climate_threshold_raster.to_numpy()
    climate_threshold_numerator, climate_threshold_denom = get_population_distributions_for_climate_threshold_quantities(climate_threshold_array, pop_array)

    #model = make_model(measure, sex_id = sex_id, age_group_id = age_group_id)

    income_df = get_ldipc_bin_proportions(lsae_location_id, year, model)
    climate_threshold_df = get_climate_threshold_proportions(model, climate_threshold_numerator, 
        threshold_col = climate_var, bins_left_closed = True, debug=False)

    if location_iso3 in model.available_locations:
        base_grid = model.weight_grid.query(f"ihme_loc_id == @location_iso3")
    else:
        base_grid = model.nocountry_grid
    result_df = base_grid.merge(income_df[['proportion_at_income', income_var_bin]], how='left', on=income_var_bin)\
        .merge(climate_threshold_df[['over_threshold_proportion', climate_var_bin]], how='left', on=climate_var_bin)

    pred_df = result_df 
    pred_df['lsae_location_id'] = lsae_location_id
    pred_df['fhs_location_id'] = fhs_location_id
    pred_df['year'] = year
    pred_df['scenario'] = scenario
    #pred_df['threshold'] = threshold
    pred_df['age_group_id'] = age_group_id
    pred_df['sex_id'] = sex_id
    pred_df['measure'] = measure

    fhs_pop_total = fhs_pop_raster.to_numpy()[~fhs_pop_raster.no_data_mask].sum()
    lsae_pop_total = pop_array[~pop_raster.no_data_mask].sum()
    pred_df['fhs_population'] = fhs_pop_total
    pred_df['lsae_population'] = lsae_pop_total
    pred_df['lsae_fhs_pop_proportion'] = lsae_pop_total / fhs_pop_total
    for col in ['grid_cell', climate_var_bin, income_var_bin]:
        pred_df[col] = pred_df[col].astype(str)
    if True:
        return pred_df
    projected_population_total = get_forecasted_population(location_id, year, sex_id, age_group_id)

    pred_df['population'] = projected_population_total
    pred_df['susceptible'] =  pred_df.population * pred_df.over_threshold_proportion * pred_df.proportion_at_income
    pred_df['prediction'] =  pred_df.population * pred_df.over_threshold_proportion * pred_df.proportion_at_income * pred_df.prediction_weight

    return pred_df


def get_predictions_countrylevel(measure, location_iso3, scenario, year, sex_id, age_group_id, model):
    #measure, location_identifier, scenario, year, threshold = 'stunting', 'NGA', 'ssp245', 2050, 30
    #location_identifier = 'NGA'
    #scenario = 'ssp245'
    #year = 2050
    income_var = 'ldi_pc_pd'
    climate_var = 'over_30'

    location_var = 'ihme_loc_id'
    
    income_var_bin = income_var + '_bin'
    climate_var_bin = climate_var + '_bin'
    location_var = 'ihme_loc_id'
    # sex_id = 1
    # age_group_id = 5
    #measure = 'stunting'

    assert(scenario in ['ssp245', 'ssp119'])

    location_id, location_iso3 = get_location_pair(location_iso3)
    loc_mapping = get_fhs_lsae_location_mapping(short=True).query("lsae_location_id == @lsae_location_id")
    assert(len(loc_mapping) == 1)
    loc_mapping = loc_mapping.iloc[0]
    lsae_location_id = loc_mapping['lsae_location_id']
    location_iso3 = loc_mapping['worldpop_iso3']
    
    pop_raster = get_worldpop_population_raster(location_iso3)
    logging.debug(f"WorldPop population raster CRS {pop_raster.crs}" )

    #for later use
    fhs_location_id = loc_mapping['fhs_location_id']
    fhs_shapefile = get_FHS_shapefile(fhs_location_id).to_crs(MOLLEWEIDE_CRS)
    fhs_pop_raster = pop_raster.clip(fhs_shapefile).mask(fhs_shapefile)

    shapefile = get_LSAE_shapefile(lsae_location_id)
    shapefile = shapefile.to_crs(MOLLEWEIDE_CRS)#(pop_raster.crs)
    pop_raster = pop_raster.clip(shapefile).mask(shapefile)

    climate_threshold_raster = get_climate_variable_raster(location_iso3, scenario, year, climate_var, shapefile, pop_raster, nodata=-99)

    pop_array = pop_raster.to_numpy()
    climate_threshold_array = climate_threshold_raster.to_numpy()
    climate_threshold_numerator, climate_threshold_denom = get_population_distributions_for_climate_threshold_quantities(climate_threshold_array, pop_array)

    #model = make_model(measure, sex_id = sex_id, age_group_id = age_group_id)

    income_df = get_ldipc_bin_proportions(lsae_location_id, year, model)
    climate_threshold_df = get_climate_threshold_proportions(model, climate_threshold_numerator, 
        threshold_col = climate_var, bins_left_closed = True, debug=False)

    if location_iso3 in model.available_locations:
        base_grid = model.weight_grid.query(f"ihme_loc_id == @location_iso3")
    else:
        base_grid = model.nocountry_grid
    result_df = base_grid.merge(income_df[['proportion_at_income', income_var_bin]], how='left', on=income_var_bin)\
        .merge(climate_threshold_df[['over_threshold_proportion', climate_var_bin]], how='left', on=climate_var_bin)

    pred_df = result_df #model.pred_grid.query("iso3 == @location_iso3").merge(result_df, how='left', on=['grid_cell', 'over30_avgperyear_bin', income_col_bin])
    pred_df['lsae_location_id'] = lsae_location_id
    pred_df['fhs_location_id'] = fhs_location_id
    pred_df['year'] = year
    pred_df['scenario'] = scenario
    #pred_df['threshold'] = threshold
    pred_df['age_group_id'] = age_group_id
    pred_df['sex_id'] = sex_id
    pred_df['measure'] = measure

    fhs_pop_total = fhs_pop_raster.to_numpy()[~fhs_pop_raster.no_data_mask].sum()
    lsae_pop_total = pop_array[~pop_raster.no_data_mask].sum()
    pred_df['fhs_population'] = fhs_pop_total
    pred_df['lsae_population'] = lsae_pop_total
    pred_df['lsae_fhs_pop_proportion'] = lsae_pop_total / fhs_pop_total
    for col in ['grid_cell', climate_var_bin, income_var_bin]:
        pred_df[col] = pred_df[col].astype(str)
    if True:
        return pred_df
    projected_population_total = get_forecasted_population(location_id, year, sex_id, age_group_id)

    pred_df['population'] = projected_population_total
    pred_df['susceptible'] =  pred_df.population * pred_df.over_threshold_proportion * pred_df.proportion_at_income
    pred_df['prediction'] =  pred_df.population * pred_df.over_threshold_proportion * pred_df.proportion_at_income * pred_df.prediction_weight

    return pred_df

def run_model_and_save(measure, model_identifier, sex_id, age_group_id, model_spec, grid_vars):
    model = make_model(measure, model_spec, grid_list = grid_vars, sex_id = sex_id, age_group_id = age_group_id)
    model_filepath = paths.MODEL_ROOTS / model_identifier / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
    model_filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def run_model_and_predict(measure, model_identifier, lsae_location_id, scenario, year, sex_id, age_group_id, model_spec, grid_vars, save_model=False):
    #Make model and predict
    model = make_model(measure, model_spec, grid_list = grid_vars, sex_id = sex_id, age_group_id = age_group_id)
    pred_df = get_predictions(measure, lsae_location_id, scenario, year, sex_id, age_group_id, model)
    if save_model:
        model_filepath = paths.MODEL_ROOTS / model_identifier / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
    pred_df['model_identifier'] = model_identifier
    pred_df['model_spec'] = model_spec
    pred_df['grid_vars'] = ','.join(grid_vars)

    out_filepath = paths.MODEL_ROOTS / model_identifier / 'predictions' / measure / scenario / year / \
        f'mp_{lsae_location_id}_{scenario}_{year}_{age_group_id}_{sex_id}.parquet'
    out_filepath.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_filepath)

def predict_on_model(measure, model_identifier, lsae_location_id, scenario, year, sex_id, age_group_id):
    MODEL_ROOTS = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/output/models/')
    model_filepath = paths.MODEL_ROOTS / model_identifier / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    pred_df = get_predictions(measure, lsae_location_id, scenario, year, sex_id, age_group_id, model)
    out_filepath = MODEL_ROOTS / model_identifier / 'predictions' / measure / scenario / str(year) / \
        f'mp_{lsae_location_id}_{scenario}_{year}_{age_group_id}_{sex_id}.parquet'
    out_filepath.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_filepath)


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
    'ldi_pc_pd': {'bin_category': 'household', 'bin_strategy' : 'quantiles', 'nbins': 10, 'type': 'income'},
    'over_30': {'bin_category': 'location', 'bin_strategy' : 'custom_daysover', 'nbins': 10, 'type':'climate'},
    'temp' : {'bin_category': 'location', 'bin_strategy' : 'quantiles', 'nbins': 10, 'type':'climate'}
}

def make_model(cgf_measure, model_spec, grid_list = None, grid_spec = None, 
    sex_id = None, age_group_id = None, filter = None, location_var = 'ihme_loc_id'):
    import re
    import cgf_utils
    import sklearn
    from pymer4 import Lmer
    df = get_modeling_input_data(cgf_measure)

    if 'grid_cell' in model_spec and grid_list is None and grid_spec is None:
        raise ValueError("Model specification includes grid_cell but no grid_list or grid_spec is provided")
    if grid_list is not None:
        grid_spec = {'grid_order' : grid_list}
        for var in grid_list:
            grid_spec[var] = STANDARD_BINNING_SPECS[var]
    #model_spec = f'{cgf_measure} ~ (1 | {location_var}) + (1 | grid_cell)'
    # grid_spec = {'grid_order' : ['ldi_pc_pd', 'over_30'],
    #     'ldi_pc_pd': {'bin_category': 'household', 'bin_strategy' : 'quantiles', 'nbins': 10, 'type': 'income'},
    #     'over_30': {'bin_category': 'location', 'bin_strategy' : 'custom_daysover', 'nbins': 10, 'type':'climate'}}

    binned_df = df.copy()

    for var in grid_spec['grid_order']:
        grid_spec[var]['var_bin'] = var + '_bin'
        grid_spec[var]['bins'], binned_df = cgf_utils.group_and_bin_column_definition(binned_df, var, 
            grid_spec[var]['bin_category'], grid_spec[var]['nbins'], bin_strategy = grid_spec[var]['bin_strategy'], retbins = True) 

    binned_vars = [grid_spec[var]['var_bin'] for var in grid_spec['grid_order']]
    binned_df['grid_cell'] = binned_df[binned_vars].astype(str).apply('_'.join, axis=1)

    cols_to_scale = []#'over30_avgperyear', 'precip', 'precip_cumavg_5' ,'temp', 'temp_cumavg_5', 'income_per_day']
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

    weight_grid = binned_df[model_vars[2:] + binned_vars].drop_duplicates().merge(pd.DataFrame({location_var:binned_df[location_var].unique()}), how = 'cross')
    weight_grid['prediction_weight'] = model.predict(weight_grid, verify_predictions=False)
    modelled_locations = binned_df[location_var].unique()

    nocountry_grid = binned_df[model_vars[2:] + binned_vars].drop_duplicates()
    nocountry_grid[location_var] = np.nan
    nocountry_grid['prediction_weight'] = model.predict(nocountry_grid, verify_predictions=False)

    model.model_vars = model_vars
    model.nocountry_grid = nocountry_grid
    model.weight_grid = weight_grid
    model.available_locations = modelled_locations
    model.grid_spec = grid_spec
    return model

def make_model_old(cgf_measure, sex_id = None, age_group_id = None, filter = None, climate_var = 'over_30', 
    income_var = 'ldi_pc_pd', location_var = 'ihme_loc_id'):
    import re
    import cgf_utils
    import sklearn
    df = get_modeling_input_data(cgf_measure)

    nbins = 10

    income_var_bin = income_var + '_bin'
    climate_var_bin = climate_var + '_bin'

    model_spec = f'{cgf_measure} ~ (1 | {location_var}) + (1 | grid_cell)'

    over30_bins, binned_df = cgf_utils.group_and_bin_column_definition(df, climate_var, 'location', nbins, bin_strategy = 'custom_daysover', retbins = True) #custom_daysover 0_more_readable
    income_bins, binned_df = cgf_utils.group_and_bin_column_definition(binned_df, income_var, 'household', nbins, retbins = True)

    binned_df['grid_cell'] = binned_df[climate_var_bin].astype(str) +'_'+ binned_df[income_var_bin].astype(str)

    cols_to_scale = []#'over30_avgperyear', 'precip', 'precip_cumavg_5' ,'temp', 'temp_cumavg_5', 'income_per_day']
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
    binned_vars = [climate_var_bin, income_var_bin]
    bins = {climate_var:over30_bins, income_var : income_bins}

    weight_grid = binned_df[model_vars[2:] + binned_vars].drop_duplicates().merge(pd.DataFrame({location_var:binned_df[location_var].unique()}), how = 'cross')
    weight_grid['prediction_weight'] = model.predict(weight_grid, verify_predictions=False)
    modelled_locations = binned_df[location_var].unique()

    nocountry_grid = binned_df[model_vars[2:] + binned_vars].drop_duplicates()
    nocountry_grid[location_var] = np.nan
    nocountry_grid['prediction_weight'] = model.predict(nocountry_grid, verify_predictions=False)

    model.binned_vars = bins
    model.model_vars = model_vars
    model.nocountry_grid = nocountry_grid
    model.weight_grid = weight_grid
    model.available_locations = modelled_locations
    return model


def get_fhs_lsae_location_mapping(short = False, extra_checks = False):
    import paths
    import db_queries
    lsae_loc_meta = db_queries.get_location_metadata(125, release_id = 16)
    lsae_loc_most_detailed = lsae_loc_meta.query("most_detailed == 1").copy()[['location_id', 'ihme_loc_id', 'location_name', 'path_to_top_parent']].rename(columns = {'location_id':'lsae_location_id', 'ihme_loc_id':'lsae_ihme_loc_id', 'location_name':'lsae_location_name'})

    fhs_loc_meta = db_queries.get_location_metadata(39, release_id = 10)
    # FIX THE ONE PROBLEMATIC ETHIOPIA SUBNATIONAL
    SNNP_lsae_locid = lsae_loc_meta.query("location_name == 'Southern Nations, Nationalities, and Peoples'").location_id.item()
    SNNP_fhs_locid = fhs_loc_meta.query("location_name == 'Southern Nations, Nationalities, and Peoples'").location_id.item()
    fhs_loc_meta.loc[fhs_loc_meta.location_id == SNNP_fhs_locid, 'location_id'] = SNNP_lsae_locid
    fhs_loc_most_detailed = fhs_loc_meta.query("most_detailed == 1").copy()[['location_id', 'ihme_loc_id', 'local_id', 'location_name']].rename(columns = {'location_id':'fhs_location_id', 'ihme_loc_id':'fhs_ihme_loc_id', 'local_id':'fhs_local_id', 'location_name':'fhs_location_name'})

    worldpop_locs = [name.parts[-1] for name in paths.WORLDPOP_FILEPATH.glob('*/')]

    lsae_loc_most_detailed['hierarchy'] = lsae_loc_most_detailed['path_to_top_parent'].str.split(',')
    exploded_lsae_loc = lsae_loc_most_detailed.explode('hierarchy')
    exploded_lsae_loc['hierarchy'] = exploded_lsae_loc['hierarchy'].astype(int)
    exploded_lsae_loc = exploded_lsae_loc.merge(lsae_loc_meta[['location_id', 'location_name']].rename(columns = {'location_id': 'hierarchy', 'location_name':'lsae_hierarchy_name'}), on='hierarchy', how='left')
    lsae_fhs_mapping = exploded_lsae_loc.merge(fhs_loc_most_detailed, left_on='hierarchy', right_on='fhs_location_id', how='right')
    fhs_loc_meta.loc[fhs_loc_meta.location_id == SNNP_lsae_locid, 'location_id'] = SNNP_fhs_locid
    fhs_loc_most_detailed.loc[fhs_loc_most_detailed.fhs_location_id == SNNP_lsae_locid, 'fhs_location_id'] = SNNP_fhs_locid

    lsae_fhs_mapping.loc[lsae_fhs_mapping.fhs_location_id == SNNP_lsae_locid, 'fhs_location_id'] = SNNP_fhs_locid

        # Now, FHS, check that the locations match between the fhs shapefile, the most detailed loc metadata and the forecasted population data
    
    #assert len(set(fhs_loc_most_detailed.fhs_location_id.unique()) - set(locs_in_forecasted_population)) == 0
    assert set(fhs_loc_most_detailed.fhs_location_id.unique()) == set(lsae_fhs_mapping.fhs_location_id.unique())

    # For income, not all LSAE locations end up being needed so it won't match the income ones, but make sure the ones we do need are included in both ldi and in the shapefile
    #assert len(set(lsae_fhs_mapping.lsae_location_id.unique()) - set(ldipc.location_id.unique())) == 0
    #assert len(set(lsae_fhs_mapping.lsae_location_id.unique()) - set(lsae_shapes.loc_id.unique())) == 0

    # For worldpop, make sure all the locations we need are included in the worldpop data
    lsae_fhs_mapping['worldpop_iso3'] = lsae_fhs_mapping.fhs_ihme_loc_id.str[0:3]
    assert len(set(lsae_fhs_mapping.worldpop_iso3.unique()) - set(worldpop_locs)) == 0
    if extra_checks:
        gbd_shapes = gpd.read_file(paths.GBD_SHAPEFILE_LOCATION)
        assert len(set(fhs_loc_most_detailed.fhs_location_id.unique()) - set(gbd_shapes.loc_id.unique())) == 0

    return lsae_fhs_mapping if not short else lsae_fhs_mapping[['fhs_location_id', 'lsae_location_id', 'worldpop_iso3']]
