from ctypes import cdll
cdll.LoadLibrary('/mnt/share/homes/victorvt/envs/cgf_temperature/lib/libstdc++.so.6')
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
import rasterra as rt
from pathlib import Path
import geopandas as gpd
from pymer4 import Lmer
import glob
import logging
import pickle
import argparse
import sys
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

def get_worldpop_population_raster(location_iso3, return_year = True):
    import glob
    OTHER_GRIDDED_PROJECTS_ROOTPATH = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/other-gridded-pop-projects/')#worldpop-bespoke/NGA/2019.tif')
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
    year_versions = [int(Path(x).stem) for x in file_paths]
    year_version = max(year_versions)
    WORLDPOP_FILEPATH = OTHER_GRIDDED_PROJECTS_ROOTPATH / 'worldpop-constrained' / location_iso3 / f"{year_version}.tif"
    logging.info(f"Using WorldPop {worldpop_product} year {year_version} for {location_iso3}")
    pop_raster = rt.load_raster(WORLDPOP_FILEPATH)
    if return_year:
        return year_version, pop_raster
    return pop_raster


def get_shapefile(location_iso3):
    SHAPEFILE_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/shapefiles/GLOBAL/2022/admin0.parquet')
    #shapefile = Path('/mnt/team/rapidresponse/pub/population/data/01-raw-data/shapefiles/NGA_adm/NGA_adm0.shp')
    shapefile = gpd.read_parquet(SHAPEFILE_FILEPATH)
    shapefile = shapefile.loc[shapefile.shape_id == location_iso3]
    assert(len(shapefile) == 1)
    return shapefile

def xarray_to_raster(ds: xr.DataArray, nodata: float | int) -> rt.RasterArray:
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

def get_CHELSA_projection(location_iso3, threshold, scenario, year):
    CLIMATE_ROOT = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/human-niche/chelsa-downscaled-projections')
    OVER30_FSTR = "days_above_threshold_{threshold}_{scenario}.nc"
    projection_filepath = CLIMATE_ROOT / location_iso3 / OVER30_FSTR.format(threshold=threshold, scenario=scenario)
    #over30_refscenario_filepath = CLIMATE_ROOT / location_iso3 / OVER30_FSTR.format(threshold="30-0", scenario='ssp245')
    #over30_testscenario_filepath = CLIMATE_ROOT / location_iso3 / OVER30_FSTR.format(threshold="30-0", scenario='ssp126')
    projection_ds = xr.open_dataset(projection_filepath)
    # TODO: Check if year is there
    projection_ds =projection_ds.loc[dict(year=year)]
    return projection_ds
    
def get_CHELSA_projection_raster(location_iso3, threshold, scenario, year, shapefile, reference_raster, untreated=False):
    NODATA_CLIMATE = -99
    projection_ds = get_CHELSA_projection(location_iso3, threshold, scenario, year)
    # TODO check that variable 'tas' is there
    result_raster = xarray_to_raster(projection_ds.tas, NODATA_CLIMATE) 
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
    income_df_raw = pd.read_csv(LDIPC_FILEPATH)
    income_df_raw['ldi_pc_pd'] = income_df_raw['ldipc'] / 365
    income_df_raw = income_df_raw[['population_percentile', 'ldi_pc_pd', 'location_id', 'year_id']]

    income_bins = model.binned_vars['ldi_pc_pd']

    income_df = income_df_raw.query("location_id == @location_id and year_id == @year")[['population_percentile', 'ldi_pc_pd']]
    income_df = pd.concat([income_df, pd.DataFrame({'population_percentile': np.nan, 'ldi_pc_pd':income_bins})]).sort_values(['ldi_pc_pd']).set_index('ldi_pc_pd').interpolate(method='slinear', limit_area='inside').reset_index()
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
    income_df = income_df.rename(columns={'population_percentile':'pop_percent'})
    return income_df



def get_climate_threshold_proportions(model, threshold_counts, threshold_col = 'over_30', threshold_col_bin = 'over_30_bin', bins_left_closed = True, debug=False):
    over_threshold_bins_df = pd.DataFrame({'lower_bound':model.binned_vars[threshold_col][:-1],
        'upper_bound':model.binned_vars[threshold_col][1:], 
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

    pred_df['population'] = current_pop

def get_model(measure, threshold, age_group_id, sex_id):
    MODEL_PICKLE_FILEPATH = f'/ihme/scratch/users/victorvt/threshold_inv/{measure}/{threshold}/model_gdp.pkl'
    with open(MODEL_PICKLE_FILEPATH, 'rb') as f:
        model = pickle.load(f)
    return model

def get_predictions(measure, location_identifier, scenario, year, threshold, sex_id, age_group_id):
    #measure, location_identifier, scenario, year, threshold = 'stunting', 'NGA', 'ssp245', 2050, 30
    #location_identifier = 'NGA'
    #scenario = 'ssp245'
    #year = 2050
    income_col = 'gdp_pc_pd'
    income_col_bin = 'gdp_pc_pd_bin'
    climate_threshold_col = 'over_30'
    climate_threshold_col_bin = 'over_30_bin'
    location_var = 'ihme_loc_id'
    # sex_id = 1
    # age_group_id = 5
    #threshold = 30
    threshold_str = str(float(threshold)).replace(".", "-")#"30-0"
    #measure = 'stunting'

    assert(scenario in ['ssp245', 'ssp126'])

    location_id, location_iso3 = get_location_pair(location_identifier)
    pop_year, pop_raster = get_worldpop_population_raster(location_iso3)

    shapefile = get_shapefile(location_iso3)
    CHELSA_CRS = rio.crs.CRS.from_string('WGS84')
    MOLLEWEIDE_CRS = rio.crs.CRS.from_string('ESRI:54009')

    logging.debug(f"WorldPop population raster CRS {pop_raster.crs}" )
    shapefile = shapefile.to_crs(MOLLEWEIDE_CRS)#(pop_raster.crs)
    pop_raster = pop_raster.clip(shapefile).mask(shapefile)

    climate_threshold_raster = get_CHELSA_projection_raster(location_iso3, threshold_str, scenario, pop_year, shapefile, pop_raster)

    pop_array = pop_raster.to_numpy()
    climate_threshold_array = climate_threshold_raster.to_numpy()
    climate_threshold_numerator, climate_threshold_denom = get_population_distributions_for_climate_threshold_quantities(climate_threshold_array, pop_array)

    model = make_model(measure, sex_id = sex_id, age_group_id = age_group_id)
    income_df = get_ldipc_bin_proportions(location_id, year, model)
    climate_threshold_df = get_climate_threshold_proportions(model, climate_threshold_numerator, 
        threshold_col = climate_threshold_col, bins_left_closed = True, debug=False)

    if location_iso3 in model.available_locations:
        base_grid = model.weight_grid.query(f"ihme_loc_id == @location_iso3")
    else:
        base_grid = model.nocountry_grid
    result_df = base_grid.merge(income_df[['pop_percent', income_col_bin]], how='left', on=income_col_bin)\
        .merge(climate_threshold_df[['over_threshold_proportion', climate_threshold_col_bin]], how='left', on=climate_threshold_col_bin)

    pred_df = result_df #model.pred_grid.query("iso3 == @location_iso3").merge(result_df, how='left', on=['grid_cell', 'over30_avgperyear_bin', income_col_bin])

    projected_population_total = get_forecasted_population(location_id, year, sex_id, age_group_id)

    pred_df['population'] = projected_population_total
    pred_df['susceptible'] =  pred_df.population * pred_df.over_threshold_proportion * pred_df.pop_percent
    pred_df['prediction'] =  pred_df.population * pred_df.over_threshold_proportion * pred_df.pop_percent * pred_df.pred_weight

    for col in ['grid_cell', climate_threshold_col_bin, income_col_bin]:
        pred_df[col] = pred_df[col].astype(str)

    return pred_df

def fit_and_predict_LMER(data:pd.DataFrame, model_spec:str):
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


def make_model(cgf_measure, sex_id = None, age_group_id = None, filter = None):
    import re
    import cgf_utils
    import sklearn
    df = pd.read_parquet(f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/new_{cgf_measure}.parquet')

    nbins = 10
    climate_var = 'over_30'
    climate_var_bin = 'over_30_bin'
    income_var = 'gdp_pc_pd'
    income_var_bin = 'gdp_pc_pd_bin'
    location_var = 'ihme_loc_id'
    
    over30_bins, binned_df = cgf_utils.group_and_bin_column_definition(df, climate_var, 'location', nbins, bin_strategy = 'custom_daysover', retbins = True) #custom_daysover 0_more_readable
    income_bins, binned_df = cgf_utils.group_and_bin_column_definition(binned_df, income_var, 'household', nbins, retbins = True)

    binned_df['grid_cell'] = binned_df[climate_var_bin].astype(str) +'_'+ binned_df[income_var_bin].astype(str)

    cols_to_scale = []#'over30_avgperyear', 'precip', 'precip_cumavg_5' ,'temp', 'temp_cumavg_5', 'income_per_day']
    scaler = sklearn.preprocessing.MinMaxScaler()
    if cols_to_scale:
        scaled_cols = [f'sc_{col}' for col in cols_to_scale]
        binned_df[scaled_cols] = scaler.fit_transform(binned_df[cols_to_scale])

    model_spec = 'cgf_value ~ (1 | ihme_loc_id) + (1 | grid_cell)'
    
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
    weight_grid['pred_weight'] = model.predict(weight_grid, verify_predictions=False)
    modelled_locations = binned_df[location_var].unique()

    nocountry_grid = binned_df[model_vars[2:] + binned_vars].drop_duplicates()
    nocountry_grid[location_var] = np.nan
    nocountry_grid['pred_weight'] = model.predict(nocountry_grid, verify_predictions=False)

    model.binned_vars = bins
    model.model_vars = model_vars
    model.nocountry_grid = nocountry_grid
    model.weight_grid = weight_grid
    model.available_locations = modelled_locations
    return model



        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = """ Predictorrr. \
            Example usage: python get_climate_vars.py /mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/bmi/bmi_data_outliered.csv /mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/bmi_climate.parquet -lat_col latnum -long_col longnum"""
    )
    #def get_predictions(measure, location_identifier, scenario, year, threshold, sex_id, age_group_id):

    parser.add_argument("measure", help="filepath with lat longs to process", type=str)
    parser.add_argument("location_identifier", help="desired filepath for output", type=str)
    parser.add_argument("scenario", help='input file column with latitude', type=str)
    parser.add_argument("year", help='input file column with longitude', type=int)
    #parser.add_argument("threshold", help='year to extract, optional', type=float)
    parser.add_argument("sex_id", help='year to extract, optional', type=int)
    parser.add_argument("age_group_id", help='year to extract, optional', type=int)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    result_df = get_predictions(args.measure, args.location_identifier, args.scenario, args.year, 30, args.sex_id, args.age_group_id)
    total = result_df.prediction.sum()

    output_df = pd.DataFrame({'total': [total], 'measure':[args.measure], 
        'ihme_loc_id':[ args.location_identifier], 'scenario':[args.scenario], 'year':[args.year], 'threshold':[30], 'sex_id':[args.sex_id], 'age_group_id':[args.age_group_id]})

    OUTPUT_FOLDER = Path('/mnt/share/scratch/users/victorvt/predictions_good')
    OUTPUT_FILEPATH = OUTPUT_FOLDER / f'pred_{args.measure}_{args.location_identifier}_{args.scenario}_{args.year}_{args.sex_id}_{args.age_group_id}.csv'
    output_df.to_csv(OUTPUT_FILEPATH, index=False)
