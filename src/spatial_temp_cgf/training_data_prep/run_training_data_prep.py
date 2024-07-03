from pathlib import Path

import click
import pandas as pd
import numpy as np
import xarray as xr
import multiprocessing as mp
from functools import partial
import rioxarray
import xarray as xr
import spatial_temp_cgf.cli_options as clio
from spatial_temp_cgf.data import ClimateMalnutritionData, DEFAULT_ROOT, get_run_directory
from spatial_temp_cgf import paths

SURVEY_DATA_ROOT = Path(
    '/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data'
)
LSAE_CGF_GEOSPATIAL_ROOT = Path('/mnt/share/limited_use/LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/pre_collapse/')
CGF_FILEPATH_LSAE = Path('/mnt/share/limited_use/LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/pre_collapse/cgf_lbw_2020_06_15.csv')

# TODO: Carve out the piece of the R script that makes this file, move output someplace better
WEALTH_FILEPATH = Path('/mnt/share/scratch/users/victorvt/cgfwealth_spatial/dhs_wealth_uncollapsed_again.parquet')
LDIPC_FILEPATH = Path('/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/national_ldipc_estimates.csv')

SURVEY_DATA_PATHS = {
    "bmi": {'gbd': SURVEY_DATA_ROOT / "bmi" / "bmi_data_outliered_wealth_rex.csv"},
    "cgf": {'gbd' : SURVEY_DATA_ROOT / "wasting_stunting" / "wasting_stunting_outliered_wealth_rex.csv",
        'lsae': CGF_FILEPATH_LSAE},
    "wealth": {'lsae': WEALTH_FILEPATH},
}

DATA_SOURCE_TYPE = {'stunting':'cgf', 'wasting':'cgf', 'low_adult_bmi':'bmi'}
MEASURES_IN_SOURCE = {'cgf': ['stunting', 'wasting'], 'bmi': ['low_adult_bmi']}

############################
# Wasting/Stunting columns #
############################

ihme_meta_columns = [
    'nid',
    'file_path',    
]
survey_meta_columns = [
    'survey_name',
    'survey_module',
    'year_start',
    'year_end',
]
sample_meta_columns = [
    'psu',    
    'psu_id',
    'pweight',
    'pweight_admin_1'
    'urban',
    'strata'    
    'strata_id',
    'hh_id',
    'hhweight',
    'line_id',
    'int_year', 
    'int_month',
    'int_day',
]
location_meta_columns = [
    'ihme_loc_id', 
    'location_name',
    'super_region_name',
    'geospatial_id',
    'admin_1',
    'admin_1_id',
    'admin_1_mapped',
    'admin_1_urban_id',
    'admin_1_urban_mapped',        
    'admin_2',
    'admin_2_id',
    'admin_2_mapped',
    'admin_3',
    'admin_4',
    'latnum',
    'longnum',
]
individual_meta_columns = [
    'individual_id',
    'sex_id',
    'age_year',
    'age_month', 
    'age_day',
    'age_categorical'
]
value_columns = [
    'metab_height',
    'metab_height_unit',
    'metab_weight',
    'metab_weight_unit',
    'bmi',
    'overweight',
    'obese',
    'pregnant',
    'birth_weight',
    'birth_weight_unit',
    'birth_order',
    'mother_weight',
    'mother_height',
    'mother_age_month',
    'maternal_ed_yrs',
    'paternal_ed_yrs',
    'wealth_factor',
    'wealth_index_dhs',
    'suspicious.heights',
    'suspicious.weights',
    'HAZ', 'HAZ_b1', 'HAZ_b2', 'HAZ_b3', 'HAZ_b4',
    'WAZ', 'WAZ_b1', 'WAZ_b2', 'WAZ_b3', 'WAZ_b4',
    'WHZ', 'WHZ_b1', 'WHZ_b2', 'WHZ_b3', 'WHZ_b4'
]


def examine_survey_schema(df, columns):
    print('Records:', len(df))
    print()
    
    template = "{:<20} {:>10} {:>10} {:>10}"
    header = template.format('COLUMN', 'N_UNIQUE', 'N_NULL', "DTYPE")
    
    print(header)
    print('='*len(header))
    for col in columns:
        unique = df[col].nunique()
        nulls = df[col].isnull().sum()
        dtype = str(df[col].dtype)
        print(template.format(col, unique, nulls, dtype))

COLUMN_NAME_TRANSLATOR = {
    'country': 'ihme_loc_id',
    'year_start': 'year_start',
    'end_year': 'year_end',
    'psu_id': 'psu',
    'strata_id': 'strata',
    'sex': 'sex_id',
    'age_mo': 'age_month',
    'stunting_mod_b': 'stunting',
    'wasting_mod_b': 'wasting',
    'underweight_mod_b': 'underweight',
    'HAZ_b2': 'stunting', 
    'WHZ_b2': 'wasting', 
    'WAZ_b2': 'underweight',
    'latnum': 'lat',
    'longnum': 'long',
    'latitude': 'lat',
    'longitude': 'long',
}

def run_training_data_prep_main(
    output_root: str | Path,
    data_source_type: str,
) -> None:
    
    if data_source_type != 'cgf':
        raise NotImplementedError(f"Data source {data_source_type} not implemented yet.")

    survey_data_path = SURVEY_DATA_PATHS[data_source_type]
    print(f"Running training data prep for {data_source_type}...")
    #print(f"Survey data path: {survey_data_path}")

    #df = pd.read_csv(survey_data_path)
    print("Processing gbd extraction survey data...")
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)

    lsae_cgf_data_raw = pd.read_csv(survey_data_path['lsae'], dtype={'hh_id': str, 'year_start': int, 'int_year':int, 'year_end': int})
    new_cgf_data_raw = pd.read_csv(survey_data_path['gbd'], dtype={'hh_id': str, 'year_start': int, 'int_year':int, 'year_end': int})

    # Translator to harmonize column names between both sets
    #gbd_columns = ['nid', 'ihme_loc_id', 'year_start', 'year_end', 'int_year', 'geospatial_id', 'psu_id', 'strata_id', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month','HAZ_b2', 'WHZ_b2', 'WAZ_b2', 'latnum', 'longnum']
    #lsae_columns = ['nid', 'country', 'year_start', 'end_year','int_year', 'geospatial_id', 'psu', 'strata', 'hh_id', 'sex', 'age_year', 'age_mo', 'int_year', 'int_month', 'stunting_mod_b', 'wasting_mod_b', 'underweight_mod_b']
    #desired_columns = ['nid', 'ihme_loc_id', 'year_start', 'year_end', 'int_year', 'geospatial_id', 'psu', 'strata', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month', 'stunting', 'wasting', 'underweight', 'lat', 'long']
    #gbd_column_translator = dict(zip(gbd_columns, desired_columns))
    #lsae_column_translator = dict(zip(lsae_columns, desired_columns))

    # subset to columns of interest
    gbd_cgf_data = new_cgf_data_raw[['nid', 'ihme_loc_id', 'year_start', 'year_end', 'geospatial_id', 'psu_id', 'strata_id', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month','HAZ_b2', 'WHZ_b2', 'WAZ_b2', 'wealth_index_dhs','latnum', 'longnum']]
    gbd_cgf_data = gbd_cgf_data.rename(columns=COLUMN_NAME_TRANSLATOR)

    # Take away NIDs without wealth information, to see later if we can add it from second source
    print(len(gbd_cgf_data))
    nids_without_wealth = gbd_cgf_data[gbd_cgf_data.wealth_index_dhs.isna()].nid.unique()
    to_add_wealth = gbd_cgf_data[gbd_cgf_data.nid.isin(nids_without_wealth)]
    gbd_cgf_data = gbd_cgf_data[~gbd_cgf_data.nid.isin(nids_without_wealth)]
    print(len(gbd_cgf_data))

    # We get wealth data to be merged with GBD CGF data and merge
    print("Processing wealth data...")
    gbd_data_wealth_distribution = gbd_cgf_data.groupby(['nid', 'ihme_loc_id', 'year_start', 'psu', 'hh_id']).agg(wealth_index_dhs = ('wealth_index_dhs', 'first'), check = ('wealth_index_dhs', 'nunique')).reset_index()
    assert((gbd_data_wealth_distribution.check == 1).all())
    gbd_data_wealth_distribution = gbd_data_wealth_distribution.merge(loc_meta, how='left', on='ihme_loc_id')

    gbd_data_wealth_distribution = get_ldipc_from_asset_score(gbd_data_wealth_distribution, asset_score_col = 'wealth_index_dhs')
    gbd_data_wealth_distribution = gbd_data_wealth_distribution[['nid', 'ihme_loc_id', 'location_id', 'year_start', 'psu', 'hh_id', 'ldipc']]
    check = gbd_cgf_data.merge(gbd_data_wealth_distribution, how='left', on=['nid', 'ihme_loc_id', 'year_start', 'psu', 'hh_id'])
    assert(len(gbd_cgf_data) == len(check))
    gbd_cgf_data = check

    # Now the old LSAE data
    # We get wealth data to be merged with CGF data and the lsae cgf data with standard column names
    wealth_df = get_ldipc_from_asset_score(get_wealth_dataset(), asset_score_col = 'wealth_index_dhs')
    lsae_cgf_data = lsae_cgf_data_raw[['nid', 'country', 'year_start', 'end_year',  'geospatial_id', 'psu', 'strata', 'hh_id', 'sex', 'age_year', 'age_mo', 'int_year', 'int_month', 'stunting_mod_b', 'wasting_mod_b', 'underweight_mod_b']]
    lsae_cgf_data = lsae_cgf_data.rename(columns=COLUMN_NAME_TRANSLATOR)

    print("Processing LSAE data...")
    # Take away bad NIDs, without household information
    print(len(lsae_cgf_data))
    no_hhid_nids = lsae_cgf_data.groupby('nid').filter(lambda x: x['hh_id'].isna().all()).nid.unique()
    lsae_cgf_data = lsae_cgf_data[~lsae_cgf_data.nid.isin(no_hhid_nids)]

    # Get NIDs in common between wealth and LSAE extraction CGF data and subset to that
    common_nids = set(lsae_cgf_data['nid'].unique()).intersection(set(wealth_df['nid'].unique()))
    wealth_lsae_df = wealth_df[wealth_df['nid'].isin(common_nids)][['nid', 'ihme_loc_id', 'location_id', 'year_start', 'psu', 'hh_id', 'geospatial_id', 'lat', 'long', 'wealth_index_dhs', 'ldipc']]

    # Try to make household id usable to merge on 
    # For some reason in some extractions hh_id is a string with household id and psu, in others it's just the household id
    lsae_cgf_data = lsae_cgf_data[lsae_cgf_data['nid'].isin(common_nids)].copy()
    lsae_cgf_data = lsae_cgf_data.rename(columns = {'latitude':'lat', 'longitude':'long', 'hh_id':'old_hh_id'})
    wealth_lsae_df = wealth_lsae_df.rename(columns = {'hh_id':'old_hh_id'})
    lsae_cgf_data['hh_id'] = lsae_cgf_data['old_hh_id'].str.split(r'[_ ]').str[-1]
    wealth_lsae_df['hh_id'] = wealth_lsae_df['old_hh_id'].str.split(r'[_ ]').str[-1]
    lsae_cgf_data['psu'] = lsae_cgf_data['psu'].astype(int)
    print(len(lsae_cgf_data))

    #Some NIDs need extra cleaning so that hh_id can be merged.
    # Take those out and merge LSAE CGF data with wealth
    print(len(lsae_cgf_data))

    maybe_fixable_df = lsae_cgf_data[lsae_cgf_data['nid'].isin([157057, 286780, 341838])].copy()
    lsae_cgf_data = lsae_cgf_data[~lsae_cgf_data['nid'].isin([157057, 286780, 341838])]
    merge_cols = ['nid', 'ihme_loc_id', 'hh_id', 'psu', 'year_start']

    lsae_merged = lsae_cgf_data.drop(columns=['old_hh_id']).merge(wealth_lsae_df[['ihme_loc_id', 'location_id', 'nid', 'psu', 'hh_id', 'year_start', 'wealth_index_dhs', 'ldipc', 'lat', 'long']], on=merge_cols, how='left')
    #print(len(lsae_cgf_data_raw))
    print(len(lsae_cgf_data))
    print(len(lsae_merged))

    print(len(maybe_fixable_df))
    maybe_fixable_df.loc[maybe_fixable_df.nid == 157057, 'hh_id'] = maybe_fixable_df.loc[maybe_fixable_df.nid == 157057].psu.astype(int).astype(str).str.zfill(4) + maybe_fixable_df.loc[maybe_fixable_df.nid == 157057].old_hh_id.astype(int).astype(str).str.zfill(3)
    maybe_fixable_df.loc[maybe_fixable_df.nid == 286780, 'hh_id'] = maybe_fixable_df.loc[maybe_fixable_df.nid == 286780].psu.astype(int).astype(str).str.zfill(4) + maybe_fixable_df.loc[maybe_fixable_df.nid == 286780].old_hh_id.astype(int).astype(str).str.zfill(4)
    maybe_fixable_df.loc[maybe_fixable_df.nid == 341838, 'hh_id'] = maybe_fixable_df.loc[maybe_fixable_df.nid == 341838].psu.astype(int).astype(str).str.zfill(4) + maybe_fixable_df.loc[maybe_fixable_df.nid == 341838].old_hh_id.astype(int).astype(str).str.zfill(4)
    merged_fixed_nids_df = maybe_fixable_df.drop(columns=['old_hh_id']).merge(wealth_lsae_df[['ihme_loc_id', 'location_id', 'nid', 'psu', 'hh_id', 'year_start', 'ldipc', 'lat', 'long']], on=merge_cols, how='left')
    merged_fixed_nids_df = merged_fixed_nids_df.dropna(subset=['location_id', 'lat', 'long', 'ldipc'], how='any')
    merged_fixed_nids_df = merged_fixed_nids_df.dropna(subset=['stunting', 'wasting', 'underweight'], how='all')
    print(len(merged_fixed_nids_df))

    #Only interested in rows have have either stunting, wasting or underweight information, with both wealth and location
    lsae_merged = lsae_merged.dropna(subset=['stunting', 'wasting', 'underweight'], how='all')
    lsae_merged = lsae_merged.dropna(subset=['ldipc', 'lat', 'long'], how='any')
    lsae_merged.loc[lsae_merged.sex_id == 0, 'sex_id'] = 2
    print(len(lsae_merged))

    # Drop rows from GBD dataset with missing location information
    #gbd_cgf_data = gbd_cgf_data.drop(columns = ['wealth_index_dhs'], errors='ignore')
    gbd_cgf_data = gbd_cgf_data.dropna(subset=['lat', 'long'], how='any')  

    # For the GBD NIDs that need wealth information, attempt to get it
    gbd_nids_need_wealth = [x for x in nids_without_wealth if x in wealth_df.nid.unique() and x not in lsae_cgf_data.nid.unique()]
    extra_nids = new_cgf_data_raw.query("nid in @gbd_nids_need_wealth")[['nid', 'ihme_loc_id', 'year_start', 'year_end',  'geospatial_id', 'psu_id', 'strata_id', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month','HAZ_b2', 'WAZ_b2', 'WHZ_b2','latnum', 'longnum', 'wealth_index_dhs']]
    extra_nids = extra_nids.rename(columns=COLUMN_NAME_TRANSLATOR)
    extra_nids['hh_id'] = extra_nids['hh_id'].str.split(r'[_ ]').str[-1]
    extra_nids = extra_nids.merge(wealth_df[['ihme_loc_id', 'location_id', 'nid', 'psu', 'hh_id', 'year_start', 'ldipc']], on=['nid', 'ihme_loc_id', 'year_start', 'psu', 'hh_id'], how='left')
    extra_nids = extra_nids.dropna(subset=['ldipc'])

    # Bring the two datasets (LSAE and GBD) together, giving preference to the new extractions
    new_extraction_nids = gbd_cgf_data.nid.unique()
    lsae_only = lsae_merged.loc[~lsae_merged.nid.isin(new_extraction_nids)]
    #assert(set(lsae_only.columns) == set(gbd_cgf_data.columns))
    #temporary thing TODO take this out
    if not (set(lsae_only.columns) == set(gbd_cgf_data.columns)):
        print(lsae_only.columns)
        print(gbd_cgf_data.columns)

    # Merge the two datasets and the NIDs that needed wealth information
    cgf_consolidated = pd.concat([gbd_cgf_data, lsae_only, extra_nids], ignore_index=True).reset_index(drop=True)

    cgf_consolidated = cgf_consolidated.drop(columns = ['strata', 'geospatial_id'])
    cgf_consolidated['ldi_pc_pd'] = cgf_consolidated['ldipc'] / 365
    
    # Assign age group
    cgf_consolidated = assign_age_group(cgf_consolidated)
    cgf_consolidated = cgf_consolidated.dropna(subset=['age_group_id'])

    #Merge with climate data
    print("Processing climate data...")
    climate_df = get_climate_vars_for_dataframe(cgf_consolidated)
    #climate_df = pd.concat([pd.read_parquet(f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/cgf_climate_years/cgf_{year}.parquet') for year in range(1979, 2017)]).rename(columns={'year':'year_start'})
    cgf_consolidated = cgf_consolidated.merge(climate_df, on=['int_year', 'lat', 'long'], how='left')

    print("Adding elevation data...")
    cgf_consolidated = get_elevation_for_dataframe(cgf_consolidated)

    # Write to output
    for measure in MEASURES_IN_SOURCE[data_source_type]:
        measure_df = cgf_consolidated[cgf_consolidated[measure].notna()].copy()
        measure_df['cgf_measure'] = measure
        measure_df['cgf_value'] = measure_df[measure]
        measure_root = Path(output_root) / measure
        cm_data = ClimateMalnutritionData(measure_root)
        version = cm_data.new_training_version()
        cm_data.save_training_data(measure_df, version)
    print("Done!")

def get_climate_vars_for_year(year_df:pd.DataFrame, climate_variables, 
        lat_col = 'lat', long_col = 'long', year_col = 'int_year'):
        HISTORICAL_CLIMATE_ROOT = Path('/mnt/share/erf/climate_downscale/results/annual/historical/')

        assert(year_df[year_col].nunique() == 1)
        yr = year_df[year_col].iloc[0]

        temp_df = year_df.copy()
        lats = xr.DataArray(temp_df[lat_col], dims='point')
        lons = xr.DataArray(temp_df[long_col], dims='point')
        years = xr.DataArray(temp_df[year_col], dims='point')
        for climate_variable in climate_variables:
            #temp_df = year_df.loc[year_df.year_start == yr].copy()
            climate_ds = xr.open_dataset(HISTORICAL_CLIMATE_ROOT / climate_variable /f'{yr}.nc')
            temp_df[climate_variable] = climate_ds['value'].sel(
                latitude=lats,
                longitude=lons,
                year=years,
                method='nearest'
            ).values
            #assert temp_df[climate_variable].notna().all()
        return temp_df

def get_climate_vars_for_dataframe(df:pd.DataFrame, lat_col = 'lat', long_col = 'long', year_col = 'int_year'):    

    HISTORICAL_CLIMATE_ROOT = Path('/mnt/share/erf/climate_downscale/results/annual/historical/')
    var_names = [child.name for child in HISTORICAL_CLIMATE_ROOT.iterdir() if child.is_dir()]
    temp_dfs = []

    unique_coords = df[[lat_col, long_col, year_col]].drop_duplicates()

    df_splits = [year_df for _, year_df in  unique_coords.groupby(year_col)]
    p = mp.Pool(processes=5)
    results_df = pd.concat(p.map(partial(get_climate_vars_for_year,climate_variables=var_names), df_splits))
    #results_df = pd.concat(p.map(find_peaks_and_troughs_in_df, df_splits))
    p.close()
    p.join()
    return results_df

def get_elevation_for_dataframe(df:pd.DataFrame, lat_col = 'lat', long_col = 'long'):    
    ELEVATION_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/srtm_elevation.tif')

    unique_coords = df[[lat_col, long_col]].drop_duplicates()
    elevation_ds = rioxarray.open_rasterio(ELEVATION_FILEPATH)
    unique_coords['elevation'] = elevation_ds.sel(x=xr.DataArray(unique_coords.long, dims='z'), 
        y=xr.DataArray(unique_coords.lat, dims='z'), 
        band = 1, method='nearest').values
        
    unique_coords['elevation'] = unique_coords['elevation'].astype(int)
    unique_coords.loc[unique_coords.elevation == -32768, 'elevation'] = 0

    results_df = df.merge(unique_coords, on=[lat_col, long_col], how='left')
    assert results_df['elevation'].notna().all()
    return results_df



def get_ldipc_from_asset_score(asset_df:pd.DataFrame,  asset_score_col= 'wealth_index_dhs', year_df_col = 'year_start'):
#    INCOME_FILEPATH = '/mnt/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_fixGUY/gdppc_estimates.csv'

    income_raw = pd.read_csv(LDIPC_FILEPATH)
    income_raw = income_raw[['population_percentile', 'ldipc', 'location_id', 'year_id']]
    #income_raw['ihme_loc_id'] = income_raw.location_id.astype(int)
    income_raw['year_id'] = income_raw.year_id.astype(int)

    wdf = asset_df.copy()

    # We get which asset score is what percentile of that asset score for each nid
    get_percentile = lambda x: x.rank() / len(x) 

    assert('percentile' not in wdf.columns)
    assert('nid' in wdf.columns)
    assert('location_id' in wdf.columns)
    wdf['percentile'] = wdf.groupby(['nid'], group_keys=False)[asset_score_col].apply(get_percentile)

    wdf = wdf.sort_values(['percentile'])
    income_raw = income_raw.sort_values(['population_percentile'])

    income_interp = income_raw.merge(
        wdf[['location_id', year_df_col, 'percentile']].drop_duplicates().rename(columns={year_df_col:'year_id', 'percentile':'population_percentile'}),
        on=['location_id', 'year_id', 'population_percentile'],
        how='outer')
    # We then interpolate to get the percentiles that aren't included in the income dataset
    income_interp = income_interp.sort_values(['location_id', 'year_id', 'population_percentile'])
    income_interp = income_interp.set_index(['location_id', 'year_id', 'population_percentile'])

    income_interp['ldipc'] = income_interp.groupby(level=[0, 1], group_keys=False)['ldipc'].apply(lambda x: x.interpolate(method='linear', limit_area='inside'))
    income_interp = income_interp.reset_index()

    rolling_mean_window_f = lambda x: x.rolling(window=5).mean()
    income_interp['ldipc_last5'] = income_interp.groupby(['location_id', 'population_percentile'])['ldipc'].transform(rolling_mean_window_f)

    wdf = wdf.merge(income_interp, how='left', left_on=['location_id', 'year_start', 'percentile'],
        right_on = ['location_id', 'year_id', 'population_percentile'])

    # Only NAs should be because of newer surveys for which we don't have income distribution yet
    #assert((wdf.loc[wdf.income_per_day.isna()].year_id > 2020).all())
    assert(len(wdf.loc[wdf.ldipc.isna()]) == 0)
    assert(len(wdf) == len(asset_df))
    return wdf

def get_wealth_dataset():
    loc_meta = pd.read_parquet(paths.FHS_LOCATION_METADATA_FILEPATH)
    wealth_raw = pd.read_parquet(SURVEY_DATA_PATHS['wealth']['lsae'])
    wealth_df = wealth_raw[wealth_raw['point'] == 1]
    wealth_df = wealth_df.rename(columns = {'wealth_score':'wealth_index_dhs', 'iso3':'ihme_loc_id'})
    wealth_df = wealth_df.rename(columns = COLUMN_NAME_TRANSLATOR)
    wealth_df = wealth_df[['nid', 'ihme_loc_id', 'year_start', 'year_end', 'geospatial_id', 'psu', 'strata', 'hh_id', 'wealth_index_dhs', 'lat', 'long']]
    wealth_df = wealth_df[['ihme_loc_id', 'nid', 'psu', 'hh_id', 'lat', 'long', 'year_start', 'wealth_index_dhs', 'geospatial_id']].drop_duplicates()

    # In the wealth team's dataset, sometimes there are multiple asset scores for a given household id.
    # Take away those NIDs

    #Make sure that by nid, psu and hh_id they all have the same lat and long
    grouped = wealth_df.groupby(['nid', 'psu', 'hh_id'])
    assert((grouped['lat'].nunique().eq(1) & grouped['long'].nunique().eq(1)).all())
    assert(grouped.size().max() == 1)
    # Sometimes an nid has more than a year
    assert(wealth_df.groupby(['nid', 'hh_id', 'year_start', 'psu', 'geospatial_id']).size().sort_values().max() == 1)


    bad_wealth = wealth_df.groupby(['nid', 'hh_id', 'year_start', 'psu',]).size()
    bad_nid_wealth = list(bad_wealth[bad_wealth.gt(1)].reset_index().nid.unique())
    bad_nid_wealth = bad_nid_wealth + [20315, 20301, 20537]
    wealth_df = wealth_df[~wealth_df.nid.isin(bad_nid_wealth)]
    # Make sure that by nid, psu and hh_id they all have the same lat and long
    #grouped = wealth_df.groupby(['nid', 'psu', 'hh_id'])
    #assert((grouped['lat'].nunique().lt(2) & grouped['long'].nunique().lt(2)).all())
    # Sometimes an nid has more than a year
    assert(wealth_df.groupby(['nid', 'hh_id', 'year_start', 'psu']).size().sort_values().max() == 1)
    wealth_df = wealth_df.merge(loc_meta[['location_id', 'ihme_loc_id']], on='ihme_loc_id', how='left')
    assert(wealth_df.location_id.notna().all())
    wealth_df['year_start'] = wealth_df['year_start'].astype(int)
    wealth_df['nid'] = wealth_df['nid'].astype(int)
    return wealth_df

def assign_age_group(df:pd.DataFrame):
    age_group_spans = pd.read_parquet(paths.AGE_SPANS_FILEPATH)
    age_group_spans = age_group_spans.query('age_group_id in [388, 389, 238, 34]')
    df['age_group_id'] = np.nan
    for i, row in age_group_spans.iterrows():
        df.loc[(df.age_year >= row.age_group_years_start.round(5)) & (df.age_year < row.age_group_years_end), 'age_group_id'] = row.age_group_id
    for i, row in age_group_spans.iterrows():
        df.loc[(df.age_group_id.isna()) & (df.age_year == 0) & (df.age_month/12 >= row.age_group_years_start) & (df.age_month/12 < row.age_group_years_end), 'age_group_id'] = row.age_group_id
    #Fix for some kids that had age_year = 0.076 but month == 1
    df.loc[(df.age_group_id.isna()) & (df.age_year > 0.076) & (df.age_month == 1), 'age_group_id'] = 388

    #TODO: Delete when (or if) we get population at the more granular age group level
    df.loc[df.age_group_id == 388, 'age_group_id'] = 4
    df.loc[df.age_group_id == 389, 'age_group_id'] = 4
    df.loc[df.age_group_id == 238, 'age_group_id'] = 5
    df.loc[df.age_group_id == 34, 'age_group_id'] = 5
    return df


@click.command()
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_source_type(allow_all=False)
def run_training_data_prep(output_root: str, source_type: str):
    """Run training data prep."""
    print(f"Running training data prep for {source_type}...")
    #for src in source_type:
    run_training_data_prep_main(output_root, source_type)
