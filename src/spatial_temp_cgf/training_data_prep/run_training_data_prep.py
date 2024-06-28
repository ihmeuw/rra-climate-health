from pathlib import Path

import click
import pandas as pd

from db_queries import get_location_metadata

import spatial_temp_cgf.cli_options as clio
from spatial_temp_cgf.data import ClimateMalnutritionData, DEFAULT_ROOT, get_run_directory


SURVEY_DATA_ROOT = Path(
    '/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data'
)

# TODO: Put these in a better format
CGF_FILEPATH_LSAE = Path('/mnt/share/limited_use/LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/pre_collapse/cgf_lbw_2020_06_15.csv')
WEALTH_FILEPATH = Path('/mnt/share/scratch/users/victorvt/cgfwealth_spatial/dhs_wealth_uncollapsed_again.parquet')
LDIPC_FILEPATH = Path('/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/national_ldipc_estimates.csv')

SURVEY_DATA_PATHS = {
    "bmi": {'gbd': SURVEY_DATA_ROOT / "bmi" / "bmi_data_outliered_wealth_rex.csv"},
    "wasting": {'gbd' : SURVEY_DATA_ROOT / "wasting_stunting" / "wasting_stunting_outliered_wealth_rex.csv",
        'lsae': CGF_FILEPATH_LSAE},
    "stunting": {'gbd' : SURVEY_DATA_ROOT / "wasting_stunting" / "wasting_stunting_outliered_wealth_rex.csv",
        'lsae': CGF_FILEPATH_LSAE},
    "wealth": {'lsae': WEALTH_FILEPATH},
}



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


def run_training_data_prep_main(
    output_root: str | Path,
    measure: str,
) -> None:
    measure_root = Path(output_root) / measure
    cm_data = ClimateMalnutritionData(measure_root)
    version = cm_data.new_training_version()

    if measure not in ['stunting', 'wasting']:
        raise NotImplementedError(f"Measure {measure} not implemented.")

    survey_data_path = SURVEY_DATA_PATHS[measure]
    print(f"Running training data prep for {measure}...")
    print(f"Survey data path: {survey_data_path}")

    df = pd.read_csv(survey_data_path)

    loc_meta = get_location_metadata(39, release_id = 9)[['location_id', 'ihme_loc_id']]

    lsae_cgf_data_raw = pd.read_csv(survey_data_path['lsae'], dtype={'hh_id': str, 'year_start': int, 'year_end': int})
    new_cgf_data_raw = pd.read_csv(survey_data_path['gbd'], dtype={'hh_id': str, 'year_start': int, 'year_end': int})

    # Translator to harmonize column names between both sets
    gbd_columns = ['nid', 'ihme_loc_id', 'year_start', 'year_end', 'geospatial_id', 'psu_id', 'strata_id', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month','HAZ_b2', 'WHZ_b2', 'WAZ_b2', 'latnum', 'longnum']
    lsae_columns = ['nid', 'country', 'year_start', 'end_year', 'geospatial_id', 'psu', 'strata', 'hh_id', 'sex', 'age_year', 'age_mo', 'int_year', 'int_month', 'stunting_mod_b', 'wasting_mod_b', 'underweight_mod_b']
    desired_columns = ['nid', 'ihme_loc_id', 'year_start', 'year_end', 'geospatial_id', 'psu', 'strata', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month', 'stunting', 'wasting', 'underweight', 'lat', 'long']
    gbd_column_translator = dict(zip(gbd_columns, desired_columns))
    lsae_column_translator = dict(zip(lsae_columns, desired_columns))

    # subset to columns of interest
    gbd_cgf_data = new_cgf_data_raw[['nid', 'ihme_loc_id', 'year_start', 'year_end', 'geospatial_id', 'psu_id', 'strata_id', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month','HAZ_b2', 'WHZ_b2', 'WAZ_b2', 'wealth_index_dhs','latnum', 'longnum']]
    gbd_cgf_data = gbd_cgf_data.rename(columns=gbd_column_translator)

    # Take away NIDs without wealth information, to see later if we can add it from second source
    print(len(gbd_cgf_data))
    nids_without_wealth = gbd_cgf_data[gbd_cgf_data.wealth_index_dhs.isna()].nid.unique()
    to_add_wealth = gbd_cgf_data[gbd_cgf_data.nid.isin(nids_without_wealth)]
    gbd_cgf_data = gbd_cgf_data[~gbd_cgf_data.nid.isin(nids_without_wealth)]
    print(len(gbd_cgf_data))

    # We get wealth data to be merged with GBD CGF data and merge
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
    lsae_cgf_data = lsae_cgf_data_raw[['nid', 'country', 'year_start', 'end_year', 'geospatial_id', 'psu', 'strata', 'hh_id', 'sex', 'age_year', 'age_mo', 'int_year', 'int_month', 'stunting_mod_b', 'wasting_mod_b', 'underweight_mod_b']]
    lsae_cgf_data = lsae_cgf_data.rename(columns=lsae_column_translator)

    # Take away bad NIDs, without household information
    print(len(lsae_cgf_data))
    no_hhid_nids = lsae_cgf_data.groupby('nid').filter(lambda x: x['hh_id'].isna().all()).nid.unique()
    lsae_cgf_data = lsae_cgf_data[~lsae_cgf_data.nid.isin(no_hhid_nids)]

    # Get NIDs in common between wealth and LSAE extraction CGF data and subset to that
    common_nids = set(lsae_cgf_data['nid'].unique()).intersection(set(wealth_df['nid'].unique()))
    wealth_lsae_df = wealth_df[wealth_df['nid'].isin(common_nids)][['nid', 'ihme_loc_id', 'location_id', 'year_start', 'psu', 'hh_id', 'geospatial_id', 'lat', 'long', 'ldipc']]

    # Try to make household id usable to merge on 
    # For some reason in some extractions hh_id is a string with household id and psu, in others it's just the household id
    # TODO: why is it so different between the datasets if it's ostensibly the same source?
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

    lsae_merged = lsae_cgf_data.drop(columns=['old_hh_id']).merge(wealth_lsae_df[['ihme_loc_id', 'location_id', 'nid', 'psu', 'hh_id', 'year_start', 'ldipc', 'lat', 'long']], on=merge_cols, how='left')
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
    extra_nids = new_cgf_data_raw.query("nid in @gbd_nids_need_wealth")[['nid', 'ihme_loc_id', 'year_start', 'year_end', 'geospatial_id', 'psu_id', 'strata_id', 'hh_id', 'sex_id', 'age_year', 'age_month', 'int_year', 'int_month','HAZ_b2', 'WAZ_b2', 'WHZ_b2','latnum', 'longnum']]
    extra_nids = extra_nids.rename(columns=gbd_column_translator)
    extra_nids['hh_id'] = extra_nids['hh_id'].str.split(r'[_ ]').str[-1]
    extra_nids = extra_nids.merge(wealth_df[['ihme_loc_id', 'location_id', 'nid', 'psu', 'hh_id', 'year_start', 'ldipc']], on=['nid', 'ihme_loc_id', 'year_start', 'psu', 'hh_id'], how='left')
    extra_nids = extra_nids.dropna(subset=['ldipc'])

    # Bring the two datasets (LSAE and GBD) together, giving preference to the new extractions
    new_extraction_nids = gbd_cgf_data.nid.unique()
    lsae_only = lsae_merged.loc[~lsae_merged.nid.isin(new_extraction_nids)]
    assert(set(lsae_only.columns) == set(gbd_cgf_data.columns))

    # Merge the two datasets
    cgf_consolidated = pd.concat([gbd_cgf_data, lsae_only, extra_nids], ignore_index=True)

    cgf_consolidated = cgf_consolidated.drop(columns = ['strata', 'geospatial_id'])
    cgf_consolidated['ldi_pc_pd'] = cgf_consolidated['ldipc'] / 365
    
    # Assign age group
    cgf_consolidated = assign_age_group(cgf_consolidated)
    cgf_consolidated = cgf_consolidated.dropna(subset=['age_group_id'])

    #Merge with climate data
    climate_df = pd.concat([pd.read_parquet(f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/cgf_climate_years/cgf_{year}.parquet') for year in range(1979, 2017)]).rename(columns={'year':'year_start'})
    cgf_consolidated = cgf_consolidated.merge(climate_df, on=['year_start', 'lat', 'long'], how='left').query('year_start < 2017')

    # OUT_ROOT = Path("/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi")

    # cgf_dfs = dict()
    # geo_dfs = dict()
    # cgf_measures = ['stunting', 'wasting']

    # for measure in cgf_measures:
    #     cgf_dfs[measure] = cgf_consolidated[cgf_consolidated[measure].notna()].copy()
    #     cgf_dfs[measure]['cgf_measure'] = measure
    #     cgf_dfs[measure]['cgf_value'] = cgf_dfs[measure][measure]
    #     cgf_dfs[measure].to_parquet(OUT_ROOT / f"new_{measure}.parquet")
    # cgf_consolidated.to_parquet(OUT_ROOT / f"cgf_all_measures.parquet")

    # Write to output
    #cm_data.save_training_data(cgf_consolidated, version)

    print("Done!")

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
    # Now the old LSAE data
    #loc_meta = get_location_metadata(release_id = 9, location_set_id = 35)

    wealth_raw = pd.read_parquet(WEALTH_FILEPATH)
    wealth_df = wealth_raw[wealth_raw['point'] == 1]
    wealth_df = wealth_df.rename(columns = {'wealth_score':'wealth_index_dhs', 'iso3':'ihme_loc_id'})
    wealth_df = wealth_df.rename(columns = lsae_column_translator)
    wealth_df = wealth_df.rename(columns = gbd_column_translator)
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
    from db_queries import get_age_spans
    age_group_spans = get_age_spans().query('age_group_id in [388, 389, 238, 34]')
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
@clio.with_measure(allow_all=True)
def run_training_data_prep(output_root: str, measure: list[str]):
    """Run training data prep."""

    for m in measure:
        run_training_data_prep_main(output_root, m)
