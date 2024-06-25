import pandas as pd
from spatial_temp_cgf import paths


def load_binned_income_distribution_proportions(fhs_location_id = None, model = None, measure = None, year_id = None):
    if model is not None:
        model_identifier = model.model_identifier
        ldipc_distribution_bin_proportions_filepath = paths.MODELS / model_identifier / f'ldipc_bin_proportions.parquet'
    elif measure is not None:
        filepath = paths.LDIPC_DISTRIBUTION_BIN_PROPORTIONS_DEFAULT_FILEPATH_FORMAT.format(measure = measure)
        load_filters = []
        if fhs_location_id: load_filters.append(('fhs_location_id', '==', fhs_location_id))
        if year_id : load_filters.append(('year_id', '==', year_id))
        return pd.read_parquet(filepath, filters = load_filters)
    else:
        raise ValueError("Either model or measure must be specified")

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