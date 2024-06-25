import argparse
import logging
import sys
from ctypes import cdll
from pathlib import Path

import numpy as np
import pandas as pd
import xarray

from spatial_temp_cgf import paths

cdll.LoadLibrary(paths.LIBSTDCPP_PATH)

CLIMATE_ROOT = paths.CHELSA_HISTORICAL_ROOT
OVER30_FSTR = "days_above_{threshold}_{year}.nc"
PRECIP_FSTR = "precipitation_{year}.nc"
TEMPERATURE_FSTR = "temperature_{year}.nc"

def extract_climate_data_for_year(year:int, lats:pd.Series, longs:pd.Series, threshold = 30)-> pd.DataFrame:
    if isinstance(threshold, int) or threshold.is_integer():
        threshold_str = str(threshold)
    else:
        threshold_str = str(threshold).replace(".", "-")
    try:
        over30_x = xarray.open_dataset(CLIMATE_ROOT / OVER30_FSTR.format(year= year, threshold=threshold_str))
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Year {year} not available. FILE {CLIMATE_ROOT / OVER30_FSTR}') from e
    
    try:
        temp_x = xarray.open_dataset(CLIMATE_ROOT / TEMPERATURE_FSTR.format(year= year))
        precip_x = xarray.open_dataset(CLIMATE_ROOT / PRECIP_FSTR.format(year= year))
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Year {year} not available.') from e
    
    year_locs = pd.concat([lats, longs], axis=1)
    year_locs['year'] = year
    
    temp_over30_series = np.diag(over30_x.interp(lat=lats, lon=longs, method='nearest').tas)
    temp_series = np.diag(temp_x.interp(lat=lats, lon=longs, method='nearest').tas)
    precip_series = np.diag(precip_x.interp(lat=lats, lon=longs, method='nearest').pr)
    year_locs[f'over_{threshold}'] = temp_over30_series
    year_locs['temp'] = temp_series
    year_locs['precip'] = precip_series
    #dfs.append(year_locs)\
    return year_locs

def extract_days_over_threshold_for_year(year:int, lats:pd.Series, longs:pd.Series, threshold = 30)-> pd.DataFrame:
    if isinstance(threshold, int) or threshold.is_integer():
        threshold_str = str(threshold)
    else:
        threshold_str = str(threshold).replace(".", "-")
    try:
        over30_x = xarray.open_dataset(CLIMATE_ROOT / OVER30_FSTR.format(year= year, threshold=threshold_str))
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Year {year} not available. FILE {CLIMATE_ROOT / OVER30_FSTR}') from e
    
    year_locs = pd.concat([lats, longs], axis=1)
    
    temp_over30_series = np.diag(over30_x.interp(lat=lats, lon=longs, method='nearest').tas)
    
    year_locs[f'over_{threshold}'] = temp_over30_series
    year_locs['year'] = year

    #dfs.append(year_locs)\
    return year_locs

def extract_climate_data(input_df:pd.DataFrame, lat_col:str, long_col:str) -> pd.DataFrame:
    #Get unique combination of locations in input dataframe

    #Could detect what years are available, or only extract user-select years, for now let's hardcode it
    years = range(1979, 2017)
    dfs = []
    for year in years:
        logging.info(f'Processing year {year}')
        dfs.append(extract_climate_data_for_year(year, unique_locs_df[lat_col], unique_locs_df[long_col]))
    
    return pd.concat(dfs)

def process_input_file(in_filepath:Path, lat_col, long_col) -> pd.DataFrame:
    extension = in_filepath.suffix
    if extension == '.csv':
        df = pd.read_csv(in_filepath)
    elif extension == '.parquet':
        df = pd.read_parquet(in_filepath)
    else:
        raise NotImplementedError(f"File type {extension} not supported")

    return df[[lat_col, long_col]].drop_duplicates()


def write_output(df:pd.DataFrame, out_filepath:Path, **kwargs) -> None:
    extension = out_filepath.suffix
    if extension == '.csv':
        return df.to_csv(out_filepath)
    elif extension == '.parquet':
        return df.to_parquet(out_filepath)
    else:
        raise NotImplementedError(f"File type {extension} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" Tool to extract climate data for a dataset that has latitude and longitude. \
            Example usage: python get_climate_vars.py /mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/bmi/bmi_data_outliered.csv /mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/bmi_climate.parquet -lat_col latnum -long_col longnum"""
    )
    parser.add_argument("input_file", help="filepath with lat longs to process", type=str)
    parser.add_argument("output_file", help="desired filepath for output", type=str)
    parser.add_argument("-lat_col", help='input file column with latitude', default='lat')
    parser.add_argument("-long_col", help='input file column with longitude', default = 'long')
    parser.add_argument("-year", help='year to extract, optional')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    in_path = Path(args.input_file)
    logging.info(f"Processing file {in_path}")
    unique_locs_df = process_input_file(in_path, args.lat_col, args.long_col)
    if args.year:
        result = extract_climate_data_for_year(int(args.year), unique_locs_df[args.lat_col],
            unique_locs_df[args.long_col], threshold = 30)
    else:
        result = extract_climate_data(unique_locs_df, args.lat_col, args.long_col)
    write_output(result, Path(args.output_file))
