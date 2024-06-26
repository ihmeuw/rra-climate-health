import argparse
import logging
import sys
from ctypes import cdll
from pathlib import Path

import numpy as np
import pandas as pd
from pymer4 import Lmer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            """Testing threshold.
            Example usage: python threshold_worker_linear.py 30 stunting"""
        ),
    )
    parser.add_argument("threshold", help="threshold", type=float)
    parser.add_argument("measure", help="desired filepath for output", type=str)
    args = parser.parse_args()

    threshold = args.threshold
    if threshold.is_integer():
        threshold = int(threshold)

    years = list(range(1979, 2017))  # 1979 2017
    measure = args.measure
    if measure != 'low_adult_bmi':
        in_filepath = f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/cgf_{measure}_cal.parquet'
    else:
        in_filepath = f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/low_adult_bmi_cal.parquet'
    #in_filepath = f'/mnt/team/rapidresponse/pub/population/data/02-processed-data/cgf_bmi/new_{measure}_cal.parquet'
    data_df = pd.read_parquet(in_filepath)
    print(f"Doing threshold {threshold} for {measure}")

    data_df = data_df[['nid', 'country', 'year_start', 'end_year', 'psu', 'hh_id', 'sex', 'age_year', 'age_mo', 'int_year', 'int_month', 'cgf_value', 'income_per_day', 'lat', 'long']]

    file_path = climate_utils.CLIMATE_ROOT / climate_utils.OVER30_FSTR.format(year=1981, threshold=str(threshold).replace(".","-"))
    file_exists = Path(file_path).exists()
    if not file_exists:
        print(f"File not found: {file_path}")

    lat_col = 'lat'
    long_col = 'long'

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    unique_locs_df = climate_utils.process_input_file(Path(in_filepath), lat_col, long_col)

    temp_dfs = []
    for year in years:
        print(f"Doing year {year}")
        climate_data_for_year = climate_utils.extract_days_over_threshold_for_year(year, threshold, unique_locs_df[lat_col], unique_locs_df[long_col])
        temp_dfs.append(climate_data_for_year)
    
    climate_df = pd.concat(temp_dfs)
    temp_dfs = None
    
    data_df = data_df.merge(climate_df, left_on=['lat', 'long', 'year_start'],
        right_on=['lat', 'long', 'year'], how='left')
    data_df = data_df.loc[data_df.year_start < 2017]
    data_df = data_df.loc[data_df.year_start > 1978]

    nbins = 10
    threshold_col = f"over_{threshold}"
    threshold_binned_col = f"over_{threshold}_bin"
    data_df = cgf_utils.group_and_bin_column_definition(data_df, threshold_col, 'location', nbins, bin_strategy = 'custom_daysover', result_column = threshold_binned_col) #custom_daysover 0_more_readable
    data_df = cgf_utils.group_and_bin_column_definition(data_df, 'income_per_day', 'household', nbins)

    data_df['grid_cell'] = data_df[threshold_binned_col].astype(str) +'_'+ data_df.income_per_day_bin.astype(str)

    # cols_to_scale = []#'over30_avgperyear', 'precip', 'precip_cumavg_5' ,'temp', 'temp_cumavg_5', 'income_per_day']
    # scaler = sklearn.preprocessing.MinMaxScaler()
    # scaled_cols = [f'sc_{col}' for col in cols_to_scale]
    # data_df[scaled_cols] = scaler.fit_transform(data_df[cols_to_scale])

    def fit_and_predict_LMER(data:pd.DataFrame, model_spec:str, value_col = 'cgf_value', location_col = 'country'):
        print("fitting")
        model = Lmer(model_spec, data=data, family='binomial')
        model.fit()
        pred_df = data.copy()
        pred_df[location_col] = np.nan
        pred_df['model_fit'] = model.fits
        pred_df['model_residual'] = pred_df[value_col] - pred_df.model_fit
        print("predicting")
        pred_df['model_fit_nocountry'] = model.predict(pred_df)
        pred_df['model_fit_nocountry_res'] = pred_df['model_fit_nocountry'] + pred_df['model_residual']
        return pred_df, model

    pred_df, model = fit_and_predict_LMER(data_df, 'cgf_value ~ (1 | country) + (1 | grid_cell)')


    OUTPUT_ROOT = Path('/mnt/share/scratch/users/victorvt/threshold_inv')
    OUTPUT_FOLDER = OUTPUT_ROOT / measure / str(threshold)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    data_df['model_fit_nocountry_res'] = pred_df['model_fit_nocountry_res']
    cgf_utils.printable_names['model_fit_nocountry_res'] = measure
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_file = PdfPages(OUTPUT_FOLDER / f"threshold_{str(threshold).replace('.', '-')}.pdf")

    cgf_utils.plot_heatmap(data_df, threshold_binned_col, value_col='cgf_value', 
        title_addin='Data', counts=True, pdf_handle = pdf_file)

    cgf_utils.plot_heatmap(data_df, threshold_binned_col, value_col='model_fit_nocountry_res', 
        title_addin='Model', counts=False, pdf_handle = pdf_file)

    pdf_file.close()

    fitness_df = pd.DataFrame.from_records([{'threshold':threshold, 
        'formula':model.formula, 
        'aic':model.AIC, 
        'bic':model.BIC, 
        'log_likelihood':model.logLike,
        'measure':measure}])
    fitness_df.to_csv(OUTPUT_FOLDER/f"fitness_{measure}.csv", index=False)

    import pickle
    with open(OUTPUT_FOLDER/f"model_{measure}.pkl", 'wb') as model_pickle_obj:
        pickle.dump(model, model_pickle_obj)

