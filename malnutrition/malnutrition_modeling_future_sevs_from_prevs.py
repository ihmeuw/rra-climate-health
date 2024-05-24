'''
This script contains a function that takes the complete file path to the dataframe that contains the future prevalence of stunting and wasting and the full output path where you would like to store results of the model. 
The function reads the historical data of prevalence and SEVs for the risks of interest (stunting/wasting) and the future prevalence data (stunting/wasting), fits a mixed effects linear regression model to the historical data and predicts the future SEV of stunting and wasting based off the model using prevalence. 
The function then combines the historical data with the predicted future data and saves the results to the output path. The output file sev_val column contains the original SEV value estimates pulled from the db and the future SEV values that were predicted by the model.
The model specifications are as follows: Mixed Linear Regression set with sev_val as the dependent variable and prev_val as the independent variable with random effects set to location_id.

How to run the function: 
model_data(
    "full file path to where you have stored the dataframe that contains future prevalence data", 
    "full file path to where you want to save the output of the model"
)

Columns that should be present in the input data 
- ['location_id', 'year_id', 'rei_id', 'age_group_id', 'sex_id', 'prev_val', 'age_group_name', 'location_name', 'rei_name', 'sex']

Columns that should be present in the output data 
- ['location_id', 'year_id', 'rei_id', 'age_group_id', 'sex_id', 'prev_val', 'age_group_name', 'location_name', 'rei_name', 'sex', 'sev_val']

Currently the function expects the future prevalence data to have prevalence by the following groups:
- age_group_name : ['2 to 4', '12 to 23 months', '1-5 months', '6-11 months']
- rei_name : ['Child wasting', 'Child stunting']
- sex : ['Male', 'Female']
- locations : all most detailed locations in the Forecasting location set (location_set_id=39)
'''

import pandas as pd
import statsmodels.formula.api as smf

def model_data(future_prev_input_path, past_future_sev_prev_output_path):
    historical_prev_sev_df=pd.read_csv(f"/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/child_malnutrition/prev_sev_models/20240524_prev_sev_data_loc_set_39.csv")
    future_prev_df=pd.read_csv(future_prev_input_path)
    df=historical_prev_sev_df
    df['year_id'] = df['year_id'].astype(int)
    df['location_id'] = df['location_id'].astype(str)
    age_groups = df['age_group_name'].unique()
    risk_groups = df['rei_name'].unique()
    future_predictions = []

    for risk_group in risk_groups:
        for age_group in age_groups:
            df_filtered = df[(df['age_group_name'] == age_group) & (df['rei_name'] == risk_group)].dropna()
            future_df_filtered = future_prev_df[(future_prev_df['age_group_name'] == age_group) & (future_prev_df['rei_name'] == risk_group)].dropna()
            rei_name = df_filtered['rei_name'].unique()[0]
            model = smf.mixedlm("sev_val ~ prev_val", df_filtered, groups=df_filtered["location_id"], re_formula="1")
            model_fit = model.fit()
            future_df_filtered['sev_val'] = model_fit.predict(future_df_filtered)

            future_predictions.append(future_df_filtered)

    # Combine predictions into a single dataframe
    df_predictions = pd.concat(future_predictions)
    full_df = pd.concat([historical_prev_sev_df, df_predictions])
    # save full dataframe with forecasts
    full_df.to_csv(past_future_sev_prev_output_path, index=False)