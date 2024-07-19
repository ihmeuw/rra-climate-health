import numpy as np
import pandas as pd
import xarray as xr
from db_queries import get_model_results, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

## CONSTANTS
GBD_RELEASE_ID=9
SDG_RELEASE_ID=27
NONFORECASTED_YEARS=list(range(1950, 2024, 1))
locs_meta=get_location_metadata(
    location_set_id=35, 
    release_id=GBD_RELEASE_ID
)
PLOT_VERSION="re_intercept_slope_sep_sex_age"

## POPULATION - GBD2021
gbd_pop = get_population(
    age_group_id=[34, 238, 388, 389], 
    year_id=NONFORECASTED_YEARS, 
    release_id=GBD_RELEASE_ID, 
    sex_id=[1,2], 
    location_id='all'
)

## MODEL RESULTS - Victor's Wasting 
file_name='wasting_2024_07_12.04'
model_type="base_mean_temp_elevation"
model_title="base model + mean temperature + elevation"
model = pd.read_parquet(f'/mnt/share/scratch/users/victorvt/for/ncorto/{file_name}.parquet', engine='pyarrow')
model_df=model.reset_index()
model_df=model_df[['location_id', 'year_id', 'sex_id', 'age_group_id', 'ssp245']]
model_df=model_df.rename(columns={"ssp245": "model_value"})

## GBD estimates - GBD2021
severe_wasting = get_model_results(
    'epi',
    gbd_id=8945,
    location_id='all', 
    year_id=NONFORECASTED_YEARS, 
    release_id=GBD_RELEASE_ID
)
moderate_wasting = get_model_results(
    'epi',
    gbd_id=8946,
    location_id='all', 
    year_id=NONFORECASTED_YEARS, 
    release_id=GBD_RELEASE_ID
)
wasting = pd.concat([severe_wasting, moderate_wasting])
wasting_df=pd.merge(wasting, gbd_pop, on=["location_id", "year_id", "sex_id", "age_group_id"], how="inner")
wasting_df = wasting_df[['location_id', 'year_id', 'age_group_id','sex_id', 'measure_id', 'measure', 'mean', 'population']]
wasting_df = wasting_df.rename(columns={"mean": "val"})
wasting_df["value_count"] = wasting_df["val"]*wasting_df["population"]
wasting_df_agg = wasting_df.groupby(["location_id", "year_id", "age_group_id", "sex_id"]).agg({"value_count":"sum", "population":"sum"}).reset_index()
# split into two dataframes by age 
gbd_prev_pop_pnn=wasting_df_agg[wasting_df_agg["age_group_id"].isin([388,389])]
gbd_prev_pop_1_to_4=wasting_df_agg[wasting_df_agg["age_group_id"].isin([238,34])]
gbd_prev_pop_pnn = gbd_prev_pop_pnn.groupby(["location_id", "year_id", "sex_id"]).agg({"value_count":"sum", "population":"sum"}).reset_index()
gbd_prev_pop_pnn["age_group_id"]=4
gbd_prev_pop_1_to_4 = gbd_prev_pop_1_to_4.groupby(["location_id", "year_id", "sex_id"]).agg({"value_count":"sum", "population":"sum"}).reset_index()
gbd_prev_pop_1_to_4["age_group_id"]=5
gbd_prev_pop_age_agg = pd.concat([gbd_prev_pop_pnn, gbd_prev_pop_1_to_4])
gbd_prev_pop_age_agg["gbd_value"]=gbd_prev_pop_age_agg["value_count"]/gbd_prev_pop_age_agg["population"]
gbd_df = gbd_prev_pop_age_agg.drop(["value_count", "population"], axis=1)

## SDI - Future + Past
past_sdi = xr.open_dataset('/mnt/share/forecasting/data/7/past/sdi/20240531_gk24/sdi.nc')
past_sdi_df = past_sdi.to_dataframe().reset_index()
past_sdi_df_agg = past_sdi_df.groupby(["location_id", "year_id"]).agg({"sdi":"mean"}).reset_index()
future_sdi= xr.open_dataset('/mnt/share/forecasting/data/7/future/sdi/20240531_gk24/sdi.nc')
future_sdi_df = future_sdi.to_dataframe().reset_index()
future_sdi_df=future_sdi_df[future_sdi_df["scenario"]==0]
future_sdi_df=future_sdi_df.drop(columns=["scenario"])
future_sdi_df_agg = future_sdi_df.groupby(["location_id", "year_id"]).agg({"sdi":"mean"}).reset_index()
sdi_df=pd.concat([past_sdi_df_agg, future_sdi_df_agg])

## HAQI - Future + Past
haq_df=pd.read_csv('/share/resource_tracking/forecasting/haq/GK_2024/summary_files/scenario_stats_w_subnats.csv')
haq_df=haq_df[haq_df["scenario"]==0]
haq_df=haq_df.drop(columns=["scenario"])
haq_df=haq_df.rename(columns={"mean": "haq_value"})

## PREP DATA FOR MODELING
## Merge 
gbd_model_df=pd.merge(gbd_df, model_df, on=['location_id', 'year_id', 'sex_id', 'age_group_id'], how='inner')
gbd_model_df['residual_value'] = gbd_model_df['gbd_value'] - gbd_model_df['model_value']
effects_df=pd.merge(sdi_df, haq_df[["year_id", "location_id", "haq_value"]], on=['location_id', 'year_id'], how='left')
effects_df=effects_df[effects_df["location_id"].isin(gbd_model_df["location_id"].unique())]
modeling_df=pd.merge(gbd_model_df, effects_df, on=['location_id', 'year_id'], how='inner')

## DROP LOCATION_ID = 44585 BECAUSE WE DO NOT HAVE A HAQ FOR IT
modeling_df=modeling_df[modeling_df["location_id"]!=44858]
sdi_df=sdi_df[sdi_df["location_id"]!=44858]
haq_df=haq_df[haq_df["location_id"]!=44858]
future_sdi_modeling_fe=sdi_df[sdi_df["year_id"]>2022]
future_sdi_modeling_re=sdi_df[sdi_df["year_id"]>2022]
future_haq_modeling_fe=haq_df[haq_df["year_id"]>2022]
future_haq_modeling_re=haq_df[haq_df["year_id"]>2022]

unique_combinations = modeling_df[['age_group_id', 'sex_id']].drop_duplicates()
prediction_results_sdi_fe = []
prediction_results_sdi_re = []
prediction_results_haq_fe = []
prediction_results_haq_re = []

for _, row in unique_combinations.iterrows():
    # Filtering the DataFrame for each unique combination
    temp_df = modeling_df[(modeling_df['age_group_id'] == row['age_group_id']) & 
                          (modeling_df['sex_id'] == row['sex_id'])]
    ## REGRESSION - RE on SDI 
    fe_sdi_model = smf.ols("residual_value ~ sdi", data=temp_df).fit()
    print(fe_sdi_model.summary())
    future_sdi_modeling_fe['predicted_residual_value_sdi'] = fe_sdi_model.predict(future_sdi_modeling_fe['sdi'])
    future_sdi_modeling_fe['age_group_id'] = row['age_group_id']
    future_sdi_modeling_fe['sex_id'] = row['sex_id']
    ## MIXED LINEAR REGRESSION - RE on SDI AND RE (slope) on YEAR BY LOCATION
    sdi_year_model_formula = smf.mixedlm("residual_value ~ sdi", temp_df, groups=temp_df["location_id"], re_formula="~1 + year_id")
    fe_sdi_re_year_model = sdi_year_model_formula.fit()
    print(fe_sdi_re_year_model.summary())
    future_sdi_modeling_re['predicted_residual_value_sdi_year'] = fe_sdi_re_year_model.predict(future_sdi_modeling_re)
    future_sdi_modeling_re['age_group_id'] = row['age_group_id']
    future_sdi_modeling_re['sex_id'] = row['sex_id']
    ## REGRESSION - RE on HAQI 
    fe_haq_model = smf.ols("residual_value ~ haq_value", data=temp_df).fit()
    print(fe_haq_model.summary())
    future_haq_modeling_fe['predicted_residual_value_haq'] = fe_haq_model.predict(future_haq_modeling_fe['haq_value'])
    future_haq_modeling_fe['age_group_id'] = row['age_group_id']
    future_haq_modeling_fe['sex_id'] = row['sex_id']
    ## MIXED LINEAR REGRESSION - RE on HAQI AND RE (slope) on YEAR BY LOCATION
    haq_year_model_formula = smf.mixedlm("residual_value ~ haq_value", temp_df, groups=temp_df["location_id"], re_formula="~1 + year_id")
    fe_haq_re_year_model = haq_year_model_formula.fit()
    print(fe_haq_re_year_model.summary())
    future_haq_modeling_re['predicted_residual_value_haq_year'] = fe_haq_re_year_model.predict(future_haq_modeling_re)
    future_haq_modeling_re['age_group_id'] = row['age_group_id']
    future_haq_modeling_re['sex_id'] = row['sex_id']
    
    # Store the predictions for both SDI and HAQI
    prediction_results_sdi_fe.append(future_sdi_modeling_fe[['location_id', 'year_id', 'age_group_id', 'sex_id', 'sdi', 'predicted_residual_value_sdi']])
    prediction_results_sdi_re.append(future_sdi_modeling_re[['location_id', 'year_id', 'age_group_id', 'sex_id', 'sdi', 'predicted_residual_value_sdi_year']])
    prediction_results_haq_fe.append(future_haq_modeling_fe[['location_id', 'year_id', 'age_group_id', 'sex_id', 'haq_value', 'predicted_residual_value_haq']])
    prediction_results_haq_re.append(future_haq_modeling_re[['location_id', 'year_id', 'age_group_id', 'sex_id', 'haq_value', 'predicted_residual_value_haq_year']])
prediction_results_sdi_fe = pd.concat(prediction_results_sdi_fe, ignore_index=True)
prediction_results_sdi_re = pd.concat(prediction_results_sdi_re, ignore_index=True)
prediction_results_haq_fe = pd.concat(prediction_results_haq_fe, ignore_index=True)
prediction_results_haq_re = pd.concat(prediction_results_haq_re, ignore_index=True)
full_predictions_sdi = pd.merge(prediction_results_sdi_fe, prediction_results_sdi_re, on=['location_id', 'year_id', 'age_group_id', 'sex_id', 'sdi'], how='inner')
full_predictions_haq = pd.merge(prediction_results_haq_fe, prediction_results_haq_re, on=['location_id', 'year_id', 'age_group_id', 'sex_id', 'haq_value'], how='inner')

future_sdi_modeling = pd.merge(full_predictions_sdi, model_df, on=['location_id', 'year_id', 'sex_id', 'age_group_id'], how='inner')
future_haq_modeling = pd.merge(full_predictions_haq, model_df, on=['location_id', 'year_id', 'sex_id', 'age_group_id'], how='inner')

## PLOTTING 
plot_df=pd.concat([future_sdi_modeling, gbd_model_df])
plot_df=pd.merge(plot_df, locs_meta[['location_id', 'location_name']], on='location_id', how='inner')
plot_path=f'/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/plot/diagnostic_plots/stunting_wasting_model_gbd_comparison/model_predictions/wasting/wasting_sdi_{PLOT_VERSION}.pdf'
# Timeseries plotting
with PdfPages(plot_path) as pdf:
    for location in plot_df['location_id'].unique():
        df_location = plot_df[plot_df['location_id'] == location]

        # Ensure df_location is sorted by year_id
        df_location = df_location.sort_values(by='year_id')

        # Calculate local min and max values for setting the y-axis scale for all subplots on this page
        local_min = df_location[['gbd_value', 'model_value', 'residual_value', 'predicted_residual_value_sdi', 'predicted_residual_value_sdi_year']].min().min()
        local_max = df_location[['gbd_value', 'model_value', 'residual_value', 'predicted_residual_value_sdi', 'predicted_residual_value_sdi_year']].max().max()

        # Add a padding to ensure lines aren't cut off
        padding = (local_max - local_min) * 0.05  # 5% padding
        local_min -= padding
        local_max += padding

        # Setup a figure for the current location
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        location_name = df_location.get('location_name', 'Unknown Location').iloc[0]
        fig.suptitle(f'wasting Comparison\n{model_title} ({file_name}) | GBD estimates\nLocation : {location_name}', fontsize=16)

        # Counter for subplot index
        plot_count = 0

        for sex_id in [1, 2]:
            for age_group_id in [4, 5]:
                ax = axs[plot_count // 2, plot_count % 2]
                plot_count += 1

                # Sort df_plot by year_id
                df_plot = df_location[(df_location['sex_id'] == sex_id) & (df_location['age_group_id'] == age_group_id)].sort_values(by='year_id')

                sex_label = "Male" if sex_id == 1 else "Female"
                age_group_label = "Post Neonatal" if age_group_id == 4 else "1 to 4"

                # Plot model_value, gbd_value, residual_value, and predicted values
                ax.plot(df_plot['year_id'], df_plot['model_value'], color='green', label='Model Value')
                #ax.plot(df_plot['year_id'], df_plot['sdi'], color='black', label='SDI')
                ax.plot(df_plot['year_id'], df_plot['gbd_value'], color='blue', label='GBD Value')
                filtered_df_plot = df_plot[df_plot['year_id'] <= 2022]
                ax.plot(filtered_df_plot['year_id'], filtered_df_plot['residual_value'], color='red', label='Residual Value')
                future_df_plot = df_plot[df_plot['year_id'] >= 2023]
                if not future_df_plot.empty:
                    ax.plot(future_df_plot['year_id'], future_df_plot['predicted_residual_value_sdi'], 'r:', label='Predicted Residual : FE SDI')
                    ax.plot(future_df_plot['year_id'], future_df_plot['predicted_residual_value_sdi_year'], 'r--', label='Predicted Residual : FE SDI & RE Year per loc (intercept+slope)')
                    #ax.plot(future_df_plot['year_id'], future_df_plot['sdi'], color='black')
                ax.axvline(x=2022, color='black', linestyle='-', linewidth=1)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')
                ax.set_ylim(local_min, local_max)
                ax.set_title(f'Sex: {sex_label}, Age Group: {age_group_label}')
                ax.set_xlabel('Year')
                ax.set_ylabel('Value')
                #ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        handles, labels = [], []
        for value_column, color, linestyle in zip(['gbd_value', 'model_value', 'residual_value', 'predicted_residual_value_sdi', 'predicted_residual_value_sdi_year'], ['blue', 'green', 'red', 'red', 'red'], ['-', '-', '-', ':', '--']):
            handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2))
            labels.append(value_column)

        fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
        pdf.savefig(fig)
        plt.close(fig)

plot_df=pd.concat([future_haq_modeling, gbd_model_df])
plot_df=pd.merge(plot_df, locs_meta[['location_id', 'location_name']], on='location_id', how='inner')
plot_path=f'/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/plot/diagnostic_plots/stunting_wasting_model_gbd_comparison/model_predictions/wasting/wasting_haqi_{PLOT_VERSION}.pdf'
# Timeseries plotting
with PdfPages(plot_path) as pdf:
    for location in plot_df['location_id'].unique():
        df_location = plot_df[plot_df['location_id'] == location]

        # Ensure df_location is sorted by year_id
        df_location = df_location.sort_values(by='year_id')

        # Calculate local min and max values for setting the y-axis scale for all subplots on this page
        local_min = df_location[['gbd_value', 'model_value', 'residual_value', 'predicted_residual_value_haq', 'predicted_residual_value_haq_year']].min().min()
        local_max = df_location[['gbd_value', 'model_value', 'residual_value', 'predicted_residual_value_haq', 'predicted_residual_value_haq_year']].max().max()

        # Add a padding to ensure lines aren't cut off
        padding = (local_max - local_min) * 0.05  # 5% padding
        local_min -= padding
        local_max += padding

        # Setup a figure for the current location
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        location_name = df_location.get('location_name', 'Unknown Location').iloc[0]
        fig.suptitle(f'wasting Comparison\n{model_title} ({file_name}) | GBD estimates\nLocation : {location_name}', fontsize=16)

        # Counter for subplot index
        plot_count = 0

        for sex_id in [1, 2]:
            for age_group_id in [4, 5]:
                ax = axs[plot_count // 2, plot_count % 2]
                plot_count += 1

                df_plot = df_location[(df_location['sex_id'] == sex_id) & (df_location['age_group_id'] == age_group_id)].sort_values(by='year_id')

                sex_label = "Male" if sex_id == 1 else "Female"
                age_group_label = "Post Neonatal" if age_group_id == 4 else "1 to 4"

                # Plot model_value, gbd_value, residual_value, and predicted values
                ax.plot(df_plot['year_id'], df_plot['model_value'], color='green', label='Model Value')
                #ax.plot(df_plot['year_id'], df_plot['haq_value'], color='black', label='HAQI')
                ax.plot(df_plot['year_id'], df_plot['gbd_value'], color='blue', label='GBD Value')
                filtered_df_plot = df_plot[df_plot['year_id'] <= 2022]
                ax.plot(filtered_df_plot['year_id'], filtered_df_plot['residual_value'], color='red', label='Residual Value')
                future_df_plot = df_plot[df_plot['year_id'] >= 2023]
                if not future_df_plot.empty:
                    ax.plot(future_df_plot['year_id'], future_df_plot['predicted_residual_value_haq'], 'r:', label='Predicted Residual : FE HAQI')
                    ax.plot(future_df_plot['year_id'], future_df_plot['predicted_residual_value_haq_year'], 'r--', label='Predicted Residual : FE HAQI & RE Year per loc (intercept+slope)')
                    #ax.plot(future_df_plot['year_id'], future_df_plot['haq_value'], color='black')
                ax.axvline(x=2022, color='black', linestyle='-', linewidth=1)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')
                ax.set_ylim(local_min, local_max)
                ax.set_title(f'Sex: {sex_label}, Age Group: {age_group_label}')
                ax.set_xlabel('Year')
                ax.set_ylabel('Value')
                # ax.legend()  # This line is commented out to remove subplot legends

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        handles, labels = [], []
        for value_column, color, linestyle in zip(['gbd_value', 'model_value', 'residual_value', 'predicted_residual_value_haq', 'predicted_residual_value_haq_year'], ['blue', 'green', 'red', 'red', 'red'], ['-', '-', '-', ':', '--']):
            handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2))
            labels.append(value_column)

        fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
        pdf.savefig(fig)
        plt.close(fig)