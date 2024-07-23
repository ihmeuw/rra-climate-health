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
from patsy import dmatrix


## CONSTANTS 
GBD_RELEASE_ID=16
SDG_RELEASE_ID=27
NONFORECASTED_YEARS=list(range(1950, 2024, 1))
locs_meta=get_location_metadata(
    location_set_id=35, 
    release_id=GBD_RELEASE_ID
)
RESID_PLOT_VERSION="20240722_attentuated"
RESID_PLUS_MODEL_PLOT_VERSION="20240722_combined"

## POPULATION - Goalkeepers ETL'd
gbd_pop_da = xr.open_dataset('/mnt/share/forecasting/data/9/past/population/20240517_etl_gbd_9/population.nc')
gbd_pop = gbd_pop_da.to_dataframe().reset_index()

## MODEL RESULTS - Victor's Wasting 
file_name='stunting_2024_07_12.08'
model_type="base_mean_temp_elevation"
model_title="base model + mean temperature + elevation"
model = pd.read_parquet(f'/mnt/share/scratch/users/victorvt/for/ncorto/{file_name}.parquet', engine='pyarrow')
model_df=model.reset_index()
model_df=model_df[['location_id', 'year_id', 'sex_id', 'age_group_id', 'ssp245']]
model_df=model_df.rename(columns={"ssp245": "model_value"})

## GBD estimates - GBD2021
sdgs_df = get_draws(
    "indicator_component_id",
    35,
    source="sdg",
    release_id=SDG_RELEASE_ID
)
max_draw = max([int(col.split("_")[1]) for col in sdgs_df.columns if "draw" in col])
draw_cols = ["draw_" + str(i) for i in range(max_draw + 1)]
sdgs_df = pd.melt(
    sdgs_df,
    id_vars=["location_id", "year_id", "age_group_id", "sex_id"],
    value_vars=draw_cols,
    var_name="draw",
)
stunting = sdgs_df.groupby(["location_id", "year_id", "age_group_id", "sex_id"]).agg({"value":"mean"}).reset_index()
stunting_df=pd.merge(stunting, gbd_pop, on=["location_id", "year_id", "sex_id", "age_group_id"], how="inner")
#age_group_4
stunting_age_group_4 = stunting_df[stunting_df["age_group_id"].isin([388, 389])]
stunting_age_group_4["value_count"] = stunting_age_group_4["value"] * stunting_age_group_4["population"]
stunting_age_group_4_agg = stunting_age_group_4.groupby(["location_id", "year_id", "sex_id"]).agg({"value_count":"sum", "population":"sum"}).reset_index()
stunting_age_group_4_agg["gbd_value"] = stunting_age_group_4_agg["value_count"] / stunting_age_group_4_agg["population"]
stunting_age_group_4_agg=stunting_age_group_4_agg[['gbd_value', 'location_id', 'year_id', 'sex_id']]
stunting_age_group_4_agg["age_group_id"]=4
#age_group_5
stunting_age_group_5 = stunting_df[stunting_df["age_group_id"].isin([34, 238])]
stunting_age_group_5["value_count"] = stunting_age_group_5["value"] * stunting_age_group_5["population"]
stunting_age_group_5_agg = stunting_age_group_5.groupby(["location_id", "year_id", "sex_id"]).agg({"value_count":"sum", "population":"sum"}).reset_index()
stunting_age_group_5_agg["gbd_value"] = stunting_age_group_5_agg["value_count"] / stunting_age_group_5_agg["population"]
stunting_age_group_5_agg=stunting_age_group_5_agg[['gbd_value', 'location_id', 'year_id', 'sex_id']]
stunting_age_group_5_agg["age_group_id"]=5
gbd_df=pd.concat([stunting_age_group_4_agg, stunting_age_group_5_agg])

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

## PREP DATA FOR MODELING
## Merge 
gbd_model_df=pd.merge(gbd_df, model_df, on=['location_id', 'year_id', 'sex_id', 'age_group_id'], how='inner')
gbd_model_df['residual_value'] = gbd_model_df['gbd_value'] - gbd_model_df['model_value']
#effects_df=pd.merge(sdi_df, haq_df[["year_id", "location_id", "haq_value"]], on=['location_id', 'year_id'], how='left')
effects_df=sdi_df[sdi_df["location_id"].isin(gbd_model_df["location_id"].unique())]
modeling_df=pd.merge(gbd_model_df, effects_df, on=['location_id', 'year_id'], how='inner')
# DROP LOCATION_ID = 44585 BECAUSE WE DO NOT HAVE A HAQ FOR IT
modeling_df=modeling_df[modeling_df["location_id"]!=44858]
modeling_locs=list(modeling_df["location_id"].unique())
combinations_df = pd.DataFrame({
    'sex_id': [1, 1, 2, 2],
    'age_group_id': [4, 5, 4, 5]
})
sdi_df['key'] = 1
combinations_df['key'] = 1
sdi_df = pd.merge(sdi_df, combinations_df, on='key')
sdi_df.drop('key', axis=1, inplace=True)

sdi_df=sdi_df[sdi_df["location_id"].isin(modeling_locs)]
#haq_df=haq_df[haq_df["location_id"].isin(modeling_locs)]
future_sdi_modeling_nte=sdi_df[sdi_df["year_id"]>2023]
future_sdi_modeling_te=sdi_df[sdi_df["year_id"]>2023]
# apply linear time scale on year 
def calculate_year_scale(year):
    # Apply scaling only for years 2023 to 2100
    if year >= 2023:
        return max(0, (2100 - year) / (2100 - 2023))
    else:
        # No scaling applied before 2023
        return 1

# Apply this function to your DataFrame
modeling_df['year_scale'] = modeling_df['year_id'].apply(calculate_year_scale)
future_sdi_modeling_te['year_scale'] = future_sdi_modeling_te['year_id'].apply(calculate_year_scale)

##MODELING
## REGRESSION - RE on SDI + Interaction with sdi*location + location*age*sex (NO TIME EFFECT)
no_time_effect_model = smf.ols("residual_value ~ sdi + sdi*C(location_id) + C(location_id)*C(age_group_id)*C(sex_id)" , data=modeling_df).fit()
print(no_time_effect_model.summary())
predicted_values = no_time_effect_model.predict(future_sdi_modeling_nte)
future_sdi_modeling_nte['predicted_residual_no_time_effect'] = predicted_values

## REGRESSION - RE on SDI + Interaction on location + location*year (without main effect of year by itself) + location*age*sex (TIME EFFECT)
modeling_df['time_id'] = (modeling_df['year_id'] - 2100)/100
time_effect_model = smf.ols("residual_value ~ sdi + C(location_id) + C(location_id):time_id + C(location_id)*C(age_group_id)*C(sex_id)", data=modeling_df).fit()
print(time_effect_model.summary())
#future_sdi_modeling_te['year_backup'] = future_sdi_modeling_te['year_id']
future_sdi_modeling_te['time_id'] = ((future_sdi_modeling_te['year_id'] - 2100)/100 ) * future_sdi_modeling_te['year_scale']
predicted_values = time_effect_model.predict(future_sdi_modeling_te)
#future_sdi_modeling_te = future_sdi_modeling_te.rename(columns={"year_backup": "year_id"})
future_sdi_modeling_te['predicted_residual_time_effect'] = predicted_values

## combine predicted datasets
future_sdi_modeling=pd.merge(future_sdi_modeling_nte, future_sdi_modeling_te, on=['location_id',  'sdi', 'sex_id', 'age_group_id',  'year_id'], how='inner')
future_sdi_modeling_df=pd.merge(future_sdi_modeling, model_df, on=['location_id',  'sex_id', 'age_group_id',  'year_id'], how='inner')


## PLOTTING 
plot_df=pd.concat([future_sdi_modeling_df, modeling_df])
plot_df=pd.merge(plot_df, locs_meta[['location_id', 'location_name']], on='location_id', how='inner')
plot_path=f'/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/plot/diagnostic_plots/stunting_wasting_model_gbd_comparison/model_predictions/stunting/stunting_sdi_{RESID_PLOT_VERSION}.pdf'
# Timeseries plotting
with PdfPages(plot_path) as pdf:
    for location in plot_df['location_id'].unique():
        df_location = plot_df[plot_df['location_id'] == location]

        # Ensure df_location is sorted by year_id
        df_location = df_location.sort_values(by='year_id')

        # Calculate local min and max values for setting the y-axis scale for all subplots on this page
        local_min = df_location[['gbd_value', 'model_value', 'residual_value',  'predicted_residual_no_time_effect', 'predicted_residual_time_effect']].min().min()
        local_max = df_location[['gbd_value', 'model_value', 'residual_value',  'predicted_residual_no_time_effect', 'predicted_residual_time_effect']].max().max()

        # Add a padding to ensure lines aren't cut off
        padding = (local_max - local_min) * 0.05  # 5% padding
        local_min -= padding
        local_max += padding

        # Setup a figure for the current location
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        location_name = df_location.get('location_name', 'Unknown Location').iloc[0]
        fig.suptitle(f'Stunting Comparison\n{model_title} ({file_name}) | GBD estimates\nLocation : {location_name}', fontsize=16)

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
                filtered_df_plot = df_plot[df_plot['year_id'] <= 2023]
                ax.plot(filtered_df_plot['year_id'], filtered_df_plot['residual_value'], color='red', label='Residual Value')
                #ax.plot(filtered_df_plot['year_id'], filtered_df_plot['predicted_residual_no_time_effect_in_sample'],  color='red', linestyle=':')
                #ax.plot(filtered_df_plot['year_id'], filtered_df_plot['predicted_residual_time_effect_in_sample'],  color='red', linestyle='-.')
                #ax.plot(filtered_df_plot['year_id'], filtered_df_plot['sdi_re_predictions'], 'r--', label='Predicted (in-sample) Residual : FE SDI & RE Year per loc (intercept+slope)')
                future_df_plot = df_plot[df_plot['year_id'] >= 2024]
                if not future_df_plot.empty:
                    ax.plot(future_df_plot['year_id'], future_df_plot['predicted_residual_no_time_effect'], 'r:', label='FE on SDI + IE on location*sdi and location*age*sex')
                    ax.plot(future_df_plot['year_id'], future_df_plot['predicted_residual_time_effect'], 'r-.', label='FE on SDI + IE on location*year')
                    #ax.plot(future_df_plot['year_id'], future_df_plot['sdi'], color='black')
                ax.axvline(x=2023, color='black', linestyle='-', linewidth=1)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')
                ax.set_ylim(local_min, local_max)
                ax.set_title(f'Sex: {sex_label}, Age Group: {age_group_label}')
                ax.set_xlabel('Year')
                ax.set_ylabel('Value')
                #ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        handles, labels = [], []
        for value_column, color, linestyle in zip(['GBD Estimate', 'Victor Model Estimate', 'Residual Value', 'FE on SDI + IE on location*sdi and location*age*sex', 'FE on SDI + IE on location and location:year and location*age*sex (attenuated)'], ['blue', 'green', 'red', 'red', 'red'], ['-', '-', '-',  ':', '-.']):
            handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2))
            labels.append(value_column)

        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
        pdf.savefig(fig)
        plt.close(fig)

plot_df['residual_val_nte'] = plot_df['residual_value'].combine_first(plot_df['predicted_residual_no_time_effect'])
plot_df['residual_val_te'] = plot_df['residual_value'].combine_first(plot_df['predicted_residual_time_effect'])
plot_df=plot_df[['location_id', 'year_id', 'sdi', 'sex_id','age_group_id', 'model_value', 'gbd_value', 'location_name', 'residual_val_nte', 'residual_val_te' ]]
plot_df['model_val_plus_resid_te'] = plot_df['model_value'] + plot_df['residual_val_te']
plot_df['model_val_plus_resid_nte'] = plot_df['model_value'] + plot_df['residual_val_nte']

#file_path = '/mnt/share/scratch/users/victorvt/for/ncorto/20240722_stunting_residuals.csv'
#plot_df.to_csv(file_path, index=False) 

# Plot residual + victor model value 
plot_path=f'/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/plot/diagnostic_plots/stunting_wasting_model_gbd_comparison/model_predictions/stunting/stunting_sdi_{RESID_PLUS_MODEL_PLOT_VERSION}.pdf'
# Timeseries plotting
with PdfPages(plot_path) as pdf:
    for location in plot_df['location_id'].unique():
        df_location = plot_df[plot_df['location_id'] == location]

        # Ensure df_location is sorted by year_id
        df_location = df_location.sort_values(by='year_id')

        # Calculate local min and max values for setting the y-axis scale for all subplots on this page
        local_min = df_location[['gbd_value', 'model_value', 'residual_val_te', 'residual_val_nte', 'model_val_plus_resid_te', 'model_val_plus_resid_nte']].min().min()
        local_max = df_location[['gbd_value', 'model_value', 'residual_val_te', 'residual_val_nte', 'model_val_plus_resid_te', 'model_val_plus_resid_nte']].max().max()

        # Add a padding to ensure lines aren't cut off
        padding = (local_max - local_min) * 0.05  # 5% padding
        local_min -= padding
        local_max += padding

        # Setup a figure for the current location
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        location_name = df_location.get('location_name', 'Unknown Location').iloc[0]
        fig.suptitle(f'Stunting Joint Model : Residual Models + Stunting Model\nLocation : {location_name}', fontsize=16)

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
                ax.plot(df_plot['year_id'], df_plot['gbd_value'], color='blue', label='GBD Value')
                ax.plot(df_plot['year_id'], df_plot['model_val_plus_resid_nte'], 'k:', label='Model Value + Residual Value (no time effect)')
                ax.plot(df_plot['year_id'], df_plot['model_val_plus_resid_te'], 'k-.', label='Model Value + Residual Value (time effect)')
                filtered_df_plot = df_plot[df_plot['year_id'] <= 2023]
                ax.plot(filtered_df_plot['year_id'], filtered_df_plot['residual_val_nte'], color='red', label='Residual Value (no time effect)')
                ax.plot(filtered_df_plot['year_id'], filtered_df_plot['residual_val_te'], color='red', label='Residual Value (time effect)')
                future_df_plot = df_plot[df_plot['year_id'] >= 2024]
                if not future_df_plot.empty:
                    ax.plot(future_df_plot['year_id'], future_df_plot['residual_val_nte'], 'r:', label='Residual Value (no time effect)')
                    ax.plot(future_df_plot['year_id'], future_df_plot['residual_val_te'], 'r-.', label='Residual Value (time effect)')
                ax.axvline(x=2023, color='black', linestyle='-', linewidth=1)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')
                ax.set_ylim(local_min, local_max)
                ax.set_title(f'Sex: {sex_label}, Age Group: {age_group_label}')
                ax.set_xlabel('Year')
                ax.set_ylabel('Value')
                #ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        handles, labels = [], []
        for value_column, color, linestyle in zip(['Model Value', 'GBD Value', 'Model Value + Residual Value (no time effect)', 'Model Value + Residual Value (time effect)', 'Residual Value (no time effect)', 'Residual Value (time effect)'], ['green', 'blue', 'black', 'black', 'red', 'red'], ['-', '-', ':', '-.', ':', '-.']):
            handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2))
            labels.append(value_column)

        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
        pdf.savefig(fig)
        plt.close(fig)

