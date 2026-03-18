import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from lifelines import CoxPHFitter  # for Cox survival models
from pymer4.models import Lmer
import os
from datetime import date

DATE_STR = date.today().strftime("%Y_%m_%d")

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_08.01/data.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_08.01/"
os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

df = pd.read_parquet(DATA_PATH)

# Prep data
df["line_id"] = df["line_id"].astype(int)
df["indv_id"] = df[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
print(f"{df['indv_id'].nunique():,} unique individuals in data")

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df["child_mortality"] = 1 - df["child_alive"]

df.rename(columns={"ldipc_weighted_no_match": "consumption"}, inplace=True)


event_col = "child_mortality"
id_col = "indv_id"
time_col = "age_year_at_year_end"
covariate_cols = [
    "consumption",
    "ihme_loc_id",  # r.e. not yet supported in lifelines package
    # "mean_temperature",
    # "total_precipitation",
    # "relative_humidity",
    # "mean_high_temperature",
    # "mean_low_temperature",
    # "precipitation_days",
    "days_over_30C",
    # "days_over_26C",
    "sex_id",
]
df_model_data = df[[event_col, time_col] + covariate_cols]


# One-hot encode ihme_loc_id
df_model_data = pd.get_dummies(df_model_data, columns=["ihme_loc_id"], drop_first=True)
df_model_data["sex_id"] = (
    df_model_data["sex_id"].map({"1": "Male", "2": "Female"}).astype("category")
)
df_model_data = pd.get_dummies(df_model_data, columns=["sex_id"], drop_first=True)

# Fit Cox model with fixed effects for countries
cph = CoxPHFitter()
cph.fit(df_model_data, duration_col=time_col, event_col=event_col)
cph.print_summary()

# Save the model summary as text
with open(RESULTS_PATH + "fe_model_do30_summary" + DATE_STR + ".txt", "w") as f:
    f.write(str(cph.summary))

# Predict survival function only at each individual's observed time
pred_surv = []
for i in range(len(df_model_data)):
    t = df_model_data.iloc[i][time_col]
    surv = cph.predict_survival_function(df_model_data.iloc[[i]], times=[t]).values[0][
        0
    ]
    pred_surv.append(surv)

df_model_data["predicted_survival"] = pred_surv
df_model_data["predicted_mortality"] = 1 - df_model_data["predicted_survival"]

# Save predictions
df_model_data.to_csv(
    RESULTS_PATH + "fe_model_do30_predictions_" + DATE_STR + ".csv", index=False
)
