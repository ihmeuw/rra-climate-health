"""
model <- glmer(
  child_mortality ~ consumption_pd +
    days_over_30C_prev_0_mo +
    total_precipitation_prev_0_mo +
    sex_id +
    birth_year +
    (1 | ihme_loc_id),
  data = df_model,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)
"""

import pandas as pd
from mrtool import MRData, MRBRT, CovModel


# Load data and set constants/paths
summary_file = "nnm_1_mo_q9_summary"

results_dir = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/"
neo_version = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/training_data/2025_12_09.01/neonatal/neonatal_data.parquet"

data_raw = pd.read_parquet(neo_version)

# Format data
data_raw = data_raw[
    [
        "child_mortality",
        "consumption_pd",
        "days_over_30C_prev_0_mo",
        "total_precipitation_prev_0_mo",
        "sex_id",
        "birth_year",
        "ihme_loc_id",
    ]
].dropna()

# Set up MR-BRT
data = MRData()
data.load_df(
    data=data_raw,
    col_obs="child_mortality",  # Dependent variable (outcome)
    col_covs=[
        "consumption_pd",
        "days_over_30C_prev_0_mo",
        "total_precipitation_prev_0_mo",
        "sex_id",
        "birth_year",
    ],  # Covariates
    col_study_id="ihme_loc_id",  # Random effect grouping variable
)

cov_models = [
    CovModel("consumption_pd", use_spline=True),
    CovModel("days_over_30C_prev_0_mo", use_spline=True),
    CovModel("total_precipitation_prev_0_mo"),
    CovModel("sex_id"),
    CovModel("birth_year"),
]


# Set up the MR-BRT model
model = MRBRT(
    data=data,  # Pass the MRData object
    cov_models=cov_models,  # Covariate models
)

model.fit_model()


model = MRBRT(
    data=data_raw,
    col_obs="child_mortality",  # Binary outcome
    covariates=[
        {"name": "consumption_pd", "type": "spline", "spline_info": spline_cov1},
        {
            "name": "days_over_30C_prev_0_mo",
            "type": "spline",
            "spline_info": spline_cov2,
        },
    ],
    study_id="ihme_loc_id",  # Random effect on location
    family="binomial",  # Logistic regression
)

# Fit the model
model.fit_model()

# (Optional) Plot the results
model.plot()

# (Optional) Access results
results = model.summary()
print(results)


predictions = model.predict(data)
