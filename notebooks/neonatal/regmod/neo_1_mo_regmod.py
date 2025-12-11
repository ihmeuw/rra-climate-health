import pandas as pd
from regmod.data import Data
from regmod.variable import Variable
from regmod.models import GaussianModel
import os

# Load data and set constants/paths
summary_file = "nnm_1_mo_q9_summary"

results_dir = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/"
neo_version = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/training_data/2025_12_09.01/neonatal/neonatal_data.parquet"

data = pd.read_parquet(neo_version)


# Format data
df = Data(
    col_obs="child_mortality",
    col_covs=[
        "consumption_pd",
        "days_over_30C_prev_0_mo",
        "total_precipitation_prev_0_mo",
        "sex_id",
        "birth_year",
        "ihme_loc_id",
    ],
    # col_weights="weights",
    df=data,
)
