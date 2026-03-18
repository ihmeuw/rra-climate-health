"""
Since some months contain no events, convert the data to yearly format
by aggregating months into years.
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data.parquet"

# Raw data
df = pd.read_parquet(DATA_PATH)

# check count of mortality by age month
df_grouped = df.groupby("age_month")["child_mortality"].sum()

# create year column
df["age_year_recoded"] = (1 + df["age_month"] // 12).astype(int)
df_regrouped = df.groupby("age_year_recoded")["child_mortality"].sum()


df.to_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data_yearly.parquet"
)
