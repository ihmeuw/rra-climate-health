"""
Create new data version with censored survivors removed and
7-year cutoff
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data.parquet"

# Raw data
df = pd.read_parquet(DATA_PATH)

# test
len(df.query("(age_month_original<60)&(child_mortality==0)")) / len(df)
# numb to remove
len(df.query("(age_month_original<60)&(child_mortality==0)"))  # 802560

expected_n = len(df) - len(df.query("(age_month_original<60)&(child_mortality==0)"))

df_out = df.query("(age_month_original>=60)|(child_mortality==1)").copy()

assert len(df_out) == expected_n

# also impose 7 year cutoff (84 months)
df_out = df_out.query("int_birth_year_diff_months <= 84")

df_out.to_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data_filtered.parquet"
)
