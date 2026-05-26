"""
Format neonatal and child mortality as required for training pipeline.
"""

import pandas as pd
import os

##  Neonatal ###################################################################
# 5/26/2026
# check neonatal for missing outcome vars
neo_path = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_05_20.01/data.parquet"
neo_df = pd.read_parquet(neo_path)

# load corrected neonatal outcome var
neo_corrected_path = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_05_19.01/neonatal_data_old_climate_vars.parquet"
neo_corrected_df = pd.read_parquet(neo_corrected_path)
neo_corrected_df = neo_corrected_df[["indv_id", "neonatal_mortality"]]

neo_df.drop(columns=["neonatal_mortality"], inplace=True)
neo_df = neo_df.merge(neo_corrected_df, on="indv_id", how="left")
neo_df["neonatal_mortality"].isna().sum()  # 0 missing neonatal mortality values
neo_df["neonatal_mortality"].value_counts()


outpath = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_05_20.01/data.parquet"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
neo_df.to_parquet(outpath, index=False)

raw_df_path = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_05_19.01/data.parquet"
raw_df = pd.read_parquet(raw_df_path)

# compare against previous formatted training data
prev_fmt_path = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_03_19.06/data.parquet"
prev_fmt_df = pd.read_parquet(prev_fmt_path)

prev_fmt_df["int_birth_year_diff_months"].max()  # 60

# format
# raw_df.rename(columns={"child_mortality": "neonatal_mortality"}, inplace=True)
raw_df["int_birth_year_diff_months"].max()  # 532
raw_df = raw_df[raw_df["int_birth_year_diff_months"] <= 60]

# save formatted data
raw_df.to_parquet(outpath, index=False)
