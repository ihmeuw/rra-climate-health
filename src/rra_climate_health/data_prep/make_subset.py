"""
Ad-hoc script to make subsets of data based on samples of countries and individuals.

"""

import os
import pandas as pd
from pathlib import Path


# Parameters
sample_percent = 0.02


data_version = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_03_19.04/data.parquet"
output_dir = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_03_19.05/"
os.makedirs(output_dir, exist_ok=True)

# Read and format data
df = pd.read_parquet(data_version)
df["ihme_loc_id"] = df["ihme_loc_id"].astype("category")

# Sample individuals per country, keeping all rows for sampled individuals
indv_dt = df[["indv_id", "ihme_loc_id"]].drop_duplicates()
indv_counts = indv_dt.groupby("ihme_loc_id", observed=True).size().reset_index(name="N")
indv_dt = indv_dt.merge(indv_counts, on="ihme_loc_id")
indv_dt["n_sample"] = (sample_percent * indv_dt["N"]).apply(int)

sampled_indv = indv_dt.groupby("ihme_loc_id", observed=True, group_keys=False).apply(
    lambda g: g.sample(n=g["n_sample"].iloc[0], random_state=42)
)["indv_id"]

df_sample = df[df["indv_id"].isin(sampled_indv)]

print(f"{len(df_sample):,} rows")
df_sample.to_parquet(Path(output_dir) / "data.parquet")


# Make second sample with interview time cutoff max 5 years
# Add int_birth_year_diff_months back into columns
df = pd.read_parquet(data_version)
output_dir = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_03_19.06/"
os.makedirs(output_dir, exist_ok=True)

df["int_birth_year_diff_months"] = df["age_month_original"]
df_sample = df[df["int_birth_year_diff_months"] <= 60]
df_sample.to_parquet(Path(output_dir) / "data.parquet")
