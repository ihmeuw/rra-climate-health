import pandas as pd
from pathlib import Path


PREV_WEALTH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/DHS_wealth.parquet"
UPDATED_WEALTH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/DHS_wealth_3_26_2026.parquet"

df_prev = pd.read_parquet(PREV_WEALTH)
df_current = pd.read_parquet(UPDATED_WEALTH)

df_prev["nid"] = df_prev["nid"].astype(int)
df_current["nid"] = df_current["nid"].astype(int)

# examine differences in nids
prev_nids = set(df_prev["nid"].unique())
current_nids = set(df_current["nid"].unique())

# nids in previous but not in current
nids_removed = prev_nids - current_nids
# nids in current but not in previous
nids_added = current_nids - prev_nids

df_dif = df_current[df_current["nid"].isin(nids_added)]
df_dif["nid"].nunique()  # 12
df_dif["iso3"].unique()
"""
'UGA', 'SEN', 'MOZ', 'JOR', 'TJK', 'LSO', 'COD', 'NGA', 'ZMB',
       'AGO', 'MLI', 'BGD'
"""
df_dif["year_start"].min()  # 2019
df_dif["year_end"].max()  # 2024

df_dif.to_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/DHS_wealth_3_26_2026_new_add.parquet"
)


# Compare to raw mortality data
## 1. Load and format neonatal data from DEM_BR module
mortality_raw = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_matched_2025_10_14.parquet"
)

# drop rows with missing key variables
key_vars = [
    "nid",
    "psu",
    "birth_year",
    "birth_month",
    "age_month",
    "hh_id",
    "geospatial_id",
    "line_id",
    "lat",
    "long",
    "child_alive",
]
mortality_raw.dropna(subset=key_vars, inplace=True)

mortality_nids = set(mortality_raw["nid"].unique())

# check for new NIDs in mortality data
new_mortality_nids = df_dif["nid"].isin(mortality_nids).sum()  # 0
