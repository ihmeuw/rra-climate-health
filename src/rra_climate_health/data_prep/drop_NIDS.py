import pandas as pd

exclusion_list = pd.read_csv(
    "/ihme/homes/elyeb/rank_investigation/coarse_wealth_nids_all_causes.csv"
)

exclusion_list = exclusion_list["nid"].to_list()
full_data = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_08_24.01/data.parquet"
)

outpath = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_09_01.01/data.parquet"

# Remove any missing consumption_pd values
assert (
    len(full_data[full_data["consumption_pd_cumul"].isnull()]) == 0
), "There are missing consumption_pd_cumul values in the data."
full_data = full_data[full_data["consumption_pd_cumul"].notnull()]

# Drop rows with NIDs in the exclusion list
full_data = full_data[~full_data["nid"].isin(exclusion_list)]
full_data.to_parquet(outpath, index=False)
