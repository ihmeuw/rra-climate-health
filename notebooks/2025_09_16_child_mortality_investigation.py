import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_09_15.01/data.parquet"

df = pd.read_parquet(DATA_PATH)

df["line_id"] = df["line_id"].astype(int)

df["indv_id"] = df[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
print(f"{df['indv_id'].nunique():,} unique individuals in data")

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df["child_mortality"] = 1 - df["child_alive"]

# get unique individuals per year
agg_yr = (
    df.groupby(["int_year"])["indv_id"]
    .nunique()
    .reset_index()
    .rename(columns={"indv_id": "unique_individuals"})
)
plt.figure(figsize=(20, 5))
ax = agg_yr.plot(x="int_year", y="unique_individuals", kind="bar")
plt.title("Unique Individuals per Year")
plt.xticks(fontsize=5, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ",")))
plt.tight_layout()
plt.show()

# get max age of children who did not die
children_died_ids = df.query("child_alive == 0")["indv_id"].unique()
children_alive = df.query("indv_id not in @children_died_ids")
children_alive["age_month_original"] = children_alive["age_month_original"].astype(int)
children_alive["over_5_at_survey"] = children_alive["age_month_original"] > 60
over5_at_survey = children_alive[children_alive["over_5_at_survey"]][
    "indv_id"
].nunique()
under5_at_survey = children_alive[~children_alive["over_5_at_survey"]][
    "indv_id"
].nunique()
print(
    f"{under5_at_survey:,} children out of {over5_at_survey + under5_at_survey:,} "
    f"who did not die were under 5 at survey ({under5_at_survey/(over5_at_survey + under5_at_survey):.1%})"
)

# Aggregate data
agg_df = (
    df.groupby(["nid", "ihme_loc_id", "int_year"], as_index=False)
    .mean(numeric_only=True)
    .rename(
        columns={
            "ldipc_weighted_no_match": "consumption",
            "child_mortality": "total_mortality",
        }
    )
)
px.scatter(agg_df, x="mean_temperature", y="total_mortality", color="ihme_loc_id")


columns_to_bin = [
    "mean_temperature",
    "total_precipitation",
    "relative_humidity",
    "mean_high_temperature",
    "mean_low_temperature",
    "precipitation_days",
    "days_over_30C",
    "days_over_26C",
]
heatmap_df = df.copy()  # .query("ihme_loc_id == 'ETH'")
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(
    heatmap_df.ldipc_weighted_no_match, 10, retbins=True
)
col = "days_over_30C"
sns.heatmap(
    heatmap_df.groupby(["consumption", f"{col}_bin"])["child_mortality"]
    .mean()
    .unstack(),
    annot=True,
    fmt=".2f",
    cmap="YlOrBr",
)
col = "mean_temperature"
sns.heatmap(
    heatmap_df.groupby(["consumption", f"{col}_bin"])["child_mortality"]
    .mean()
    .unstack(),
    annot=True,
    fmt=".2f",
    cmap="YlOrBr",
)
