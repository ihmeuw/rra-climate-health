import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from lifelines import CoxPHFitter  # for Cox survival models
import os

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_09_15.01/data.parquet"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_09_15.01/"
os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

df = pd.read_parquet(DATA_PATH)

## 1. Get basic info about data
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

## 2. Make scatterplots and heatmaps based on raw data
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
heatmap_df = df.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(
    heatmap_df.ldipc_weighted_no_match, 10, retbins=True
)

for col in columns_to_bin:
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(
        heatmap_df.groupby(["consumption", f"{col}_bin"])["child_mortality"]
        .mean()
        .unstack(),
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
    )
    plt.title(f"Child Mortality by Consumption and {col.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"heatmap_child_mortality_{col}.png"))
    plt.close()


## 3. Make simple model
df.rename(columns={"ldipc_weighted_no_match": "consumption"}, inplace=True)
event_col = "child_mortality"
id_col = "indv_id"
time_col = "age_month_at_year_end"
covariate_cols = [
    "consumption",
    "ihme_loc_id",  # r.e. not yet supported in lifelines package
    "mean_temperature",
    # "total_precipitation",
    # "relative_humidity",
    # "mean_high_temperature",
    # "mean_low_temperature",
    # "precipitation_days",
    "days_over_30C",
    # "days_over_26C",
]
df_model_data = df[[event_col, time_col] + covariate_cols]

cph = CoxPHFitter()
cph.fit(df_model_data, duration_col=time_col, event_col=event_col)
cph.print_summary()

## 4. Plot data sources by country
