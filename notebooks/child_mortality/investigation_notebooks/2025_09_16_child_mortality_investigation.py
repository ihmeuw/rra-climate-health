import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from lifelines import CoxPHFitter  # for Cox survival models
from pymer4.models import Lmer
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
ax = agg_yr.plot(x="int_year", y="unique_individuals", kind="bar", legend=False)
plt.title("Unique Individuals per Year")
# Only label every 5 years
years = agg_yr["int_year"].values
xticks_idx = [i for i, y in enumerate(years) if (y - years[0]) % 5 == 0]
ax.set_xticks(xticks_idx)
ax.set_xticklabels(
    [str(years[i]) for i in xticks_idx], fontsize=8, rotation=45, ha="right"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ",")))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, f"unique_indv_children_per_year.png"))
plt.close()

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

# Head maps of variables
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
    heatmap_data = (
        heatmap_df.groupby(["consumption", f"{col}_bin"])["child_mortality"]
        .mean()
        .unstack()
    )
    ax1 = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
    )
    plt.title(f"Child Mortality by Consumption and {col.replace('_', ' ').title()}")

    # Set rounded axis labels
    ax1.set_xticklabels(
        [f"{int(x.left)}–{int(x.right)}" for x in heatmap_data.columns],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax1.set_yticklabels(
        [f"{int(y.left)}–{int(y.right)}" for y in heatmap_data.index],
        rotation=0,
        fontsize=10,
    )

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

# Attempt logistic regression for end result on lifetime mean temperatures and
# cumulative days over 30C based on subset of data.

## modify data

# remove children who were less than 5 at survey and did not die
children_died_ids = df.query("child_alive == 0")["indv_id"].unique()
children_alive = df.query("indv_id not in @children_died_ids")
children_alive["age_month_original"] = children_alive["age_month_original"].astype(int)
children_alive["over_5_at_survey"] = children_alive["age_month_original"] > 60
under5_at_survey = children_alive[~children_alive["over_5_at_survey"]][
    "indv_id"
].unique()

df_subset = df[~df["indv_id"].isin(under5_at_survey)]  # dropping 248068
df_subset.rename(columns={"ldipc_weighted_no_match": "consumption"}, inplace=True)
df_subset_agg = df_subset.groupby(["indv_id", "sex_id", "ihme_loc_id"]).agg(
    {
        "child_mortality": "max",  # if they died at any point
        "consumption": "mean",
        "days_over_30C": "sum",
        "mean_temperature": "mean",
        "age_month_at_year_end": "max",
    }
)
df_subset_agg = df_subset_agg.reset_index()

# check dups
df_subset_agg_duplicated = df_subset_agg[
    df_subset_agg["indv_id"].duplicated(keep=False)
]
df_subset_agg_duplicated = df_subset_agg_duplicated.sort_values("indv_id")
# These are problematic rows that should be dropped

df_subset_agg = df_subset_agg.drop_duplicates(subset=["indv_id"], keep="first")

assert len(df_subset_agg) == df_subset_agg["indv_id"].nunique()
df_subset_sample = df_subset_agg.sample(n=10000, random_state=42)
df_subset_sample.rename(
    columns={
        "mean_temperature": "mean_temperature_lifetime",
        "days_over_30C": "total_days_over_30C_lifetime",
    },
    inplace=True,
)

model = Lmer(
    "child_mortality ~ consumption + "
    "sex_id + "
    "mean_temperature_lifetime + "
    "total_days_over_30C_lifetime + "
    "(1|ihme_loc_id)",
    data=df_subset_sample,
)

model.fit()
print(model.summary())
# Linear mixed model fit by REML [’lmerMod’]
# Formula: child_mortality~consumption+sex_id+mean_temperature_lifetime+total_days_over_30C_lifetime+(1|ihme_loc_id)

# Family: gaussian	 Inference: parametric

# Number of observations: 10000	 Groups: {'ihme_loc_id': 15.0}

# Log-likelihood: 1123.654 	 AIC: -2233.308

# Random effects:

#                     Name    Var    Std
# ihme_loc_id  (Intercept)  0.001  0.030
# Residual                  0.046  0.215

# No random effect correlations specified

# Fixed effects:

#                               Estimate  2.5_ci  97.5_ci     SE        DF  \
# (Intercept)                      0.060   0.029    0.090  0.016   109.955
# consumption                     -0.000  -0.000   -0.000  0.000  9700.429
# sex_id                          -0.009  -0.017   -0.001  0.004  9984.652
# mean_temperature_lifetime        0.001  -0.000    0.002  0.001   396.674
# total_days_over_30C_lifetime    -0.000  -0.000   -0.000  0.000  3740.613

#                               T-stat  P-val  Sig
# (Intercept)                    3.825  0.000  ***
# consumption                   -6.676  0.000  ***
# sex_id                        -2.085  0.037    *
# mean_temperature_lifetime      1.958  0.051    .
# total_days_over_30C_lifetime  -7.648  0.000  ***
