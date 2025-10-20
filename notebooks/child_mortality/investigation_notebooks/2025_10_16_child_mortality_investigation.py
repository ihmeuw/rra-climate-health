import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from lifelines import CoxPHFitter  # for Cox survival models
from pymer4.models import Lmer
import os

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_16.01/data.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_16.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_16.01/"

os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)


## FUNCTIONS:
def plot_indvs_by_age_group(data):
    # Get individuals per age_month
    if "indv_id" not in data.columns:
        data = data.copy()
        data["indv_id"] = (
            data[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
        )
    agg_age = (
        data.groupby(["age_month"])["indv_id"]
        .nunique()
        .reset_index()
        .rename(columns={"indv_id": "unique_individuals"})
    )
    plt.figure(figsize=(30, 5))
    ax = agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False)
    plt.title("Unique Individuals per Age Month in Raw Data")
    months = agg_age["age_month"].values
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.tight_layout()


def plot_heat_map(
    data: pd.DataFrame,
    outfile: str,
    title: str,
    bin_cols: list,
    format: str = ".2f",
    vmin: float = None,
    vmax: float = None,
):

    heatmap_df = data.copy()
    for col in bin_cols:
        heatmap_df[f"{col}_bin"] = pd.qcut(
            heatmap_df[col], 10, retbins=False, duplicates="drop"
        )
    heatmap_df["consumption"], ldi_bins = pd.qcut(
        heatmap_df.consumption, 10, retbins=True, duplicates="drop"
    )

    # Create a discrete colormap with steps matching your rounded values
    # bounds = np.round(np.arange(vmin, vmax + 0.01, 0.01), 5)  # steps of 0.0001
    # norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    for col in columns_to_bin:
        plt.figure(figsize=(10, 8))
        heatmap_data = (
            heatmap_df.groupby(["consumption", f"{col}_bin"])["model_predictions"]
            .mean()
            .unstack()
        )

        if vmin and vmax:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
                vmin=vmin,
                vmax=vmax,
                # norm=norm,
            )
        else:
            ax1 = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=format,
                cmap="YlOrBr",
            )
        plt.title(
            f"{title}\nby Consumption and {col.replace('_', ' ').title()}",
            fontsize=20,
        )

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
        ax1.set_xlabel("Binned " + col.replace("_", " ").title(), fontsize=16)
        ax1.set_ylabel("Consumption Bin", fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"{outfile}_{col}.png"))
        plt.close()


## READ IN DATA

# Raw data
df = pd.read_parquet(DATA_PATH)
df.rename(columns={"ldipc_weighted_no_match": "consumption"}, inplace=True)

# Modeled data
# df_model = pd.read_parquet(
#     RESULTS_PATH + "predictions_subset_100pct_model_do30.parquet"
# )
df_model = pd.read_parquet(
    RESULTS_PATH + "predictions_subset_100pct_model_do30.parquet"
)

# Fixed effects model results
df_model_fe = df_model.copy()
df_model_fe.rename(columns={"mortality_fe_manual": "model_predictions"}, inplace=True)

# Mixed effects model results
df_model_me = df_model.copy()
df_model_me.rename(columns={"mortality_me_manual": "model_predictions"}, inplace=True)

# Neonatal predictions
# df_neo = pd.read_parquet(RESULTS_PATH + "neonatal/neonatal_mortality_1_mo.parquet")
df_neo = pd.read_parquet(
    RESULTS_PATH + "neonatal/neonatal_mortality_subset_100pct_model_do30.parquet"
)

# Fixed effects model results
df_neo_fe = df_neo.copy()
df_neo_fe.rename(columns={"mortality_fe_manual": "model_predictions"}, inplace=True)

# Mixed effects model results
df_neo_me = df_neo.copy()
df_neo_me.rename(columns={"mortality_me_manual": "model_predictions"}, inplace=True)


## CONSTANTS

# Heat maps of variables
columns_to_bin = [
    # "mean_temperature",
    # "total_precipitation",
    # "relative_humidity",
    # "mean_high_temperature",
    # "mean_low_temperature",
    # "precipitation_days",
    "days_over_30C",
    # "days_over_26C",
]


## Make bar plot of death rates by year
by_year = df.groupby("int_year")["child_mortality"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=by_year, x="int_year", y="child_mortality", color="skyblue")
plt.title("Child Mortality Rate by Year", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Child Mortality Rate", fontsize=14)
plt.ylim(0, by_year["child_mortality"].max() * 1.1)
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, f"child_mortality_by_year_10_16.png"))
plt.close()

# Make bar plot of death rates by age in months
by_age = df.groupby("age_month")["child_mortality"].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=by_age, x="age_month", y="child_mortality", color="skyblue")
plt.title("Child Mortality Rate by Age in Months", fontsize=16)
plt.xlabel("Age in Months", fontsize=14)
plt.ylabel("Child Mortality Rate", fontsize=14)
plt.ylim(0, by_age["child_mortality"].max() * 1.1)
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, f"child_mortality_by_age_month.png"))
plt.close()
# The above must be wrong, but also curious why it is so different from below.

# Try the above by first exploding by month for every obs
# reduce raw data to max age
df_max_age = df.copy()

# create list of months between 1 and the age_month.
df_max_age["months_list"] = df_max_age["age_month"].apply(
    lambda x: list(range(1, x + 1))
)

# explode data on months_list
df_exploded = df_max_age.explode("months_list").reset_index(drop=True)

# All preultimate age_month should have child_mortality = 0
df_exploded["months_list"] = df_exploded["months_list"].astype(int)
df_exploded["age_month"] = df_exploded["age_month"].astype(int)
df_exploded.loc[
    (df_exploded["child_mortality"] == 1)
    & (df_exploded["months_list"] < df_exploded["age_month"]),
    "child_mortality",
] = 0

# override age_month
df_exploded.drop(columns=["age_month"], inplace=True)
df_exploded.rename(columns={"months_list": "age_month"}, inplace=True)

by_age = df_exploded.groupby("age_month")["child_mortality"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=by_age, x="age_month", y="child_mortality", color="skyblue")
plt.title("Child Mortality Rate by Age in Months (exploded)", fontsize=16)
plt.xlabel("Age in Months", fontsize=14)
plt.ylabel("Child Mortality Rate", fontsize=14)
plt.ylim(0, by_age["child_mortality"].max() * 1.1)
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, f"child_mortality_by_age_month_exploded_10_14.png"))
plt.close()

# save neonatal raw data
# neonatal = df_exploded[df_exploded["age_month"] == 1]
# neonatal.to_parquet(
#     "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_16.01/neonatal.parquet"
# )

# or read it in
neonatal = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_16.01/neonatal.parquet"
)

# Get individuals in surveys by year
# get unique individuals per year
agg_yr = (
    df.groupby(["int_year"])["indv_id"]
    .nunique()
    .reset_index()
    .rename(columns={"indv_id": "unique_individuals"})
)
plt.figure(figsize=(25, 5))
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

# Get individuals per age_month
agg_age = (
    df.groupby(["age_month"])["indv_id"]
    .nunique()
    .reset_index()
    .rename(columns={"indv_id": "unique_individuals"})
)
plt.figure(figsize=(30, 5))
ax = agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False)
plt.title("Unique Individuals per Age Month")
months = agg_age["age_month"].values
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ",")))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, f"unique_indv_children_per_age_month.png"))
plt.close()

# Make side-by-side scatters for overall data (on left) vs neonatal (on right)

# scatter vars: consumption, mean_temperature, days_over_30C, sex_id, ihme_loc_id, int_year

# differences between me and fe
px.scatter(df_model, x="mortality_me_manual", y="mortality_fe_manual")
px.scatter(
    df_model,
    x="mortality_me_manual",
    y="mortality_fe_manual",
    color="days_over_30C",
    color_continuous_scale=["white", "orange", "darkred"],
    labels={"days_over_30C": "Days > 30°C"},
    title="Mortality: Mixed vs Fixed Effects Colored by Days Over 30°C",
)

# by sex
px.scatter(
    df_model,
    x="mortality_me_manual",
    y="days_over_30C",
    color="sex_id",
    labels={"sex_id": "Sex"},
)
# by location
px.scatter(
    df_model,
    x="mortality_me_manual",
    y="days_over_30C",
    color="ihme_loc_id",
)
px.scatter(df_model, x="mortality_fe_manual", y="days_over_30C")

px.scatter(df_model, x="mortality_me_manual", y="mean_temperature")
px.scatter(df_model, x="mortality_fe_manual", y="mean_temperature")
px.scatter(df_model, x="mortality_me_manual", y="consumption")
px.scatter(df_model, x="mortality_fe_manual", y="consumption")

# color actual mortality
px.scatter(
    df_model,
    x="mortality_me_manual",
    y="days_over_30C",
    color="child_mortality",
    color_continuous_scale=["green", "red"],
    labels={"child_mortality": "Child Mortality"},
)


columns_to_corr = [
    "mean_temperature",
    "total_precipitation",
    "relative_humidity",
    "precipitation_days",
    "days_over_30C",
    "days_over_26C",
]


df_loc_group = (
    df_model.groupby("ihme_loc_id")
    .agg(
        {
            "child_mortality": "mean",
            "mortality_me_manual": "mean",
            "mortality_fe_manual": "mean",
            "days_over_30C": "mean",
            "mean_temperature": "mean",
            "total_precipitation": "mean",
            "relative_humidity": "mean",
            "mean_high_temperature": "mean",
            "mean_low_temperature": "mean",
            "precipitation_days": "mean",
            "days_over_26C": "mean",
            "consumption": "mean",
        }
    )
    .reset_index()
)

px.scatter(
    df_loc_group, x="child_mortality", y="mortality_me_manual", color="ihme_loc_id"
)
px.scatter(
    df_loc_group, x="child_mortality", y="mortality_fe_manual", color="ihme_loc_id"
)
# days over 30
px.scatter(df_loc_group, x="child_mortality", y="days_over_30C", color="ihme_loc_id")
px.scatter(
    df_loc_group, x="mortality_me_manual", y="days_over_30C", color="ihme_loc_id"
)
px.scatter(
    df_loc_group, x="mortality_fe_manual", y="days_over_30C", color="ihme_loc_id"
)
# mean temp
px.scatter(df_loc_group, x="child_mortality", y="mean_temperature", color="ihme_loc_id")
px.scatter(
    df_loc_group, x="mortality_me_manual", y="mean_temperature", color="ihme_loc_id"
)
px.scatter(
    df_loc_group, x="mortality_fe_manual", y="mean_temperature", color="ihme_loc_id"
)
# consumption
px.scatter(df_loc_group, x="child_mortality", y="consumption", color="ihme_loc_id")
px.scatter(df_loc_group, x="mortality_me_manual", y="consumption", color="ihme_loc_id")
px.scatter(df_loc_group, x="mortality_fe_manual", y="consumption", color="ihme_loc_id")

# group by age-month
df_age_group = (
    df_model.groupby("age_month")
    .agg(
        {
            "child_mortality": "mean",
            "mortality_me_manual": "mean",
            "mortality_fe_manual": "mean",
            "days_over_30C": "mean",
            "mean_temperature": "mean",
            "total_precipitation": "mean",
            "relative_humidity": "mean",
            "mean_high_temperature": "mean",
            "mean_low_temperature": "mean",
            "precipitation_days": "mean",
            "days_over_26C": "mean",
            "consumption": "mean",
        }
    )
    .reset_index()
)
px.scatter(df_age_group, y="child_mortality", x="age_month")
px.scatter(df_age_group, y="mortality_me_manual", x="age_month")
px.scatter(df_age_group, y="mortality_fe_manual", x="age_month")
px.scatter(
    df_age_group, x="child_mortality", y="mortality_me_manual", color="age_month"
)
px.scatter(
    df_age_group, x="child_mortality", y="mortality_fe_manual", color="age_month"
)

## MAKE SIDE-BY-SIDE HEATMAPS TOGETHER

# get min and max values for color scale consistency across plots
# df_non_neo = df[df["age_month"] > 0]

# try new version of mortality: age_month/60 as a weight for numerator
# df["time_alived_weight"] = df["age_month"] / 60
# df["child_mortality_scaled"] = df["child_mortality"] * df["time_alived_weight"]

# df = df_model.copy()
df["child_mortality_scaled"] = 1 - (df["age_month"] / 60)  # 60 months = 5 years
df["child_mortality_scaled"] *= df["child_mortality"]  # 0 if alive, 1 if died
# df.drop(columns=["mortality_me_manual", "mortality_fe_manual"], inplace=True)

raw_heatmap_df = df.copy()
for col in columns_to_bin:
    raw_heatmap_df[f"{col}_bin"] = pd.qcut(
        raw_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
raw_heatmap_df["consumption"], ldi_bins = pd.qcut(
    raw_heatmap_df.consumption, 10, retbins=True
)

fe_heatmap_df = df_model_fe.copy()
for col in columns_to_bin:
    fe_heatmap_df[f"{col}_bin"] = pd.qcut(
        fe_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
fe_heatmap_df["consumption"], ldi_bins = pd.qcut(
    fe_heatmap_df.consumption, 10, retbins=True
)

me_heatmap_df = df_model_me.copy()
for col in columns_to_bin:
    me_heatmap_df[f"{col}_bin"] = pd.qcut(
        me_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
me_heatmap_df["consumption"], ldi_bins = pd.qcut(
    me_heatmap_df.consumption, 10, retbins=True
)

all_values = []
for data in [raw_heatmap_df, fe_heatmap_df, me_heatmap_df]:
    for col in columns_to_bin:
        if "model_predictions" in data.columns:
            vals = (
                data.groupby(["consumption", f"{col}_bin"])["model_predictions"]
                .mean()
                .values
            )
            print(f"model: {vals}")
            all_values.append(vals)
        elif "child_mortality_scaled" in data.columns:
            vals = (
                data.groupby(["consumption", f"{col}_bin"])["child_mortality_scaled"]
                .mean()
                .values
            )
            print(f"transformed: {vals}")
            all_values.append(vals)
        elif "child_mortality" in data.columns:
            vals = (
                data.groupby(["consumption", f"{col}_bin"])["child_mortality"]
                .mean()
                .values
            )
            print(f"raw: {vals}")
            all_values.append(vals)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

# plot raw data heatmaps with consistent color scale
plot_heat_map(
    data=df.rename(
        columns={
            "child_mortality_scaled": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_16_6",
    title="Raw Data Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot fe model heatmaps with consistent color scale
plot_heat_map(
    data=df_model_fe.copy(),
    outfile="fe_100pc_do30_heatmap_child_mortality",
    title="Modeled Child Mortality without Random Effects",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot me model heatmaps with consistent color scale
plot_heat_map(
    data=df_model_me.copy(),
    outfile="me_100pc_do30_heatmap_child_mortality",
    title="Modeled Child Mortality with Random Effects",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

## Neonatal
neonatal.rename(columns={"ldipc_weighted_no_match": "consumption"}, inplace=True)

# get min and max values for color scale consistency across plots
# df_non_neo = df[df["age_month"] > 0]
raw_heatmap_df = neonatal.copy()
# raw_heatmap_df = df_exploded[df_exploded["age_month"] == 1]
for col in columns_to_bin:
    raw_heatmap_df[f"{col}_bin"] = pd.qcut(
        raw_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
raw_heatmap_df["consumption"], ldi_bins = pd.qcut(
    raw_heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)

me_heatmap_df = df_neo_me.copy()
for col in columns_to_bin:
    me_heatmap_df[f"{col}_bin"] = pd.qcut(
        me_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
me_heatmap_df["consumption"], ldi_bins = pd.qcut(
    me_heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)

fe_heatmap_df = df_neo_fe.copy()
for col in columns_to_bin:
    fe_heatmap_df[f"{col}_bin"] = pd.qcut(
        fe_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
fe_heatmap_df["consumption"], ldi_bins = pd.qcut(
    fe_heatmap_df.consumption, 10, retbins=True, duplicates="drop"
)

all_values = []
for data in [raw_heatmap_df, me_heatmap_df, fe_heatmap_df]:
    for col in columns_to_bin:
        if "model_predictions" in data.columns:
            append_val = (
                data.groupby(["consumption", f"{col}_bin"])["model_predictions"]
                .mean()
                .values
            )
            print(append_val)
            all_values.append(append_val)
        elif "child_mortality" in data.columns:
            append_val = (
                data.groupby(["consumption", f"{col}_bin"])["child_mortality"]
                .mean()
                .values
            )
            print(append_val)
            all_values.append(append_val)

all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

plot_heat_map(
    data=neonatal.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_neonatal_child_mortality_10_17",
    title="Raw Data Neonatal Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=df_neo_fe.copy(),
    outfile="neo_heatmap_child_mortality_100pc_do30_fe_10_17",
    title="Neonatal Modeled Child Mortality (without random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

plot_heat_map(
    data=df_neo_me.copy(),
    outfile="neo_heatmap_child_mortality_100pc_do30_me_10_17",
    title="Neonatal Modeled Child Mortality (with random effects)",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

## Explore raw data
df_raw = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_matched_10_06_2025.parquet"
)
key_vars = [
    "nid",
    "psu",
    "birth_year",
    "birth_month",
    "int_year",
    "int_month",
    "age_month",
    "hh_id",
    "geospatial_id",
    "line_id",
    "lat",
    "long",
    "child_alive",
]
df_raw.dropna(subset=key_vars, inplace=True)
df_raw["age_month"] = df_raw["age_month"].astype(int)
df_raw = df_raw[df_raw["age_month"] <= 60]
df_raw["indv_id"] = (
    df_raw[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
)

# Get individuals per age_month
agg_age = (
    df_raw.groupby(["age_month"])["indv_id"]
    .nunique()
    .reset_index()
    .rename(columns={"indv_id": "unique_individuals"})
)
plt.figure(figsize=(30, 5))
ax = agg_age.plot(x="age_month", y="unique_individuals", kind="bar", legend=False)
plt.title("Unique Individuals per Age Month in Raw Data")
months = agg_age["age_month"].values
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ",")))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, f"unique_indv_children_per_age_month_raw_data.png"))
plt.close()

## Plot predictions by country, me and fe, in bar charts
# Calculate mean predictions by country
country_preds = (
    df_model.groupby("ihme_loc_id")
    .agg({"mortality_fe_manual": "mean", "mortality_me_manual": "mean"})
    .reset_index()
)

# Sort by mixed effects prediction for better visualization
country_preds = country_preds.sort_values("mortality_me_manual", ascending=False)
# country_preds[['mortality_fe_manual', 'mortality_me_manual']]
# Create figure
fig, ax = plt.subplots(figsize=(20, 8))

# Set up bar positions
x = np.arange(len(country_preds))
width = 0.35

# Create bars
bars1 = ax.bar(
    x - width / 2,
    country_preds["mortality_fe_manual"],
    width,
    label="Fixed Effects",
    color="skyblue",
    alpha=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    country_preds["mortality_me_manual"],
    width,
    label="Mixed Effects (with frailty)",
    color="coral",
    alpha=0.8,
)

# Customize plot
ax.set_xlabel("Country (ihme_loc_id)", fontsize=14)
ax.set_ylabel("Average Mortality Probability", fontsize=14)
ax.set_title(
    "Comparison of Fixed Effects vs Mixed Effects Predictions by Country", fontsize=16
)
ax.set_xticks(x)
ax.set_xticklabels(country_preds["ihme_loc_id"], rotation=90, ha="right", fontsize=8)
ax.legend(fontsize=12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# Add grid for easier reading
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "fe_vs_me_predictions_by_country.png"), dpi=300)
plt.close()

## Plot scatterplot between two predictions
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=country_preds,
    x="mortality_fe_manual",
    y="mortality_me_manual",
    hue="ihme_loc_id",
    s=100,
    palette="tab20",
)
plt.plot([0, 0.2], [0, 0.2], color="gray", linestyle="--")  # 45-degree line
plt.xlabel("Fixed Effects Prediction", fontsize=14)
plt.ylabel("Mixed Effects Prediction", fontsize=14)
plt.title("Scatterplot of Fixed vs Mixed Effects Predictions by Country", fontsize=16)
plt.xlim(
    0, country_preds[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1
)
plt.ylim(
    0, country_preds[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1
)
plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.legend(
    title="Country (ihme_loc_id)",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8,
)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "fe_vs_me_predictions_scatter_by_country.png"), dpi=300
)
plt.close()

## Plot scatterplot between two predictions for full dataset
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_model,
    x="mortality_fe_manual",
    y="mortality_me_manual",
    hue="ihme_loc_id",
    s=20,
    alpha=0.5,
    palette="tab20",
)
plt.plot([0, 0.2], [0, 0.2], color="gray", linestyle="--")  # 45-degree line
plt.xlabel("Fixed Effects Prediction", fontsize=14)
plt.ylabel("Mixed Effects Prediction", fontsize=14)
plt.title(
    "Scatterplot of Fixed vs Mixed Effects Predictions for Individuals", fontsize=16
)
plt.xlim(0, df_model[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1)
plt.ylim(0, df_model[["mortality_fe_manual", "mortality_me_manual"]].max().max() * 1.1)
plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "fe_vs_me_predictions_scatter_for_individuals.png"), dpi=300
)
plt.close()

## Get correlation between country effects and days over 30
cntry_effects = pd.read_csv(
    RESULTS_PATH + "model_summaries/frailty_estimates_subset_05pct_model_do30.csv"
)
df_do30 = df_model.groupby("ihme_loc_id")["days_over_30C"].mean().reset_index()
cntry_effects = cntry_effects.merge(df_do30, on="ihme_loc_id", how="left")
cntry_effects.rename(columns={"frailty": "country_frailty"}, inplace=True)
cntry_effects = pd.DataFrame(cntry_effects)
# scatter
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=cntry_effects,
    x="days_over_30C",
    y="country_frailty",
    hue="ihme_loc_id",
    s=100,
    palette="tab20",
)
plt.xlabel("Average Days Over 30C", fontsize=14)
plt.ylabel("Country Frailty Estimate", fontsize=14)
plt.title(
    "Scatterplot of Country Frailty vs Average Days Over 30C by Country", fontsize=16
)
# plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ",")))
plt.legend(
    title="Country (ihme_loc_id)",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8,
)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_PATH, "country_frailty_vs_days_over_30C_scatter.png"), dpi=300
)
plt.close()
