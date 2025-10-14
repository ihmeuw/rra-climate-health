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

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_13.01/data_avg_climate.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_13.01/"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_13.01/"
os.makedirs(PLOT_PATH, exist_ok=True, mode=0o777)

## FUNCTIONS:


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
        heatmap_df.consumption, 10, retbins=True
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

# Modeled data
# df_model = pd.read_parquet(RESULTS_PATH + "predictions_50pc_do30_fe.parquet")
# df_model = pd.read_csv(RESULTS_PATH + "predictions_subset_05pct_model_do30_sdi.csv")
df_model = pd.read_csv(RESULTS_PATH + "predictions_subset_25pct_model_do30_sdi.csv")

# Fixed effects model results
# df_model_data = pd.read_csv(RESULTS_PATH + "fe_model_predictions_2025_10_08.csv")
# df_model_data = pd.read_csv(RESULTS_PATH + "fe_model_do30_predictions_2025_10_10.csv")
df_model_fe = df_model.copy()
df_model_fe.rename(columns={"model_predictions_fe": "model_predictions"}, inplace=True)

# Mixed effects model results
# df_me_25pc_model_data = pd.read_csv(RESULTS_PATH + "subset_25pct_model_results.csv")
# df_me_model_data = pd.read_csv(RESULTS_PATH + "subset_5pct_model_do30_results.csv")
df_model_me = df_model.copy()
df_model_me.rename(columns={"model_predictions_me": "model_predictions"}, inplace=True)

# Neonatal predictions
# df_neo = pd.read_parquet(RESULTS_PATH + "neonatal/neonatal_mortality_1_mo.parquet")
df_neo = pd.read_parquet(RESULTS_PATH + "neonatal/neonatal_mortality_1_mo_do30.parquet")

## CONSTANTS

# Heat maps of variables
columns_to_bin = [
    "mean_temperature",
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
plt.savefig(os.path.join(PLOT_PATH, f"child_mortality_by_year_10_14.png"))
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
neonatal = df_exploded[df_exploded["age_month"] == 1]
neonatal.to_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_13.01/neonatal.parquet"
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
df_group = (
    df.groupby(["nid", "psu", "hh_id", "int_year"])[["child_mortality", "consumption"]]
    .mean()
    .reset_index()
)
px.scatter(df_group, x="consumption", y="child_mortality")


## MAKE SIDE-BY-SIDE PLOTS TOGETHER

# get min and max values for color scale consistency across plots
# df_non_neo = df[df["age_month"] > 0]
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
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_child_mortality_10_14",
    title="Raw Data Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot fe model heatmaps with consistent color scale
plot_heat_map(
    data=df_model_fe.copy(),
    outfile="fe_25pc_do30_heatmap_child_mortality_sdi",
    title="Modeled (25% of individuals) Child Mortality without Random Effects",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
# plot me model heatmaps with consistent color scale
plot_heat_map(
    data=df_model_me,
    outfile="me_25pc_do30_heatmap_child_mortality_sdi",
    title="Modeled (25% of individuals) Child Mortality with Random Effects",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)

## Neonatal

# get neonatal raw data
df_neo_raw = neonatal.copy()

# Plot neonatal predictions
df_neo = df_neo.rename(columns={"mortality_1_mo": "model_predictions"})


# get min and max values for color scale consistency across plots
# df_non_neo = df[df["age_month"] > 0]
# raw_heatmap_df = df_neo_raw.copy()
raw_heatmap_df = df_exploded[df_exploded["age_month"] == 1]
for col in columns_to_bin:
    raw_heatmap_df[f"{col}_bin"] = pd.qcut(
        raw_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
raw_heatmap_df["consumption"], ldi_bins = pd.qcut(
    raw_heatmap_df.consumption, 10, retbins=True
)

me_heatmap_df = df_neo.copy()
for col in columns_to_bin:
    me_heatmap_df[f"{col}_bin"] = pd.qcut(
        me_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
me_heatmap_df["consumption"], ldi_bins = pd.qcut(
    me_heatmap_df.consumption, 10, retbins=True
)

all_values = []
for data in [raw_heatmap_df, me_heatmap_df]:
    for col in columns_to_bin:
        if "model_predictions" in data.columns:
            append_val = (
                data.groupby(["consumption", f"{col}_bin"])["model_predictions"]
                .mean()
                .values
            )
            # print(append_val)
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
    data=df_neo_raw.rename(
        columns={
            "child_mortality": "model_predictions",
        }
    ),
    outfile="raw_heatmap_neonatal_child_mortality",
    title="Raw Data Neonatal Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    # vmin=vmin,
    # vmax=vmax,
)

plot_heat_map(
    data=df_neo,
    outfile="neo_heatmap_child_mortality_do30",
    title="Neonatal Modeled Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    vmin=vmin,
    vmax=vmax,
)
