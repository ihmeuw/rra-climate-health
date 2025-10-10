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

DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_08.01/data.parquet"
RESULTS_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"
# DATA_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.csv"
PLOT_PATH = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_08.01/"
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

# Fixed effects model results
df_model_data = pd.read_csv(RESULTS_PATH + "fe_model_predictions_2025_10_08.csv")

# Mixed effects model results
# df_me_25pc_model_data = pd.read_csv(RESULTS_PATH + "subset_25pct_model_results.csv")
df_me_model_data = pd.read_csv(RESULTS_PATH + "subset_25pct_model_do30_results.csv")


# Neonatal predictions
# df_neo = pd.read_parquet(RESULTS_PATH + "neonatal/neonatal_mortality_1_mo.parquet")
df_neo = pd.read_parquet(RESULTS_PATH + "neonatal/neonatal_mortality_1_mo_do30.parquet")


# df = pd.read_csv(DATA_PATH)

## 1. Get basic info about data
df["line_id"] = df["line_id"].astype(int)

df["indv_id"] = df[["nid", "psu", "hh_id", "line_id"]].astype(str).agg("_".join, axis=1)
print(f"{df['indv_id'].nunique():,} unique individuals in data")

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df["child_mortality"] = 1 - df["child_alive"]

## 2. Make scatterplots and heatmaps based on aggregated raw data
# Aggregate data
final_outcome_df = (
    df.groupby(["nid", "ihme_loc_id", "int_year"], as_index=False)
    .mean(numeric_only=True)
    .rename(
        columns={
            "ldipc_weighted_no_match": "consumption",
            "child_mortality": "total_mortality",
        }
    )
)

scatter_vars = [
    "consumption",
    "mean_temperature",
    "mean_low_temperature",
    "mean_high_temperature",
    "precipitation_days",
    "total_precipitation",
    "relative_humidity",
    "elevation",
    "days_over_26C",
    "days_over_27C",
    "days_over_28C",
    "days_over_29C",
    "days_over_30C",
    "days_over_31C",
    "days_over_32C",
    "days_over_33C",
]
# make indv scatters
for var in scatter_vars:
    plt.figure(figsize=(6, 4))
    plt.scatter(
        final_outcome_df[var], final_outcome_df["total_mortality"], s=10, alpha=0.7
    )
    plt.xlabel(var)
    plt.ylabel("total_mortality")
    plt.title(f"{var} vs total_mortality")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"scatter_child_mortality_{var}.png"))
    plt.close()

# make single scatter
fig, axes = plt.subplots(8, 2, figsize=(16, 32))
axes = axes.flatten()

for i, var in enumerate(scatter_vars):
    axes[i].scatter(
        final_outcome_df[var], final_outcome_df["total_mortality"], s=10, alpha=0.7
    )
    axes[i].set_xlabel(var, fontsize=12)
    axes[i].set_ylabel("total_mortality", fontsize=12)
    axes[i].set_title(f"{var} vs total_mortality", fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "scatter_child_mortality_all.png"))
plt.close()

px.scatter(
    final_outcome_df, x="mean_temperature", y="total_mortality"  # , color="ihme_loc_id"
)

px.scatter(
    final_outcome_df, x="mean_temperature", y="total_mortality", color="ihme_loc_id"
)

# Examine binary variables
ct = pd.crosstab(df["child_mortality"], df["sex_id"])
ct_prop = pd.crosstab(df["child_mortality"], df["sex_id"], normalize="index")
corr = df["child_mortality"].corr(df["sex_id"].astype(int))

# Heat maps of variables
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
    plt.title(
        f"Raw Data Child Mortality by Consumption\nand {col.replace('_', ' ').title()}",
        fontsize=20,
    )

    # Set rounded axis labels
    ax1.set_xticklabels(
        [f"{int(x.left)}–{int(x.right)}" for x in heatmap_data.columns],
        rotation=45,
        ha="right",
        fontsize=14,
    )
    ax1.set_yticklabels(
        [f"{int(y.left)}–{int(y.right)}" for y in heatmap_data.index],
        rotation=0,
        fontsize=14,
    )
    ax1.set_xlabel("Binned " + col.replace("_", " ").title(), fontsize=16)
    ax1.set_ylabel("Consumption Bin", fontsize=16)

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
    "sex_id",
]
df_model_data = df[[event_col, time_col] + covariate_cols]

# One-hot encode ihme_loc_id
df_model_data = pd.get_dummies(df_model_data, columns=["ihme_loc_id"], drop_first=True)
df_model_data["sex_id"] = (
    df_model_data["sex_id"].map({"1": "Male", "2": "Female"}).astype("category")
)
df_model_data = pd.get_dummies(df_model_data, columns=["sex_id"], drop_first=True)

# Fit Cox model with fixed effects for countries
cph = CoxPHFitter()
cph.fit(df_model_data, duration_col=time_col, event_col=event_col)
cph.print_summary()

# Save the model summary as text
# with open("coxph_model_summary.txt", "w") as f:
#     f.write(str(cph.summary))

# Predict survival function only at each individual's observed time
pred_surv = []
for i in range(len(df_model_data)):
    t = df_model_data.iloc[i][time_col]
    surv = cph.predict_survival_function(df_model_data.iloc[[i]], times=[t]).values[0][
        0
    ]
    pred_surv.append(surv)

df_model_data["predicted_survival"] = pred_surv
df_model_data["predicted_mortality"] = 1 - df_model_data["predicted_survival"]

# Save predictions
df_model_data.to_csv(RESULTS_PATH + "fe_model_predictions.csv", index=False)

## Heat maps of modeled predictions

# Read FE model back in and make plots

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
heatmap_df = df_model_data.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

for col in columns_to_bin:
    plt.figure(figsize=(10, 8))
    heatmap_data = (
        heatmap_df.groupby(["consumption", f"{col}_bin"])["predicted_mortality"]
        .mean()
        .unstack()
    )
    ax1 = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
    )
    plt.title(
        f"Fixed Effects Child Mortality by Consumption\nand {col.replace('_', ' ').title()}",
        fontsize=20,
    )

    # Set rounded axis labels
    ax1.set_xticklabels(
        [f"{int(x.left)}–{int(x.right)}" for x in heatmap_data.columns],
        rotation=45,
        ha="right",
        fontsize=14,
    )
    ax1.set_yticklabels(
        [f"{int(y.left)}–{int(y.right)}" for y in heatmap_data.index],
        rotation=0,
        fontsize=14,
    )
    ax1.set_xlabel("Binned " + col.replace("_", " ").title(), fontsize=16)
    ax1.set_ylabel("Consumption Bin", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"fe_heatmap_child_mortality_{col}.png"))
    plt.close()


# Read in 25% mixed effects model predictions and make plots

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
heatmap_df = df_me_50pc_model_data.copy()
for col in columns_to_bin:
    heatmap_df[f"{col}_bin"] = pd.qcut(
        heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
heatmap_df["consumption"], ldi_bins = pd.qcut(heatmap_df.consumption, 10, retbins=True)

for col in columns_to_bin:
    plt.figure(figsize=(10, 8))
    heatmap_data = (
        heatmap_df.groupby(["consumption", f"{col}_bin"])["model_predictions"]
        .mean()
        .unstack()
    )
    ax1 = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
    )
    plt.title(
        f"Mixed Effects (50% of individuals) Modeled Child Mortality\nby Consumption and {col.replace('_', ' ').title()}",
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
    plt.savefig(os.path.join(PLOT_PATH, f"me_50pc_heatmap_child_mortality_{col}.png"))
    plt.close()


# get min and max values for color scale consistency across plots
# df_non_neo = df[df["age_month"] > 0]
raw_heatmap_df = df.copy()
for col in columns_to_bin:
    raw_heatmap_df[f"{col}_bin"] = pd.qcut(
        raw_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
raw_heatmap_df["consumption"], ldi_bins = pd.qcut(
    raw_heatmap_df.ldipc_weighted_no_match, 10, retbins=True
)

fe_heatmap_df = df_model_data.copy()
for col in columns_to_bin:
    fe_heatmap_df[f"{col}_bin"] = pd.qcut(
        fe_heatmap_df[col], 10, retbins=False, duplicates="drop"
    )
fe_heatmap_df["consumption"], ldi_bins = pd.qcut(
    fe_heatmap_df.consumption, 10, retbins=True
)

me_heatmap_df = df_me_50pc_model_data.copy()
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
        if "child_mortality" in data.columns:
            all_values.append(
                data.groupby(["consumption", f"{col}_bin"])["child_mortality"]
                .mean()
                .values
            )
        elif "predicted_mortality" in data.columns:
            all_values.append(
                data.groupby(["consumption", f"{col}_bin"])["predicted_mortality"]
                .mean()
                .values
            )
        elif "model_predictions" in data.columns:
            all_values.append(
                data.groupby(["consumption", f"{col}_bin"])["model_predictions"]
                .mean()
                .values
            )
all_values = np.concatenate(all_values)
vmin = all_values.min()
vmax = all_values.max()

# plot raw data heatmaps with consistent color scale
plot_heat_map(
    data=df.rename(
        columns={
            "child_mortality": "model_predictions",
            "ldipc_weighted_no_match": "consumption",
        }
    ),
    outfile="raw_heatmap_child_mortality",
    title="Raw Data Child Mortality",
    bin_cols=columns_to_bin,
    vmin=vmin,
    vmax=vmax,
)
# plot fe model heatmaps with consistent color scale
plot_heat_map(
    data=df_model_data.rename(columns={"predicted_mortality": "model_predictions"}),
    outfile="fe_heatmap_child_mortality",
    title="FE Model Child Mortality",
    bin_cols=columns_to_bin,
    vmin=vmin,
    vmax=vmax,
)
# plot me model heatmaps with consistent color scale
plot_heat_map(
    data=df_me_model_data,
    outfile="me_25pc_do30_heatmap_child_mortality",
    title="ME (25% of individuals) Modeled Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
    # vmin=vmin,
    # vmax=vmax,
)

# Plot neonatal predictions
df_neo = df_neo.rename(columns={"mortality_1_mo": "model_predictions"})


plot_heat_map(
    data=df_neo,
    outfile="neo_heatmap_child_mortality",
    title="Neonatal Modeled Child Mortality",
    bin_cols=columns_to_bin,
    format=".3f",
)
