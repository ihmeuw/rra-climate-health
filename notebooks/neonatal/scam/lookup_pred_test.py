import pandas as pd
import numpy as np
import os

import xarray as xr
import rasterra as rt
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from pathlib import Path


# 1. Load data and make functions  #############################################
################################################################################
DATA_DIR = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/"
INFERENCE_DIR = os.path.join(DATA_DIR, "inference_format")
summary_version = "nnm_1_mo_do30_scam_summary"

df_me = pd.read_parquet(
    os.path.join(DATA_DIR, "predictions_me_only_" + summary_version + ".parquet")
)
df_fe = pd.read_parquet(
    os.path.join(DATA_DIR, "predictions_" + summary_version + ".parquet")
)
me_categories = list(df_me["ihme_loc_id"].cat.categories)
df_fe["ihme_loc_id"] = pd.Categorical(["CAF"] * len(df_fe), categories=me_categories)

# Load lookup tables and coefficients
climate_lookup = pd.read_csv(os.path.join(INFERENCE_DIR, "climate_smooth_lookup.csv"))
consumption_lookup = pd.read_csv(
    os.path.join(INFERENCE_DIR, "consumption_smooth_lookup.csv")
)
linear_coefs = pd.read_csv(os.path.join(INFERENCE_DIR, "linear_coefficients.csv"))
random_effects = pd.read_csv(os.path.join(INFERENCE_DIR, "random_effects.csv"))


def interpolate_smooth(x_values, lookup_df, x_col, y_col):
    """Linear interpolation from lookup table"""
    return np.interp(
        x_values,
        lookup_df[x_col],
        lookup_df[y_col],
        left=lookup_df[y_col].iloc[0],  # extrapolate with boundary values
        right=lookup_df[y_col].iloc[-1],
    )


# Make predictions
def predict_mortality(new_data):
    # Interpolate smooth terms
    smooth_climate = interpolate_smooth(
        new_data["days_over_30C_prev_0_mo"],
        climate_lookup,
        "days_over_30C_prev_0_mo",
        "smooth_contribution",
    )

    smooth_consumption = interpolate_smooth(
        new_data["consumption_pd"],
        consumption_lookup,
        "consumption_pd",
        "smooth_contribution",
    )

    # Get coefficients
    intercept = linear_coefs[linear_coefs["variable"] == "(Intercept)"][
        "coefficient"
    ].values[0]
    beta_precip = linear_coefs[
        linear_coefs["variable"] == "total_precipitation_prev_0_mo"
    ]["coefficient"].values[0]
    beta_sex = linear_coefs[linear_coefs["variable"] == "sex_id"]["coefficient"].values[
        0
    ]

    # Get birth year effects
    # Convert birth_year to string to ensure consistent matching
    birth_year_effect = (
        new_data["birth_year"]
        .astype(str)
        .apply(
            lambda year: (
                linear_coefs[linear_coefs["variable"] == f"birth_year{year}"][
                    "coefficient"
                ].values[0]
                if f"birth_year{year}" in linear_coefs["variable"].values
                else 0.0
            )
        )
    )

    # Get random effects for ihme_loc_id
    # Convert to string first to handle categorical
    random_effect_dict = random_effects.set_index("ihme_loc_id")[
        "coefficient"
    ].to_dict()
    random_effect = (
        new_data["ihme_loc_id"]
        .astype(str)
        .map(random_effect_dict)
        .fillna(0.0)
        .astype(float)
    )

    # Calculate linear predictor
    linear_pred = (
        intercept
        + new_data["total_precipitation_prev_0_mo"] * beta_precip
        + new_data["sex_id"] * beta_sex
        + smooth_climate
        + smooth_consumption
        + birth_year_effect
        + random_effect
    )

    # Add birth year and random effects as needed
    # (simplified for brevity)

    # Inverse logit
    return 1 / (1 + np.exp(-linear_pred))


################################################################################

# 2. Test that predictions are similar to R predict() results  #################

# Test 1: predict on ME data
df_me["pred_me_interpolated"] = predict_mortality(df_me)

# save results
df_me.to_parquet(
    os.path.join(
        DATA_DIR, "predictions_me_only_interpolated_" + summary_version + ".parquet"
    )
)

# Load if it's already been run:
df_me = pd.read_parquet(
    os.path.join(
        DATA_DIR, "predictions_me_only_interpolated_" + summary_version + ".parquet"
    )
)

df_me["difference"] = df_me["pred_me_interpolated"] - df_me["pred_me"]
df_me["difference"].abs().max()  # 0.0000022015050127

# Test 2: predict on FE data
df_fe["pred_fe_interpolated"] = predict_mortality(df_fe)
df_fe["difference"] = df_fe["pred_fe_interpolated"] - df_fe["pred_fe"]
df_fe["difference"].abs().max()  # 0.0000005987208661559218


# save results
df_fe.to_parquet(
    os.path.join(
        DATA_DIR, "predictions_fe_only_interpolated_" + summary_version + ".parquet"
    )
)

################################################################################

# 3. Draft lookups for xarrays of climate data  ################################

# Notes: data is yearly avg DO30, whereas my variable is for birth month DO30
# However, methods should be the same.

cm_data = ClimateMalnutritionData(Path(DEFAULT_ROOT) / "stunting")
do30 = cm_data.load_climate_raster("days_over_30C", "ssp126", 2005, 0)

# alt monthly
do30 = xr.open_dataarray(
    "/mnt/share/erf/climate_downscale/results/monthly/raw/historical/days_over_30C/2005_era5.nc"
)
# Convert climate_lookup to xarray DataArray (FAST)
climate_lookup.head()

# Create xarray DataArray from climate_lookup
climate_lookup_xr = xr.DataArray(
    climate_lookup["smooth_contribution"].values,
    dims=["days_over_30C_prev_0_mo"],
    coords={
        "days_over_30C_prev_0_mo": climate_lookup["days_over_30C_prev_0_mo"].values
    },
)

climate_lookup_xr.to_netcdf(os.path.join(INFERENCE_DIR, "climate_smooth_lookup.nc"))

smooth_contributions = climate_lookup_xr.interp(
    days_over_30C_prev_0_mo=do30,
    method="nearest",  # Use nearest neighbor instead of linear
)
# something like the above should be saved out
smooth_contributions_df = smooth_contributions.to_dataframe(
    name="smooth_contribution"
).reset_index()
smooth_contributions_df["smooth_contribution"].min()  # = -0.0217827964849219
smooth_contributions_df["smooth_contribution"].max()  # = 0.142065473059706
