"""
Convert a csv or parquet dataframe to an xarray.
"""

import pandas as pd


INFILE = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/inference_format/consumption_smooth_lookup.parquet"

infile = pd.read_parquet(INFILE)


# Create xarray DataArray from climate_lookup
climate_lookup_xr = xr.DataArray(
    climate_lookup["smooth_contribution"].values,
    dims=["days_over_30C_prev_0_mo"],
    coords={
        "days_over_30C_prev_0_mo": climate_lookup["days_over_30C_prev_0_mo"].values
    },
)

climate_lookup_xr.to_netcdf(os.path.join(INFERENCE_DIR, "climate_smooth_lookup.nc"))


OUTFILE = INFILE.replace(".csv", ".parquet")
print(OUTFILE)
infile.to_parquet(OUTFILE, index=True)
