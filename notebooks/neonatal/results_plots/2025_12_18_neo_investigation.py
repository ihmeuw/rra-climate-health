"""
Map the various variables of interest by lat-long, rather than country.

1. Group data by lat-long and calculate mean of variables.
2. Map mortality
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

df = pd.read_parquet(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2025_12_16.01/neonatal_threshold_averages.parquet"
)


# 1. Group data by lat-long and calculate mean of variables. ###################

df_grouped = (
    df.groupby(["lat", "long"])
    .agg(
        {
            "indv_id": "count",
            "child_mortality": "mean",
            "mean_temperature_prev_0_mo": "mean",
            "days_over_30C_prev_0_mo": "mean",
            "consumption_pd": "mean",
            "q75_prev_0_mo": "mean",
            "q8_prev_0_mo": "mean",
            "q85_prev_0_mo": "mean",
            "q9_prev_0_mo": "mean",
            "q95_prev_0_mo": "mean",
            "q99_prev_0_mo": "mean",
            "q9_prev_3_mo_avg": "mean",
            "q9_prev_6_mo_avg": "mean",
            "q9_prev_9_mo_avg": "mean",
            "q95_prev_3_mo_avg": "mean",
            "q95_prev_6_mo_avg": "mean",
            "q95_prev_9_mo_avg": "mean",
            "q99_prev_3_mo_avg": "mean",
            "q99_prev_6_mo_avg": "mean",
            "q99_prev_9_mo_avg": "mean",
        }
    )
    .reset_index()
)

################################################################################
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create a GeoDataFrame from the grouped DataFrame
gdf_mortality = gpd.GeoDataFrame(
    df_grouped,
    geometry=gpd.points_from_xy(df_grouped["long"], df_grouped["lat"]),
    crs="EPSG:4326",  # WGS 84 coordinate reference system
)
# gdf_mortality = gdf_mortality.to_crs("EPSG:4326")
land = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

land = cfeature.NaturalEarthFeature(
    "physical", "land", "50m", edgecolor="black", facecolor="white"
)
borders = cfeature.NaturalEarthFeature(
    "cultural", "admin_0_boundary_lines_land", "50m", edgecolor="black"
)
# Create the map
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})

# Add a white background and country borders
# ax.add_feature(cfeature.LAND, facecolor="white")
# ax.add_feature(cfeature.BORDERS, edgecolor="black")

# Add features to the map
ax.add_feature(land)
ax.add_feature(borders)

# Plot the points
gdf_mortality.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    markersize=gdf_mortality["indv_id"] / 10,
    c=gdf_mortality["child_mortality"],
    cmap="Reds",
    alpha=0.7,
    legend=True,
)
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
# Add title
ax.set_title("Global Map of Child Mortality by Latitude-Longitude", fontsize=16)

# Show the plot
plt.show()
