from pathlib import Path

LDIPC_FILEPATH = Path('/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/national_ldipc_estimates.csv')
LDIPC_SUBNAT_FILEPATH = Path('/share/resource_tracking/forecasting/poverty/GK_2024_income_distribution_forecasts/income_forecasting_through2100_admin2_final_nocoviddummy_intshift/admin2_ldipc_estimates.csv')

#On one hand, we have population on country level
WORLDPOP_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/worldpop-constrained')

#On the other hand, we have forecasted populations on subnational level at the GBD hierarchy
FORECASTED_POPULATIONS_FILEPATH = Path('/mnt/share/forecasting/data/7/future/population/20240529_500d_2100_lhc_ref_squeeze_hiv_shocks_covid_all_gbd_7_shifted/population.nc')
GBD_SHAPEFILE_LOCATION = Path('/snfs1/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2021/master/shapefiles/GBD2021_analysis_final.shp')

LSAE_SHAPEFILE_LOCATION = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2023_10_30/lbd_standard_admin_2.shp')

CLIMATE_PROJECTIONS_ROOT = Path("/mnt/share/erf/climate_downscale/results/annual")
CLIMATE_HISTORICAL_ROOT = Path("/mnt/share/erf/climate_downscale/results/annual")


MODEL_ROOTS = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/output/models/')

LDIPC_DISTRIBUTION_BIN_PROPORTIONS_DEFAULT_FILEPATH_FORMAT = '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/default_ldipc_bin_proportions_{measure}.parquet'

GLOBAL_POPULATION_FILEPATH = '/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/global-human-settlement-layer/2020/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif'