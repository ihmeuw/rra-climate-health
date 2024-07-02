from pathlib import Path


################################
# Locations, hierarchies, etc. #
################################

FHS_LOCATION_METADATA_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/fhs_location_metadata.parquet')
AGE_SPANS_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/age_spans.parquet')

####################
# Demographic data #
####################


GLOBAL_POPULATION_FILEPATH = '/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/global-human-settlement-layer/2020/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif'
FORECASTED_POPULATIONS_FILEPATH = Path('/mnt/share/forecasting/data/7/future/population/20240529_500d_2100_lhc_ref_squeeze_hiv_shocks_covid_all_gbd_7_shifted/population.nc')

################
# Climate data #
################

CHELSA_HISTORICAL_ROOT = Path('/mnt/team/rapidresponse/pub/population/data/02-processed-data/human-niche/chelsa-downscaled-historical')
CLIMATE_HISTORICAL_ROOT = Path("/mnt/share/erf/climate_downscale/results/annual")
CLIMATE_PROJECTIONS_ROOT = Path("/mnt/share/erf/climate_downscale/results/annual")


MODEL_ROOTS = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/output/models/')

################
# Output paths #
################

OUTPUT_ROOT = Path('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition')
MODELS = OUTPUT_ROOT / 'models'
RESULTS = OUTPUT_ROOT / 'results'

