from pathlib import Path

################################
# Locations, hierarchies, etc. #
################################

FHS_LOCATION_METADATA_FILEPATH = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/fhs_location_metadata.parquet"
)

LBD_ADMIN2_METADATA_FILEPATH = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/lbd_admin2_metadata.parquet"
)

AGE_SPANS_FILEPATH = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/age_spans.parquet"
)

####################
# Demographic data #
####################


GLOBAL_POPULATION_FILEPATH = "/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/global-human-settlement-layer/2020/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
FORECASTED_POPULATIONS_FILEPATH = Path(
    "/mnt/share/forecasting/data/7/future/population/20240529_500d_2100_lhc_ref_squeeze_hiv_shocks_covid_all_gbd_7_shifted/population.nc"
)

########################
# Socio-demographic index #
########################

# FHS SDI, used as the covariate in the residual model.  The scenario dimension
# of the future file is subset to scenario 130 (reference).
PAST_SDI_FILEPATH = Path(
    # "/mnt/share/forecasting/data/9/past/sdi/20250404_rcp45_first_sub_climate_ref/sdi.nc"
    # "/mnt/share/forecasting/data/9/past/sdi/past_sdi_s131v3/sdi.nc"
    "/mnt/share/forecasting/data/32/past/sdi/past_sdi_s130v89/sdi.nc"
)
FUTURE_SDI_FILEPATH = Path(
    # "/mnt/share/forecasting/data/9/future/sdi/20250404_rcp45_first_sub_climate_ref/sdi.nc"
    # "/mnt/share/forecasting/data/9/future/sdi/future_sdi_s131v3/sdi.nc"
    "/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v89/sdi.nc"
)
FUTURE_SDI_SCENARIO = 130

################
# Climate data #
################


MODEL_ROOTS = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/output/models/"
)

################
# Output paths #
################

OUTPUT_ROOT = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition"
)
MODELS = OUTPUT_ROOT / "models"
RESULTS = OUTPUT_ROOT / "results"
