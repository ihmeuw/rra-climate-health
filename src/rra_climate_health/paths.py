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
# of the future file is subset to FUTURE_SDI_SCENARIO, the reference (RCP 4.5).
#
# Each future release carries its own scenario id, so FUTURE_SDI_SCENARIO has to
# move with FUTURE_SDI_FILEPATH.  A stale id does not raise - the scenario filter
# in residual_data.get_sdi just returns nothing, and the residual is then
# predicted for historical years only.
#
# The round-38 release also publishes the other two RCPs, unused for now because
# the residual model holds SDI at reference across every climate scenario:
#   RCP 2.6  /mnt/share/forecasting/data/38/future/sdi/future_sdi_s160v7/sdi.nc  (scenario 160)
#   RCP 8.5  /mnt/share/forecasting/data/38/future/sdi/future_sdi_s161v7/sdi.nc  (scenario 161)
#
# v5 -> v8 on 2026-10-20: the v5 directories were moved to _to_delete upstream on
# 2026-10-02 and v8 published on 2026-10-19, so the old paths raised
# FileNotFoundError mid-run. Checked on the v8 files: past covers 1980-2023 and
# future 2024-2100, both over 484 locations and 100 draws, and the future
# scenario dimension is [159] - so FUTURE_SDI_SCENARIO is unchanged. This is a
# new SDI VINTAGE, not a relocation: runs scored against v5 and v8 are not
# like-for-like.
PAST_SDI_FILEPATH = Path(
    # "/mnt/share/forecasting/data/9/past/sdi/20250404_rcp45_first_sub_climate_ref/sdi.nc"
    # "/mnt/share/forecasting/data/9/past/sdi/past_sdi_s131v3/sdi.nc"
    # "/mnt/share/forecasting/data/32/past/sdi/past_sdi_s130v89/sdi.nc"
    # "/mnt/share/forecasting/data/38/past/sdi/past_sdi_s159v5/sdi.nc"  (retired upstream)
    "/mnt/share/forecasting/data/38/past/sdi/past_sdi_s159v8/sdi.nc"
)
FUTURE_SDI_FILEPATH = Path(
    # "/mnt/share/forecasting/data/9/future/sdi/20250404_rcp45_first_sub_climate_ref/sdi.nc"
    # "/mnt/share/forecasting/data/9/future/sdi/future_sdi_s131v3/sdi.nc"
    # "/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v89/sdi.nc"
    # "/mnt/share/forecasting/data/38/future/sdi/future_sdi_s159v5/sdi.nc"  (retired upstream)
    "/mnt/share/forecasting/data/38/future/sdi/future_sdi_s159v8/sdi.nc"
)
FUTURE_SDI_SCENARIO = 159

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
