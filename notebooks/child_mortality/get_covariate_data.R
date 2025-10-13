################################################################################
# DESCRIPTION: Try adding covariates such as SDI to data
# PROJECT: Climate nutrition
# DATE: 2025-10-13
################################################################################

#==============================================================================
# SECTION 0: PACKAGE LOADING AND ENVIRONMENT SETUP
#==============================================================================
# Clear workspace
rm(list = ls())

# Username is pulled automatically
username <- Sys.info()[["user"]]
if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/"
  h <- paste0("/homes/", username, "/")
  r <- "/mnt/"
  l <-"/ihme/limited_use/"
  functions_dir <- "/ihme/cc_resources/libraries/current/r/"
} else {
  j <- "J:/"
  h <- "H:/"
  r <- "R:/"
  l <- "L:/"
}

library(data.table)
source(paste0(functions_dir,"get_location_metadata.R"))
source(paste0(functions_dir,"get_covariate_estimates.R"))


cov_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/covariates/"
dir.create(cov_dir, recursive = TRUE, showWarnings = FALSE)

# initialize release_id variable for shared functions
release = 16
# get location IDs from metadata

locs <- get_location_metadata(location_set_id=22, release_id = release) 
# location_set_id=22: -> covariate computation
loc_ids <- locs$location_id

######################### Socio-demographic Index (SDI) ########################
# Getting SDI (covariate_id: 881)
sdi <- get_covariate_estimates(covariate_id = 881, age_group_id = 22, location_id = loc_ids, 
                               release_id = release, sex_id = 3, year_id=c(1980:2024))

write.csv(sdi,paste0(cov_dir,"sdi.csv"),row.names = FALSE)
