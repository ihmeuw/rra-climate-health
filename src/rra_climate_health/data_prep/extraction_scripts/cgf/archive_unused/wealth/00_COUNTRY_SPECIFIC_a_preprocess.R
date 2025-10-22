#' @Title: [00_COUNTRY_SPECIFIC_a_preprocess_.R]  
#' @Authors: Bianca Zlavog
#' @contact: zlavogb@uw.edu
#' @Purpose: Preprocess COUNTRY SPECIFIC surveys for UbCov extraction
#' 
rm(list = ls())
require(pacman)
p_load(data.table, haven, dplyr,readstata13,janitor)

if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", Sys.getenv("USER"), "/")
  k <- "/ihme/cc_resources/"
  l <- "/ihme/limited_use/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
  l <- "L:/"
}

out_dir <- "/ihme/resource_tracking/LSAE_income/0_preprocessing/COUNTRY_SPECIFIC/"

# Preprocess surveys by merging on variables saved across different files

  # MEX 2016 339653
  # The geospatial_id variables ubica_geo and ageb are not available in their most detailed form from the main HH file,
  # need to merge on correct values from a different file
  mex_2016_hh <- data.table(read_dta(paste0(j, "DATA/MEX/SURVEY_INCOME_AND_HOUSEHOLD_EXPENDITURE_ENIGH/2016/MEX_ENIGH_2016_HH_VARIABLES_Y2021M09D03.DTA")))[, c("ubica_geo", "ageb") := NULL]
  mex_2016_geo <- data.table(read_dta(paste0(j, "DATA/MEX/SURVEY_INCOME_AND_HOUSEHOLD_EXPENDITURE_ENIGH/2016/MEX_ENIGH_2016_HH_CHARACTERISTICS_Y2018M09D19.DTA")))[, .(ubica_geo, ageb, folioviv)]
  mex_2016_all <- merge(mex_2016_hh, mex_2016_geo, by = c("folioviv"), all = T)
  fwrite(mex_2016_all, paste0(out_dir, "MEX_COUNTRY_SPECIFIC_2016_preprocess.csv"))
  