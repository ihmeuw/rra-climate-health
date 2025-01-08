#' @Title: [00_USHD_extract_data]  
#' @Authors: Bianca Zlavog
#' @contact: zlavogb@uw.edu
#' @Date_code_last_updated:  11/02/2022
#' @Date_data_last_updated: 11/02/2022
#' @Purpose: Extract USA admin2 income per capita data from US Health Disparities (USHD) team database
#' Data comes from the BEA and currently spans 1980-2021. In future, this data may be updated to span through more recent years.
#' USHD DB info: https://docs.sae.ihme.washington.edu/svc-sae-jenkins/ushd.dbr/
#' USHD covariates prep code: https://stash.ihme.washington.edu/projects/UC/repos/covariates/browse
#' USHD contact details:
#' Vanessa Calder – Database permissions and general architecture
#' Katya Kelly – Contents and application
#' Slack - #ushd_databases – Technical issues/questions; often monitored by Mike Richards as well as USHD team members
#' 
#' @File_Input(s)
#'  
#' @File_Output(s)
#' Compiled data in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_USHD.csv`, 
#' and identical archive file at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_USHD_[datestamp].csv`

##### Setup

rm(list = ls())
library(data.table)

if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", Sys.getenv("USER"), "/")
  k <- "/ihme/cc_resources/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
}

source(paste0(h, "/indicators/1_retrospective/1_GDP_LDI/LSAE/helper_functions.R"))

locs <- get_LSAE_locations(admin_level = "subnational")
locs <- locs[country_ISO3 == "USA" & location_type=="admin2"]

##### Pull in USHD income per capita data

# TODO: This doesn't work currently, try again in future
# library(lbd.loader, lib.loc = sprintf("/share/geospatial/code/geospatial-libraries/lbd.loader-%s.%s", 
#                                       R.version$major, 
#                                       strsplit(R.version$minor, '.', fixed = TRUE)[[1]][[1]]))
# library("ushd.dbr", lib.loc = lbd.loader::pkg_loc("ushd.dbr"))
# tt <- get_covariate_data(covariate_name = "income_pc")

# The equivalent SQL command to pull this data is:
# select * from ushd_shared.covariate_data where covariate_dataset_id = 40

incomepc_path <- paste0(j, "/Project/us_counties/covariates/counties/prepped/income_pc/2022_12_19_09_16_30.rds")
incomepc <- readRDS(incomepc_path)
location_mapping <- fread(paste0(j, "/Project/us_counties/locations/counties/merged_counties.csv"))[, c("mcnty", "cnty_name", "location_id")]

# Subset to LSAE location set
incomepc <- merge(incomepc, location_mapping, by = c("mcnty"), all.x = T)
incomepc <- incomepc[location_id %in% unique(locs$location_id)]

# Merge on location info - location_code, shapefile
# Merge on geolocation info from shapefiles
incomepc_before_merge <- copy(incomepc)
incomepc <- merge_shapefile_locations(incomepc)
validate_merge_nrows(incomepc_before_merge, incomepc)

# Format per extraction template
incomepc <- incomepc[, .(nid = 512023, source = "USHD", data_type = "admin_tabulated", 
                         file_path = incomepc_path, year, iso3 = "USA", 
                         source_location_id = NA, source_location_type = NA,
                         location_id, location_type = "admin2", location_name = cnty_name, 
                         lat = NA, long = NA, location_code, shapefile, 
                         measure = "income", denominator = "per capita", 
                         multiplier = 1, value = income_pc, value_type = "observed", 
                         currency = "LCU", base_year = 2021, currency_detail = NA, 
                         notes = "", geomatching_notes = NA, initials = "BZ")]

# Run validations and save
validate_extractions(incomepc)

validate_geomatched_locations(incomepc, 2010)

save_extraction(incomepc)

