#' @Title: [00_COUNTRY_SPECIFIC_c_extract_data.R]  
#' @Authors: Bianca Zlavog
#' @contact: zlavogb@uw.edu
#' @Date_code_last_updated: 10/2023
#' @Date_data_last_updated: 10/2023
#' @Purpose: Process COUNTRY SPECIFIC wealth data surveys into extraction template format
#' Includes cleaning, geomatching, point-to-polygon matching, aggregation, and validation
#' @File_Input(s)
#' COUNTRY SPECIFIC survey files extracted by UbCov at 
#' `/ihme/resource_tracking/LSAE_income/0_preprocessing/COUNTRY SPECIFIC/extractions/`
#' @File_Output(s)
#' Compiled COUNTRY SPECIFIC surveys in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_COUNTRY SPECIFIC_point_and_polygon.csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_COUNTRY SPECIFIC_all_polygon.csv`
#' and identical archive file at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_COUNTRY SPECIFIC_point_and_polygon_[datestamp].csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_COUNTRY SPECIFIC_all_polygon_[datestamp].csv`

##### Setup
rm(list = ls())
require(pacman)
p_load(data.table, dplyr, readxl, haven, tidyverse, testthat, Hmisc)

lbd_loader_dir <- paste0(
  "/share/geospatial/code/geospatial-libraries/lbd.loader-", R.version$major, ".",
  strsplit(R.version$minor, '.', fixed = TRUE)[[1]][[1]]
)
library(lbd.loader, lib.loc = lbd_loader_dir)
suppressWarnings(suppressMessages(
  library(lbd.mbg, lib.loc = lbd.loader::pkg_loc("lbd.mbg"))
))


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


source(paste0(h, "/indicators/1_retrospective/1_GDP_LDI/LSAE/helper_functions.R"))

modeling_shapefile_version <- get_LSAE_location_versions()
modeling_shapefile_version <- modeling_shapefile_version[
  location_set_version_ids == max(modeling_shapefile_version$location_set_version_ids), shapefile_dates]
global_admin2_shapefile_fp <- get_admin_shapefile(
  admin_level = 2,
  version = modeling_shapefile_version
)
global_admin2_sf <- sf::st_read(global_admin2_shapefile_fp)
global_admin1_shapefile_fp <- get_admin_shapefile(
  admin_level = 1,
  version = modeling_shapefile_version
)
global_admin1_sf <- sf::st_read(global_admin1_shapefile_fp)


#############################
## Load and read all files ##
#############################

country_specific_path <- "/ihme/resource_tracking/LSAE_income/0_preprocessing/COUNTRY_SPECIFIC/"
country_specific_files <- list.files(path = paste0(country_specific_path, 'extractions'), pattern = "*.dta", recursive = T, ignore.case = T)
country_specific_hh_files <- paste0(country_specific_path, "extractions/", country_specific_files)

#read in all extracted surveys and combine###
country_specific_extracts <- lapply(country_specific_hh_files,function(i){
  read_dta(i)
})

country_specific_extracts <- rbindlist(country_specific_extracts, fill=TRUE) 

#Check on NID count share folder vs. codebook
# used comma seperating tool for econ_indicators codebook with total variables here: https://delim.co/#
x <- c(25136, 25154, 25173, 25194, 25215, 25235, 25254, 25273, 25293, 25317, # MEX ENIGH SURVEYS
       25335, 25358, 93321, 165610, 223663, 339653, 424269, 483875, 534143)  # MEX ENIGH SURVEYS


#list of nids from share drive
y <- unique(country_specific_extracts$nid)

#missing NIDS
nid_diff <- setdiff(x,y)

if(length(nid_diff) > 0){
  print(paste("Check why survey data is missing: ", paste(nid_diff, collapse = ", ")))
}


#We only want these specific columns for total consumption and total income if available. Add sum columns at later date.
country_specific_extracts <- country_specific_extracts[, c("currency", "base_year", "currency_detail", "unit", "wealth_variables",
                                   "survey_name", "nid", "ihme_loc_id", "year_start", "year_end", "int_year", 
                                   "survey_module", "file_path" ,"strata" ,"psu", "hhweight" ,"household_size",
                                   "tot_exp", "spend_total_recall", 
                                   "income_total_lcu", "income_total_recall", "geospatial_id","hh_id")]

###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################

# pull all available MBG and SAE geocodebooks
source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
geo_sae <- get_geocodebooks(nids = unique(country_specific_extracts$nid), data_type = "sae")
geo_mbg <- get_geocodebooks(nids = unique(country_specific_extracts$nid), data_type = "mbg")

# Identify surveys with both MBG and SAE geocodebooks, and keep the preferred geomatching info
# Generally, we prefer MBG point geomatched data when available, or SAE polygon geomatched data for lbd_standard_admin_1/2 shapefiles
# Also identify any surveys missing geocodebooks
geocodebooks_overlap <- intersect(unique(geo_sae$nid), unique(geo_mbg$nid))

# Prefer MBG codebooks if they have only point data, 
# OR if they have lbd_standard_admin_1/2 geomatching AND the SAE codebook does not have lbd_standard_admin_1/2 geomatching (may have stable shapefile matching)
# Otherwise, prefer SAE codebooks for all other surveys
preferred_mbg <- c()
preferred_sae <- c()
geo_mbg_subset <- geo_mbg[!(is.na(point) & is.na(lat) & is.na(long) & is.na(shapefile))]
geo_sae_subset <- geo_sae[!(is.na(point) & is.na(lat) & is.na(long) & is.na(shapefile))]
for(nid_check in geocodebooks_overlap) {
  if(all(unique(geo_mbg_subset[nid == nid_check]$point) == 1)) {
    preferred_mbg <- c(preferred_mbg, nid_check)
  } else {
    if(all(unique(geo_mbg_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_") & 
       (!unique(geo_sae_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_")) {
      preferred_mbg <- c(preferred_mbg, nid_check)
    } else {
      preferred_sae <- c(preferred_sae, nid_check)
    }
  }
}

geo_mbg <- geo_mbg[!nid %in% preferred_sae]
geo_sae <- geo_sae[!nid %in% preferred_mbg]
geocodebooks_overlap <- intersect(unique(geo_sae$nid), unique(geo_mbg$nid)) # should now be an empty list
sae_mbg_bind <- rbind(geo_sae, geo_mbg)
sae_mbg_bind <- unique(sae_mbg_bind) # Removing duplicate rows from buggy geocodebooks

# Identify any surveys missing geocodebooks
surveys_missing_geocodebooks <- c(setdiff(y, unique(sae_mbg_bind$nid))) # none

# Temporary method to use geospatial_ids for geocodebooks matched to admin1 data
country_specific_extracts[!(nid %in% c(165610, 223663, 339653)), geospatial_id := substr(geospatial_id, 1, nchar(geospatial_id)-3)]
country_specific_extracts[(nid %in% c(165610, 223663)), geospatial_id := substr(geospatial_id, 1, nchar(geospatial_id)-13)]

# Merge extracted dataset and geography dataset together
country_specific_all <- merge(sae_mbg_bind, country_specific_extracts, by.x=c("nid", "iso3", "geospatial_id"),
                  by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T)

##################################
## Check survey information ##
#################################

##Finds nas in  all location columns
geo_missing <- country_specific_all %>% filter_at(vars('lat','long','location_code','shapefile'), all_vars(is.na(.)))

#INSERT CHECKED NIDS:
#These NIDS have been checked by an analyst, and they are surveys known to have missingness in their geocodebooks, or it is not possible to create geocodebooks for these
geo_checked <- c()
# for(n in geo_checked) {
#   print(paste0("NID: ", n, ", number of missing geospatial_ids: ", length(unique(geo_missing[nid==n]$geospatial_id))))
# }
geo_missing <- geo_missing[!geo_missing$nid %in% geo_checked,]
geo_missing <- geo_missing[!geo_missing$nid %in% surveys_missing_geocodebooks,]

if(nrow(geo_missing) > 0){
  print(paste("Check why there is missing location information from survey data: ", paste(unique(geo_missing$nid), collapse = ", ")))
}


#Check also for missing hhweight 

test_hhweight <- country_specific_all[(is.na(hhweight))]


#Checked by an analyst. See documentation.
hhw_checked <- c()
test_hhweight <- test_hhweight[!test_hhweight$nid %in% hhw_checked,]

if(length(unique(test_hhweight$nid)) > 0){
  print(paste("Check why there are missing hhweight from survey data: ", paste(unique(test_hhweight$nid), collapse = ", ")))
}

country_specific_all <- country_specific_all[!(nid %in% hhw_checked & is.na(hhweight))]


#Make sure housesize is available for all surveys
missing_hs <- country_specific_all[is.na(country_specific_all$hhsize) & unit == "per household" & 
                         !(is.na(country_specific_all$tot_cons) & is.na(country_specific_all$tot_conexp) 
                           & is.na(country_specific_all$tot_exp) & is.na(country_specific_all$income_total_lcu))]

if(length(unique(missing_hs$nid)) > 0){
  print(paste("Check why there are missing housesizes from survey data: ", paste(unique(missing_hs$nid), collapse = ", ")))
}


## drop all rows with per household data missing hhsize
country_specific_all <- country_specific_all[!(is.na(country_specific_all$household_size) & unit == 'per household')]


#########################
## Column Adjustments ##
########################

#Adjust for recall period to get to annual data 
consump_list <- c("tot_exp", "income_total_lcu")

for (vars in consump_list){
  if (vars == "income_total_lcu") {
    recall_var <- "income_total_recall"
  } else {
    recall_var <- "spend_total_recall"
  }
 country_specific_all[get(recall_var) == 28 | get(recall_var) == 30, paste0(vars) := 12 * get(vars)]
 country_specific_all[get(recall_var) == 7, paste0(vars) := get(vars) * 52 ]
}

#If classified under "per household", divide by householdsize to get to per capita
country_specific_all <- country_specific_all[!(country_specific_all$household_size == 0),]
for (vars in consump_list){
 country_specific_all[unit == "per household", paste0(vars) := get(vars)/household_size ,]
}


# Renaming exp. classifications
setnames(country_specific_all, "tot_exp", "expenditure")
setnames(country_specific_all, "income_total_lcu", "income")


#Confirm  unit variable values are correct
variables <- c("unit")
expected_unit <- c("per household", "per capita")
                   
for(var in variables) {
  if(var %in% colnames(country_specific_all)) {
    observed_var <- unique(country_specific_all[, c(get(var))])
    for(i in observed_var) {
      if(!(i %in% get(paste0("expected_", var)))) {
        message("Error: Value \'", i, "\' in column \'", var, "\' outside of expected range: \'", paste(get(paste0("expected_", var)), collapse = ', '), "\'")
      }
    }
  }
}


#########################
## Duplicate Removal ##
#########################
#Find duplicated rows from surveys where multiple people in HH were interviewed and income and expenditure stayed the same for multiple rows.
#Checked by an analyst. See documentation.
duplicate_check <- c()
find_dup <- country_specific_all[duplicated(country_specific_all) | duplicated(country_specific_all, fromLast=TRUE), ]
find_dup <- find_dup[!nid %in% duplicate_check]

if(length(unique(find_dup$nid)) > 0){
  print(paste0("These NIDS have duplicate data, check why: ", paste(unique(find_dup$nid), collapse = ", ")))
}

#duplicate removal      
country_specific_all <- country_specific_all[!duplicated(country_specific_all),]

####################################
## Reshape data from wide to long ##
####################################

country_specific_all <- country_specific_all %>%
  pivot_longer(
    cols = c("expenditure", "income"),
    names_to = "measure",
    values_to = "value",
    values_drop_na = TRUE   # removing NAs here 
  )

country_specific_all <- country_specific_all %>% filter(measure %in% c("expenditure", "income"))
country_specific_all <- data.table(country_specific_all)

######################################
##Rename and Format Columns##
#####################################

# Replace year (the year the survey started in) with int_year (the median interview year for each location)
setnames(country_specific_all, "int_year", "year")
setnames(country_specific_all, "survey_series", "source")
setnames(country_specific_all, "hhweight", "weight")
setnames(country_specific_all, "admin_level", "location_type")
#matching taulated data
country_specific_all[location_type=="1", location_type := "admin1"]
country_specific_all[location_type=="2", location_type := "admin2"]

#Add column data based on extraction template
country_specific_all$source <- "COUNTRY_SPECIFIC"
country_specific_all$data_type <- "survey"
country_specific_all$location_id <- NA
country_specific_all$source_location_id <- NA
country_specific_all$source_location_type <- NA 
#measure column classifications
country_specific_all$multiplier <- 1
country_specific_all$denominator <- "per capita"
country_specific_all$value_type <- "aggregated"
country_specific_all$currency_detail <- NA #This comes from econ_indicators codebook
country_specific_all$currency <- "LCU"
country_specific_all$notes <- NA
country_specific_all$geomatching_notes <- NA
country_specific_all$initials <- "BZ"


# #Check for negative values and NAs in value columns and why:
country_specific_debt <- country_specific_all[(country_specific_all$value < 0),]
country_specific_debt_checked <- c()
country_specific_debt <- country_specific_debt[!country_specific_debt$nid %in% country_specific_debt_checked,]

if(length(country_specific_debt$value) > 0) {
  print(paste("Check these nids for negative values:", paste(unique(country_specific_debt$nid), collapse = ", ")))
}

country_specific_all <- data.table(country_specific_all)

# Pull location_ids based on location_codes from standard shapefiles in a few surveys where they are missing from geocodebooks
ad1_locs_info <- data.table(global_admin1_sf)[, .(location_code = ADM1_CODE, location_id_fill_ad1 = loc_id)]
ad2_locs_info <- data.table(global_admin2_sf)[, .(location_code = ADM2_CODE, location_id_fill_ad2 = loc_id)]

country_specific_all_ad1_fill <- country_specific_all[grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)]
country_specific_all_ad1_fill <- merge(country_specific_all_ad1_fill, ad1_locs_info, by = c("location_code"), all.x = TRUE)
nrow(country_specific_all_ad1_fill[is.na(location_id_fill_ad1)]) # should be 0
country_specific_all_ad1_fill[, location_id := location_id_fill_ad1][, location_id_fill_ad1 := NULL]
country_specific_all_ad2_fill <- country_specific_all[grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code)]
country_specific_all_ad2_fill <- merge(country_specific_all_ad2_fill, ad2_locs_info, by = c("location_code"), all.x = TRUE)
nrow(country_specific_all_ad2_fill[is.na(location_id_fill_ad2)]) # should be 0
country_specific_all_ad2_fill[, location_id := location_id_fill_ad2][, location_id_fill_ad2 := NULL]

country_specific_fill <- rbind(country_specific_all_ad1_fill, country_specific_all_ad2_fill)
country_specific_all_before <- copy(country_specific_all)
country_specific_all <- country_specific_all[!(grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)) & 
                         !(grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code))]
country_specific_all <- rbind(country_specific_all, country_specific_fill)
validate_merge_nrows(country_specific_all_before, country_specific_all)

# Fill location_ids based on standard shapefiles from stable shapefiles
# TODO - revisit this section once this ticket (https://jira.ihme.washington.edu/browse/LSAE-556) is resolved;
# MEX admin2 - 339653
stable_shapefile_nids <- unique(country_specific_all[shapefile_type=="stable"]$nid)
stable_shapefile_fill <- copy(country_specific_all)[nid %in% c(stable_shapefile_nids)]
stable_shapefile_fill_all <- data.table()
for(stable_nid in stable_shapefile_nids) {
  stable_shapefile_fill_nid <- stable_shapefile_fill[nid == stable_nid]
  stable_nid_country <- unique(stable_shapefile_fill_nid$iso3)
  stable_nid_admin_level <- unique(stable_shapefile_fill_nid[!is.na(location_type)]$location_type)
  stable_nid_admin_level <- gsub("admin", "", stable_nid_admin_level)
  stable_shapefile_version <- unique(stable_shapefile_fill_nid[!is.na(shapefile)]$shapefile)
  stable_shapefile_version <- gsub(paste0("/home/j/WORK/11_geospatial/admin_shapefiles/sae/", 
                                          stable_nid_country, "/"), "", stable_shapefile_version)
  stable_shapefile_version <- gsub(paste0("J:/WORK/11_geospatial/admin_shapefiles/sae/", 
                                          stable_nid_country, "/"), "", stable_shapefile_version)
  stable_shapefile_version <- gsub(paste0("/", stable_nid_country, "_admin_", 
                                          stable_nid_admin_level, ".shp"), "", stable_shapefile_version)
  reference_table <- fread(paste0("/mnt/team/sae/pub/reference_table/", 
                                  stable_nid_country, "/", stable_shapefile_version, 
                                  "/reference_table_admin_", 
                                  stable_nid_admin_level, "_", stable_nid_country, ".csv"))
  reference_table <- reference_table[stable_version == stable_shapefile_version & 
                                       standard_version == modeling_shapefile_version]
  reference_table <- reference_table[, .(stable_poly_id, standard_poly_id)]
  stable_shapefile_fill_nid <- merge(stable_shapefile_fill_nid, reference_table, 
                                     by.x = "location_code", by.y = "stable_poly_id", all.x = T, allow.cartesian = T)
  stable_shapefile_fill_nid[, location_id := standard_poly_id][, standard_poly_id := NULL]
  stable_shapefile_fill_all <- rbind(stable_shapefile_fill_all, stable_shapefile_fill_nid, fill = T)
}
country_specific_all <- rbind(country_specific_all[!nid %in% stable_shapefile_nids], stable_shapefile_fill_all)


###########################################
## Collapse data ##
###########################################

# split the data into point and polygon datasets
dpoint <- subset(country_specific_all, point == 1)
dpoly  <- subset(country_specific_all, point == 0)

# alternate dataset for SAE modeling where point data is matched and aggregated to admin2 polygons
# note that the CRS projection used here should align with that used by the shapefile you're matching to; 
# this can be checked by running `sf::st_crs(global_admin2_sf)`
dpoint_to_poly <- sf::st_as_sf(dpoint, coords = c('long', 'lat'), crs = sf::st_crs(4326))
dpoint_to_poly <- sf::st_join(x = dpoint_to_poly, y = global_admin2_sf)
dpoint_to_poly <- data.table(dpoint_to_poly)
dpoint_to_poly <- dpoint_to_poly[, c("geo_id", "ad2_id", "ad0_parent",
                                     "ADM1_CODE", "ADM1_NAME", "ADM0_CODE") := NULL]
dpoint_to_poly <- dpoint_to_poly[, shapefile := paste0("/home/j/WORK/11_geospatial/admin_shapefiles/", modeling_shapefile_version, "/lbd_standard_admin_2.dbf")]
dpoint_to_poly <- dpoint_to_poly[, location_id := loc_id][, location_code := ADM2_CODE][, location_name := ADM2_NAME][, location_type := "admin2"]
dpoint_to_poly[, value_type := "aggregated_pixel_to_polygon"]

#cast weight to numeric values
#dpoint[, weight := as.numeric(weight)]
dpoly[, weight := as.numeric(weight)]
dpoint_to_poly[, weight := as.numeric(weight)]

# take the mean of the indicator and count the number of houses in the survey-year (N) 
dpoint <- dpoint[, .(value = mean(value), sd_weighted = sqrt(wtd.var(value, weights = weight, na.rm = TRUE)), sd_unweighted = sd(value, na.rm = TRUE), N_households = .N),
                 by = .(iso3, nid, source, year, lat, long, data_type, file_path,location_type, location_id, location_name, location_code, source_location_type, source_location_id, measure, denominator, multiplier, value_type, currency, base_year, currency_detail, geomatching_notes, notes, initials)]

# take the weighted mean of the indicator
dpoly <- dpoly[, .(value = weighted.mean(value, w=weight), sd_weighted = sqrt(wtd.var(value, weights = weight, na.rm = TRUE)), sd_unweighted = sd(value, na.rm = TRUE), N_households = .N),
               by = .(iso3, nid, source, year, shapefile, location_code, data_type, location_id, file_path, location_name, location_type,source_location_type, source_location_id, measure, denominator, multiplier, value_type, currency, base_year, currency_detail, geomatching_notes, notes, initials)]
dpoint_to_poly <- dpoint_to_poly[, .(value = weighted.mean(value, w=weight), sd_weighted = sqrt(wtd.var(value, weights = weight, na.rm = TRUE)), sd_unweighted = sd(value, na.rm = TRUE), N_households = .N),
               by = .(iso3, nid, source, year, shapefile, location_code, data_type, location_id, file_path, location_name, location_type,source_location_type, source_location_id, measure, denominator, multiplier, value_type, currency, base_year, currency_detail, geomatching_notes, notes, initials)]



# Combine dpoint and dpoly back together
country_specific_all <- rbindlist(list(dpoint, dpoly), fill=T)
country_specific_all_to_poly <- rbindlist(list(dpoint_to_poly, dpoly), fill=T)
country_specific_all_to_poly <- copy(dpoly)

# Process SD
# Drop some observations where only one household was surveyed, since this is not representative of the region, 
# and because the very small SD throws off the SAE models
country_specific_all <- country_specific_all[!(N_households == 1 & sd_weighted < 1)]
country_specific_all_to_poly <- country_specific_all_to_poly[!(N_households == 1 & sd_weighted < 1)]
nrow(country_specific_all[is.na(sd_unweighted)]) + nrow(country_specific_all[is.na(sd_weighted)]) # should be 0
nrow(country_specific_all_to_poly[is.na(sd_unweighted)]) + nrow(country_specific_all_to_poly[is.na(sd_weighted)]) # should be 0

#Select columns for final dataset
country_specific_all <- country_specific_all[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name",
                         "source_location_type",  "source_location_id", "lat",  "long", "location_code",  "shapefile", "measure", "denominator",
                         "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",
                         "currency_detail", "notes", "geomatching_notes", "initials")]
country_specific_all_to_poly <- country_specific_all_to_poly[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                         "source_location_type",  "source_location_id", "location_code",  "shapefile", "measure", "denominator",
                         "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",
                         "currency_detail", "notes", "geomatching_notes", "initials")][, c("lat", "long") := NA]

###########################################
## Validate and Save ##
###########################################
validate_extractions(country_specific_appended)
validate_extractions(country_specific_all_to_poly)

#list of nids from share drive
y <- unique(country_specific_all$nid)
z <- unique(country_specific_all_to_poly$nid)

#Check NID count, x is count of codebook nids from top of script
nid_diff <- setdiff(x,y)
nid_diff <- nid_diff[!nid_diff %in% surveys_missing_geocodebooks]
nid_diff_to_poly <- setdiff(x,z)
nid_diff_to_poly <- nid_diff_to_poly[!nid_diff_to_poly %in% surveys_missing_geocodebooks]

if(length(nid_diff) > 0){
  print(paste("Check that these NIDS have been dropped correctly: ", paste(unique(nid_diff), collapse = ", ")))
}
if(length(nid_diff_to_poly) > 0){
  print(paste("Check that these NIDS have been dropped correctly: ", paste(unique(nid_diff_to_poly), collapse = ", ")))
}

save_extraction(country_specific_appended, point_to_polygon = FALSE)
save_extraction(country_specific_all_to_poly, point_to_polygon = TRUE)
