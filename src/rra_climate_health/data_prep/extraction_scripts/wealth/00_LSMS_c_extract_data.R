#' @Title: [00_process_LSMS_data.R]  
#' @Authors: Bianca Zlavog, Audrey Serfes
#' @contact: zlavogb@uw.edu, aserfe@uw.edu
#' @Date_code_last_updated: 08/2023
#' @Date_data_last_updated: 08/2023
#' @Purpose: Process LSMS wealth data into extraction template format
#' Includes cleaning, geomatching, point-to-polygon matching, aggregation, and validation
#' @File_Input(s)
#' LSMS survey files extracted by UbCov at 
#' `/ihme/resource_tracking/LSAE_income/0_preprocessing/WB_LSMS/latest_extract/`
#' @File_Output(s)
#' Compiled LSMS surveys in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_LSMS_point_and_polygon.csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_LSMS_all_polygon.csv`
#' and identical archive file at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_LSMS_point_and_polygon_[datestamp].csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_LSMS_all_polygon_[datestamp].csv`

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

lsms_path <- "/ihme/resource_tracking/LSAE_income/0_preprocessing/WB_LSMS/"
lsms_files <- list.files(path = paste0(lsms_path, 'latest_extract'), pattern = "*.dta", recursive = T, ignore.case = T)
lsms_hh_files <- paste0(lsms_path, "latest_extract/", lsms_files)
lsms_hh_manual <- read_csv(paste0(lsms_path, "lsms_manual_extraction.csv"))
lsms_hh_manual <- subset(lsms_hh_manual, select = -c(table_page, source_location_type, currency_detail))

#read in all extracted surveys and combine###
lsms_extracts <- lapply(lsms_hh_files,function(i){
  read_dta(i)
})

lsms_extracts <- rbindlist(lsms_extracts, fill=TRUE) 

#Check on NID count share folder vs. codebook
# used comma seperating tool for econ_indicators codebook with total variables here: https://delim.co/#
x <- c(46517, 311265, 224096, 283013, 44844, 44812, 438439, 165101, 45718, 236205, 34524, 45856,7222,
       80626,141572,141573,141574,168020,417512,44645,46480,10224,11606,46580,11580,46317,
       45852,45854,583,46804,46837,4905,4679,9212,7575,45989,45942,10277,12101,11117,14340,9919,10877,46255,137374,
       45782,7245,45784,45786,45850,7248,45779,1775,4580,4250,9422,9310,12584,12863,25006,1698,1813,12455,12489,
       13572,13614,13693,46123,46563,46682,151802,157392,157393,235348,274160,286657,430032,327852,464878,
       468213,22106,81005,9370,
       10555,10675,135713,209103,209104,4043,4075,438798,44126,44253,45283,45376,45464,45549,
       7224,7233,264959,476073,455387,
       272529, 510026, 510027, 510028, 510029, 510030, 510031, 510032, 508581,
       508302, 508569, 508570, 508582, 508579, 508580, 508576, 523766)


#list of nids from share drive
y <- unique(lsms_extracts$nid)

#missing NIDS
nid_diff <- setdiff(x,y)

if(length(nid_diff) > 0){
  print(paste("Check why survey data is missing: ", paste(nid_diff, collapse = ", ")))
}


#We only want these specific columns for total consumption and total income if available. Add sum columns at later date.
lsms_extracts <- lsms_extracts[, c("currency", "base_year", "currency_detail", "unit", "wealth_variables", "econ_notes", 
                                   "survey_name", "nid", "ihme_loc_id", "year_start", "year_end", "int_year", 
                                   "survey_module", "file_path" ,"strata" ,"psu", "pweight", "hhweight" ,"household_size",
                                   "tot_cons", "tot_conexp","tot_exp", "spend_total_recall", "spend_total_quest", 
                                   "income_total_lcu", "income_total_recall", "income_total_quest","geospatial_id","hh_id")]

###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################

# pull all available MBG and SAE geocodebooks
source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
geo_sae <- get_geocodebooks(nids = unique(lsms_extracts$nid), data_type = "sae")
geo_mbg <- get_geocodebooks(nids = unique(lsms_extracts$nid), data_type = "mbg")

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
sae_mbg_bind <- sae_mbg_bind[nid != 135713] # drop one survey for which a SAE geocodebook exists but cannot be mapped to survey locations due to insufficient information in the survey data
surveys_missing_geocodebooks <- c(setdiff(y, unique(sae_mbg_bind$nid))) # 10555, 10675, 135713, 141573, 141574 [these previous 5 are not possible to create geocodebooks for], 455387, 523766

# Correct geospatial_ids for a subnational survey
lsms_extracts <- lsms_extracts[nid == 1813, geospatial_id := substring(geospatial_id, 2, nchar(geospatial_id))]

# Drop a few observations with missing location data
lsms_extracts <- lsms_extracts[!(nid %in% c(209103,209104) & is.na(geospatial_id))]

# Merge extracted dataset and geography dataset together
lsms_all <- merge(sae_mbg_bind, lsms_extracts, by.x=c("nid", "iso3", "geospatial_id"),
                  by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T)

##################################
## Check survey information ##
#################################

##Finds nas in  all location columns
geo_missing <- lsms_all %>% filter_at(vars('lat','long','location_code','shapefile'), all_vars(is.na(.)))

#INSERT CHECKED NIDS:
#These NIDS have been checked by an analyst, and they are surveys known to have missingness in their geocodebooks, or it is not possible to create geocodebooks for these
# See documentation at https://uwnetid-my.sharepoint.com/:x:/r/personal/hashimig_uw_edu/_layouts/15/Doc.aspx?sourcedoc=%7BD9D8B39B-DE1F-44B8-B753-1068A47764FF%7D&file=LSMS_geomatch_hhweight_notes.xlsx&action=default&mobileredirect=true
geo_checked <- c(1775, 4580, 9370, 9422, 10877, 12101, 12584, 13572, 44645, 
                 44844, 46255, 46563, 46682, 46804, 80626, 141572, 
                 157393, 224096, 235348, 236205, 464878, 4043, 264959, 44253, 438798)
# for(n in geo_checked) {
#   print(paste0("NID: ", n, ", number of missing geospatial_ids: ", length(unique(geo_missing[nid==n]$geospatial_id))))
# }
geo_missing <- geo_missing[!geo_missing$nid %in% geo_checked,]
geo_missing <- geo_missing[!geo_missing$nid %in% surveys_missing_geocodebooks,]

if(nrow(geo_missing) > 0){
  print(paste("Check why there is missing location information from survey data: ", paste(unique(geo_missing$nid), collapse = ", ")))
}


#Check also for missing hhweight 

# Fill in weights for surveys that are self-weighted
lsms_all[nid %in% c(4043, 4075, 7233, 10555, 209103, 209104), hhweight := 1]

test_hhweight <- lsms_all[(is.na(hhweight))]


#Checked by an analyst. See documentation.
hhw_checked <- c(4905, 9370, 44844, 80626, 141572, 157393)
test_hhweight <- test_hhweight[!test_hhweight$nid %in% hhw_checked,]

if(length(unique(test_hhweight$nid)) > 0){
  print(paste("Check why there are missing hhweight from survey data: ", paste(unique(test_hhweight$nid), collapse = ", ")))
}

lsms_all <- lsms_all[!(nid %in% hhw_checked & is.na(hhweight))]


#Make sure housesize is available for all surveys
missing_hs <- lsms_all[is.na(lsms_all$hhsize) & unit == "per household" & 
                         !(is.na(lsms_all$tot_cons) & is.na(lsms_all$tot_conexp) 
                           & is.na(lsms_all$tot_exp) & is.na(lsms_all$income_total_lcu))]

if(length(unique(missing_hs$nid)) > 0){
  print(paste("Check why there are missing housesizes from survey data: ", paste(unique(missing_hs$nid), collapse = ", ")))
}


## drop all rows with per household data missing hhsize
lsms_all <- lsms_all[!(is.na(lsms_all$household_size) & unit == 'per household')]


#########################
## Column Adjustments ##
########################


#Change data type so all columns melting are numeric
lsms_all[, spend_total_quest := as.numeric(spend_total_quest)]
lsms_all[, income_total_quest := as.numeric(income_total_quest)]

#Adjust for recall period to get to annual data 
consump_list <- c("tot_cons", "tot_conexp", "tot_exp", "income_total_lcu")

for (vars in consump_list){
  if (vars == "income_total_lcu") {
    recall_var <- "income_total_recall"
  } else {
    recall_var <- "spend_total_recall"
  }
 lsms_all[get(recall_var) == 28 | get(recall_var) == 30, paste0(vars) := 12 * get(vars)]
 lsms_all[get(recall_var) == 7, paste0(vars) := get(vars) * 52 ]
}

#If classified under "per household", divide by householdsize to get to per capita
lsms_all <- lsms_all[!(lsms_all$household_size == 0),]
for (vars in consump_list){
 lsms_all[unit == "per household", paste0(vars) := get(vars)/household_size ,]
}


# Renaming exp. classifications
setnames(lsms_all, "tot_cons", "consumption")
setnames(lsms_all, "tot_conexp", "consumption_expenditure")
setnames(lsms_all, "tot_exp", "expenditure")
setnames(lsms_all, "income_total_lcu", "income")

#Change the one raw data row from year "1908" to "1998"
lsms_all[nid == 9370 & int_year == 1908, int_year := 1998]

#Confirm  unit variable values are correct
variables <- c("unit")
expected_unit <- c("per household", "per capita")
                   
for(var in variables) {
  if(var %in% colnames(lsms_all)) {
    observed_var <- unique(lsms_all[, c(get(var))])
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
duplicate_check <- c(4679, 7245, 7248, 9370, 9919, 11117, 11606, 12863, 14340, 
                     22106, 45782, 45784, 45786, 46480, 46837, 80626, 81005, 
                     137374, 141572, 157392, 165101, 274160, 236205, 10675, 264959)
find_dup <- lsms_all[duplicated(lsms_all) | duplicated(lsms_all, fromLast=TRUE), ]
find_dup <- find_dup[!nid %in% duplicate_check]

if(length(unique(find_dup$nid)) > 0){
  print(paste0("These NIDS have duplicate data, check why: ", paste(unique(find_dup$nid), collapse = ", ")))
}

#duplicate removal      
lsms_all <- lsms_all[!duplicated(lsms_all),]

####################################
## Reshape data from wide to long ##
####################################

setnames(lsms_all, "consumption_expenditure", "consumption expenditure")

lsms_all <- lsms_all %>%
  pivot_longer(
    cols = c(consumption:income_total_quest),
    names_to = "measure",
    values_to = "value",
    values_drop_na = TRUE   # removing NAs here 
  )

lsms_all <- lsms_all %>% filter(measure %in% c("consumption", "consumption expenditure", "expenditure", "income"))
lsms_all <- data.table(lsms_all)

######################################
##Rename and Format Columns##
#####################################

# Replace year (the year the survey started in) with int_year (the median interview year for each location)
setnames(lsms_all, "int_year", "year")
setnames(lsms_all, "survey_series", "source")
setnames(lsms_all, "hhweight", "weight")
setnames(lsms_all, "admin_level", "location_type")
#matching taulated data
lsms_all[location_type=="1", location_type := "admin1"]
lsms_all[location_type=="2", location_type := "admin2"]
lsms_all[location_type=="3", location_type := "admin3"]
lsms_all[location_type=="4", location_type := "admin4"]
lsms_all[location_type=="6", location_type := "admin6"]
lsms_all[location_type %in% c(">1", "<1", "<3"), location_type := NA]

#Add column data based on extraction template
lsms_all$source <- "LSMS"
lsms_all$data_type <- "survey"
lsms_all$location_id <- NA
lsms_all$source_location_id <- NA
lsms_all$source_location_type <- NA 
#measure column classifications
lsms_all$multiplier <- 1
lsms_all$denominator <- "per capita"
lsms_all$value_type <- "aggregated"
lsms_all$currency_detail <- NA #This comes from econ_indicators codebook
lsms_all$currency <- "LCU"
lsms_all$notes <- NA
lsms_all$geomatching_notes <- NA
lsms_all$initials <- "BZ"


# #Check for negative values and NAs in value columns and why:
lsms_debt <- lsms_all[(lsms_all$value < 0),]
lsms_debt_checked <- c(1698,1775,4250,4580,4679,9212,12101,12455,12489,14340,
                       46123,45283,45464)
lsms_debt <- lsms_debt[!lsms_debt$nid %in% lsms_debt_checked,]

if(length(lsms_debt$value) > 0) {
  print(paste("Check these nids for negative values:", paste(unique(lsms_debt$nid), collapse = ", ")))
}

lsms_all <- data.table(lsms_all)

# Pull location_ids based on location_codes from standard shapefiles in a few surveys where they are missing from geocodebooks
ad1_locs_info <- data.table(global_admin1_sf)[, .(location_code = ADM1_CODE, location_id_fill_ad1 = loc_id)]
ad2_locs_info <- data.table(global_admin2_sf)[, .(location_code = ADM2_CODE, location_id_fill_ad2 = loc_id)]

lsms_all_ad1_fill <- lsms_all[grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)]
lsms_all_ad1_fill <- merge(lsms_all_ad1_fill, ad1_locs_info, by = c("location_code"), all.x = TRUE)
nrow(lsms_all_ad1_fill[is.na(location_id_fill_ad1)]) # should be 0
lsms_all_ad1_fill[, location_id := location_id_fill_ad1][, location_id_fill_ad1 := NULL]
lsms_all_ad2_fill <- lsms_all[grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code)]
lsms_all_ad2_fill <- merge(lsms_all_ad2_fill, ad2_locs_info, by = c("location_code"), all.x = TRUE)
nrow(lsms_all_ad2_fill[is.na(location_id_fill_ad2)]) # should be 0
lsms_all_ad2_fill[, location_id := location_id_fill_ad2][, location_id_fill_ad2 := NULL]

lsms_fill <- rbind(lsms_all_ad1_fill, lsms_all_ad2_fill)
lsms_all_before <- copy(lsms_all)
lsms_all <- lsms_all[!(grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)) & 
                         !(grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code))]
lsms_all <- rbind(lsms_all, lsms_fill)
validate_merge_nrows(lsms_all_before, lsms_all)

# Fill location_ids based on standard shapefiles from stable shapefiles
# TODO - revisit this section once this ticket (https://jira.ihme.washington.edu/browse/LSAE-556) is resolved;
# ALB admin1 - 137374. matched and validated reference table. slight differences in polygons but pretty much a 1:1 match. ref table not moved yet to central directory, so not yet matched.
# ALB admin2 - 44126, 44253. to check reference table. ref table not moved yet to central directory, so not yet matched.
# BIH admin1 - 44812, 44844. matched and validated reference table. polygons are identical.
stable_shapefile_nids <- unique(lsms_all[shapefile_type=="stable"]$nid)
stable_shapefile_fill <- copy(lsms_all)[nid %in% c(stable_shapefile_nids)]
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
lsms_all <- rbind(lsms_all[!nid %in% stable_shapefile_nids], stable_shapefile_fill_all)


###########################################
## Collapse data ##
###########################################

# split the data into point and polygon datasets
dpoint <- subset(lsms_all, point == 1)
dpoly  <- subset(lsms_all, point == 0)

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
dpoint[, weight := as.numeric(weight)]
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
lsms_all <- rbindlist(list(dpoint, dpoly), fill=T)
lsms_all_to_poly <- rbindlist(list(dpoint_to_poly, dpoly), fill=T)

# Process SD
# Drop some observations where only one household was surveyed, since this is not representative of the region, 
# and because the very small SD throws off the SAE models
lsms_all <- lsms_all[!(N_households == 1 & sd_weighted < 1)]
lsms_all_to_poly <- lsms_all_to_poly[!(N_households == 1 & sd_weighted < 1)]
nrow(lsms_all[is.na(sd_unweighted)]) + nrow(lsms_all[is.na(sd_weighted)]) # should be 0
nrow(lsms_all_to_poly[is.na(sd_unweighted)]) + nrow(lsms_all_to_poly[is.na(sd_weighted)]) # should be 0

#Select columns for final dataset
lsms_all <- lsms_all[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                         "source_location_type",  "source_location_id", "lat",  "long", "location_code",  "shapefile", "measure", "denominator",
                         "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",
                         "currency_detail", "notes", "geomatching_notes", "initials")]
lsms_all_to_poly <- lsms_all_to_poly[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                         "source_location_type",  "source_location_id", "location_code",  "shapefile", "measure", "denominator",
                         "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",
                         "currency_detail", "notes", "geomatching_notes", "initials")][, c("lat", "long") := NA]


###########################################
## Append and Template Set-Up ##
###########################################

#Append lsms manual extraction to dataset
lsms_appended <- do.call("rbind", list(lsms_all, lsms_hh_manual, fill=TRUE))
lsms_appended_to_poly <- do.call("rbind", list(lsms_all_to_poly, lsms_hh_manual, fill=TRUE))

###########################################
## Validate and Save ##
###########################################
validate_extractions(lsms_appended)
validate_extractions(lsms_appended_to_poly)

#list of nids from share drive
y <- unique(lsms_appended$nid)
z <- unique(lsms_appended_to_poly$nid)

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

save_extraction(lsms_appended, point_to_polygon = FALSE)
save_extraction(lsms_appended_to_poly, point_to_polygon = TRUE)
