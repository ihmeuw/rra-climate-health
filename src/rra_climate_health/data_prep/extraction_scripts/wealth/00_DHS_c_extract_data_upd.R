#' @Title: [00_DHS_extract_data.R]  
#' @Authors: Bianca Zlavog, Audrey Serfes
#' @contact: zlavogb@uw.edu, aserfe@uw.edu
#' @Date_code_last_updated: 08/2023
#' @Date_data_last_updated: 08/2023
#' @Purpose: Process DHS wealth data into extraction template format
#' Includes cleaning, geomatching, point-to-polygon matching, aggregation, and validation
#' @File_Input(s)
#' DHS survey files extracted by UbCov at 
#' `/ihme/resource_tracking/LSAE_income/0_preprocessing/DHS/extractions/`
#' @File_Output(s)
#' Compiled DHS surveys in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_DHS_point_and_polygon.csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_DHS_all_polygon.csv`
#' and identical archive file at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_DHS_point_and_polygon_[datestamp].csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_DHS_all_polygon_[datestamp].csv`

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



source(paste0(h, "/repos/indicators/1_retrospective/1_GDP_LDI/02_subnational_LSAE/helper_functions.R"))

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

dhs_path <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/1_raw_extractions/wealth"
dhs_files <- list.files(path = dhs_path, pattern = "*.dta", recursive = T, ignore.case = T)
dhs_hh_files <- paste0(dhs_path, "/", dhs_files)

#read in all extracted surveys and combine###
dhs_extracts <- lapply(dhs_hh_files,function(i){
  read_dta(i)
})

dhs_extracts <- rbindlist(dhs_extracts, fill=TRUE)

#Check on NID count share folder vs. codebook
# used comma seperating tool for non-EXCLUDED DHS in wealth_data codebook with total variables here: https://delim.co/#
x <- c(77819,398033,21014,157018,218555,18834,393876,18854,31750,218563,18865,30431,286766,18959,79839,218565,19088,
       19133,18902,18913,55956,157021,446877,19001,19016,18533,19211,19274,413167,19381,76878,19391,56151,19324,21281,
       218566,76850,19456,19521,26842,154897,19557,21301,218568,76706,19627,21188,157027,19683,69761,396957,77384,459854,
       157031,21348,19728,95440,19720,65118,218574,20011,20021,76705,286781,19963,157050,20083,77517,356955,20145,21365,
       157057,77518,19167,30379,157024,20191,77385,459845,326837,20167,21382,157058,20361,20339,20223,21409,21311,218582,
       20274,77388,157061,20394,55975,20263,21393,218581,20428,150382,20499,74393,20567,21433,77390,408484,20462,21240,286782,
       20595,77521,286783,275090,270404,270469,270470,270471,210231,20699,21421,142943,337877,426238,20740,56040,157063,462482,
       26855,56063,111432,191270,218592,286772,353526,450419,460813,21258,131467,425283,26866,20829,19315,157025,77515,74460,341838,
       21274,286785,20954,32421,223669,20875,21331,218593,56021,286780,21024,112500,157064,21117,77516,411301,21163,55992,157066,19539,
       21058,18843,18938,18950,19064,19076,18889,18878,26826,18990,18971,19046,19292,18519,18531,19188,19198,19341,19350,19359,19370,19421,
       19444,19493,19511,19546,19571,19579,19604,19614,19670,19637,19695,19708,19999,19950,19787,20051,20060,20092,20103,20132,20154,
       19156,20212,20301,20315,20382,20322,20235,20404,20417,20537,20552,20478,20487,20437,20450,20584,20626,20638,20649,20674,20683,
       20608,20711,20722,20780,19305,20909,20936,20947,20852,20865,20976,20993,21033,21049,20796,21090,21102,21139,21151,20073,459865,56148,
       3133,4837,8906,157060,12630,12644,77395,55973,13544,19431,19656,19529,21206,21222,416582,672,56169,108080,188785,413556,286788,
       218579,58006,34279,56828,286768,69806,111438,218580,218587,413934,77387,157059,286769,30991,218590,77391,350836,11516,11540,77393,286773,
       359318,350798,13109,157065,439649,500002,475926,56099,18920,21198,165645,157026,218572,30777,56426,467681,
       210182,470667,13084,484506,493562,46123,506196,510177, 523643, 520553, 527435, 440151, 528571, 529017, 529525, 527622, 535004)


#list of nids from share drive
y <- unique(dhs_extracts$nid)

#missing NIDS
nid_diff <- setdiff(x,y)

if(length(nid_diff) > 0){
  print(paste("Check why survey data is missing: ", paste(nid_diff, collapse = ", ")))
}

# Hotfix for one survey's geospatial_id
dhs_extracts[nid == 19001, geospatial_id := admin_1]

#We only want these specific columns for total consumption and total income if available. Add sum columns at later date.
dhs_extracts <- dhs_extracts[, c("nid", "ihme_loc_id", "year_start",  "year_end", "int_year", 
                                   "survey_module", "file_path" ,"strata" ,"psu", "hhweight" ,"wealth_score", "quintiles", 
                                   "geospatial_id","hh_id")]


###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################

# pull all available MBG and SAE geocodebooks
source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
geo_sae <- get_geocodebooks(nids = unique(dhs_extracts$nid), data_type = "sae")
geo_mbg <- get_geocodebooks(nids = unique(dhs_extracts$nid), data_type = "mbg")

# Identify surveys with both MBG and SAE geocodebooks, and keep the preferred geomatching info
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

# Hotfix for a miscoded SAE geocodebook with GPS data
sae_mbg_bind[nid == 393876, point := 1]

# Hotfix for an incorrectly extracted geospatial_id
dhs_extracts[nid %in% c(529017, 529525), geospatial_id := psu]

# Merge extracted dataset and geography dataset together
dhs_all <- merge(sae_mbg_bind, dhs_extracts, by.x=c("nid", "iso3", "geospatial_id"),
                  by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T, allow.cartesian = T)

# Hotfix where the geocodebook is missing an entry that is available in the survey GPS file
dhs_all[nid == 529525 & geospatial_id == "7_258", lat := -2.947942][nid == 529525 & geospatial_id == "7_258", long := 11.025596]

##################################
## Check survey information ##
#################################
##Finds nas in  all location columns
geo_missing <- dhs_all %>% filter_at(vars('lat','long','location_code','shapefile'), all_vars(is.na(.)))

# #INSERT CHECKED NIDS:
# #These NIDS have been checked by an analyst, and they are surveys known to have missingness in their geocodebooks, or it is not possible to create geocodebooks for these
# See documentation at https://uwnetid-my.sharepoint.com/:x:/r/personal/hashimig_uw_edu/_layouts/15/Doc.aspx?sourcedoc=%7BF7AA8860-F8D8-49AA-9AF5-DC61F333A605%7D&file=dhs_validation_notes.xlsx&action=default&mobileredirect=true
geo_checked <- c(18533,18902,19016,19088,19133,19167,19211,19274,19324,19381,19456,19521,19557,19627,19683,19720,20011,
                 20073,20083,20145,20167,20191,20263,20274,20339,20428,20567,20595,20699,20740,20829,21014,21163,21188,21258,21281,
                 21301,21331,21348,21365,21382,21393,21409,21421,26842,26855,30379,55975,55992,56021,56063,65118,74460,76706,
                 76850,76878,77390,77516,77518,79839,95440,150382,154897,157018,157021,157027,157031,157057,191270,218563,218565,
                 218568,270404,286766,286780,286781,337877,341838,393876,408484,425283,426238,459845,19076,
                 19156, 19188, 19511, 19571, 19604, 19787, 20315, 20417, 20649, 20865, 20909, 20993, 21151, 286783, 398033,
                 11540, 12644, 19529, 21222, 30777, 56169, 58006, 69806, 77395, 157050, 188785,
                 210231, 218572, 218590, 286769, 286788, 350798, 411301, 413556, 413934, 439649, 157065, 467681, 210182, 506196, 523643,
                 19444, 21206, 440151, 21274, 223669, 18920)
# for(n in geo_checked) {
#   print(paste0("NID: ", n, ", number of missing geospatial_ids: ", length(unique(geo_missing[nid==n]$geospatial_id))))
# }
geo_missing <- geo_missing[!geo_missing$nid %in% geo_checked,]
geo_missing <- geo_missing[!geo_missing$nid %in% surveys_missing_geocodebooks,]

if(nrow(geo_missing) > 0){
  print(paste("Check why there is missing location information from survey data: ", paste(unique(geo_missing$nid), collapse = ", ")))
}


#Check also for missing household weights.
test_hhweight <- dhs_all[(is.na(dhs_all$hhweight))]
#Checked by an analyst. See documentation.
hhw_checked <- c()
test_hhweight <- test_hhweight[!test_hhweight$nid %in% hhw_checked,]
if(length(unique(test_hhweight$nid)) > 0){
  print(paste("Check why there are missing hhweight from survey data: ", paste(unique(test_hhweight$nid), collapse = ", ")))
}


#########################
## Column Adjustments ##
########################

# Drop buggy data for surveys with unexpected values in the wealth scores and quintiles
dhs_all <- dhs_all[!(quintiles == 0 & wealth_score == 0)]

##########################
## Duplicate Removal ##
#########################

#Checked by an analyst. See documentation. 

find_dup <- dhs_all[duplicated(dhs_all) | duplicated(dhs_all, fromLast=TRUE), ]
dup_checked <- c()
find_dup <- find_dup[!find_dup$nid %in% dup_checked,]


if(length(unique(find_dup$nid)) > 0){
  print(paste0("These NIDS have duplicate data, check why: ", paste(unique(find_dup$nid), collapse = ", ")))
}


#duplicate removal      
dhs_all <- dhs_all[!duplicated(dhs_all),]



####################################
## Reshape data from wide to long ##
####################################
dhs_all$quintiles <- as.numeric(dhs_all$quintiles)

dhs_long <- dhs_all %>%
  pivot_longer(
    cols = c(wealth_score:quintiles),
    names_to = "measure",
    values_to = "value",
    values_drop_na = FALSE
  )


# Replace year (the year the survey started in) with int_year (the median interview year for each location)
setnames(dhs_long, "int_year", "year")
setnames(dhs_long, "survey_series", "source")
setnames(dhs_long, "hhweight", "weight")
setnames(dhs_long, "admin_level", "location_type")

#matching tabulated data
dhs_long$location_type[dhs_long$location_type=="0"]<-"admin0"
dhs_long$location_type[dhs_long$location_type=="1"]<-"admin1"
dhs_long$location_type[dhs_long$location_type=="2"]<-"admin2"
dhs_long$location_type[dhs_long$location_type=="3"]<-"admin3"
dhs_long$location_type[dhs_long$location_type=="4"]<-"admin4"
dhs_long$location_type[dhs_long$location_type=="4"]<- NA
dhs_long$location_type[dhs_long$location_type==">1"]<- NA 
dhs_long$location_type[dhs_long$location_type=="<1"]<- NA
dhs_long$location_type[dhs_long$location_type=="<2"]<- NA



#changing values based on extraction template
dhs_long$measure[dhs_long$measure=="wealth_score"]<-"asset score"
dhs_long$measure[dhs_long$measure=="quintiles"]<-"asset percentile"




#Add column dhs_long based on extraction template
dhs_long$source <- "DHS"
dhs_long$data_type <- "survey"
dhs_long$location_id <- NA
dhs_long$source_location_id <- NA
dhs_long$source_location_type <- NA 
dhs_long$multiplier <- 1
dhs_long$denominator <- "total"
dhs_long$value_type <- "aggregated"
dhs_long$currency_detail <- NA 
dhs_long$currency <- NA
dhs_long$base_year <- NA
dhs_long$notes <- NA
dhs_long$geomatching_notes <- NA
dhs_long$initials <- "BZ"


#check for NAS in asset score/percentile values and why:
test_value <-  dhs_long[is.na(dhs_long$measure), ]

# #List of NIDS checked by an analyst. See documentation.
value_checked <- c(20595,191270,270469)
test_value <- test_value[!test_value$nid %in% value_checked,]


if(nrow(test_value) > 0){
  print(paste("Check why there are NAs in value column: ", paste(unique(test_value$nid), collapse = ", ")))
}


#NA asset score/percentile value removal      
# dhs_long <- dhs_long[!is.na(dhs_long$wealth_score),]
# the above is not running, commenting out

# Pull location_ids for data that has been geomatched to LSAE standard location set, based on location_codes
dhs_long <- data.table(dhs_long)
ad1_locs_info <- data.table(global_admin1_sf)[, .(location_code = ADM1_CODE, location_id_fill_ad1 = loc_id)]
ad2_locs_info <- data.table(global_admin2_sf)[, .(location_code = ADM2_CODE, location_id_fill_ad2 = loc_id)]

dhs_long_ad1_fill <- dhs_long[grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)]
dhs_long_ad1_fill <- merge(dhs_long_ad1_fill, ad1_locs_info, by = c("location_code"), all.x = TRUE)
nrow(dhs_long_ad1_fill[is.na(location_id_fill_ad1)]) # should be 0
dhs_long_ad1_fill[, location_id := location_id_fill_ad1][, location_id_fill_ad1 := NULL]
dhs_long_ad2_fill <- dhs_long[grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code)]
dhs_long_ad2_fill <- merge(dhs_long_ad2_fill, ad2_locs_info, by = c("location_code"), all.x = TRUE)
nrow(dhs_long_ad2_fill[is.na(location_id_fill_ad2)]) # should be 0
dhs_long_ad2_fill[, location_id := location_id_fill_ad2][, location_id_fill_ad2 := NULL]

dhs_fill <- rbind(dhs_long_ad1_fill, dhs_long_ad2_fill)
dhs_long_before <- copy(dhs_long)
dhs_long <- dhs_long[!(grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)) & 
                     !(grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code))]
dhs_long <- rbind(dhs_long, dhs_fill)
validate_merge_nrows(dhs_long_before, dhs_long)

# Fill location_ids based on standard shapefiles from stable shapefiles - currently no data match this criteria
stable_shapefile_nids <- unique(dhs_long[shapefile_type=="stable"]$nid) # none


###########################################
## Collapse data ##
###########################################

# split the data into point and polygon datasets
dpoint <- subset(dhs_long, point == 1)
dpoly  <- subset(dhs_long, point == 0)

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
dpoint_to_poly <- dpoint_to_poly[!is.na(loc_id)] # drops a set of NGA observations that fall outside of country boundaries, seem to be in the ocean
# Note that in NID 20361 for MAR, some points get geomatched to locations (location_ids 91017 and 91018) that are part of disputed Western Sahara, which are dropped in modeling
dpoint_to_poly[, value_type := "aggregated_pixel_to_polygon"]

# take the mean of the indicator and count the number of houses in the survey-year (N) 
dpoint <- as.data.table(dpoint)
dpoint <- dpoint[, .(value = mean(value), sd_weighted = sqrt(wtd.var(value, weights = weight, na.rm = TRUE)), sd_unweighted = sd(value, na.rm = TRUE), N_households = .N),
                 by = .(iso3, nid, source, year, lat, long, data_type, file_path,location_type, location_id, location_name, location_code, source_location_type, source_location_id, measure, denominator, multiplier, value_type, currency, base_year, currency_detail, geomatching_notes, notes, initials)]

# take the weighted mean of the indicator
dpoly <- as.data.table(dpoly)
dpoly <- dpoly[, .(value = weighted.mean(value, w=weight), sd_weighted = sqrt(wtd.var(value, weights = weight, na.rm = TRUE)), sd_unweighted = sd(value, na.rm = TRUE), N_households = .N),
               by = .(iso3, nid, source, year, shapefile, location_code, data_type, location_id, file_path, location_name, location_type,source_location_type, source_location_id, measure, denominator, multiplier, value_type, currency, base_year, currency_detail, geomatching_notes, notes, initials)]
dpoint_to_poly <- as.data.table(dpoint_to_poly)
dpoint_to_poly <- dpoint_to_poly[, .(value = weighted.mean(value, w=weight), sd_weighted = sqrt(wtd.var(value, weights = weight, na.rm = TRUE)), sd_unweighted = sd(value, na.rm = TRUE), N_households = .N),
               by = .(iso3, nid, source, year, shapefile, location_code, data_type, location_id, file_path, location_name, location_type,source_location_type, source_location_id, measure, denominator, multiplier, value_type, currency, base_year, currency_detail, geomatching_notes, notes, initials)]

# Combine dpoint and dpoly back together
dhs_long <- rbindlist(list(dpoint, dpoly), fill=T)
dhs_long_to_poly <- rbindlist(list(dpoint_to_poly, dpoly), fill=T)

# Process SD
# Drop some observations where only one household was surveyed, since this is not representative of the region, 
# and because the very small SD throws off the SAE models
dhs_long <- dhs_long[!(N_households == 1 & sd_weighted < 1)]
dhs_long_to_poly <- dhs_long_to_poly[!(N_households == 1 & sd_weighted < 1)]
dhs_long[is.na(sd_weighted), sd_weighted := sd_unweighted] # Correct a few clusters where SD goes missing
dhs_long_to_poly[is.na(sd_weighted), sd_weighted := sd_unweighted] 
nrow(dhs_long[is.na(sd_unweighted)]) + nrow(dhs_long[is.na(sd_weighted)]) # should be 0
nrow(dhs_long_to_poly[is.na(sd_unweighted)]) + nrow(dhs_long_to_poly[is.na(sd_weighted)]) # should be 0

#Select columns for final dataset
dhs_long <- dhs_long[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                       "source_location_type",  "source_location_id", "lat",  "long", "location_code",  "shapefile", "measure", "denominator",
                       "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",
                       "currency_detail", "notes", "geomatching_notes", "initials")]
dhs_long_to_poly <- dhs_long_to_poly[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                       "source_location_type",  "source_location_id", "location_code",  "shapefile", "measure", "denominator",
                       "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",
                       "currency_detail", "notes", "geomatching_notes", "initials")][, c("lat", "long") := NA]

## NEXT UP: Pull from dhs_all the psu, strata, hh_weight and hh_id columns and merge onto dhs_long and dhs_long_to_poly
dhs_merge <- dhs_all[, .(nid, strata, psu, hhweight, hh_id, location_code)]
dhs_merge <- unique(dhs_merge)

# Separating necessary merge columns into pieces
dhs_merge_1 <- dhs_merge[, .(nid, location_code, hh_id)]
dhs_merge_1 <- unique(dhs_merge_1)

dhs_merge_2 <- dhs_merge[, .(nid, hh_id)]

# Stepwise merge due to many-to-many relationship
dhs_long <- merge(dhs_long, dhs_merge_1, by = c("nid","location_code"), all.x = TRUE)

###########################################
## Validate and Save ##
###########################################
# validate_extractions(dhs_long)
# validate_extractions(dhs_long_to_poly)
# The above are not running properly, commenting out for now

#list of nids from share drive
y <- unique(dhs_long$nid)
z <- unique(dhs_long_to_poly$nid)

# Surveys missing all wealth information in the raw data files, as checked by an analyst
surveys_missing_all_data <- c(58006, 30777)

#Check NID count, x is count of codebook nids from top of script
nid_diff <- setdiff(x,y)
nid_diff <- nid_diff[!nid_diff %in% surveys_missing_geocodebooks & !nid_diff %in% surveys_missing_all_data]
nid_diff_to_poly <- setdiff(x,z)
nid_diff_to_poly <- nid_diff_to_poly[!nid_diff_to_poly %in% surveys_missing_geocodebooks & !nid_diff_to_poly %in% surveys_missing_all_data]

if(length(nid_diff) > 0){
  print(paste0("Check that these NIDS have been dropped correctly: ", paste(unique(nid_diff), collapse = ", ")))
}
if(length(nid_diff_to_poly) > 0){
  print(paste0("Check that these NIDS have been dropped correctly: ", paste(unique(nid_diff_to_poly), collapse = ", ")))
}

# Exporting
dhs_long$point_to_polygon = FALSE
write.csv(dhs_long, "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/00_DHS.csv")
dhs_long_to_poly$point_to_polygon = TRUE
write.csv(dhs_long_to_poly, "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/00_DHS_poly.csv")
