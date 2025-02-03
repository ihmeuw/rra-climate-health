#' @Title: [00b_MICS_extract_data.R]  
#' @Authors: Bianca Zlavog, Audrey Serfes
#' @contact: zlavogb@uw.edu, aserfe@uw.edu
#' @Date_code_last_updated: 08/2023
#' @Date_data_last_updated: 08/2023
#' @Purpose: Process MICS wealth data into extraction template format
#' Includes cleaning, geomatching, point-to-polygon matching, aggregation, and validation
#' @File_Input(s)
#' MICS survey files extracted by UbCov at 
#' `/ihme/resource_tracking/LSAE_income/0_preprocessing/MICS/extractions/`
#' @File_Output(s)
#' Compiled MICS surveys in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_MICS_point_and_polygon.csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_MICS_all_polygon.csv`
#' and identical archive file at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_MICS_point_and_polygon_[datestamp].csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_MICS_all_polygon_[datestamp].csv`

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

mics_path <- "/ihme/resource_tracking/LSAE_income/0_preprocessing/MICS/extractions/"
mics_files <- list.files(path = mics_path, pattern = "*.dta", recursive = T, ignore.case = T)
mics_hh_files <- paste0(mics_path, mics_files)

#read in all extracted surveys and combine###
mics_extracts <- lapply(mics_hh_files,function(i){
  read_dta(i)
})

mics_extracts <- rbindlist(mics_extracts, fill=TRUE) 
mics_extracts <- data.table(mics_extracts)

#Check on NID count share folder vs. codebook
# used comma seperating tool for MICS in wealth_data codebook with total variables here: https://delim.co/#
x <- c(56830,687,595,608,137208,490784,881,1994,1981,151086,206075,1927,951,436764,1385,80731,1039,80733,435732,1089,76699,264910,1289,
       194011,40028,2209,2223,82832,466384,26433,26444,218611,2053,2063,244455,3161,437955,234733,3114,125596,452953,3301,464145,27069,
       200697,488697,210614,464103,3970,429463,4694,63993,437993,303458,3922,3935,91506,424884,4808,4818,174049,457894,3655,4916,
       4926,200598,474221,490966,7054,76707,385708,141336,7340,76702,260403,7540,162283,408226,438014,7618,7629,103973,375362,7721,427778,
       149906,8734,140200,27020,399853,264590,7843,91324,453328,270627,248224,8932,90696,8819,150526,413741,8777,8788,76704,150866,427952,
       27031,8115,152783,267343,7919,161662,494148,9439,9516,76703,437724,39999,162317,462617,161587,324470,10948,125591,161590,464593,26930,
       153643,32189,200617,27044,287639,11639,11649,76700,218619,200636,11774,11551,156268,150870,459477,462027,27055,214640,464540,12280,12289
       ,81203,427983,12320,30325,200707,488613,12399,2244,76701,466891,12886,12896,40021,429991,12732,148649,296646,459434,12595,12608,13064,
       264583,453549,453346,12940,12950,332558,76709,409558,489863,13197,132739,214734,13436,13445,13516,13708,13719,57999,152735,13465,13816,
       456286,14122,35493,152720,431951,
       104042,104043,104236,12243,125594,135416,155335,189045,189048,203654,203663,203664,232763,236266,26998,308316,335994,336042,
       400526,422512,479655,494157,494214,498448,56153,56241,56420,60942,7387,7401,91507,91508, 487664, 509345, 505452, 527453, 529982,
       515231)


#list of nids from share drive
y <- unique(mics_extracts$nid)

#missing NIDS
nid_diff <- setdiff(x,y)

if(length(nid_diff) > 0){
  print(paste("Check why survey data is missing: ", paste(nid_diff, collapse = ", ")))
}

# Quick fix to keep only one observation per household for a particular survey, in future do this in preprocessing script
mics_extracts <- mics_extracts[!(nid == 26998 & line_id > 1)]

# Hotfix for one survey's geospatial_id
mics_extracts[nid == 104042, geospatial_id := paste0(admin_1, "_", admin_2)]

#We only want these specific columns for total consumption and total income if available. Add sum columns at later date.
mics_extracts <- mics_extracts[, c("nid", "ihme_loc_id", "year_start",  "year_end", "int_year", 
                                   "survey_module", "file_path" ,"strata" ,"psu", "hhweight" ,"wealth_score", "quintiles", "percentiles", 
                                   "geospatial_id","hh_id")]



###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################

# pull all available MBG and SAE geocodebooks
source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
geo_sae <- get_geocodebooks(nids = unique(mics_extracts$nid), data_type = "sae")
geo_mbg <- get_geocodebooks(nids = unique(mics_extracts$nid), data_type = "mbg")

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
surveys_missing_geocodebooks <- c(setdiff(y, unique(sae_mbg_bind$nid))) # 529982

# Correct geospatial_ids for some surveys
mics_extracts[nid == 4916, geospatial_id := sub("_[^_]+$", "", geospatial_id)]
mics_extracts[nid == 1981, geospatial_id := sub("_[^_]+$", "", sub("_[^_]+$", "", sub("_[^_]+$", "", geospatial_id)))]

# Replace a few subnational surveys for which the geocodebooks have the country iso3 assigned
mics_extracts[nid %in% c(60942, 422512, 515231), ihme_loc_id := substring(ihme_loc_id, 1, 3)]

# Merge extracted dataset and geography dataset together
mics_all <- merge(sae_mbg_bind, mics_extracts, by.x=c("nid", "iso3", "geospatial_id"),
                  by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T, allow.cartesian = T)

##################################
## Check survey information ##
#################################
##Finds nas in  all location columns
geo_missing <- mics_all %>% filter_at(vars('lat','long','location_code','shapefile'), all_vars(is.na(.)))

#INSERT CHECKED NIDS:
#These NIDS have been checked by an analyst, and they are surveys known to have missingness in their geocodebooks, or it is not possible to create geocodebooks for these
# See documentation at https://uwnetid-my.sharepoint.com/:x:/r/personal/hashimig_uw_edu/_layouts/15/Doc.aspx?sourcedoc=%7B51B70A2B-CAC0-4895-ABBD-DC69A45B2030%7D&file=mics_geomatching_hhweight_notes.xlsx&action=default&mobileredirect=true
geo_checked <- c(595, 951, 1927, 2063, 3655, 7721, 8115, 10948, 12399, 
                 12886, 26998, 57999, 63993, 80731, 150866, 151086, 
                 162317, 200598,214734, 427952, 437724,7387, 
                 91507, 135416, 155335, 203663, 335994, 505452,
                 435732, 7054, 30325, 161590, 490966)
# for(n in geo_checked) {
#   print(paste0("NID: ", n, ", number of missing geospatial_ids: ", length(unique(geo_missing[nid==n]$geospatial_id))))
# }
geo_missing <- geo_missing[!geo_missing$nid %in% geo_checked,]
geo_missing <- geo_missing[!geo_missing$nid %in% surveys_missing_geocodebooks,]

if(nrow(geo_missing) > 0){
  print(paste("Check why there is missing location information from survey data: ", paste(unique(geo_missing$nid), collapse = ", ")))
}


#Check also for missing hhweight 

# Fill in weights for one survey that is self-weighted
mics_all[nid == 7721, hhweight := 1]

test_hhweight <- mics_all[(is.na(mics_all$hhweight))]
#Checked by an analyst. See documentation.
hhw_checked <- c(4926, 27031, 35493, 125591, 296646)
test_hhweight <- test_hhweight[!test_hhweight$nid %in% hhw_checked,]

if(length(unique(test_hhweight$nid)) > 0){
  print(paste("Check why there are missing hhweight from survey data: ", paste(unique(test_hhweight$nid), collapse = ", ")))
}


#########################
## Column Adjustments ##
########################


#Change raw data years for CUB 3301 2000 
mics_all[nid == 3301 & int_year < 1980, int_year := 2000]

# Drop buggy data for surveys with unexpected values in the wealth scores and quintiles
mics_all <- mics_all[!(quintiles == 0 & wealth_score == 0)]
mics_all <- mics_all[!(quintiles == 99)]

#########################
## Duplicate Removal ##
#########################

#Checked by an analyst. See documentation. 

find_dup <- mics_all[duplicated(mics_all)]
dup_checked <- c(218611, 13708)
find_dup <- find_dup[!find_dup$nid %in% dup_checked,]


if(length(unique(find_dup$nid)) > 0){
  print(paste0("These NIDS have duplicate data, check why: ", paste(unique(find_dup$nid), collapse = ", ")))
}


#duplicate removal      
mics_all <- mics_all[!duplicated(mics_all),]


####################################
## Reshape data from wide to long ##
####################################

mics_all <- mics_all %>%
  pivot_longer(
    cols = c(wealth_score:quintiles),
    names_to = "measure",
    values_to = "value",
    values_drop_na = FALSE
  )


# Replace year (the year the survey started in) with int_year (the median interview year for each location)
setnames(mics_all, "int_year", "year")
setnames(mics_all, "survey_series", "source")
setnames(mics_all, "hhweight", "weight")
setnames(mics_all, "admin_level", "location_type")

#matching tabulated data
mics_all$location_type[mics_all$location_type=="0"]<-"admin0"
mics_all$location_type[mics_all$location_type=="1"]<-"admin1"
mics_all$location_type[mics_all$location_type=="2"]<-"admin2"
mics_all$location_type[mics_all$location_type=="3"]<-"admin3"
mics_all$location_type[mics_all$location_type==">1"]<- NA 
mics_all$location_type[mics_all$location_type=="<1"]<- NA
mics_all$location_type[mics_all$location_type=="<2"]<- NA
mics_all$location_type[mics_all$location_type=="5"]<- NA
mics_all$location_type[mics_all$location_type=="0"]<-"admin0"
mics_all$location_type[mics_all$location_type=="0.5"]<- NA 


#changing values based on extraction template
mics_all$measure[mics_all$measure=="wealth_score"]<-"asset score"
mics_all$measure[mics_all$measure=="quintiles"]<-"asset percentile"




#Add column mics_all based on extraction template
mics_all$source <- "MICS"
mics_all$data_type <- "survey"
mics_all$location_id <- NA
mics_all$source_location_id <- NA
mics_all$source_location_type <- NA 
mics_all$multiplier <- 1
mics_all$denominator <- "total"
mics_all$value_type <- "aggregated"
mics_all$currency_detail <- NA 
mics_all$currency <- NA
mics_all$base_year <- NA
mics_all$notes <- NA
mics_all$geomatching_notes <- NA
mics_all$initials <- "BZ"


#check for NAS in asset score/percentile values and why:
test_value <-  mics_all[is.na(mics_all$value), ]

#List of NIDS checked by an analyst. See documentation.
value_checked <- c(595, 608, 687, 1089, 2209, 3161, 3922, 4694, 4916, 
                   4926, 7054, 7540, 8734, 8777, 10948, 12940, 27031, 
                   27044, 27069, 32189, 35493, 103973, 150526, 161587,
                   296646, 332558,7387, 12243, 7401, 56241, 91507, 155335)
test_value <- test_value[!test_value$nid %in% value_checked,]


if(nrow(test_value) > 0){
  print(paste("Check why there are NAs in value column: ", paste(unique(test_value$nid), collapse = ", ")))
}


#NA asset score/percentile value removal      
mics_all <- mics_all[!is.na(mics_all$value),]
mics_all <- data.table(mics_all)

# Pull location_ids based on location_codes from standard shapefiles in a few surveys where they are missing from geocodebooks
ad1_locs_info <- data.table(global_admin1_sf)[, .(location_code = ADM1_CODE, location_id_fill_ad1 = loc_id)]
ad2_locs_info <- data.table(global_admin2_sf)[, .(location_code = ADM2_CODE, location_id_fill_ad2 = loc_id)]

mics_all_ad1_fill <- mics_all[grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)]
mics_all_ad1_fill <- merge(mics_all_ad1_fill, ad1_locs_info, by = c("location_code"), all.x = TRUE)
nrow(mics_all_ad1_fill[is.na(location_id_fill_ad1)]) # should be 0
mics_all_ad1_fill[, location_id := location_id_fill_ad1][, location_id_fill_ad1 := NULL]
mics_all_ad2_fill <- mics_all[grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code)]
mics_all_ad2_fill <- merge(mics_all_ad2_fill, ad2_locs_info, by = c("location_code"), all.x = TRUE)
nrow(mics_all_ad2_fill[is.na(location_id_fill_ad2)]) # should be 0
mics_all_ad2_fill[, location_id := location_id_fill_ad2][, location_id_fill_ad2 := NULL]

mics_fill <- rbind(mics_all_ad1_fill, mics_all_ad2_fill)
mics_all_before <- copy(mics_all)
mics_all <- mics_all[!(grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)) & 
                       !(grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code))]
mics_all <- rbind(mics_all, mics_fill)
validate_merge_nrows(mics_all_before, mics_all)

# Fill location_ids based on standard shapefiles from stable shapefiles
# TODO - revisit this section once this ticket (https://jira.ihme.washington.edu/browse/LSAE-556) is resolved;
# ALB admin1 - 595, 608 - matched and validated reference table. slight differences in polygons but pretty much a 1:1 match. ref table not moved yet to central directory, so not yet matched.
# ARG admin1 - 137208 - matched and validated reference table. polygons are identical.
# AZE admin2 - 881 - matched and validated reference table. slightly more granular polygons in standard shape.
# BIH admin1 - 80731 - matched and validated reference table. polygons are identical.
# BIH admin2 - 1385 - matched and validated reference table. polygons are identical.
# BLR admin1 - 80733, 1039 - matched and validated reference table. polygons are identical.
# BLR admin2 - 435732 - matched and validated reference table. polygons are identical.
# CRI admin1 - 452953 - matched and validated reference table. polygons are identical.
# MKD admin1 - 7843, 91324 - Bill said to use this one with caution. imperfect alignment between stable and standard polygons. some strange behavior around Ohrid province.
# MKD admin2 - 453328 - Bill said to use this one with caution. imperfect alignment between stable and standard polygons. some strange behavior around Ohrid province.
stable_shapefile_nids <- unique(mics_all[shapefile_type=="stable"]$nid)
stable_shapefile_fill <- copy(mics_all)[nid %in% c(stable_shapefile_nids)]
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
mics_all <- rbind(mics_all[!nid %in% stable_shapefile_nids], stable_shapefile_fill_all)


###########################################
## Collapse data ##
###########################################

# split the data into point and polygon datasets
dpoint <- subset(mics_all, point == 1)
dpoly  <- subset(mics_all, point == 0)

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
dpoint_to_poly <- dpoint_to_poly[!is.na(loc_id)]
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
mics_all <- rbindlist(list(dpoint, dpoly), fill=T)
mics_all_to_poly <- rbindlist(list(dpoint_to_poly, dpoly), fill=T)

# Process SD
# Drop some observations where only one household was surveyed, since this is not representative of the region, 
# and because the very small SD throws off the SAE models
mics_all <- mics_all[!(N_households == 1 & sd_weighted < 1)]
mics_all_to_poly <- mics_all_to_poly[!(N_households == 1 & sd_weighted < 1)]
mics_all[is.na(sd_weighted), sd_weighted := sd_unweighted] # Correct a few clusters where SD goes missing
mics_all_to_poly[is.na(sd_weighted), sd_weighted := sd_unweighted] 
nrow(mics_all[is.na(sd_unweighted)]) + nrow(mics_all[is.na(sd_weighted)]) # should be 0
nrow(mics_all_to_poly[is.na(sd_unweighted)]) + nrow(mics_all_to_poly[is.na(sd_weighted)]) # should be 0

#Select columns for final dataset
mics_all <- mics_all[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                         "source_location_type",  "source_location_id", "lat",  "long", "location_code",  "shapefile", "measure", "denominator",
                         "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",  "currency_detail", 
                         "notes", "geomatching_notes", "initials")]
mics_all_to_poly <- mics_all_to_poly[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                                         "source_location_type",  "source_location_id", "location_code",  "shapefile", "measure", "denominator",
                                         "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",  "currency_detail", 
                                         "notes", "geomatching_notes", "initials")][, c("lat", "long") := NA]

###########################################
## Validate and Save ##
###########################################
validate_extractions(mics_all) # Note that the zero wealth score value for one location in NID 7387 has been validated in the raw data
validate_extractions(mics_all_to_poly)

#list of nids from share drive
y <- unique(mics_all$nid)
z <- unique(mics_all_to_poly$nid)

#Check NID count, x is count of codebook nids from top of script
nid_diff <- setdiff(x,y)
nid_diff <- nid_diff[!nid_diff %in% surveys_missing_geocodebooks]
nid_diff_to_poly <- setdiff(x,z)
nid_diff_to_poly <- nid_diff_to_poly[!nid_diff_to_poly %in% surveys_missing_geocodebooks]

if(length(nid_diff) > 0){
  print(paste0("Check that these NIDS have been dropped correctly: ", paste(unique(nid_diff)), collapse = ", "))
}
if(length(nid_diff_to_poly) > 0){
  print(paste0("Check that these NIDS have been dropped correctly: ", paste(unique(nid_diff_to_poly)), collapse = ", "))
}


save_extraction(mics_all, point_to_polygon = FALSE)
save_extraction(mics_all_to_poly, point_to_polygon = TRUE)