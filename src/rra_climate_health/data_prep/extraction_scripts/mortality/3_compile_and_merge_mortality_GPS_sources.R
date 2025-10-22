#' @Title: [rra_compile_and_merge_GPS_sources.R]
#' @Purpose: Compile mortality sources and merge to GPS sources
#' 
#' 

#SET-UP
rm(list=ls())

if (Sys.info()[1] == "Linux"){
  j <- "/home/j/"
  h <- paste0("/ihme/homes/", Sys.info()[7], "/")
  k <- "/ihme/cc_resources/"
} else if (Sys.info()[1] == "Windows"){
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
}

library(haven)
library(dplyr)
library(readr)
library(readxl)
library(stringr)
library(data.table)
library(haven)
library(arrow)

today_date <- "2025_10_14"

## Read, and compile data and compare to old file ##
###############################################

#Enter filepath with new date
compiled_df <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_combined_2025_10_14.parquet")
compiled_old <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_combined.parquet")

#Enter filepath when there are new re_extractions
re_extracted <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_reextracted_10_10_2025.parquet")

#Remove re_extracted NIDS and add them back in at the very end.
nids_missing <- unique(re_extracted$nid)
compiled_df <- compiled_df[!compiled_df$nid %in% nids_missing,]

#survey types we know could have GPS
# Convert to data.table
combined_dt <- as.data.table(compiled_df)

# Define survey types pattern
survey_types_pattern <- "DHS|UNICEF_MICS|WHO|RHS|PER|IRQ|IND"

# Filter using %like%
compiled_dt <- combined_dt[survey_name %like% survey_types_pattern]

###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################

# pull all available MBG and SAE geocodebooks
source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
geo_sae <- get_geocodebooks(nids = unique(compiled_df$nid), data_type = "sae")
geo_mbg <- get_geocodebooks(nids = unique(compiled_df$nid), data_type = "mbg")

#hot fix:21039 Uzbekistan Special Demographic and Health Survey 2002
geo_mbg$point <- ifelse(geo_mbg$nid == "21039" & geo_mbg$data_type == "mbg", 1, geo_mbg$point)
geo_mbg$survey_series <- ifelse(geo_mbg$nid == "21039" & geo_mbg$data_type == "mbg", "DHS_SPECIAL", geo_mbg$survey_series)

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
    if(
      any(unique(geo_mbg_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_") &
      !any(unique(geo_sae_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_")
    ) {
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

###############################################

# Merge extracted dataset and geography dataset together
###############################################
compiled_dt[, nid := as.character(nid)]
sae_mbg_bind[, nid := as.character(sae_mbg_bind$nid)]

data_all <- merge(sae_mbg_bind, compiled_dt, by.x=c("nid", "iso3", "geospatial_id"),
                 by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T, allow.cartesian = T)
data_all <- unique(data_all)
setnames(data_all, "iso3", "ihme_loc_id")
# setnames(data_all, "year_start.x", "year_start")
# setnames(data_all, "year_end.x", "year_end")

#add year_start, year_end
data_all <- select(data_all, "year_start", "year_end", "nid", "survey_name", "int_year", "int_month", "sex_id", "mothers_age_year", "aod_months", "age_month",
                   "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id", "hh_id", "hhweight", "pweight","birth_year", "birth_month", "child_alive", "lat", "long")

#Adding back into merge to GPS fiiles for the following NIDs: 
data_all <- rbind(data_all, re_extracted, fill = TRUE)

# ## MERGE Missing GPS files that were found or entered differently in GCDB ##
# # ###############################################
# # 20132 Kenya Demographic and Health Survey 1998 - matched! might need to go back to geocodebook
ken_dhs_gps <- read_excel("/snfs1/DATA/DHS_PROG_DHS/KEN/1998/KEN_DHS3_1998_GPS_Y2017M04D21.XLSX")
ken_dhs_gps <- ken_dhs_gps[ , (names(ken_dhs_gps) %in% c("cluster number", "Region","Lat", "Long"))]
ken_dhs_gps$`cluster number` <- as.character(ken_dhs_gps$`cluster number`)
kenya_nid <- subset(data_all, nid == 20132)
kenya_nid$lat <- NULL
kenya_nid$long <- NULL
kenya_data <- merge(kenya_nid, ken_dhs_gps, by.x = "geospatial_id", by.y = "cluster number", all.x = T, all.y = F)
setnames(kenya_data, "Lat", "lat")
setnames(kenya_data, "Long", "long")
kenya_data <- select(kenya_data,"nid", "survey_name", "year_start", "year_end", "int_year", "int_month", "sex_id", "mothers_age_year", "aod_months", "age_month", "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id", "hh_id", "hhweight", "pweight","birth_year", "birth_month", "child_alive", "lat", "long")

# #520553	Niger Malaria Indicator Survey 2021	 - merging manually since we can't get to winnower
ner_mis_gps <- read_dta("/snfs1/DATA/DHS_PROG_MIS/NER/2021/NER_MIS8_2021_GPS_NIGE81FL_Y2023M03D03.DTA")
ner_mis_gps <- ner_mis_gps[ , (names(ner_mis_gps) %in% c("dhsclust", "dhsregna","latnum", "longnum"))]
ner_mis_gps$dhsclust <- as.character(ner_mis_gps$dhsclust)
setnames(ner_mis_gps, "latnum", "lat")
setnames(ner_mis_gps, "longnum", "long")
ner_nid <- subset(data_all, nid == 520553)
ner_nid$lat <- NULL
ner_nid$long <- NULL
ner_data <- merge(ner_nid, ner_mis_gps, by.x = "geospatial_id", by.y = "dhsclust", all.x = T, all.y = F)
ner_data <- select(ner_data,"nid", "survey_name", "year_start", "year_end", "int_year", "int_month", "sex_id", "mothers_age_year", "aod_months", "age_month", "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id", "hh_id", "hhweight", "pweight","birth_year", "birth_month", "child_alive", "lat", "long")

# #538795	Tanzania Demographic and Health Survey 2022 - latlong in file, match
tza_dhs_gps <- read_dta("/snfs1/DATA/DHS_PROG_DHS/TZA/2022/TZA_DHS8_2022_GPS_TZGE81FL_Y2023M11D03.DTA")
tza_dhs_gps <- tza_dhs_gps[ , (names(tza_dhs_gps) %in% c("dhsclust", "dhsregna","latnum", "longnum"))]
tza_dhs_gps$dhsclust <- as.character(tza_dhs_gps$dhsclust)
setnames(tza_dhs_gps, "latnum", "lat")
setnames(tza_dhs_gps, "longnum", "long")
tza_nid <- subset(data_all, nid == 538795)
tza_nid$lat <- NULL
tza_nid$long <- NULL
tza_data <- merge(tza_nid, tza_dhs_gps, by.x = "geospatial_id", by.y = "dhsclust", all.x = T, all.y = F)
tza_data <- select(tza_data,"nid", "survey_name", "year_start", "year_end", "int_year", "int_month", "sex_id", "mothers_age_year", "aod_months", "age_month", "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id",  "hh_id", "hhweight", "pweight","birth_year", "birth_month", "child_alive", "lat", "long")

#527880	Bangladesh Demographic and Health Survey 2022 - I believe tis is in the geocodebook database now
bgd_dhs_gps <- read_dta("/snfs1/DATA/DHS_PROG_DHS/BGD/2022/BGD_DHS8_2022_GPS_BDGE81FL_Y2024M09D09.DTA")
bgd_dhs_gps <- bgd_dhs_gps[ , (names(bgd_dhs_gps) %in% c("dhsclust", "dhsregna","latnum", "longnum"))]
bgd_dhs_gps$dhsclust <- as.character(bgd_dhs_gps$dhsclust)
setnames(bgd_dhs_gps, "latnum", "lat")
setnames(bgd_dhs_gps, "longnum", "long")
bgd_nid <- subset(data_all, nid == 527880)
bgd_nid$lat <- NULL
bgd_nid$long <- NULL
bgd_nid$remove <- 1
bgd_data <- merge(bgd_nid, bgd_dhs_gps, by.x = "geospatial_id", by.y = "dhsclust", all.x = T, all.y = F)
bgd_data <- select(bgd_data,"nid", "survey_name", "year_start", "year_end", "int_year", "int_month", "sex_id", "mothers_age_year", "aod_months", "age_month", "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id", "hh_id", "hhweight", "pweight","birth_year", "birth_month", "child_alive", "lat", "long")


final <- rbind(data_all, kenya_data, ner_data,tza_data, bgd_data, fill = TRUE)
final <- unique(final)


gps_missing <- c(358824,375362,427778,462027,4779,487664,490966,505452,165390)

#remove nids from data set that have no GPS
final <- final[!nid %in% gps_missing]

#remove nids where the are values that are child_alive = 0 and they are missing aod_months in raw data
aod_missing <- c(218587, 218590, 23565,  27563, 27630, 439649, 91508, 108080, 111438, 157059, 218579, 218580, 218613, 286768, 286769, 286788, 350798, 359318, 413934, 475926, 484506, 500002, 510177,56828, 69806, 7738721622, 21622)
final <- final[!nid %in% aod_missing]

# Quality checks
###############################################

missing_summary <- final %>% #Look for 21309, 20132
  group_by(nid) %>%
  summarise(
    n = n(),  # total rows in group
    n_missing_colA = sum(is.na(lat)),  # count missing colA
    n_missing_colB = sum(is.na(long)),  # count missing colB
    pct_missing_colA = 100 * n_missing_colA / n,
    pct_missing_colB = 100 * n_missing_colB / n
  )

# Assuming missing_summary is already created
missing_100 <- missing_summary %>%
  filter(pct_missing_colA == 100 | pct_missing_colB == 100)

#Take this list and run it through the GHDX Record + NID Tool: https://internal-ghdx.healthdata.org/records-files-by-nid/
# Filter to GPS Files, or GPS in Abstract and notes
write.csv(missing_100, "/mnt/team/surge/pub/aserfe/rr/100_percent_missing_mortality.csv", row.names = FALSE)

#Only include lbwsg where lat/long is not 100 percent missing
final <- final[!final$nid %in% missing_100$nid, ]
###############################################

# Quality checks
###############################################
nid_add <- setdiff(final$nid, compiled_old$nid)
cat("NIDS ADDED:", unique(nid_add), "\n")

nid_drop <- setdiff(compiled_old$nid,final$nid)
cat("NIDS DROPPED:", unique(nid_drop), "\n")


final_sub <- select(final, "nid", "survey_name")
final_sub <- unique(final_sub)
table(final_sub$survey_name)

# Combine to make full file path
final_filepath <- paste0(output_folder, today_date, ".parquet")

write_parquet(final, final_filepath)


#Flagged
#21622 is dropped because demographics teams flagged it in codebook.
#358824	Peru Demographic and Family Health Survey 2017 - lat/long in codebook but not matching - try recreating the geospatial id in the geocodebook.

##NIDS WITH NO GPS INFO - 
# 375362	Laos Multiple Indicator Cluster Survey 2017
# 427778	Lesotho Multiple Indicator Cluster Survey 2018
# 462027	Kosovo Multiple Indicator Cluster Survey 2019-2020
# 4779	Guatemala Reproductive Health Survey 2008-2009
# 487664	Fiji Multiple Indicator Cluster Survey 2021
# 490966	Honduras Multiple Indicator Cluster Survey 2019
# 505452	Nigeria Multiple Indicator Cluster Survey and National Immunization Coverage Survey 2021
# 165390 India District Level Household Survey 2012-2014 - vetted, and unable to match due to parsing of latitude and longitude degrees.
