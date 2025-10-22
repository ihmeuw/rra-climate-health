#' @Title: [compile_lbwsg_sources.R]
#' @Purpose: Compile lbwsg sources.


#SET-UP
#(list=ls())

if (Sys.info()[1] == "Linux"){
  j <- "/home/j/"
  h <- paste0("/ihme/homes/", Sys.info()[7], "/")
  k <- "/ihme/cc_resources/"
} else if (Sys.info()[1] == "Windows"){
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
}


library(haven)      # For read_dta
library(data.table) # For fast splitting and writing
library(dplyr)
library(readr)
library(readxl)
library(stringr)
library(haven)
library(arrow)
library(purrr)

# Set-up
###############################################

today_date <- "2025_10_21"# change when updating and saving data

input_folder_j <- "/mnt/team/nch/pub/lbwsg/gbd25_microdata_extractions/"

input_folder_l <- "/ihme/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/lbwsga/2025_06_30/"

input_folder_reext <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/lbwsg/re_extract/"

# Change to your folder
output_folder <- '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/lbwsg/' 

#Change file here with the last final dataset
compiled_old <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/lbwsg/2025_10_15.parquet")

# Load data
###############################################
#List .dta files, excluding those with "VR"  and vital in the path
dta_files <- list.files(path = input_folder_j, pattern = "(DHS|UNICEF_MICS|KEN_WMS|WB_LSMS).*\\.dta$", full.names = TRUE)

# Exclude files with "VR" and files with "vital"
dta_files <- dta_files[!grepl("VR", dta_files) & !grepl("vital", dta_files)]
n_files <- length(dta_files)
cat("Found", n_files, ".dta files in", input_folder_j, "(excluding files with 'VR' and 'Vital')\n")

dta_list <- list()
for (i in seq_along(dta_files)) {
  f <- dta_files[i]
  percent <- round(i / n_files * 100, 1)
  cat(sprintf("Reading file %d/%d (%.1f%%): %s\n", i, n_files, percent, basename(f)))
  dta_list[[i]] <- as.data.table(read_dta(f))
}
compiled_dt_j <- rbindlist(dta_list, fill = TRUE)
cat("All files read and combined.\n")

##############List .dta files in L drive
dta_files_l <- list.files(path = input_folder_l, pattern = "(DHS|UNICEF_MICS|BWA_FAMILY|WB_LSMS).*\\.dta$", full.names = TRUE)
n_files <- length(dta_files_l)
dta_list_l <- list()
for (i in seq_along(dta_files_l)) {
  f <- dta_files_l[i]
  percent <- round(i / n_files * 100, 1)
  cat(sprintf("Reading file %d/%d (%.1f%%): %s\n", i, n_files, percent, basename(f)))
  dta_list_l[[i]] <- as.data.table(read_dta(f))
}
compiled_dt_l <- rbindlist(dta_list_l, fill = TRUE)
cat("All files read and combined.\n")


##############List .dta files; originally re_extracted following GPS data available on DHS site, additionally there were more surveys in LBWSG codebook.
dta_files_reext <- list.files(path = input_folder_reext, pattern = "*\\.dta$", full.names = TRUE)
n_files <- length(dta_files_reext)
dta_list_reext <- list()
for (i in seq_along(dta_files_reext)) {
  f <- dta_files_reext[i]
  percent <- round(i / n_files * 100, 1)
  cat(sprintf("Reading file %d/%d (%.1f%%): %s\n", i, n_files, percent, basename(f)))
  dta_list_reext[[i]] <- as.data.table(read_dta(f))
}
compiled_dt_reext <- rbindlist(dta_list_reext, fill = TRUE)
cat("All files read and combined.\n")

# Quality check for re-extracted data and Combine
###############################################
dhs_filled_out_in_cb <- c(69761,111432,218555,218582,270404,270469,270470,270471,286772,411301,413167,425283,450419,459845,459854,460813,462482,
                             470667,467681,493562,527435,440151,528571,529017,529525,18834,19035,55956,157021,157050,157064,218565,210182,218574,218592,
                             286781,286783,337877,341838,353526,356955,393876,396957,398033,408484,413666,413667,426238,446877,449435,449446,449449,449450,
                             449451,523643,19950,20322,527622,535004,538795,275090,527726,553572,553289,560560,527880,560575) #65 NIDS



#What's in the current dataset?
compiled_dt <- rbind(compiled_dt_j, compiled_dt_l, compiled_dt_reext, fill = TRUE)

#Find which values in dhs_filled_out_in_cb are NOT in compiled dataset
missing_nids <- dhs_filled_out_in_cb[!dhs_filled_out_in_cb %in% compiled_dt$nid]

if (length(missing_nids) == 0) {
  cat("All values in dhs_filled_out_in_cb are present in compiled_dt$nid!\n")
} else {
  cat("The following values are missing from compiled_dt$nid:\n")
  print(missing_nids)
}

#If NIDS are missing go back and re-extract from winnower

#210182 no data in winnmill for br or ch
#19950 no data in winnmmill


#How many are in the current dataset?
length(unique(compiled_dt$nid))

#hot fix: changing parent nid to child nid to match gps by year
compiled_dt$nid[compiled_dt$nid == "275090" & compiled_dt$int_year == "2003"] <- 449435 
compiled_dt$nid[compiled_dt$nid == "275090" & compiled_dt$int_year == "2004"] <- 449435
compiled_dt$nid[compiled_dt$nid == "275090" & compiled_dt$int_year == "2005"] <- 449446
compiled_dt$nid[compiled_dt$nid == "275090" & compiled_dt$int_year == "2006"] <- 449449
compiled_dt$nid[compiled_dt$nid == "275090" & compiled_dt$int_year == "2007"] <- 449450
compiled_dt$nid[compiled_dt$nid == "275090" & compiled_dt$int_year == "2008"] <- 449451

unique(compiled_dt$nid)

###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################
# pull all available MBG and SAE geocodebooks 
 # source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
 # geo_sae <- get_geocodebooks(nids = unique(compiled_dt$nid), data_type = "sae")
 # geo_mbg <- get_geocodebooks(nids = unique(compiled_dt$nid), data_type = "mbg")

#ALTERNATIVE: Read in mbg, whatever doesn't have lat/long check in sae side of GCDB afterwards
geo_mbg <- read.csv("/mnt/team/surge/pub/aserfe/rr/mbg_geocodebook_results_2025_10_21.csv")
#geo_sae #<-   All extra missing lat/long NIDS were checked against the sae files in the database.

#hot fix:21039 Uzbekistan Special Demographic and Health Survey 2002
geo_mbg$point <- ifelse(geo_mbg$nid == "21039" & geo_mbg$data_type == "mbg", 1, geo_mbg$point)
geo_mbg$survey_series <- ifelse(geo_mbg$nid == "21039" & geo_mbg$data_type == "mbg", "DHS_SPECIAL", geo_mbg$survey_series)


# Identify surveys with both MBG and SAE geocodebooks, and keep the preferred geomatching info
#geocodebooks_overlap <- intersect(unique(geo_sae$nid), unique(geo_mbg$nid))

# Prefer MBG codebooks if they have only point data,
# OR if they have lbd_standard_admin_1/2 geomatching AND the SAE codebook does not have lbd_standard_admin_1/2 geomatching (may have stable shapefile matching)
# Otherwise, prefer SAE codebooks for all other surveys
#preferred_mbg <- c()
# preferred_sae <- c()
geo_mbg_subset <- geo_mbg[!(is.na("point") & is.na("lat") & is.na("long") & is.na("shapefile"))]
# geo_sae_subset <- geo_sae[!(is.na(point) & is.na(lat) & is.na(long) & is.na(shapefile))]
# for(nid_check in geocodebooks_overlap) {
#   if(all(unique(geo_mbg_subset[nid == nid_check]$point) == 1)) {
#     preferred_mbg <- c(preferred_mbg, nid_check)
#   } else {
#     if(
#       any(unique(geo_mbg_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_") &
#       !any(unique(geo_sae_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_")
#     ) {
#       preferred_mbg <- c(preferred_mbg, nid_check)
#     } else {
#       preferred_sae <- c(preferred_sae, nid_check)
#     }
#   }
# }

# geo_mbg <- geo_mbg[!nid %in% preferred_sae]
# geo_sae <- geo_sae[!nid %in% preferred_mbg]
# geocodebooks_overlap <- intersect(unique(geo_sae$nid), unique(geo_mbg$nid)) # should now be an empty list
# sae_mbg_bind <- rbind(geo_sae, geo_mbg)
# sae_mbg_bind <- unique(sae_mbg_bind) # Removing duplicate rows from buggy geocodebooks

sae_mbg_bind <- geo_mbg_subset
# Merge extracted dataset and geography dataset together
###############################################
compiled_dt[, nid := as.character(nid)]
#sae_mbg_bind[, nid := as.character(sae_mbg_bind$nid)]

data_all <- merge(sae_mbg_bind, compiled_dt, by.x=c("nid", "iso3", "geospatial_id"),
                  by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T, allow.cartesian = T)
data_all <- unique(data_all)
setnames(data_all, "iso3", "ihme_loc_id")

#hot fix: changing child nid to parent nid for PER 275090
data_all$nid[data_all$nid == "449435" & data_all$int_year == "2003"] <- "275090"
data_all$nid[data_all$nid == "449435" & data_all$int_year == "2004"] <- "275090"
data_all$nid[data_all$nid == "449446" & data_all$int_year == "2005"] <- "275090"
data_all$nid[data_all$nid == "449449" & data_all$int_year == "2006"] <- "275090"
data_all$nid[data_all$nid == "449450" & data_all$int_year == "2007"] <- "275090"
data_all$nid[data_all$nid == "449451" & data_all$int_year == "2008"] <- "275090"

data_all <- select(data_all, "year_start", "year_end", "nid","survey_name", "int_year", "int_month", "sex_id", "birth_card", "birth_order", "birth_weight", "birth_weight_unit", "maternal_ed_yrs", "mother_age_month", "mother_height", "mother_weight", "paternal_ed_yrs",
                   "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id", "hh_id", "hhweight", "pweight", "paternal_ed_yrs", "wealth_index_dhs", "lat", "long")

# ## MERGE Missing GPS files that were found or entered differently in GCDB ##
# # ###############################################
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
tza_data <- select(tza_data,"year_start", "year_end", "nid","survey_name", "int_year", "int_month", "sex_id", "birth_card", "birth_order", "birth_weight", "birth_weight_unit", "maternal_ed_yrs", "mother_age_month", "mother_height", "mother_weight", "paternal_ed_yrs",
                   "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id", "hh_id", "hhweight", "pweight", "paternal_ed_yrs", "wealth_index_dhs", "lat", "long")

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
bgd_data <- select(bgd_data, "year_start", "year_end", "nid","survey_name", "int_year", "int_month", "sex_id", "birth_card", "birth_order", "birth_weight", "birth_weight_unit", "maternal_ed_yrs", "mother_age_month", "mother_height", "mother_weight", "paternal_ed_yrs",
                   "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id", "hh_id", "hhweight", "pweight", "paternal_ed_yrs", "wealth_index_dhs", "lat", "long")

#Removing NIDS from data_all that were creating duplicates
# Define the nids to remove
nids_to_remove <- c(527880, 538795)
# Remove rows 
data_all <- data_all[!data_all$nid %in% nids_to_remove, ]


final <- rbind(data_all, tza_data, bgd_data, fill = TRUE)
final <- unique(final)

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
#Filter to GPS Files.
write.csv(missing_100, "/mnt/team/surge/pub/aserfe/rr/100_percent_missing_lbwsg.csv", row.names = FALSE)

#Only include lbwsg where lat/long is not 100 percent missing
final <- final[!final$nid %in% missing_100$nid, ]

###############################################

# Quality checks
###############################################
#Were any new nids added to the demographics team folder?
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

