#' @Title: [2_mortality_reextraction_data_compile.R]
#' @Purpose: Compile re-extracted mortality sources 
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

## List files, read, and compile data ##
###############################################
#CHANGE DATE
re_extract_date <- "2025_10_10"

output_folder <- '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/mort_rextractions/'   
  
#Added all the Winnower re-_extracted files here in a parquet file:
compiled_df <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/mort_reextractions/mort_reextract_dta_2025_10_06.parquet")
 
compiled_df <- unique(compiled_df)

# hot fix: 21622	Ghana World Health Survey 2003
#Replace geospatial id with hh_id
compiled_df$geospatial_id[compiled_df$nid == "21622"] <- compiled_df$hh_id[compiled_df$nid == "21622"]

#hot fix: changing parent nid to child nid to match gps by year
compiled_df$nid[compiled_df$nid == "275090" & compiled_df$int_year == "2003"] <- 449435 
compiled_df$nid[compiled_df$nid == "275090" & compiled_df$int_year == "2004"] <- 449435
compiled_df$nid[compiled_df$nid == "275090" & compiled_df$int_year == "2005"] <- 449446
compiled_df$nid[compiled_df$nid == "275090" & compiled_df$int_year == "2006"] <- 449449
compiled_df$nid[compiled_df$nid == "275090" & compiled_df$int_year == "2007"] <- 449450
compiled_df$nid[compiled_df$nid == "275090" & compiled_df$int_year == "2008"] <- 449451
#The year_start and year_end are all 2003 to 2008, so we can use interview year to match?

## Read in, prepare geocodebooks ##
## Currently, using MBG with "point = 1" but good to check that mbg and sae are duplicative or not.
###############################################
###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################

# pull all available MBG and SAE geocodebooks
source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
geo_sae <- get_geocodebooks(nids = unique(compiled_df$nid), data_type = "sae")
geo_mbg <- get_geocodebooks(nids = unique(compiled_df$nid), data_type = "mbg")

#hot fix: Removing admin 1 shapefiles from NID where there is now lat/long data, NID 20315 MLI 2001
# Remove rows where point == 0 and nid == 20315
geo_mbg <- geo_mbg[!(point == 0 & nid == 20315)]

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
compiled_dt <- as.data.table(compiled_df)
compiled_dt[, nid := as.character(nid)]
sae_mbg_bind[, nid := as.character(sae_mbg_bind$nid)]

data_all <- merge(sae_mbg_bind, compiled_dt, by.x=c("nid", "iso3", "geospatial_id"),
                  by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T, allow.cartesian = T)
data_all <- unique(data_all)
setnames(data_all, "iso3", "ihme_loc_id")

#hot fix: changing child nid to parent nid 
data_all$nid[data_all$nid == "449435" & data_all$int_year == "2003"] <- "275090"
data_all$nid[data_all$nid == "449435" & data_all$int_year == "2004"] <- "275090"
data_all$nid[data_all$nid == "449446" & data_all$int_year == "2005"] <- "275090"
data_all$nid[data_all$nid == "449449" & data_all$int_year == "2006"] <- "275090"
data_all$nid[data_all$nid == "449450" & data_all$int_year == "2007"] <- "275090"
data_all$nid[data_all$nid == "449451" & data_all$int_year == "2008"] <- "275090"


data_all <- select(data_all,"nid", "year_start", "year_end", "int_year", "int_month", "sex_id", "mothers_age_year", "aod_months", "age_month", "ihme_loc_id", "geospatial_id", "psu", "psu_id", "strata", "strata_id", "line_id",  "hh_id", "hhweight", "pweight","birth_year", "birth_month", "child_alive", "lat", "long")

# Quality checks
###############################################
#Were any new nids added?
nid_add <- setdiff(combined_df$nid, compiled_old$nid)
cat("NIDS ADDED:", unique(nid_add), "\n")

nid_drop <- setdiff(compiled_oldf$nid, combined_df$nid)
cat("NIDS DROPPED:", unique(nid_drop), "\n")

# Combine to make full file path
final_filepath <- paste0(output_folder, re_extract_date, ".parquet")

write_parquet(final, final_filepath)

