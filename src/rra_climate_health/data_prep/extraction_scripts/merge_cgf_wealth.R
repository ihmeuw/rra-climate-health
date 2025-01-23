## Authors: Kristin Hong, Maya Oleynikova
## Purpose: Merging CGF and wealth data files
## Date: 1/15/25

## Setup
rm(list = ls())
pacman::p_load(readr,dplyr,data.table,arrow)

## Reading in CGF and wealth data
cgf <- setDT(read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/cgf_data_prep.csv'))
wealth <- setDT(read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/extracted_ALL_compiled_processed_point_and_polygon.csv'))


# Function to convert data types of merge keys to character for consistency in merging
merge_keys_cgf <- c('hh_id','nid','strata','psu','psu_id')
merge_keys_wealth <- c('hh_id','nid','strata','psu')
convert_keys_to_char <- function(dt, keys) {
  for (key in keys) {
    dt[, (key) := as.character(get(key))]
  }
  return(dt)
}

# Convert the data types of merge keys in both datasets
cgf <- convert_keys_to_char(cgf, merge_keys_cgf)
wealth <- convert_keys_to_char(wealth, merge_keys_wealth)

## Merging
tmp <- merge(cgf, wealth[, !c('file_path','survey_module','year_start','year_end','year','geospatial_id','source')], 
             by.x = c('hh_id','nid','strata','psu_id','ihme_loc_id'), 
             by.y = c('hh_id','nid','strata','psu','iso3'), 
             all.x = T)

## Renaming weight to hhweight for clarity
tmp <- setnames(tmp, 'weight', 'hhweight')

## Combining lat/latitude and long/longitude columns together
tmp[, lat := coalesce(lat, latitude)]
tmp[, long := coalesce(long, longitude)]

tmp[, c("latitude", "longitude") := NULL]

## Rename ihme_loc_id to iso3
tmp <- setnames(tmp, 'ihme_loc_id', 'iso3')

write.csv(tmp, '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/merged_cgf_wealth.csv')
write_parquet(tmp, '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/merged_cgf_wealth.parquet')
