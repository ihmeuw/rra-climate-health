## Purpose: Merging CGF and wealth data files

## Setup
rm(list = ls())
pacman::p_load(readr)

## Reading in CGF and wealth data
cgf <- setDT(read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/2025_01_13.csv'))
wealth <- setDT(read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/extracted_ALL_compiled_processed_point_and_polygon.csv'))


# Function to convert data types of merge keys to character for consistency in merging
merge_keys <- c('hh_id','nid','strata','psu','year_start','year_end')
convert_keys_to_char <- function(dt, keys) {
  for (key in keys) {
    dt[, (key) := as.character(get(key))]
  }
  return(dt)
}

# Convert the data types of merge keys in both datasets
cgf <- convert_keys_to_char(cgf, merge_keys)
wealth <- convert_keys_to_char(wealth, merge_keys)

## Merging
tmp <- merge(cgf, wealth[, !c('file_path','survey_module')], 
             by.x = c('hh_id','nid','strata','psu','ihme_loc_id','year_start','year_end','geospatial_id','survey_name','int_year'), 
             by.y = c('hh_id','nid','strata','psu','iso3','year_start','year_end','geospatial_id','source','year'), 
             all.x = T)

## Renaming weight to hhweight for clarity
tmp <- setnames(tmp, 'weight', 'hhweight')

write.csv(tmp, '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/merged_cgf_wealth.csv')
