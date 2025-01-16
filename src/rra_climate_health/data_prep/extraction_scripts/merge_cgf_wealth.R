## Purpose: Merging CGF and wealth data files

## Setup
rm(list = ls())
pacman::p_load(readr,dplyr)

## Reading in CGF and wealth data
cgf <- setDT(read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/2025_01_16.csv'))
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

## Filtering out 275090 since CGF separates the data by year while wealth aggregates 2003-2008
cgf_filtered <- filter(cgf, nid == 275090)
wealth_filtered <- filter(wealth, nid == 275090)

## Merging 275090 separately
tmp_275090 <- merge(cgf_filtered, wealth_filtered[, !c('file_path','survey_module','year_start','year_end','year')],
             by.x = c('hh_id','nid','strata','psu','ihme_loc_id','geospatial_id','survey_name'),
             by.y = c('hh_id','nid','strata','psu','iso3','geospatial_id','source'),
             all.x = T)

## Merging 275090 onto main tmp
tmp <- merge(tmp, tmp_275090,
             by = c('hh_id', 'nid', 'strata', 'psu', 'ihme_loc_id', 'year_start', 'year_end', 'geospatial_id', 'survey_name', 
                    'int_year', 'sex_id', 'age_wks', 'age_cat_1', 'metab_height_rounded', 'age_mo', 'admin_1', 
                    'birth_order', 'birth_weight_unit', 'file_path', 'maternal_ed_yrs', 'metab_height', 'metab_height_unit', 'metab_weight', 'metab_weight_unit', 
                    'mother_height', 'mother_weight', 'paternal_ed_yrs', 'survey_module', 'urban', 'wealth_index_dhs', 
                    'pweight', 'psu_id', 'strata_id', 'line_id', 'age_year', 'age_month', 'age_day', 'int_month', 'int_day', 
                    'bmi', 'overweight', 'obese', 'admin_1_mapped', 'admin_1_id', 'birth_weight', 'mother_age_month', 'pregnant', 'wealth_factor', 
                    'admin_2', 'admin_3', 'latitude', 'longitude', 'admin_4', 'age_cat_2', 'orig.psu', 'indicator', 'N', 'BMI', 
                    'bmi_at_18yrs_16', 'bmi_at_18yrs_17', 'bmi_at_18yrs_25', 'bmi_at_18yrs_30', 'HAZ', 'WAZ', 'WHZ', 'BMIZ', 'WHZ_seas', 'WHZ_seas_ind', 
                    'overweight_iotf_b', 'obese_iotf_b', 'undernutrition_iotf_b', 'overweight_who_BMIZ', 'obese_who_BMIZ', 'undernutrition_who_BMIZ_mod_b', 'undernutrition_who_BMIZ_sev_b', 
                    'stunting_mod_b', 'stunting_sev_c', 'stunting_mil_c', 'stunting_mod_c', 'low_bw_prev', 'wasting_mod_b', 'wasting_sev_c', 'wasting_mil_c', 'wasting_mod_c', 'obese_who_b', 
                    'overweight_who_b', 'underweight_mod_b', 'underweight_sev_c', 'underweight_mil_c', 'underweight_mod_c', 'stunting_overweight', 
                    'HAZ_drop', 'HAZ_exclude', 'WHZ_drop', 'WHZ_exclude', 'WAZ_drop', 'WAZ_exclude', 
                    'V1', 'team', 'data_type', 'point', 'lat', 'long', 'shapefile', 'location_code', 'location_type', 'shapefile_type', 'weight', 
                    'measure', 'value', 'location_id', 'multiplier', 'denominator', 'value_type', 'base_year', 'initials','...1','admin_2_mapped','admin_2_id','admin_1_urban_id','admin_1_urban_mapped'),
             all.x = T)

## Renaming weight to hhweight for clarity
tmp <- setnames(tmp, 'weight', 'hhweight')

write.csv(tmp, '/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/merged_cgf_wealth.csv')
