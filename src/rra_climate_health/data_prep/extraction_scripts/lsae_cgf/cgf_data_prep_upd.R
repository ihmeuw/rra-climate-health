#####################################################################################
## Author: Rebecca Stubbs
## Modified by: Brandon Pickering, Alice Lazzar-Atwood
## Date: August 1, 2017
## Purpose: Read in and clean raw UbCov data (for child growth failure)
## Email: alicela@uw.edu
##
## Steps:
##
## 1. Set up
##
## 2. Read in and standardize age formats and columns from individual data sources
##
## 3. Set Drop and Exclude Data Criteria within a function applied to the dataset
##
## 4. Add on information from reference material, calculate Z-scores
##
## 5. Add in Ethiopia data
##
## 6. Seasonality Adjustment 
##
## 7. Generate Indicators
##
## 8. Add Somalia data, run the drop data function defined in step #3
##
## 9. Save a copy of data for GBD
##
## 10. Apply exlclude criteria
##
## 11. Define indicators
##
## 12. Collapse data by indicator
##
## 13. Run data coverage plots
##
## 14. Run data prevalence plots
##
## 15. Resample polygons
##
## 16. Final processing steps, save final datasets
##
## 17. Generate exclusions list
##
## Notes: 
## Input data is the output .csv from undernutrition_post_extraction_processing.R
## metab_height should be in centimeters, and metab_weight should be in kg.
## Sex should be coded as female-0,male=1.
## As of 8/1/2017, this code can be run locally as well as on the cluster-- this may change
## as the extracts grow.
## As of 12/6/2018, I would not recommend running this code locally. Also, the current resample method requires the cluster
############################################################################################################################################################################################

###########################################################################################################################################################################################
## 1.) SETUP: Define filepaths and variables.
##########################################################################################################################################################################################

rm(list=ls())
topic <- "cgf" # This is just to fill out some filepaths
proj <- 'dbm'  # Specify which project you want (options are 'cgf', 'dbm', 'risk_factor' or 'both') and the code will only run for relevant indicators
extract_date <- "2025_01_08" # The date that you ran the post_extraction script
previous_date <- "2025_01_08" # If you want to run coverage plots of only new data,the plots will plot all data incorporated since this date
cores <- '30'
cov_plot <- TRUE # Assign TRUE if you want to run coverage plots, assign FALSE if you just want to collapse!
plot_new_dcp <- FALSE # Assign TRUE if you just want to plot new data, and FALSE is you want to plot all data
sex_split <- FALSE # TRUE if you want to collapse indicators by sex, FALSE if you don't need to
plot_extract <- FALSE # TRUE if you want to run coverage plots for only data that was extracted from reports
new.microdata.path <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/1_raw_extractions/"
all.files <- list.files(new.microdata.path)


# Define file paths
root <- ifelse(Sys.info()[1]=="AM", "J:/", "/home/j/")
j <- ifelse(Sys.info()[1]=="Windows", "J:/", "/home/j/")
l <- ifelse(Sys.info()[1]=="Mac", "Volumes/", "/mnt/team/")
ref_dir<-paste0(root,"WORK/11_geospatial/10_mbg/child_growth_failure/01 reference/")
work_dir <- paste0(l, "rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/1_raw_extractions")

folder_out <- paste0(l, "rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/", topic, "/test")  
# 
setwd(work_dir)

# Load libraries and  MBG project functions.
message("Loading packages...")

# Load packages that aren't included in MBG setup
user<- Sys.info()['user']    
package_list <- c('survey','foreach', 'snow', 'haven', 'gdata', 'readstata13','readr','dplyr','data.table')
for(package in package_list) {
  library(package, character.only=TRUE)
}  

user <- Sys.info()['user']
core_repo  <- ""
if (!dir.exists(core_repo)) core_repo <- "/share/code/geospatial/alicela/lbd_core/"

package_list <- readLines(paste0(core_repo, "/mbg_central/share_scripts/common_inputs/package_list.csv"))

# Load libraries and  MBG project functions.
source(paste0(core_repo, 'mbg_central/setup.R'))
# mbg_setup(package_list = package_list, repos = core_repo)

# Sourcing data coverage plot functions because since it doesn't have "functions" in the file name it isn't included in mbg_setup
source(paste0(core_repo, 'mbg_central/graph_data_coverage.R'))

module_date <- Sys.Date()
module_date <- gsub("-", "_", module_date)  

input_data_csv<-paste0(l,"rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/1_raw_extractions/", all.files)
input_data_csv <- input_data_csv[grep(("archive"), input_data_csv, invert = TRUE)]
input_data_csv <- input_data_csv[grep("dta", input_data_csv, invert = TRUE)]
input_data_csv <- input_data_csv[grep("wealth", input_data_csv, invert = TRUE)]
input_data_csv <- input_data_csv[grep("excluded", input_data_csv, invert = TRUE)]

#######################################################################################################################################################
## 2.) PREPARE RAW EXTRACT DATA: Read in data, calculate age, define geospatial id, 
#######################################################################################################################################################
# Load in UbCov data
# data<-fread(input_data_csv)

data_list <- lapply(input_data_csv, function(f) {
  read_csv(f, show_col_types = FALSE)
})

data_list <- lapply(data_list, function(df) {
  df %>% mutate(across(everything(), as.character))
})

# Combine all data frames into one
data <- bind_rows(data_list)

data <- as.data.table(data)

# Reading this in as a .csv instead of a .rda or .rdata keeps data the NA values as NA instead of NaN
data_raw<-copy(data)  
startcount<-nrow(data)

# Get list of current and previous NIDs for the coverage plots
# if (plot_new_dcp==TRUE){
#   previous <- fread(paste0(l,'LIMITED_USE/LU_GEOSPATIAL/geo_matched/cgf/test/',previous_date, '.csv'))
#   previous <- unique(previous$nid)
#   current <- unique(data$nid)
# }

# Calculating Age
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Calculate age in weeks:
data$age_day <- as.numeric(data$age_day)
data$age_month <- as.numeric(data$age_month)
data[, age_wks:=as.numeric(age_day/7.0192)] # (365 days / 52 weeks = 7.0192)
data[is.na(age_wks), age_wks:=age_month*4.333] # If it's NA (they didn't have age in days), base the age on if they have age in months
data[, age_wks:=as.integer(plyr::round_any(age_wks, 1))] # rounding to integer number of weeks
data[, age_mo:=age_wks/4.333] # Define ages in months as weeks, divided by 4.333
data[, age_mo:=as.integer(plyr::round_any(age_mo, 1))]

# Define Age Categories. Changed cat_2 categories from "0-2" and "2-5" to 1 and 2 because Excel converts "2-5" to a date when reference csv is opened. 
data[, age_cat_1:=ifelse(age_wks <= 104, 1, 2)] #
data[, age_cat_2:= "0-5"]

# Setting Geospatial ID
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## (1) we take geospatial_id as PSU (we just need a unique cluster identifier)
data[,orig.psu:=psu]
data[,psu:=geospatial_id]

## unless there are more psus than geospatial_id - than take psu back (should only be an issue for early extracts)
na.geo <- aggregate(is.na(geospatial_id) ~ nid, data, mean)
na.psu <- aggregate(is.na(orig.psu) ~ nid, data, mean)
colnames(na.geo)[2] <- "nas.g"
colnames(na.psu)[2] <- "nas.p"
nas <- merge(na.geo, na.psu)
use.psu <- nas$nid[which(nas$nas.g > nas$nas.p)]
psu.r <- which(data$nid %in% use.psu)
data[psu.r, 'psu'] <- data[psu.r, 'orig.psu']

## (2) if there are more household weights than pweights,we use those instead
# na.pweight  <- aggregate(is.na(pweight) ~ nid, data, mean)
# colnames(na.pweight)[2] <- "nas.p"
# na.hhweight <- aggregate(is.na(hhweight) ~ nid, data, mean)
# colnames(na.hhweight)[2] <- "nas.h"
# nas <- merge(na.hhweight, na.pweight)
# use.hh <- nas$nid[which(nas$nas.h < nas$nas.p)]
# hh.r <- which(data$nid %in% use.hh)
# data[hh.r, 'pweight'] <- data[hh.r, 'hhweight']
# 
# ## set variables to keep and subset
# vars_to_keep <- c("nid", 
#                   "psu", 
#                   "psu_id",
#                   # "source", 
#                   "file_path",
#                   # "country",
#                   "ihme_loc_id",
#                   # "start_year",    ## The adjusted start year, based on int year. It will be changed to its average within each geography at the collapse step
#                   "year_start",    ## The start year of the survey, from ubCov 
#                   # "end_year", 
#                   "year_end",
#                   #"year_svy",
#                   #"year_exp",
#                   "admin_1",
#                   "admin_1_mapped",
#                   "admin_1_id",
#                   "admin_2",
#                   "pweight", 
#                   "strata_id",
#                   "strata",
#                   "sex_id", 
#                   "age_wks", 
#                   "age_mo", 
#                   "age_year",
#                   "age_cat_1",
#                   "age_cat_2",
#                   # "height_age_sd",    ## Pulling z-scores through for the one case where it's all we're given--NID 23017
#                   # "weight_age_sd",    ## Pulling z-scores through for the one case where it's all we're given--NID 23017
#                   # "weight_height_sd", ## Pulling z-scores through for the one case where it's all we're given--NID 23017
#                   "birth_weight", 
#                   "birth_weight_unit",
#                   "birth_order",
#                   "metab_weight",
#                   "metab_height_unit",
#                   "metab_weight_unit",
#                   "bmi",
#                   "overweight",
#                   "obese",
#                   "wealth_index_dhs",
#                   "wealth_factor",
#                   "mother_weight",
#                   "mother_height",
#                   "maternal_ed_yrs",
#                   "paternal_ed_yrs",
#                   # "point", 
#                   "latitude",
#                   "longitude",
#                   "orig.psu",
#                   #"uncertain_point", 
#                   #"buffer", 
#                   #"location_name", 
#                   #"admin_level"
#                   # "location_code",
#                   # "shapefile", 
#                   "int_month", 
#                   "int_year",
#                   "int_day",
#                   "hh_id", 
#                   "line_id",
#                   # "birth_weight_card", 
#                   "geospatial_id", 
#                   "urban",
#                   "survey_name",
#                   "survey_module",
#                   "pregnant"
#                   #"cluster_number", 
#                   # "exclude_weights",
#                   # "exclude_age_range",
#                   # "exclude_age_granularity",
#                   # "exclude_data_lbw",
#                   # "exclude_data_cgf",
#                   # "exclude_representation",
#                   # "exclude_geography",
#                   # "exclude_interview",
#                   # "exclude_longitudinal",
#                   # "exclude_BMIZ",
#                   # "exclude_duplicative"
#                   )
# 
# # Subset to only relevant variables
# data<-data[, vars_to_keep, with=F]

if((startcount-nrow(data))>0)stop("!!! WHOA! There aren't the same number of rows coming out of this process as there are going in. Check on this.")
rm(na.geo,na.hhweight,na.psu,na.pweight,nas,hh.r,psu.r,use.hh,use.psu,vars_to_keep) # Cleaning up the work environment

# Adding 'indicator' and'N' columns:
data[,indicator:=1]
data[,N:=1]

########################################################################################################################################################
## 3.) DEFINE DROP/EXCLUDE CRITERIA: Create function to drop rows that don't meet required criteria
########################################################################################################################################################

#' DropData
#'
#' @description This function describes what data are dropped from the CGF indicators.
#'
#' @param data The data frame that needs things dropped.
#' @param indicator A string input, one of the following: "HAZ","WAZ","WHZ","BMI".
#'
#' @return  This function returns a modified data.table with the following
#' columns: 'drop', with 1s where the data should be dropped due to missing information
#' or pre-processing concerns, and 'exclude', where data will be excluded based on
#' cutoffs or criteria. After having

DropData <- function(data, indicator) {
  # Check if the indicator is valid
  if (!indicator %in% c("HAZ", "WHZ", "WAZ", "BMI", "CIAF", "birth_weight", "BMIZ")) {
    stop("You have not entered a valid indicator. Must be one of: 'HAZ' 'WHZ' 'WAZ' 'BMI' 'CIAF' 'birth_weight' 'BMIZ'")
  }
  
  # 'pre-processing':
  # This describes the data that is basically useless- it doesn't
  # contain the key properties that we need-- either the children are too
  # young/old, or they don't have age, height, sex, or weight information.
  # These are the reasons why you'd have NA values in your end indicator Z-scores.
  
  # Initialize the drop and exclude columns to 0 or NA
  data[, paste0(indicator, "_drop") := 0]
  data[, paste0(indicator, "_exclude") := NA]
  
  # SEX
  if(indicator != "birth_weight") {
    data[is.na(sex_id), paste0(indicator, "_drop") := 1]
  }
  
  # AGE
  if(indicator %in% c("HAZ", "WAZ", "CIAF", "BMI", "BMIZ")) {
    data[is.na(age_mo) & !(nid %in% c(319322)), paste0(indicator, "_drop") := 1]
  }
  
  if(indicator %in% c("HAZ", "WAZ", "CIAF", "BMIZ")) {
    data[(age_mo < 3) & is.na(age_wks) & !(nid %in% c(319322)), paste0(indicator, "_drop") := 1]
  }
  
  if(indicator == "WHZ") {
    data[is.na(age_cat_1) & !(nid %in% c(319322)), paste0(indicator, "_drop") := 1]
  }
  
  if(indicator == "BMI") {
    data[age_wks <= 104, paste0(indicator, "_drop") := 1]
  }
  
  if(indicator %in% c("HAZ", "WAZ", "WHZ", "CIAF", "BMIZ", "BMI")) {
    data[age_wks > 260, paste0(indicator, "_drop") := 1]
    data[age_mo > 60, paste0(indicator, "_drop") := 1]
    data[age_year > 5, paste0(indicator, "_drop") := 1]
  }
  
  if(indicator %in% c("HAZ", "WAZ", "WHZ", "CIAF", "BMIZ", "BMI")) { # Exclude all children under 1 month
    data[age_wks < 3, paste0(indicator, "_drop") := 1]
    data[age_mo < 1, paste0(indicator, "_drop") := 1]
  }
  
  # HEIGHT
  if(indicator %in% c("HAZ", "WHZ", "CIAF", "BMI", "BMIZ")) {
    data[is.na(metab_height) & !(nid %in% c(23017, 319322)), paste0(indicator, "_drop") := 1]
    data[metab_height <= 0, paste0(indicator, "_drop") := 1]
    data[metab_height >= 180, paste0(indicator, "_drop") := 1]
  }
  
  # WEIGHT
  if(indicator %in% c("WAZ", "WHZ", "CIAF", "BMI", "BMIZ")) {
    data[is.na(metab_weight) & !(nid %in% c(23017, 319322)), paste0(indicator, "_drop") := 1]
    data[metab_weight <= 0, paste0(indicator, "_drop") := 1]
    data[metab_weight >= 45, paste0(indicator, "_drop") := 1]
  }
  
  # BIRTHWEIGHT
  if(indicator == "birth_weight") {
    data[is.na(birth_weight), paste0(indicator, "_drop") := 1]
    data[birth_weight <= 0, paste0(indicator, "_drop") := 1]
  }
  
  # 'post-processing:
  # This is where we exclude data points from the final data set
  # based on some kind of plausibility metric. This is where the data
  # that aren't NA, but aren't believable, go to die.
  
  if (indicator == "BMI") {
    data[BMI <= 10, paste0(indicator, "_exclude") := 1]
    data[BMI >= 25, paste0(indicator, "_exclude") := 1]
    print("Data excluded if BMI is above 25 or below 10.")
  }
  
  if (indicator == "HAZ") {
    data[HAZ > 6, paste0(indicator, "_exclude") := 1]
    data[HAZ < -6, paste0(indicator, "_exclude") := 1]
    print("Data excluded if HAZ is above 6 or below -6.")
  }
  
  if (indicator == "WAZ") {
    data[WAZ > 5, paste0(indicator, "_exclude") := 1]
    data[WAZ < -6, paste0(indicator, "_exclude") := 1]
    print("Data excluded if WAZ is above 5 or below -6.")
  }
  
  if (indicator == "WHZ") {
    data[WHZ_seas > 5, paste0(indicator, "_exclude") := 1]
    data[WHZ_seas < -5, paste0(indicator, "_exclude") := 1]
    print("Data excluded if WHZ_seas is above 5 or below -5.")
  }
  
  if (indicator == "CIAF") {
    data[, paste0(indicator, "_exclude") := NA]
    data[, paste0(indicator, "_exclude_reason") := NA]
    print("CIAF data exclusion criteria still TBD; update the function when you know!")
  }
  
  if(indicator == "birth_weight") {
    # using GBD 2017 exclusions based on 0.1 and 99.9 percentiles. These may need to change to align with GBD    
    data[, paste0(indicator, "_exclude") := NA]
    data[birth_weight < 0.44 & sex == 0, paste0(indicator, "_exclude") := 1]
    data[birth_weight < 0.45 & sex == 1, paste0(indicator, "_exclude") := 1]
    data[birth_weight > 4.97 & sex == 0, paste0(indicator, "_exclude") := 1]
    data[birth_weight > 5.143 & sex == 1, paste0(indicator, "_exclude") := 1]
    print("Data excluded if birth weight is below the .1 or above the 99.9 percentile!")
  }
  
  if (indicator == "BMIZ") {
    data[BMIZ > 5, paste0(indicator, "_exclude") := 1]
    data[BMIZ < -5, paste0(indicator, "_exclude") := 1]
    print("Data excluded if BMIZ is above 5 or below -5.")
  }
  
  # Return the updated data
  return(data)
}


##############################################################################################################################################################
## 4.) CALCUATE Z SCORES: Load in Growth Standards from Reference Material, Merge on Information and Calculate Metrics
##############################################################################################################################################################

# These are the reference populations used to determine whether a child
# meets the criteria for our indicators.
# For HAZ and WAZ, ages 13 weeks and earlier are calculated using separate
# tables from the older children.

## bring in growth charts
#HAZ (lfha in WHO syntax) STUNTING
HAZ_mo <- fread(paste0(ref_dir,"growth_standards/HAZ_0_60_months.csv"))
HAZ_wks <- fread(paste0(ref_dir,"growth_standards/HAZ_0_13_weeks.csv"))

#WAZ (wfa in WHO syntax) UNDERWEIGHT
WAZ_mo <- fread(paste0(ref_dir,"growth_standards/WAZ_0_60_months.csv"))
WAZ_wks <- fread(paste0(ref_dir,"growth_standards/WAZ_0_13_weeks.csv"))

#WHZ (wfl/wfh in WHO syntax) WASTING
WHZ_mo <- fread(paste0(ref_dir,"growth_standards/WHZ_0_60_months.csv"))

# Load in BMI cutoff data
# Rebecca's note: I believe this to be from:
# Establishing a standard definition for child overweight
# and obesity worldwide: international survey
# Tim J Cole, Mary C Bellizzi, Katherine M Flegal, William H Dietz

# IOTF BMI cutoffs 
iotf_bmi_under_cutoffs<-fread(paste0(ref_dir,"growth_standards/bmi_cutoffs_undernutrition.csv"))
bmi_cutoffs<-fread(paste0(ref_dir,"bmi_cutoffs.csv"))

#WHO bmi-for-age Overweight These tables taken from: http://www.who.int/childgrowth/standards/bmi_for_age/en/
bmi_mo <- fread(paste0(ref_dir,"growth_standards/who_bmi_0_60_months.csv"))
bmi_wks <- fread(paste0(ref_dir,"growth_standards/who_bmi_birth_13_weeks_corrected.csv"))

orig_names<-names(data)
keep_cols<-orig_names
start<-nrow(data)

# 1. BMI (overweight IOTF Definition)
## Calculate BMI:
## CDC: "BMI is a person's weight in kilograms divided by the square of height in meters"
data$metab_weight <- as.numeric(data$metab_weight)
data$metab_height <- as.numeric(data$metab_height)
data$sex_id <- as.integer(data$sex_id)
data[,BMI:= metab_weight/((metab_height/100)^2)]
# Where child height is zero, this will be Infinite.
# Where child height and weight is zero, this will be NA.

bmi_cutoffs$sex <- ifelse(bmi_cutoffs$sex == 0, 1, 
                          ifelse(bmi_cutoffs$sex == 1, 2, NA))

# Merge together BMI cutoffs and raw data
data<-merge(data,
            bmi_cutoffs[,list(sex,age_mo,bmi_at_18yrs_16,bmi_at_18yrs_17,bmi_at_18yrs_25,bmi_at_18yrs_30)],
            by.x=c("age_mo","sex_id"), by.y = c('age_mo','sex'),
            all.x = T, all.y=F)

# 2. HAZ (stunting)
# Combine and Merge Data
data<-rbindlist( list(merge(data[age_wks<=13],
                            HAZ_wks[,list(l,m,s,sex,age_wks=week)],
                            by.x=c("sex_id", "age_wks"), by.y=c("sex", "age_wks"),
                            all.x = TRUE, allow.cartesian = TRUE) ,
                      
                      merge(data[age_wks>13],
                            HAZ_mo[,list(l,m,s,sex,age_mo=month,age_cat_1)],
                            by.x=c("sex_id", "age_mo","age_cat_1"), by.y=c("sex", "age_mo","age_cat_1"),
                            all.x = TRUE, allow.cartesian = TRUE),
                      
                      data[is.na(age_wks)]), use.names = T, fill=T
)

# Calculate HAZ score
data[,HAZ:=(((metab_height/m) ^ l)-1)/(s*l)]
data<-data[,!c("l","m","s"),with=F]

# 3. WAZ (Underweight)
data<-rbindlist( list(merge(data[age_wks<=13],
                            WAZ_wks[,list(l,m,s,sex,age_wks=week)],
                            by.x=c("sex_id", "age_wks"),by.y=c("sex", "age_wks"),
                            all.x = TRUE, allow.cartesian = TRUE) ,
                      
                      merge(data[age_wks>13],
                            WAZ_mo[,list(l,m,s,sex,age_mo=month,age_cat_2)],
                            by.x=c("sex_id", "age_mo","age_cat_2"),by.y=c("sex", "age_mo","age_cat_2"),
                            all.x = TRUE, allow.cartesian = TRUE),
                      
                      data[is.na(age_wks)]), use.names = T, fill=T
)

# Calculate WAZ score
data[,WAZ:=(((metab_weight/m) ^ l)-1)/(s*l)]
data<-data[,!c("l","m","s"),with=F]

# 4. WHZ (Wasting/WHO-Overweight)

# Round length to the nearest half-centimeter:
data[,metab_height_rounded := plyr::round_any(metab_height, .5)]

# Merge on WHO data
data<-merge(data,
            WHZ_mo[,list(l,m,s,sex,metab_height_rounded=length,age_cat_1)],
            by.x=c("sex_id","age_cat_1","metab_height_rounded"),by.y=c("sex","age_cat_1","metab_height_rounded"),
            all.x = TRUE, allow.cartesian = TRUE)

# calculate WHZ score
data[,WHZ :=(((metab_weight/m) ^ l)-1)/(s*l)]
data<-data[,!c("l","m","s"),with=F]

# 5. WHO BMI-for-age
## Calculate BMI:
## CDC: "BMI is a person's weight in kilograms divided by the square of height in meters"
# Combine and Merge Data
data<-rbindlist( list(merge(data[age_wks<=13],
                            bmi_wks[,list(l,m,s,sex,age_wks=week)],
                            by.x=c("sex_id", "age_wks"),by.y=c("sex", "age_wks"),
                            all.x = TRUE, allow.cartesian = TRUE) ,
                      
                      merge(data[age_wks>13],
                            bmi_mo[,list(l,m,s,sex,age_mo=month,age_cat_1)],
                            by.x=c("sex_id", "age_mo","age_cat_1"),by.y=c("sex", "age_mo","age_cat_1"),
                            all.x = TRUE, allow.cartesian = TRUE),
                      
                      data[is.na(age_wks)]), use.names = T, fill=T
)

# Calculate BMIZ Z-scores
data[,BMIZ:=(((BMI/m) ^ l)-1)/(s*l)]
data<-data[,!c("l","m","s"),with=F]	

# 6. Low birthweight
# Needs no prep, is a hard cutoff. See below.

# Clean up workspace  
rm(HAZ_mo,HAZ_wks,WAZ_wks,WAZ_mo,WHZ_mo,bmi_mo,bmi_wks,iotf_bmi_under_cutoffs,bmi_cutoffs)

###################################################################################################################################################    
## 5.) ADD ETHIOPIA DATA: Add datasets for which we were given z-scores directly     
################################################################################################################################################### 

# Meskane Mareko Ethiopia Data (NID 319322) Adding dummy values in for anthro and age so the data won't be dropped as missing. This data was not extracted through ubCov
# Need to change `ethiopia[,psu:=1:nrow(ethiopia)]` to something that counts unique lat/long pairs.
# 
# ethiopia <- read_stata(paste0(root,"DATA/ETH/MESKANE_MAREKO_NUTRITION_SURVEY_2013_2014/ETH_MESKANE_MAREKO_NUTRITION_SURVEY_2013_2014_WAZ_HAZ_WHZ_GPS_Y2017M10D30.DTA"))
# 
# # Change column names
# ethiopia <- as.data.table(ethiopia)
# ethiopia <- ethiopia[, -c("latitude")]  
# old_names<-c("y_North","x_East","waz","haz","whz")
# new_names<-c("latitude","longitude","WAZ","HAZ","WHZ")
# setnames(ethiopia,old_names,new_names)  ; rm(old_names)
# 
# ethiopia[,latitude:=as.character(latitude)]
# ethiopia[,longitude:=as.character(longitude)]    
# ethiopia[,psu:=paste0(latitude, "_", longitude)]
# ethiopia[,latitude:=as.numeric(latitude)]
# ethiopia[,longitude:=as.numeric(longitude)]     
# 
# # Add all other columns
# 
# ethiopia[,sex:=3] # This data does not have a gender variable; overwriting with a 3 so that further gender analysis will not work
# ethiopia[,nid:=319322]
# ethiopia[,source:="COUNTRY_SPECIFIC"]
# ethiopia[,country:="ETH"]
# ethiopia[,ihme_loc_id:="ETH_44858"]
# ethiopia[,start_year:=2014]
# ethiopia[,year_start:=2013]
# ethiopia[,end_year:=2014]
# ethiopia[,pweight:=1]
# ethiopia[,strata:=1]
# ethiopia[,age_cat_2:="0-5"]
# ethiopia[,height_age_sd:=HAZ]
# ethiopia[,weight_age_sd:=WAZ]
# ethiopia[,weight_height_sd:=WHZ]
# ethiopia[,point:=1]
# ethiopia[,int_month:=2]
# ethiopia[,int_year:=2014]
# ethiopia[,geospatial_id:=psu]
# ethiopia[,exclude_weights:=0]
# ethiopia[,exclude_age_range:=0]
# ethiopia[,exclude_age_granularity:=0]
# ethiopia[,exclude_data_lbw:=0]
# ethiopia[,exclude_data_cgf:=0]
# ethiopia[,exclude_representation:=0]
# ethiopia[,exclude_geography:=0]
# ethiopia[,exclude_interview:=0]
# ethiopia[,exclude_longitudinal:=0]
# ethiopia[,exclude_duplicative:=0]
# ethiopia[,indicator:=1]
# ethiopia[,N:=1]
# 
# data <- rbind.fill(data, ethiopia)    
# data <- as.data.table(data)
# 
# # Unlike the Ethiopia data, CWIQ GHA 2003 was extracted through ubCov. Overwriting the the calculated variable with the given z-scores. Putting in dummy values for necessary variables so this will make it through exclude logic.
# 
# data[nid == 23017, HAZ := height_age_sd]    
# data[nid == 23017, WAZ := weight_age_sd]    
# data[nid == 23017, WHZ := weight_height_sd]
# 
# data<-data[,!c("height_age_sd","weight_age_sd","weight_height_sd"),with=F]	    

#############################################################################################################################################################
## 6.) SEASONAL ADJUSTMENT: Adjust for Seasonality, by Modeling Regions
#############################################################################################################################################################

#' AddSeasonalAdjustment
#'
#' @description This function adds a column to the data.table which is the seasonally adjusted
#' version of the column you pass as the column you want to adjust. This fits a generalised additive models (GAM)
#' across time using the month of interview and a country-level fixed effect as the explanatory variables,
#' and the desired variable (adjust_var) as the response. A 12-month periodic spline for the interview month is used, as
#' well as a spline that smooths across the whole duration of the dataset, based on .
#'
#' The following data are excluded from the model fit process, and will not be adjusted:
#' Data without interview month, year, or that exceed the minimum/maximum valid/common-sense values for the variable (provided
#' as parameters to the function).
#'
#' @param raw_table The data.table with the raw data that contains the following columns:
#'                                  -int_mo (interview month)
#'                                  -int_year (interview year)
#'                                  -country (3-char ISO code)
#'                                  -'adjust_var' (specified by you, see below)
#'                                  -start_year (the year the survey series started)
#' @param adjust_var The column you want to generate seasonally adjusted versions of.
#' @param regions A vector of region names that you want to use to fit your model by that calculates
#'                seasonal adjustments
#' @param min_bounds The minimum valid value you want to include (of your adjust_var) in the data used to fit your model
#' @param max_bounds The maximum valid value you want to include (of your adjust_var) in the data used to fit your model.
#' @return The raw_table object, with a new column named with the convention of ('adjust_var'_seas).
#'
AddSeasonalAdjustment<-function(raw_table,regions,adjust_var, min_bounds, max_bounds){
  
  #Testing 
  #raw_table<-copy(data)
  #adjust_var<-"WHZ"
  #min_bounds<- -5
  #max_bounds<-5
  
  message(paste0("Adding a column for a seasonality-adjusted version of ",adjust_var,
                 ". Adding a new column, ",paste0(adjust_var,"_seas")," to represent the seasonally adjusted version."))
  
  # Ensure the data.table has all the right columns in it:
  if(sum(c("int_month","int_year",adjust_var,"year_start","ihme_loc_id") %in% names(raw_table))!=5){stop("You need to have the following columns: int_month int_year year_start ihme_loc_id, and the column specified that you want to adjust.")}
  #raw_table <- as.data.table(raw_table) # adding this in because of weird bug
  raw_table[,sort_order:=1:nrow(raw_table)] # Generate column preserving original sort order
  
  
  # Convert ISO3 codes to GAUL codes
  gaul_table <- fread(paste0(root, "/WORK/11_geospatial/10_mbg/stage_master_list.csv")) # read in stage master list to get regions 
  raw_table <- base::merge(raw_table, gaul_table[,list(iso3, mbg_reg)], by.x = "ihme_loc_id", by.y = "iso3", all.x = TRUE) # merge tables to create a mbg_reg column
  
  # Renaming mbg_reg column to region
  raw_table <- raw_table[,region := mbg_reg]
  raw_table <- raw_table[,mbg_reg:=NULL]
  
  # Create a template matrix to in-fill monthly adjustments for each region
  reg_periods <- matrix(ncol = 15, nrow = length(seq(1, 12.9, by = .1)))
  colnames(reg_periods) <- c("month", regions)
  reg_periods[, 1] <- seq(1, 12.9, by = .1)
  
  message("Fitting models that account for seasonal variation in measurements.")
  for(reg in regions){
    message(sprintf("On region: %s", reg))
    
    Region<-unique(raw_table[region==reg & # In the specific region
                               int_year>=year_start & # The survey date is after the survey start
                               !is.na(int_year) & !is.na(int_month) &
                               !is.na(get(adjust_var)) & !is.infinite(get(adjust_var)) &
                               (get(adjust_var))>=min_bounds & (get(adjust_var))<=max_bounds # Fit only on data within the boundaries of reasonable
                             ,]) # You have a valid observations for the variable as well
    
    # Creating an index in time from the start year available in this region:
    region_min<-min(Region$int_year,na.rm=T)
    Region[,mo_ind:= (12*(int_year-region_min)) + int_month]
    
    # Check for sufficient levels of `ihme_loc_id` before fitting the model
    if(length(unique(Region$ihme_loc_id)) < 2) {
      message("Skipping model fitting for region ", reg, " due to insufficient levels in ihme_loc_id.")
      next
    }
    
    # Fit model with uncorrelated errors
    m0 <- mgcv::gamm(get(adjust_var) ~ s(int_month, bs = "cc", k = 6) + s(mo_ind, k = 4) + as.factor(ihme_loc_id),
                     data = Region)
    
    # Plot/Sanity-check the seasonality adjustments:
    # per.plot = FALSE
    # if(per.plot){
    #   png(sprintf("/homes/azimmer/wasting_seasonality/wasting_seasonality_05_24_2017/season_gamms_%s_k_4.png", reg),
    #       width = 20, height = 8, units = "in", res = 300)
    #   layout(matrix(1:2, ncol = 2))
    #   plot(m0$gam, scale = 0, main = reg, xlab = "interview month", ylab = paste0(adjust_var," seasonal peiodic bias"))
    #   layout(1)
    # }
    
    # Getting adjustments from model fit:
    ct <- Region$ihme_loc_id[!is.na(Region$ihme_loc_id)][1]
    
    pdat <- data.frame(mo_ind    = 1,
                       ihme_loc_id   = as.factor(ct), ## random ihme_loc_id
                       int_month = seq(1, 12.9, by = .1)) ## random point in time
    
    reg_periods[, (which(regions %in% reg) + 1)] <- predict(m0$gam, newdata = pdat, type = "terms", se.fit = TRUE)$fit[, 2]
  }
  
  ## now we adjust by each datapoint by the month it was in
  raw_table[,(paste0(adjust_var,"_seas")):=NA] # Start new column for adjusted data
  reg_periods<-data.table(reg_periods)
  
  message("Adjusting data based on model fit.")
  for(reg in regions){
    for(mo in 1:12){
      rows.to.adjust <- which(
        raw_table$region==reg & # In the specific region
          raw_table$int_year>=raw_table$year_start & # The survey date is after the survey start
          !is.na(raw_table$int_year) & !is.na(raw_table$int_month) &
          !is.na(raw_table[[adjust_var]]) & !is.infinite(raw_table[[adjust_var]]) &
          raw_table$int_month==mo &
          raw_table[[adjust_var]]>=min_bounds & raw_table[[adjust_var]]<=max_bounds # Fit only on data within the boundaries of reasonable
      )
      
      ## find the adjustment
      mean.period <- mean(reg_periods[[reg]]) ## we adjust to the day(s) that's at the mean of the period
      month.val <- reg_periods[[reg]][which(reg_periods[, 1] == mo)]
      delta.adjust <- month.val - mean.period
      
      ## do the adjustment
      raw_table[[paste0(adjust_var,"_seas")]][rows.to.adjust] <- raw_table[[adjust_var]][rows.to.adjust] - delta.adjust
    }
  }
  
  # Ensuring that the sorting order is the same as when it went in
  raw_table<-raw_table[order(sort_order)]; raw_table[,sort_order:=NULL]
  raw_table[,region:=NULL] # Eliminating region column
  return(raw_table)
}# Closing the function

# Calculate Seasonality Bias for Relevant Variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Add column for seasonality-adjusted WHZ (wasting) Z-scores
data$int_year <- as.numeric(data$int_year)
data$int_month <- as.numeric(data$int_month)
data<-AddSeasonalAdjustment(raw_table = data,
                            regions = c("cssa",
                                        "essa",
                                        "noaf",
                                        "sssa",
                                        "wssa",
                                        "mide",
                                        "stan",
                                        "eaas",
                                        "soas",
                                        "seas",
                                        "ocea",
                                        "caca",
                                        "ansa",
                                        "trsa"),
                            adjust_var = "WHZ",
                            min_bounds = -5,
                            max_bounds = 5)

#Using unadjusted WHZ score where int_month is missing. Creating indicator showing whether the seasonally-adjusted WHZ or the unadjusted WHZ is given.
data[, WHZ_seas_ind := 0]  
data[!is.na(WHZ_seas), WHZ_seas_ind := 1]    
data[is.na(WHZ_seas), WHZ_seas := WHZ]

##################################################################################################################################################################
## 7.) GENERATE INDICATORS: Calculate indicators based off of Z scores
##################################################################################################################################################################

# ADD IOTF OVERWEIGHT INDICATORS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add columns for whether the child is:
#   obese (would have a BMI of over 30 at the age of 18),
#   overweight (would have a BMI of over 25 at the age of 18)
data[,overweight_iotf_b:=ifelse(BMI>=bmi_at_18yrs_25,1,0)]
data[,obese_iotf_b:=ifelse(BMI>=bmi_at_18yrs_30,1,0)]

# ADD IOTF Undernutrition INDICATORS
#~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add columns for whether the child is:
# Undernutrition (would have a BMI of below 17 at the age of 18)

data[,undernutrition_iotf_b:=ifelse(BMI<=bmi_at_18yrs_17,1,0)]

# ADD WHO BMI-for-age - overweight
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,overweight_who_BMIZ :=ifelse(BMIZ >= 2, 1, 0)]
data[,obese_who_BMIZ :=ifelse(BMIZ >= 3, 1, 0)]

# ADD WHO BMI-for-age - undernutriton
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,undernutrition_who_BMIZ_mod_b :=ifelse(BMIZ <= -2, 1, 0)]
data[,undernutrition_who_BMIZ_sev_b :=ifelse(BMIZ <= -3, 1, 0)]

# ADD STUNTING INDICATORS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,stunting_mod_b :=ifelse(HAZ <= -2, 1, 0)] # For CGF
data[,stunting_sev_c :=ifelse(HAZ <= -3, 1, 0)] # For Bobby's risk-factor project.
data[,stunting_mil_c :=ifelse(HAZ <= -1 & HAZ > -2, 1, 0)] # For Bobby's risk-factor project.
data[,stunting_mod_c :=ifelse(HAZ <= -2 & HAZ > -3, 1, 0)] # For Bobby's risk-factor project.

# ADD LOW BIRTHWEIGHT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,low_bw_prev := ifelse(birth_weight < 2.5,1,0)]

# ADD WASTING INDICATORS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,wasting_mod_b :=ifelse(WHZ_seas <= -2, 1, 0)] # NOTE: Using seasonally corrected versions here.For CGF
data[,wasting_sev_c :=ifelse(WHZ_seas <= -3, 1, 0)] # NOTE: Using seasonally corrected versions here. For Bobby's risk-factor project.
data[,wasting_mil_c :=ifelse(WHZ_seas <= -1 & WHZ_seas > -2, 1, 0)] # NOTE: Using seasonally corrected versions here. For Bobby's risk-factor project.
data[,wasting_mod_c :=ifelse(WHZ_seas <= -2 & WHZ_seas > -3, 1, 0)] # NOTE: Using seasonally corrected versions here. For Bobby's risk-factor project.


# ADD WHO-STYLE OVERWEIGHT/OBESITY (extreme right of wasting)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,obese_who_b :=ifelse(WHZ >=3, 1, 0)]
data[,overweight_who_b :=ifelse(WHZ >=2, 1, 0)] # For DBM

# ADD UNDERWEIGHT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,underweight_mod_b :=ifelse(WAZ <= -2, 1, 0)] # For CGF
data[,underweight_sev_c :=ifelse(WAZ <= -3, 1, 0)]
data[,underweight_mil_c :=ifelse(WAZ <= -1 & WAZ > -2, 1, 0)] # For Bobby's risk-factor project.
data[,underweight_mod_c :=ifelse(WAZ <= -2 & WAZ > -3, 1, 0)] # For Bobby's risk-factor project.

# ADD stunting_overweight (stunting/BMIZ overweight--not currently modeled)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data[,stunting_overweight :=ifelse(overweight_who_BMIZ == 1 & stunting_mod_b == 1, 1, 0)]

# Running DropData function
indicator_types<-c("HAZ","WHZ","WAZ") #"BMI","CIAF","birth_weight","BMIZ"
for(i in indicator_types){
  data<-DropData(data=data,indicator=i)
}

############################################################################################################################################################
## 8.) ADD IN SOMALIA DATA, DROP DATA: Clean and Prepare Somalia data--has no z-sores, just headcounts of stunting, wasting, underweight, thus no calcs or adjustments are necessary 
############################################################################################################################################################
# 
# # Prepare Somalia Data
# 
# # NOTE: This strategy does NOT allow for you to use this data for CIAF, since we don't currently know the number of children that are
# # stunted and wasted, etc. Find and use the non-collapsed versions of this data, such that there is a 0 or 1 for each child for whether
# # they are wasted, stunted, and/or underweight, and collapse that on before generating CIAF indicators, if you want to do that aspect of
# # analysis with this survey.
# 
# somalia <- fread(paste0(root,"DATA/SOM/NUTRITIONAL_ASSESSMENT_SURVEY/2007_2010/SOM_NUTRIONAL_ASSESSMENT_SURVEY_2007_2010_MALNUTRITION_LAT_LONG_Y2016M10D25.CSV"))
# 
# # Change column names
# old_names<-c("Latitude","Longitude","Year of survey","Number of children examined","Number wasting","Number stunting","Number underweight")
# new_names<-c("latitude","longitude","start_year","N","wasting_mod_b","stunting_mod_b","underweight_mod_b")
# setnames(somalia,old_names,new_names)  ; rm(old_names)
# somalia[,psu:=1:nrow(somalia)]
# 
# # Repeating, row-wise, with relevant columns only, for each lat/lon/psu, by the N of children sampled in that area.
# somalia<-somalia[rep(seq(.N), N), # Repeat the rows by the number specified in the column  N
#                  c(new_names,"psu"), # Keep these columns
#                  with=F] #
# somalia[,seq:=seq(1:.N),by=c(new_names,"psu")] # Generate an index per each child within a lat/lon
# somalia[,wasting_mod_b:=ifelse(seq<=wasting_mod_b,1,0)] # Generate binary versions
# somalia[,wasting_mod_who:=ifelse(seq<=wasting_mod_b,1,0)] 
# somalia[,stunting_mod_b:=ifelse(seq<=stunting_mod_b,1,0)]
# somalia[,underweight_mod_b:=ifelse(seq<=underweight_mod_b,1,0)]
# 
# # Generate some extra columns
# 
# somalia[,nid:=270669]
# somalia[,source:="FSNAU"]
# somalia[,country:="SOM"]
# somalia[,ihme_loc_id:="SOM"]
# somalia[,exclude_weights:=0]
# somalia[,exclude_age_range:=0]
# somalia[,exclude_age_granularity:=0]
# somalia[,exclude_data_lbw:=0]
# somalia[,exclude_data_cgf:=0]
# somalia[,exclude_representation:=0]
# somalia[,exclude_geography:=0]
# somalia[,exclude_interview:=0]
# somalia[,exclude_longitudinal:=0]
# somalia[,exclude_duplicative:=0]
# somalia[,year_start:=2007]
# somalia[,end_year:=2010]
# somalia[,pweight:=1]
# somalia[,age_cat_2:="0-5"]
# somalia[,point:=1]
# somalia[,int_year:=start_year]
# somalia[,geospatial_id:=psu]
# somalia[,indicator:=1]
# somalia[,N:=1]
# somalia[,gaul:=226]
# 
# # Add in exclusions and drop columns:
# 
# indicator_types<-c("HAZ","WHZ","WAZ") #"BMI","CIAF","birth_weight","BMIZ"
# for(i in indicator_types){
#    data<-DropData(data=data,indicator=i)
#  }
# 
# fwrite(data, paste0(j, '/temp/alicela/dropdata_plot_input/pre_', extract_date, '.csv'))

# Calculate how many observations we drop for indicators

total_obs <- nrow(data)
HAZ_drop <- nrow(data[HAZ_drop == 1,])
WAZ_drop <- nrow(data[WAZ_drop == 1,])
WHZ_drop <- nrow(data[WHZ_drop == 1,])
total_HAZ <- nrow(data[is.na(HAZ_drop),])
total_WAZ <- nrow(data[is.na(WAZ_drop),])
total_WHZ <- nrow(data[is.na(WHZ_drop),])
HAZ_exclude <- nrow(data[HAZ_exclude == 1,])
WAZ_exclude <- nrow(data[WAZ_exclude == 1,])
WHZ_exclude <- nrow(data[WHZ_exclude == 1,])

drop_table <- data.table( x = c('Total Rows', 'HAZ drop', 'WAZ drop', 'WHZ drop', 'HAZ after drop', 'WAZ after drop', 'WHZ after drop', 'HAZ exclude', 'WAZ exclude', 'WHZ exclude'),
                          y = c(total_obs, HAZ_drop, WAZ_drop, WHZ_drop, total_HAZ, total_WAZ, total_WHZ, HAZ_exclude, WAZ_exclude, WHZ_exclude))

fwrite(drop_table, paste0(l, "/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/dropped_rows_summary.csv"))

# #Add Somalia data
# data<-rbind(data,somalia, fill=T)      

#####################################################################################################################################################################
## 9.) SAVE GBD DATA: Save a copy of data for GBD before we apply our own drop/exclude criteria!
#####################################################################################################################################################################

# Save a reference before subsetting to different indicators--necessary so z-score work won't be lost. Also, this is the GBD data.

# data_copy <- copy(data)

#For GBD, fixing years we adjusted to get into geospatial's 2000 bucket:
# data_gbd <- copy(data)   
# data_gbd[nid == 5827, start_year := 1997]
# data_gbd[nid == 5827, int_year := 1997]      
# data_gbd[nid == 46563, start_year := 1996]
# data_gbd[nid == 46563, int_year := 1996]      
# data_gbd[nid == 19046, start_year := 1996]       
# data_gbd[nid == 19046, int_year := 1996]  

## dropping columns created in extraction that serve no purpose
colnames(data)
data[, c("1741:", "176:","1773:","1953:","2054:","226:","227:","2466:","3050:","3051:","3053:","34111:","4322:","4966:","507:","508:","509:","6232:") := NULL]


write.csv(data, paste0(l, "/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/cgf_data_prep.csv"))

# rm(data_gbd)   


#Summaries for dashboard      
# 
# stage1 <- get_adm0_codes('stage1')
# stage2 <- c(stage1, (get_adm0_codes('stage2')))
# 
# isos <- load_adm0_lookup_table()
# noaf <- sort(isos[mbg_reg == 'noaf', toupper(iso3)])
# cssa <- sort(isos[mbg_reg == 'cssa', toupper(iso3)])
# wssa <- sort(isos[mbg_reg == 'wssa', toupper(iso3)])
# essa <- sort(isos[mbg_reg == 'essa', toupper(iso3)])
# sssa <- sort(isos[mbg_reg == 'sssa', toupper(iso3)])
# mide <- sort(isos[mbg_reg == 'mide', toupper(iso3)])
# stan <- sort(isos[mbg_reg == 'stan', toupper(iso3)])
# wssa <- sort(isos[mbg_reg == 'wssa', toupper(iso3)])
# eaas <- sort(isos[mbg_reg == 'eaas', toupper(iso3)])
# soas <- sort(isos[mbg_reg == 'soas', toupper(iso3)])
# seas <- sort(isos[mbg_reg == 'seas', toupper(iso3)])
# ocea <- sort(isos[mbg_reg == 'ocea', toupper(iso3)])
# caca <- sort(isos[mbg_reg == 'caca', toupper(iso3)])
# ansa <- sort(isos[mbg_reg == 'ansa', toupper(iso3)])
# trsa <- sort(isos[mbg_reg == 'trsa', toupper(iso3)])
# 
# 
# africa <- sort(isos[Stage == 1, toupper(iso3)])
# 
# iso_list <- sort(isos[Stage == 1 | Stage == '2a' | Stage == '2b', toupper(iso3)])

##################################################################################################################################################################
## 10.) APPLY EXCLUDE CRITERIA: Exclude whole surveys that have been determined to not be useable
##################################################################################################################################################################

#Until I rerun, these exclusions need to be fixed by hand
# data[nid==236205, exclude_data_cgf:=1]
# data[nid==9506, exclude_data_cgf:=0]

# Adding in columns needed for further data processing
# These columns were initially generated from a merge with the following: R:\share\code\geospatial\alicela\cgf\data_prep\exclusions.csv
# However, none of these NIDs align with what we currently have, so just creating the columns and filling them all with 0
# data[, c("exclude_weights", "exclude_age_range", "exclude_age_granularity", 
#          "exclude_data_lbw", "exclude_data_cgf", "exclude_representation", 
#          "exclude_geography", "exclude_interview", "exclude_longitudinal", 
#          "exclude_BMIZ", "exclude_duplicative") := 0]
# 
# #Excluding all problematic data
# data <- data[exclude_weights!=1,]
# data <- data[exclude_age_range!=1,]
# data <- data[exclude_age_granularity!=1,]
# data <- data[exclude_representation!=1,]
# data <- data[exclude_geography!=1,]
# data <- data[exclude_data_cgf!=1,]
# data <- data[exclude_duplicative!=1,]

###############################################################################################################################################################
## 11.) GENERATE HEADCOUNTS: Generate counts of point, polygons, and N values for each survey
###############################################################################################################################################################

## 1/13/25 Commenting out data_summary code since it is unnecessary. Location info will be processed in separate scripts alongside wealth.

# #Here we create cross-indicator headcounts, point counts, and poly counts by NID for the SI
# #Since we produce separate datasets for the three indicators, the totals must be calculated prior to collapse
# data_summary <- copy(data)
# 
# # Dropping all cases where none of the three z-scores: HAZ, WAZ, and WHZ, were valid.
# # This should not be necessary if you only need headcounts, point counts and poly counts for a single indicator
# data_summary[,HAZ_ind:=0]
# data_summary[is.na(HAZ_drop) & is.na(HAZ_exclude),HAZ_ind:=1 ]
# data_summary[,WAZ_ind:=0]
# data_summary[is.na(WAZ_drop) & is.na(WAZ_exclude),WAZ_ind:=1 ]
# data_summary[,WHZ_ind:=0]
# data_summary[is.na(WHZ_drop) & is.na(WHZ_exclude),WHZ_ind:=1 ]
# data_summary[, Ind:=0]
# data_summary[HAZ_ind == 1 | WAZ_ind == 1 | WHZ_ind == 1 ,ind:=1 ]
# data_summary <- as.data.table(data_summary)
# data_summary <- data_summary[ind==1,]
# 
# #Subsetting to required variables for this summary
# summary_vars <- c("ind", "nid", "latitude", "longitude", "location_code", "shapefile")
# data_summary <- data_summary[, summary_vars, with=F]
# 
# #Creating unique point and polygon indicators
# data_summary[, latitude:= as.character(latitude)]
# data_summary[, longitude:= as.character(longitude)]
# data_summary[, latlong := paste0(latitude, "_", longitude)] #creating unique lat/long indicator
# data_summary[is.na(latitude)|is.na(longitude), latlong := NA] #nulling out lat/long indicator where either lat or long is null
# # data_summary[, location_code:= as.character(location_code)]
# # data_summary[, polygon := paste0(location_code, "_", shapefile)] #Creating unique polygon count
# # data_summary[is.na(location_code)|is.na(shapefile), polygon := NA] #nulling out cases where either shapefile or location code is null
# # data_summary <- data_summary[!is.na(latlong)|!is.na(polygon),] #subsetting to cases with mappable geographies
# 
# #Collapsing data by NID to give unique point and poly counts
# points <- ddply(data_summary,~nid,summarise,points=length(unique(na.omit(latlong))))
# # polys <- ddply(data_summary,~nid,summarise,polys=length(unique(na.omit(polygon))))
# 
# #Nulling out unnecessary variables 
# data_summary$latitude <- NULL
# data_summary$longitude <- NULL
# # data_summary$location_code <- NULL
# # data_summary$shapefile <- NULL
# 
# #Collapsing to headcounts and merging point and poly counts back on
# cgf.summary <- aggregate(ind ~ nid, data = data_summary, FUN = sum)
# cgf.summary <- merge(cgf.summary, points, by.x="nid", by.y="nid", all.x=T, all.y=F)
# # cgf.summary <- merge(cgf.summary, polys, by.x="nid", by.y="nid", all.x=T, all.y=F)
# 
# colnames(cgf.summary) <- c('nid', 'count', "points")
# 
# write.csv(cgf.summary, paste0(l,"/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/headcounts_", module_date, ".csv" ))
# 
# rm(data_summary, cgf.summary)

#########################################################################################################################################################
## 12.) COLLAPSE DATA: Split data into specific indicators, drop data according to indicator, and begin collapsing point and polygon data 
#########################################################################################################################################################  

## 1/13/25 Commenting out all collapsing code due to lack of location information. 

# # This NID is actually 7 different surveys within the same series, so it needs to be separated for the collapse
# data <- data[nid == 275090 & year_start == 2005, nid := 27509005]
# data <- data[nid == 275090 & year_start == 2007, nid := 27509007]
# data <- data[nid == 275090 & year_start == 2008, nid := 27509008]
# 
# # Assign indicators to loop through based on project (otherwise you have to do it manually or it literally takes forever)
# if(proj == 'cgf'){
#   binary_ind <-c("stunting_mod_b", "wasting_mod_b", "underweight_mod_b")
# } else if (proj == 'dbm'){
#   binary_ind <- c("overweight_who_b",'wasting_mod_cond')
# } else if (proj == 'risk_factor'){
#   binary_ind <-c("stunting_mil_c", "stunting_mod_c", "stunting_sev_c", "wasting_mil_c", "wasting_mod_c", "wasting_sev_c", "underweight_mil_c", "underweight_mod_c", "underweight_sev_c")
# }
# 
# for (ind in binary_ind) {
#   all_data <- copy(data)
#   if (ind %in% c('stunting_mod_b', 'stunting_mil_c', 'stunting_mod_c','stunting_sev_c')) {
#     all_data <- all_data[is.na(HAZ_drop),]
#     all_data <- all_data[is.na(HAZ_exclude),]
#   }  
#   if (ind %in% c("wasting_mod_b", "overweight_who_b", 'wasting_mil_c', "wasting_mod_c", 'wasting_sev_c', 'obese_who_b')) {
#     all_data <- all_data[is.na(WHZ_drop),]
#     all_data <- all_data[is.na(WHZ_exclude),]
#   }
#   if (ind %in% c("underweight_mod_b", "underweight_mil_c", "underweight_mod_c", "underweight_sev_c")) {
#     all_data <- all_data[is.na(WAZ_drop),]
#     all_data <- all_data[is.na(WAZ_exclude),]
#   }  
#   if (ind %in% c("overweight_who_BMIZ", "undernutrition_who_BMIZ_mod_b")) {
#     all_data <- all_data[is.na(BMIZ_drop),]
#     all_data <- all_data[is.na(BMIZ_exclude),]
#     all_data <- all_data[exclude_BMIZ!=1,]    
#   }  
#   #undernutrition (BMIZ) modeled conditionally on the assumption that the child is not overweight
#   if (ind == "undernutrition_who_BMIZ_mod_b") {
#     all_data <- all_data[overweight_who_BMIZ!=1,]
#   }
#   if (ind == "stunting_overweight") {
#     all_data <- all_data[is.na(HAZ_drop),]
#     all_data <- all_data[is.na(HAZ_exclude),]
#     all_data <- all_data[is.na(BMIZ_drop),]
#     all_data <- all_data[is.na(BMIZ_exclude),]
#     all_data <- all_data[exclude_BMIZ!=1,]    
#   }    
#   if (ind == "underweight_who_cond") {
#     all_data <- all_data[overweight_who_b!=1,]
#     ind <- "wasting_mod_b"
#   }
#   if (ind == 'wasting_mod_cond'){
#     all_data <- all_data[overweight_who_b!=1,]
#     setnames(all_data, old = 'wasting_mod_b', new = 'wasting_mod_cond')
#   }
#   
#   
#   # write coverage data for input to stat_compiler
#   
#   # stat_data <- copy(all_data)
#   #  stat_file <- paste0(j, 'WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/statcompiler/',ind,'/input/statcompiler_',extract_date,'.csv')
#   #  fwrite(stat_data, stat_file)
#   
#   # Seperate data into points and polys for the collapse!
#   
#   point_data <- all_data[point==1, ]
  # poly_data <- all_data[point==0, ]
  
  ########################################################################################################################################################
  # PART 1: COLLAPSE POINT DATA - Collapse point data by geography, splitting by sex if desired
  ########################################################################################################################################################
  
  ## Process point_data as you normally would, collapsing to cluster means. Let's call this new dt point_data_collapsed.
  ## sum() for binomial indicators or mean() for Gaussian indicators
  # 
  # 
  # point_collapse <- function(ind, point_data, sex){
  #   
  #   if(sex=='m'){
  #     point_data <- point_data[sex==1,]
  #   } else if(sex=='f'){
  #     point_data <- point_data[sex==0,]
  #   }
  #   
  #   # Point collapse 
  #   if (!(ind %in% c('wasting_mod_who','overweight_mod_who', 'normal_mod_who'))){
  #     point_data[point==1 & is.na(pweight), pweight := 1]
  #     
  #     all_point_data <- point_data[ ,list(N = sum(N), var = sum(get(ind), na.rm = TRUE), pweight_sum = sum(pweight, na.rm = TRUE)), by=c('source', 'start_year','latitude','longitude','country', 'nid')]
  #     setnames(all_point_data, "var", ind)
  #     all_point_data <- all_point_data[!is.na(latitude)]
  #     all_point_data <- all_point_data[!is.na(longitude)]
  #     all_point_data$point <- 1
  #     setnames(all_point_data, 'nid', 'svy_id')  
  #   } # Closes the point collapse for non-scaled indicators
  #   
  #   fwrite(all_point_data, paste0(l, "/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/4_collapsed", topic, "/", ind, "/all_point_data_", sex, "_", module_date, ".csv"))
  # } # Closes the point collapse function
  
  
  
  ##############################################################################################################################################
  # PART 2: COLLAPSE POLYGON DATA - Collapse data by survey and polygon
  ##############################################################################################################################################
  # 
  # # Per Damaris, since we're no longer dropping problem strata, we're just going to remove all rows missing pweights in the poly data
  # poly_data <- poly_data[!is.na(pweight),]
  # 
  # # Dropping polygons with only 1 observation  
  # poly_data_test <- copy(poly_data)
  # poly_data_test <- poly_data_test[, list(N=sum(N)), by=c('source', 'start_year', 'country', 'location_code', 'shapefile' )]
  # poly_data_bad <- poly_data_test[N==1, ]
  # 
  # if(length(poly_data_bad[, source]) > 0) {
  #   message("This many polygons have 1 observation so will be dropped:")
  #   print(table(poly_data_bad[, source], poly_data_bad[, start_year]))
  #   ##    poly_data_test[, N:= NULL]
  #   poly_data <- merge(poly_data, poly_data_test, by=c('start_year', 'country', 'location_code', 'shapefile', 'source'))
  #   poly_data <- poly_data[N.x != N.y, ] ## n.x and n.y are equal where both are 1, i.e. where poly had one cluster
  #   setnames(poly_data, 'N.x', 'N') ## set the original N col back
  #   poly_data[, N.y := NULL] ## remove the summed N col
  # }
  # 
  # ## setnames(poly_data, 'country', 'iso3') ## AOZ edit: these "countries" are iso3. should the be full country names?
  # poly_surveys <- unique(poly_data[, nid])
  # 
  # 
  # poly_collapse <- function(ind, poly_data, sex){
  #   
  #   if(sex=='m'){
  #     poly_data <- poly_data[sex==1,]
  #   } else if(sex=='f'){
  #     poly_data <- poly_data[sex==0,]
  #   }
  #   
  #   ####Kish's  collapse  
  #   
  #   # The polygon collapse 
  #   if (!(ind %in% c('wasting_mod_who','overweight_mod_who', 'normal_mod_who'))){
  #     collapsed_dat2 <- poly_data %>%
  #       mutate(pweight2 = pweight^2) %>%
  #       group_by(country, location_code, shapefile, source, nid) %>%
  #       summarise(mean = weighted.mean(get(ind), w = pweight, na.rm = TRUE),
  #                 n_eff = ((sum(pweight, na.rm = TRUE))^2)/(sum(pweight2, na.rm = TRUE)),
  #                 pweight_sum = sum(pweight, na.rm = TRUE),
  #                 start_year=floor(weighted.mean(start_year, w=N)))
  #     
  #     all_poly_data<-data.frame(collapsed_dat2)
  #     all_poly_data[[ind]] <- round(all_poly_data$mean * all_poly_data$n_eff)
  #     # names(all_poly_data)[names(all_poly_data) == 'var'] <- get("ind")  
  #     all_poly_data$N <- round(all_poly_data$n_eff)
  #     all_poly_data$point <- 0
  #     all_poly_data <- subset( all_poly_data, select = -c(mean, n_eff ) )
  #     setnames(all_poly_data, 'nid', 'svy_id')
  #   } # Closes polygon collapse for non-scaled indicators
  #   
  #   poly_data_collapsed <- copy(all_poly_data)
  #   poly_data_collapsed <- as.data.table(poly_data_collapsed)
  #   
  #   
  #   # write the collapsed data to a folder to use for resampling!
  #   fwrite(poly_data_collapsed, paste0(l,'/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/4_collapsed', ind,'/collapsed_polys_', sex, "_", module_date, '.csv'))
  # } # Closes the polygon collapse function
  # 
  
  # Collapse stuff!
#   
#   if(sex_split==TRUE){
#     point_collapse(ind, point_data,"m")
#     point_collapse(ind, point_data, "f")
#     point_collapse(ind, point_data, "mf")
#     # poly_collapse(ind, poly_data, "m")
#     # poly_collapse(ind, poly_data, "f")
#     # poly_collapse(ind, poly_data, "mf")
#   }
#   
#   if(sex_split==FALSE){
#     point_collapse(ind, point_data, "mf")
#     # poly_collapse(ind, poly_data, "mf")
#   }
# } # Closes the entire for loop for collapsing data by indicator

##############################################################################################################################################################
## 13.) DATA COVERAGE PLOTS: Read in collapsed points and polys, add in data extracted from reports, and plot!
##############################################################################################################################################################  

## 1/13/25 Commenting out all plotting code due to lack of point/poly data.

# source('/share/code/geospatial/alicela/lbd_core/mbg_central/graph_data_coverage.R')
# if (cov_plot == TRUE){
#   sex <- 'mf'
#   if(proj == 'cgf'){
#     coverage_ind <-c("stunting_mod_b", "wasting_mod_b", "underweight_mod_b")
#   } else if (proj == 'dbm'){
#     coverage_ind <- c("overweight_who_b", 'wasting_mod_cond') #, 'normal_mod_who'
#   } else if (proj == 'risk_factor'){
#     coverage_ind <-c("stunting_mil_c", "stunting_mod_c", "stunting_sev_c", "wasting_mil_c", "wasting_mod_c", "wasting_sev_c", "underweight_mil_c", "underweight_mod_c", "underweight_sev_c")
#   }
#   
#   for (ind in coverage_ind){
#     coverage_point <- fread(paste0("ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/", topic, "/", ind, "/all_point_data_", sex, "_", module_date, ".csv"))
#     coverage_poly <- fread(paste0('ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind,'/collapsed_polys_', sex, "_", module_date, '.csv'))
#     
#     
#     if (file.exists(paste0('ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind,'/extractions/collapsed_polys_extractions_', sex, '.csv'))){
#       print("This indicator utilizes report extractions")
#       extract_data <- fread(paste0('ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind,'/extractions/collapsed_polys_extractions_', sex, '.csv'))
#       coverage_data <- rbind(coverage_point, coverage_poly, fill=TRUE)
#       coverage_data <- coverage_data[,type:=0]
#       coverage_data <- rbind(coverage_data, extract_data, fill=TRUE)
#     } else {
#       print("No report extractions for this indicator")
#       coverage_data <- rbind(coverage_point, coverage_poly, fill=TRUE)
#       coverage_data <- coverage_data[,type:=0]
#     }
#     
#     # drop china survey for now
#     
#     # coverage_data <- coverage_data[svy_id != 283812,]
#     
#     # Creating a unique geography id and calling it cluster_id, because the plotting code errors without it  
#     coverage_data[, cluster_id := paste0(latitude, "_", longitude)]
#     coverage_data[, location_code := as.character(location_code)]
#     coverage_data[is.na(latitude) & is.na(longitude), cluster_id := location_code]  
#     coverage_data[, latitude := as.numeric(latitude)]
#     coverage_data[, longitude := as.numeric(longitude)]
#     coverage_data[, location_code := as.numeric(location_code)]
#     
#     # Also, the new coverage plot code requires an 'nid' variable, which I renamed svy_id. Bah.
#     coverage_data[,nid := svy_id]
#     
#     # Convert to rate space and replace old count variable with generic variable name
#     coverage_data[, (ind):= get(ind) / N ]
#     
#     # Drop pre 2000 surveys
#     
#     coverage_data <- coverage_data[start_year > 1999,]
#     
#     # Drop data for DBM
#     
#     # if (proj == 'dbm'){
#     #   exclude_dbm <- c(76704, 79839, 14015, 14027, 14063, 14105, 58660, 9516, 60942, 24143, 369294, 283812)
#     #   coverage_data <- coverage_data[(!nid %in% exclude_dbm),]
#     # }
#     # 
#     # # Drop South Africa 2003-04 for now
#     # coverage_data <- coverage_data[nid != 394227,]
#     # coverage_data <- coverage_data[nid != 20798,]
#     # 
#     # # test fix a peru issue
#     # 
#     # coverage_data <- coverage_data[shapefile == 'PER_ADM3', shapefile := 'PER_adm3']
#     
#     # Change data source name to be more user friendly - for stage 2 paper, can skip this step for regular coverage plots  
#     coverage_data <- coverage_data[source == 'UNICEF_MICS', source:= "UNICEF MICS"]
#     coverage_data <- coverage_data[source == 'MACRO_DHS', source:= "Macro DHS"]
#     coverage_data <- coverage_data[source == 'COUNTRY_SPECIFIC', source:= "Other"]
#     coverage_data <- coverage_data[source == 'ARAB_LEAGUE_PAPFAM', source:= "PAPFAM"]
#     coverage_data <- coverage_data[source == 'WB_CWIQ', source:= "CWIQ"]
#     coverage_data <- coverage_data[source == 'WB_LSMS', source:= "LSMS"]
#     coverage_data <- coverage_data[source == 'WB_LSMS_ISA', source:= "LSMS ISA"]
#     coverage_data <- coverage_data[source == 'WB_PRIORITY_SURVEY', source:= "Priority Survey"]
#     coverage_data <- coverage_data[source == 'CDC_RHS', source:= "CDC RHS"]
#     coverage_data <- coverage_data[source == 'ABD_DHS', source:= "ABD DHS"]
#     coverage_data <- coverage_data[source == 'RAND_FLS', source:= "RAND FLS"]
#     coverage_data <- coverage_data[source == 'CVD_GEMS', source:= "CVD GEMS"]
#     
#     # Adding this next bit in to only run plots with new data. Date ranges are specified by extract_date and previous_date in the beginning. If plot_new_dcp is set to
#     # TRUE, then it will run with new data only. If you want a full coverage plot, set to FALSE
#     
#     
#     if (plot_new_dcp == TRUE){
#       new_nids <- setdiff(current, previous)
#       coverage_data_new <- coverage_data[nid %in% new_nids,]
#       sad_nids <- setdiff(new_nids, coverage_data_new$nid)
#       print(paste0('These NIDs were extracted since ', previous_date,' but have been dropped somewhere in the data processing process:', sad_nids))
#       save_directory <- paste0(j,"WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/data coverage plots/",ind, "_", extract_date,  "_", sex, "_new/")
#       coverage_data <- copy(coverage_data_new)
#     } else {
#       save_directory <- paste0(j, "WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/data coverage plots/",ind, "_", extract_date, "_", sex, "_all/")
#     }
#     
#     if (plot_extract == TRUE){
#       coverage_data <- coverage_data[type == 1,]
#       save_directory <- paste0(j, "WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/data coverage plots/",ind, "_", extract_date, "_", sex, "_extract/")
#     } else {
#       save_directory <- paste0(j, "WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/data coverage plots/",ind, "_", extract_date, "_", sex, "_all/")
#     }
#     
#     # Assoicate indicator with a more readable name for labels
#     if (ind == 'stunting_mod_b'){
#       nice_name <- 'Stunting'
#     } else if (ind == 'wasting_mod_b'){
#       nice_name <- "Wasting"
#     } else if (ind == 'underweight_mod_b'){
#       nice_name <- "Underweight"
#     } else if (ind == 'overweight_who_b'){
#       nice_name <- 'Overweight'
#     } else if (ind == 'overweight_mod_who'){
#       nice_name <- 'Standardized Oerweight'
#     } else if (ind == 'wasting_mod_who'){
#       nice_name <- 'Wasting'
#     } else if (ind == 'normal_mod_who'){
#       nice_name <- 'Standardized Normal'
#     } else if (ind == 'wasting_mod_cond'){
#       nice_name <- 'Wasting'
#     } else if (ind == 'stunting_sev_c'){
#       nice_name <- 'Severe Stunting only'
#     } else if (ind == 'stunting_mil_c'){
#       nice_name <- 'Mild Stunting only'
#     } else if (ind == 'stunting_mod_c'){
#       nice_name <- 'Moderate Stunting only'
#     } else if (ind == 'wasting_sev_c'){
#       nice_name <- 'Severe Wasting only'
#     } else if (ind == 'wasting_mil_c'){
#       nice_name <- 'Mild Wasting only'
#     } else if (ind == 'wasting_mod_c'){
#       nice_name <- 'Moderate Wasting only'
#     } else if (ind == 'obese_who_b'){
#       nice_name <- 'Obesity'
#     } else if (ind == 'underweight_sev_c'){
#       nice_name <- 'Severe Underweight only'
#     } else if (ind == 'underweight_mil_c'){
#       nice_name <- 'Mild Underweight only'
#     } else if (ind == 'underweight_mod_c'){
#       nice_name <- 'Moderate Underweight only'
#     }
#     
#     ## make the plots now that the data is ready 
#     plot_regions<- c("se_asia", "south_asia", "middle_east-TUR", "africa","latin_america")
#     
#     for (reg in plot_regions) {    
#       
#       coverage_maps <- graph_data_coverage_values(df = coverage_data,
#                                                   var = ind,
#                                                   title = nice_name,
#                                                   year_min = 2000,
#                                                   year_max = 2017,
#                                                   year_var = 'start_year',
#                                                   region = reg,
#                                                   cores = 20,
#                                                   indicator = ind,
#                                                   since_date = '2017-11-02',
#                                                   high_is_bad = TRUE,
#                                                   return_maps = TRUE,
#                                                   legend_title = paste0("Prevalence \n of ", nice_name),
#                                                   fast_shapefiles = T,
#                                                   simplify_polys = T,
#                                                   tolerance = 0.01,
#                                                   save_on_share = F,
#                                                   base_font_size = 18,
#                                                   out_dir = save_directory,
#                                                   prep_shiny = F,
#                                                   color_scheme = "classic",
#                                                   #annual_period_maps = TRUE,
#                                                   #save_period_maps = TRUE,
#                                                   new_data_plots = FALSE)
#       #core_repo = "/snfs2/HOME/bvp/lbd_core/lbd_core")  
#     } # Closes for loop that loops through regions to generage coverage plots for each one
#   } # Closes for loop that iterates through indicators and makes a coverage plot for every region for that indicator
# } # Closes the entire coverage plot loop, including prepping data for graph_data_coverage_values() and evaluating if cov_plot == TRUE

###########################################################################################################################################################
## 14.) DATA PREVALENCE PLOTS: These use to happen pre-collapse, but it makes more sense to happen post, plus this way we can include report extractions :)
############################################################################################################################################################

## 1/13/25 Commenting out due to lack of point/poly data.

# if(proj == 'cgf'){
#   binary_ind <-c("stunting_mod_b", "wasting_mod_b", "underweight_mod_b")
# } else if (proj == 'dbm'){
#   binary_ind <-c("wasting_mod_cond", 'overweight_who_b')
# } else if (proj == 'risk_factor'){
#   binary_ind <-c("stunting_mil_c", "stunting_mod_c", "stunting_sev_c", 'wasting_mil_c', 'wasting_mod_c','wasting_sev_c', "underweight_mil_c", "underweight_mod_c", "underweight_sev_c")
# }
# 
# #This code loops through the indicators and creates plots for each one
# 
# # Create a dated folder for the prevalence plots if needed
# # if (!file.exists(paste0(j,"WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/data prevalence plots/", module_date))){
# #   dir.create(paste0(j,"WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/data prevalence plots/", module_date))
# # }
# 
# for (ind in binary_ind){
#   sex <- 'mf'
#   prev_point <- fread(paste0("/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/", topic, "/", ind, "/all_point_data_", sex, "_", module_date, ".csv"))
#   prev_poly <- fread(paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind,'/collapsed_polys_', sex, "_", module_date, '.csv'))
#   
#   if (file.exists(paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind,'/extractions/collapsed_polys_extractions_', sex, '.csv'))){
#     print("This indicator utilizes report extractions")
#     ext <- fread(paste0('ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind,'/extractions/collapsed_polys_extractions_', sex, '.csv'))
#     all_data <- rbind(prev_point, prev_poly, fill=TRUE)
#     all_data <- all_data[,type:=0]
#     all_data <- rbind(all_data, ext, fill=TRUE)
#   } else {
#     print("This indicator does not utilize report extractions")
#     all_data <- rbind(prev_point, prev_poly, fill=TRUE)
#     all_data <- all_data[,type:=0]
#   }
#   
#   
#   indi_name <- ind
#   
#   
#   #names(all_data)[names(all_data) == indi_name] <- "indi_true"
#   # all_data <- all_data[!(is.na(all_data$pweight) | is.na(all_data$indi_true)),]  
#   #all_data$pweight[all_data$point == 1 & is.na(all_data$pweight)] <- 1        
#   
#   #alldat <- mutate(all_data, wt_num = pweight*indi_true)
#   #alldat <- alldat %>% group_by(country, year_start, nid, point) %>% summarize(indicator = sum(wt_num, na.rm = TRUE)/sum(pweight, na.rm = TRUE), total_N = sum(N))
#   names(all_data)[names(all_data) == indi_name] <- "indicator"
#   names(all_data)[names(all_data) == "svy_id"] <- "nid"
#   all_data$point <- as.factor(all_data$point)
#   
#   all_data <- as.data.table(all_data)
#   
#   #all_data <- all_data[is.na(pweight_sum), pweight_sum := N]
#   #all_data <- all_data[,wt_num := pweight_sum*indicator]
#   
#   all_data <- all_data[,.(indicator = sum(indicator, na.rm = TRUE), N = sum(N, na.rm = TRUE)), by = .(country, start_year, nid, point)]
#   all_data <- all_data[,indicator := indicator/N]
#   all_data <- all_data[start_year >= 2000,]
#   # Fix this weird Mexico thing
#   all_data <- all_data[N >10,]
#   
#   plotname = paste0(indi_name, "_", module_date)
#   
#   pdf(file = paste0(j,"WORK/11_geospatial/10_mbg/child_growth_failure/03 diagnostics/data prevalence plots/", module_date, "/", plotname, ".pdf")) 
#   all_data <- as.data.frame(all_data)
#   #write.csv(alldat, paste0("/snfs2/HOME/bvp/child_growth_failure_h/data_validation/", plotname, "_data.csv"))
#   
#   alldat_clean <- subset(all_data, !is.na(all_data$point))  
#   alldat_clean$year_start <- as.numeric(alldat_clean$start_year)
#   message(paste0("Creating prevalence plots for ",ind))    
#   for (i in iso_list) {
#     #If you want to see each ISO as it's plotted, uncomment this:
#     #message(i)
#     plotdat <- filter(alldat_clean, country == i)
#     if (nrow(plotdat) > 0) {
#       print(
#         ggplot(plotdat) + 
#           geom_point(aes(x = start_year, y = indicator, shape = point, size = N,
#                          col = indi_name, fill = "black")) +
#           geom_smooth(aes(x = start_year, y = indicator, col = indi_name, weight = N),
#                       size = 0.5, se = F, fullrange = F, method = lm) +
#           geom_text(aes(x = start_year, y = indicator, label = nid)) +
#           ggtitle(i) + ylim(0,1) + xlim(1996,2018) + 
#           xlab("Year") + ylab("Prevalence") + theme_bw()
#         #+ scale_shape_identity()
#       )
#     } # if statement if nrow is > 0
#   } # for loop that iterates throught iso codes
#   dev.off()  
# }  # Closes the data prevalence plot code   
# 
# rm(all_data)



#############################################################################################################################################################################
## 15.) RESAMPLE POLYGONS TO POINTS: Assign points to all polygon surveys
##      This method runs getPoints on each shapefile in parallel. It is much faster than resample_polygons(), and breaks far less often :) Each shapefile is written to a csv.
##############################################################################################################################################################################

## 1/13/25 Commenting out due to lack of point/poly data.

# if(proj == 'cgf'){
#   binary_ind <-c("stunting_mod_b", "wasting_mod_b", "underweight_mod_b")
# } else if (proj == 'dbm'){
#   binary_ind <- c("overweight_who_b",'wasting_mod_cond')
# } else if (proj == 'risk_factor'){
#   binary_ind <-c("stunting_mil_c", "stunting_mod_c", "stunting_sev_c", "wasting_mil_c", "wasting_mod_c", "wasting_sev_c", "underweight_mil_c", "underweight_mod_c", "underweight_sev_c")
# }
# 
# for (ind in binary_ind){
#   nodes <- ''
#   project <- ifelse(nodes == 'geos',
#                     '-P proj_geo_nodes -l gn=TRUE',                      		
#                     '-P proj_geospatial')
#   user <- "alicela"
#   date <- extract_date
#   
#   #setwd('/homes/ihme/alicela/repos/cgf/data_prep/')
#   indicator <- ind
#   run_date <- module_date
#   
#   if(sex_split==TRUE){
#     sex <- c('m', 'f', 'mf')
#   } else{
#     sex <- 'mf'
#   }
#   
#   for(s in sex){
#     polydat <- read.csv(paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', indicator,'/collapsed_polys_', s, "_", run_date, '.csv'))
#     polydat <- as.data.table(polydat)
#     polydat <- polydat[,type:=0]
#     polydat <- as.data.frame(polydat)
#     
#     if (file.exists(paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind,'/extractions/collapsed_polys_extractions_', sex, '.csv'))){
#       exdat <- read.csv(paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/',indicator, '/extractions/collapsed_polys_extractions_', s, '.csv'))
#       
#       polydat <- rbind(polydat, exdat)
#     }
#     
#     
#     
#     #polydat <- filter(polydat, is.na(latitude) & !is.na(shapefile) & !is.na(location_code))
#     polydat <- subset(polydat, point == 0)
#     
#     for (shp in unique(polydat$shapefile)) { 
#       jname <- paste(indicator, shp, sep = "_")
#       mycores <- 4
#       sys.sub <- paste0("qsub ",project,paste0(" -l m_mem_free=10G -l fthread=2 -q all.q -l archive=TRUE -e /share/temp/sgeoutput/",user,"/errors -o /share/temp/sgeoutput/",user,"/output "),"-cwd -N ", jname, " ")
#       script <- "/share/homes/alicela/repos/cgf/data_prep/child.R"
#       r_shell <- ifelse(nodes == 'geos',
#                         'r_shell_geos.sh',                      		
#                         '/share/singularity-images/rstudio/shells/r_shell_singularity_3501.sh')
#       
#       args <- paste(shp, indicator, run_date, s)
#       system(paste(sys.sub, r_shell, script, args)) 
#     } # Closes for loop that iterates through shapefiles and submits qsubs for each one
#     rm(polydat)
#   } # Closes for loop that iterates through each sex and runs the resample process
#   
# } # Closes resample code

#######################################################################################################################################################################################
## 16.) FINAL PROCESSING STEPS: Append point data and resampled polygon data together, rename columns, drop specific surveys, do some final indicator specific tasks, write to folder!
#######################################################################################################################################################################################

## 1/13/25 Commenting out due to lack of point/poly data.

# if(sex_split==TRUE){
#   sex <- c('m', 'f', 'mf')} else {
#     sex <- 'mf'
#   }   
# 
# for (ind in binary_ind) {
#   # Write all resampled shapefile files to a list
#   for (s in sex){
#     resampled_polys <- list.files(paste0(j, "temp/alicela/collapse_temp/resample_", ind, "_", s), full.names=T, pattern = ".csv$", ignore.case=T, recursive = T)
#     # Read in each file and bind together
#     for (poly in resampled_polys){
#       
#       # if the merged dataset doesn't exist, create it
#       print(poly)
#       if (!exists("dataset")){
#         dataset <- read.table(poly, header=TRUE, sep=",")
#       }
#       
#       # if the merged dataset does exist, append to it
#       if (exists("dataset")){
#         temp_dataset <-read.table(poly, header=TRUE, sep=",")
#         dataset<-rbind(dataset, temp_dataset)
#         rm(temp_dataset)
#       }
#       
#     }
#     
#     # Convert resampled poly data to a data table
#     resampled_poly_data <- as.data.table(dataset) 
#     rm(dataset) # need to remove dataset or else this will mess up the next indicator
#     
#     # format resampled poly data so that it can be binded onto collapsed point data
#     setcolorder(resampled_poly_data, c('source', 'start_year', 'lat', 'long', 'country', 'svy_id', 'N', ind, 'pweight_sum', 'point', 'weight', 'shapefile', 'location_code', 'X', 'type'))
#     setnames(resampled_poly_data, c('lat', 'long'), c('latitude', 'longitude'))
#     
#     # Format point data so it can be binded to resampled poly data
#     # all_point_data <- all_point_data[, pseudocluster := NULL]
#     
#     all_point_data <- fread(paste0("/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/", topic, "/", ind, "/all_point_data","_", s, "_", module_date, ".csv"))
#     all_point_data <- all_point_data[, weight := 1]
#     all_point_data <- all_point_data[, shapefile := ""]
#     all_point_data <- all_point_data[, location_code := ""]
#     all_point_data <- all_point_data[, X := NA]
#     all_point_data <- all_point_data[, type := 0]
#     
#     
#     ##############################################################################################################################################
#     ## Append point and polygon collapsed data
#     all_processed_data <- rbind(all_point_data, resampled_poly_data)
#     setnames(all_processed_data, 'start_year', 'year')
#     all_collapsed <- copy(all_processed_data)
#     
#     ## Replace year with period 1998-2002, 2003-2007, 2008-2012, 2013-2017
#     all_collapsed <- subset(all_collapsed, year > 1997)
#     names(all_collapsed)[names(all_collapsed) == "year"] = "original_year"
#     all_collapsed <- all_collapsed[original_year >= 1998 & original_year <= 2002, year := 2000]
#     all_collapsed <- all_collapsed[original_year >= 2003 & original_year <= 2007, year := 2005]
#     all_collapsed <- all_collapsed[original_year >= 2008 & original_year <= 2012, year := 2010]
#     all_collapsed <- all_collapsed[original_year >= 2013 & original_year <= 2017, year := 2015]
#     
#     all_collapsed <- all_collapsed[, latitude := as.numeric(latitude)]
#     all_collapsed <- all_collapsed[, longitude := as.numeric(longitude)]
#     all_collapsed <- all_collapsed[!is.na(latitude)]
#     all_collapsed <- all_collapsed[!is.na(longitude)]
#     all_collapsed <- all_collapsed[latitude>=-90 & latitude<=90]
#     all_collapsed <- all_collapsed[longitude>=-180 & longitude<=180]
#     
#     # Fix the NIDs I split up for China
#     weird_nids <- c('20083890', '20083896', '20083897', '20083891', '20083893', '20083894', '20083899')
#     all_collapsed[svy_id %in% weird_nids, svy_id:= 200838]
#     
#     
#     all_collapsed <- all_collapsed[, var := get(ind)]
#     all_collapsed <- all_collapsed[, var := round(var)]
#     ## In clusters where ind > N (due to tiny samples and every child having ind = 1), cap at N
#     all_collapsed <- all_collapsed[var > N, var := N]
#     all_collapsed[, (ind) := NULL]
#     names(all_collapsed)[names(all_collapsed) == 'var'] <- get("ind")  
#     
#     
#     # Remove rows that have a NA value in column 'N' or indicator column
#     all_collapsed <- na.omit(all_collapsed, cols=c("N", ind))
#     
#     
#     # Change column names
#     
#     setnames(all_collapsed, "year", "period_year")
#     setnames(all_collapsed, "original_year", "year")
#     
#     # Include 1998 Iran before subsetting by year for DBM
#     
#     if(proj == 'dbm'){
#       all_collapsed[svy_id == 19046, year := 2000]
#     }
#     
#     # Subset by stage and year
#     
#     all_collapsed <- all_collapsed[year >= 2000,]
#     all_collapsed <- all_collapsed[country %in% iso_list,]
#     
#     # Drop weird surveys
#     
#     if(proj == 'cgf'){
#       all_collapsed <- all_collapsed[svy_id != 79839,]
#       all_collapsed <- all_collapsed[svy_id != 24143,]
#     } else if (proj == 'dbm'){
#       exclude_dbm <- c(76704, 79839, 14015, 14027, 14063, 14105, 58660, 9516, 60942, 24143, 369294, 283812)
#       all_collapsed <- all_collapsed[(!svy_id %in% exclude_dbm),]
#     } else if (proj == 'risk_factor'){
#       all_collapsed <- all_collapsed[svy_id != 79839,]
#       all_collapsed <- all_collapsed[svy_id != 24143,]
#     }
#     
#     # Drop report extractions that are from the same year as microdata
#     
#     exclude_report <- c(56883, 306329, 52147, 151732, 151729)
#     all_collapsed <- all_collapsed[(!svy_id %in% exclude_report),]
#     
#     # Subset by indicator
#     
#     if (ind %in% c('overweight_who_b', 'wasting_mod_cond')){
#       assign(paste0("collapsed_", ind, "_", s), all_collapsed)
#     }
#     
#     if (proj == 'cgf'){
#       write.csv(all_collapsed, file = paste0(l, "rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/4_collapsed/", ind, "_",s,"_", module_date, ".csv"), row.names = FALSE)
#     }
#     
#     if (proj == 'risk_factor'){
#       write.csv(all_collapsed, file = paste0(l, "rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/4_collapsed/", ind, "_",s,"_", module_date, ".csv"), row.names = FALSE)
#     }
#     
#     
#     
#     print(paste0("Finished ", ind, " ", s))
#     rm(all_collapsed)
#     rm(resampled_poly_data)
#     rm(all_point_data)
#     rm(all_processed_data)
#     #  Sys.time() - start_time
#     
#   } # Closes the for loop that iterates through each sex for a given indicator and does final processing on each dataset
# } # Closes the for loop that iterates through indicators and does final processing on each data set
# 
# # Create an index to match up overweight_mod_b and wasting_mod_cond - has to be done outside of the loop since we need both datasets to create the index.            
# 
# if (proj == 'dbm'){                 
#   create_dbm_index <- function(collapsed_overweight_who_b, collapsed_wasting_mod_cond, s){
#     # Creating index for overweight_who_b and wasting_mod_cond ONLY
#     collapsed_overweight_who_b[,index := .GRP, by =.(latitude, longitude)] # give unique numbers for each unique lat/long combo
#     index_over <- collapsed_overweight_who_b[,c('latitude', 'longitude', 'index')] # subset to create an index data table
#     index_over <- unique(index_over[]) # Remove duplicates to index table
#     
#     # applying the index from wasting to overweight
#     collapsed_wasting_mod_cond_index <- merge(collapsed_wasting_mod_cond, index_over, by = c('latitude', 'longitude'), all.x = TRUE, all.y = FALSE) # merge indexes to wasting data
#     collapsed_wasting_mod_cond_index[is.na(index),no_index:=1] # create a new variable that identifies rows that didn't match lat/longs with the index table so weren't assigned an index
#     collapsed_wasting_mod_cond_index[is.na(index),index := .GRP, by =.(latitude, longitude)] # create a new index for those rows
#     collapsed_wasting_mod_cond_index[no_index ==1,index := index+277131] # Add 277131 to each new index number so it won't overlap with the old numbers
#     collapsed_wasting_mod_cond_index[,no_index:=NULL] # remove the variable that identified NA index values
#     collapsed_wasting_mod_cond <- copy(collapsed_wasting_mod_cond_index)
#     
#     write.csv(collapsed_wasting_mod_cond, file = paste0(l, "rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/1_raw_extractions/", "wasting_mod_cond_", s,"_", module_date, ".csv"), row.names = FALSE)
#     write.csv(collapsed_overweight_who_b, file = paste0(l, "rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/1_raw_extractions/", "overweight_who_b_", s,"_",module_date, ".csv"), row.names = FALSE) #writing as wasting_who_b to keep seperate from cgf moving forward since the index is different
#   }
#   
#   
#   if (sex_split == TRUE){
#     create_dbm_index(collapsed_overweight_who_b_mf, collapsed_wasting_mod_cond_mf, "mf")
#     create_dbm_index(collapsed_overweight_who_b_m, collapsed_wasting_mod_cond_m, "m")
#     create_dbm_index(collapsed_overweight_who_b_f, collapsed_wasting_mod_cond_f, "f")
#   } else {
#     create_dbm_index(collapsed_overweight_who_b_mf, collapsed_wasting_mod_cond_mf, "mf")
#   }
# } # Closes if statement that does final processing for dbm indicators

###################################################################################################################################################################################
## 17.) Generate exclusions list
###################################################################################################################################################################################

## 1/13/25 Commenting out as BMI_drop, BMI_exclude, BMIZ_drop, BMIZ_exclude, birth_weight_drop, and birth_weight_exclude do not exist in data.

# exclude_vars <- c("HAZ_drop", "HAZ_exclude", "WAZ_drop", "WAZ_exclude", "WHZ_drop", "WHZ_exclude", "BMI_drop", "BMI_exclude", "BMIZ_drop", "BMIZ_exclude", "birth_weight_drop", "birth_weight_exclude", "nid")
# 
# cgf_exclude <- data[, exclude_vars, with=F]
# 
# cgf.exclusions <- aggregate(cbind(is.na(HAZ_drop),
#                                   is.na(HAZ_exclude),
#                                   is.na(WAZ_drop),
#                                   is.na(WAZ_exclude),
#                                   is.na(WHZ_drop),
#                                   is.na(WHZ_exclude),
#                                   is.na(BMI_drop),
#                                   is.na(BMI_exclude),
#                                   is.na(BMIZ_drop),
#                                   is.na(BMIZ_exclude),                                        
#                                   #CIAF_drop,
#                                   #CIAF_exclude,
#                                   is.na(birth_weight_drop),
#                                   is.na(birth_weight_exclude)) ~ nid,
#                             data = cgf_exclude, FUN = mean)
# colnames(cgf.exclusions) <- c('nid', 'HAZ_drop', 'HAZ_exclude', 'WAZ_drop', 'WAZ_exclude', 'WHZ_drop', 'WHZ_exclude',
#                               'BMI_drop', 'BMI_exclude', 'BMIZ_drop', 'BMIZ_exclude', 'birth_weight_drop', 'birth_weight_exclude')
# 
# write.csv(cgf.exclusions, paste0(folder_out, "/exclude_diagnostic_", module_date, ".csv"), row.names = F)

################################################################################################################################################################      