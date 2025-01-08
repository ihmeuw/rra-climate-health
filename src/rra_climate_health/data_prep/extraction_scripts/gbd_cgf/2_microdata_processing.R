# Script found from: https://stash.ihme.washington.edu/projects/MNCH/repos/cgf/browse/processing/2_microdata_processing_functions.R?at=gbd2022
# Edited copy for goalkeepers 2024 
# - changing all file paths to integrated analytics
# - changing location of raw data directory
# - deleting any references to LIMITED_USE drives

# Ryan Fitzgerald
# Script found from https://stash.ihme.washington.edu/projects/MNCH/repos/cgf/browse/processing/2_microdata_processing.R?at=gbd2022
# Edited copy for goalkeepers 2024 
# - source goalkeepers-edited version of functions 

# CGF DATA PROCESSING CONTROL SCRIPT

# library(mrbrt003, lib.loc = "/ihme/code/mscm/Rv4/dev_packages/") # run this is mrbrt isn't working Sys.setenv("RETICULATE_PYTHON" = "/ihme/code/mscm/miniconda3/envs/mrtool_0.0.1/bin/python")
# attempt 2
reticulate::use_python("/ihme/code/mscm/miniconda3/envs/mrtool_0.0.2/bin/python")
# library(mrbrt003)
library(mrbrt003, lib.loc = "/ihme/code/mscm/Rv4/dev_packages/")
# reticulate::use_python("/ihme/code/mscm/miniconda3/envs/mrtool_0.0.1/bin/python")
mr <- import("mrtool")

# # attempt 3
# library(reticulate)
# reticulate::use_python("/opt/miniconda/envs/mrtool-0.0.3/bin/python")
# mr <- reticulate::import("mrtool")
# 
# # attempt 4
# library(reticulate)
# reticulate::use_python("/ihme/code/mscm/miniconda3/envs/mrtool_0.0.1/bin/python")
# mr <- reticulate::import("mrtool")

source("/share/code/ubcov/ubcov_central/modules/collapse/launch.r")
source("~/repos/goal_keepers/yearly_reports/2024/exploratory/CGF_scripts/2_microdata_processing_functions.R")
# library(anthro, lib.loc = "~/repos/cgf") # Had to clone from gbd2022 branch of repo
library(anthro, lib.loc = "/mnt/team/nch/pub/packages") # changed after referring to .Rprofile script
library(anthro)
library(data.table)
library(plyr)
library(dplyr)
library(haven)
library(ggplot2)
library(survey)
library(Hmisc)
"%ni%" <- Negate("%in%")
"%notlike%" <- Negate("%like%")



################################################################
######################### STEP 1 ###############################
################## Extracting Microdata ########################
################################################################
# Step one is extracting microdata itself using UbCov or Winnower
# Make sure that data is extracted to a folder at this kind of directory if on J Drive
# /ihme/mnch/cgf/data/ubcov_extraction_process/<data_date>/1_raw_extractions/
# And this kind of directory if on the L Drive
# /mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/<data_date>/1_raw_extractions/







################################################################
######################### STEP 2 ###############################
################### INITIAL PROCESSING #########################
################################################################
# In this section we read in raw extracted microdata and apply some basic cleaning criteria
# We also do some checks to see if everything looks okay, flaging any studies that look problematic



# First assign a "data processing date" name - this is how we'll keep track of which microdata was processed when. This can be a date or a cycle or any label.
data.date = "10_09_2024"
data_date = "10_09_2024"
re.pre.process.some.sources = TRUE


# Creates the 2_initial_processing, 3_after_mrbrt_outliering, 4_collapsed, and potential_issue_sources folders on the J and L drive which are adjacent to the raw_extraction folders
create_data_processing_folders(data.date)


### ### ### ### ### ### ### ### ### ### 
### STARTING PRE-PROCESSING FOR MICRODATA THAT IS NOT LIMITED USE, THEREFORE LOCATED ON THE J DRIVE
### ### ### ### ### ### ### ### ### ### 


# vector of all files that have been extracted and stored on the J drive
# hard-code below if not rerunning create_data_processing_folders
# new.microdata.j.path <- "/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/raw_filtered/" 
new.microdata.j.path <- "/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/data_10_09_2024/1_raw_extractions/"
all.j.files <- list.files(new.microdata.j.path)

# # vector of all files that have been through first step of processing
# already_initial_processed <- list.files(paste0("/mnt/share/mnch/cgf/data/ubcov_extraction_process/data_", data.date, "/2_initial_processing/"))
# # remove the last four characters from each file name to get the NID
# already_initial_processed <- substr(already_initial_processed, 1, nchar(already_initial_processed)-4)
# all.j.names <- substr(all.j.files, 1, nchar(all.j.files)-4)
# all.j.names <- all.j.names[all.j.names %ni% already_initial_processed]
# all.j.files <- paste0(all.j.names, ".dta")


# looping through each file and pre-processing
# # looping through each file and pre-processing
# for (file in all.j.files) {
#   process_files(file = file,
#                 path = file.path(new.microdata.j.path),
#                 team = "integrated_analytics",
#                 type = "j",
#                 all_files = all.j.files)
# }

for (file in all.j.files) {
  tryCatch({
    
    one.file <- as.data.table(fread(paste0(new.microdata.j.path, file))) 
    
    # Need to update data types so that they don't get converted to logical
    one.file$age_day <- as.numeric(one.file$age_day)
    one.file$age_month <- as.numeric(one.file$age_month)
    one.file$age_year <- as.numeric(one.file$age_year)
    
    print(paste0("Starting NID ", unique(one.file$nid), " which is source ", which(all.j.files == file), " out of ", length(all.j.files), " you're working on processing."))
    
    if(("metab_height" %in% names(one.file) | "metab_weight" %in% names(one.file)) & "sex_id" %in% names(one.file)){
      one.file <- subset_to_usable_rows(one.file) 
      if(nrow(one.file) >10){
        one.file <- extraction_or_unit_issue_check(one.file, file.loc = "j")
        if(unique(one.file$suspicious.heights) == FALSE & unique(one.file$suspicious.weights) == FALSE){
          one.file <- calculate_z_scores(one.file)
          save_preprocessed_j_file(file, one.file)
        }
        if(unique(one.file$suspicious.heights) == TRUE & unique(one.file$suspicious.weights) == TRUE){
          log_small_or_excluded_j_sources(file, data.date, one.file, issue.type = "suspicious_values")
        }
      }
    } else {
      if("sex_id" %ni% names(one.file)){
        log_small_or_excluded_j_sources(file, data.date, one.file, issue.type = "no_sex_info")
      } else {
        log_small_or_excluded_j_sources(file, data.date, one.file, issue.type = "no_height_weight")
      }
    }
    
    if(nrow(one.file) <= 10){
      log_small_or_excluded_j_sources(file, data.date, one.file, issue.type = "small_sample_size")
    }
    
  }, error = function(e) {
    print(paste0("Error processing file ", file, ": ", e$message))
  })
}


### ### ### ### ### ### ### ### ### ### 
### STARTING PRE-PROCESSING FOR MICRODATA THAT IS  LIMITED USE, THEREFORE LOCATED ON THE l DRIVE
### ### ### ### ### ### ### ### ### ### 




################################################################
######################### STEP 3 ###############################
#################### MRBRT OUTLIERING ##########################
################################################################
# Here, we're going to outlier microdata points that we think are unreasonable
# If we only have HAZ or WAZ, we outlier if it's below -6 Z-scores or above +6 Z-scores
# If we have HAZ, WAZ, & WHZ, we outlier based on a MRBRT model describing the expected relationship of those 3 variables


# enter all the data.dates you want to contribute towards the MRBRT model to predict HAZ, WAZ, WHZ relationship
all.microdata.data.dates <- c("data_10_09_2024")

# enter the data.dates that you want to potentially outlier data points from (potentially do not list data.dates that have already been outliered)
new.microdata.data.dates <- c("data_10_09_2024")



# Creates a dataframe that has a column with all the filepaths for every microdata source to be input to the mrbrt model
source.df <- get_all_cleaned_data_filepaths(all.microdata.data.dates)



# We're going to perform the MRBRT model 10 times
ten.grids <- lapply(1:10, function(v){
  
  
  print(paste0("Taking microdata sample ", v, " out of 10 and creating an expected grid of HAZ, WAZ, and WHZ."))
  
  # here we sample the dataframe that is input to the mrbrt model
  # We randomly select up to 100 observations from each 1x1 Z score bin from -6 to +6 
  mrbrt.microdata.sample <- get_sample_in_each_bin(source.df)
  
  # We run the mrbrt model and predict out on a grid that ranges from -10 to +10
  plane.grid <- create_haz_waz_whz_grid(mrbrt.microdata.sample)
  
  
})


final.grid <- get_average_of_ten_grids(ten.grids)

allowable.diff <- get_average_residual_allowed(ten.grids)


# Creates a dataframe that has a column with all the filepaths for every microdata source to be potentially split from the MRBRT model
sources.to.mrbrt.process <- get_all_cleaned_data_filepaths(new.microdata.data.dates)




# loop through every survey we're applying this model to and save the post-mrbrt haz/waz/whz files
for (fp in sources.to.mrbrt.process$fp) {
  
  outlier_based_on_mrbrt(fp, allowable.diff, final.grid)
  
  
}


## FOR GOALKEEPERS, STOP HERE





