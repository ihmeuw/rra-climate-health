################################################################################
# DESCRIPTION: Script to evaluate model predictions from cross-validation
# PROJECT: Climate nutrition
# DATE: 2025-09-17
################################################################################

#==============================================================================
# SECTION 0: PACKAGE LOADING AND ENVIRONMENT SETUP
#==============================================================================
# Clear workspace
rm(list = ls())

# Username is pulled automatically
username <- Sys.info()[["user"]]
if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/"
  h <- paste0("/homes/", username, "/")
  r <- "/mnt/"
  l <-"/ihme/limited_use/"
} else {
  j <- "J:/"
  h <- "H:/"
  r <- "R:/"
  l <- "L:/"
}

# install.packages('frailtyEM',lib = "/homes/elyeb/rlibs") # able to handle mixed effects and predict on new data
library(data.table)
library(arrow) # to read parquet

options(scipen = 999) # turn off scientific notation


#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"
folds_dir <- paste0(results_dir,"folds/")
folds_data_subdir <- paste0(folds_dir,"data_subsets/")
folds_results_subdir <- paste0(folds_dir,"folds_results/")

#==============================================================================
# SECTION 2: READ AND COMBINE RESULTS
#==============================================================================

list.files(folds_results_subdir)

ex_df <- read_parquet(paste0(folds_results_subdir,"predictions_fold_5_vars_precipitation_days.parquet"))
