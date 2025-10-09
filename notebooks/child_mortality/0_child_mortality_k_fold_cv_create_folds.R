################################################################################
# DESCRIPTION: Create folds of the full data set, or a subset percentage that's
# a percent of individuals (as opposed to observations), and balanced by country.
# These folds then get used to model different specifications for cross-validation.
# PROJECT: Climate nutrition
# DATE: 2025-10-07
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

library(data.table)
library(caret) # for createFolds function
library(dplyr) # for anti_join function
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

sample_percent <- 0.1
num_folds <- 10

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_08.01/data.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"
folds_dir <- paste0(results_dir,"folds/")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(folds_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_parquet(data_version)
df <- data.table(df)


#==============================================================================
# SECTION 2: GET SAMPLE AND MAKE FOLDS
#==============================================================================


# Sample data, keeping all observations for any sampled individual child, and 
# balancing countries
df_model <- data.table(df)

# Make index variable
df_model[,index_col := .I]

# make indv ID
df_model[,indv_id := paste(nid,psu,hh_id,line_id,sep="_")]
# get sample
indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
indv_counts <- indv_dt[, .N, by = ihme_loc_id]
indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
indv_dt[, n_sample := floor(sample_percent * N)]

set.seed(42)
sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
df_sample <- df_model[indv_id %in% sampled_indv]

# randomize k folds of individuals and save index lists
# maintain country balance

# Use same approach as getting above % sample, but for the size of the fold
# (e.g., if the number of folds is 10, the fold size would be 10% of df_sample)

# get sample
s_indv_dt <- unique(df_sample[, .(indv_id, ihme_loc_id)])
# assign fold numbers
set.seed(42)
s_indv_dt[, fold := sample(rep(1:num_folds, length.out=.N)), by = ihme_loc_id]

df_sample <- merge(df_sample, s_indv_dt[, .(indv_id, fold)], by = "indv_id", all.x = TRUE)

# Get indices for the folds from the full data set
fold_info <- unique(df_sample[, .(indv_id, fold)])
df_model <- merge(df_model, fold_info, by = "indv_id", all.x = TRUE)


# Save fold indices as rds for quick loading by downstream scripts
for (k in 1:num_folds) {
  fold_indices <- df_model[fold == k, index_col]
  saveRDS(fold_indices, file = paste0(folds_dir,"fold_indices_", k, ".rds"))
}




