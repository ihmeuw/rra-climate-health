################################################################################
# DESCRIPTION: Parent script to submit jobs for each fold in a k-fold cross-validation
# task
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

results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"
folds_dir <- paste0(results_dir,"folds/")

#==============================================================================
# SECTION 1: Call k jobs
#==============================================================================

## Test secondary climate variables with k-fold cross-validation
child_script <- "/ihme/homes/elyeb/repos/rra-climate-health/notebooks/child_mortality/1b_child_mortality_k_fold_cv_child_call_models.R"

# Update output_log and error_log to personal directory in slurmoutput

job_name_root <- 'child_mortality'

folds <- list.files(path = folds_dir, pattern = "\\.rds$", full.names = FALSE)

for (fold in folds) {
  
  output_log <- paste0('/ihme/temp/slurmoutput/elyeb/output/%x.o%j','_',fold)
  error_log <- paste0('/ihme/temp/slurmoutput/elyeb/errors/%x.e%j','_',fold)
  
  fold_number <- as.integer(gsub(".*_(\\d+)\\.rds$", "\\1", fold))
  job_name <- paste0(job_name_root,fold_number)
  
  qsub_str <- paste("sbatch -J",job_name,"--mem=100G -c 6 -A proj_integrated_analytics -t 4-24 -p long.q",
                    "-o ",output_log,"-e",error_log, 
                    "/ihme/singularity-images/rstudio/shells/execR.sh",
                    "-s ", child_script, fold,sep=" ")
  
  system(qsub_str)
  
  Sys.sleep(5.0)
}