################################################################################
# DESCRIPTION: Script to call series of specifications as sub-tasks, over a single
# fold of the full data set. 
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



fold_file <- commandArgs()[4]

fold <- as.integer(gsub(".*_(\\d+)\\.rds$", "\\1", fold_file))

print(paste0("running on fold file ",fold_file))

#==============================================================================
# SECTION 1: SUBMIT JOBS FOR DIFFERENT MODEL SPECIFICATIONS
#==============================================================================


child_script <- "/ihme/homes/elyeb/repos/rra-climate-health/notebooks/child_mortality/1c_child_mortality_k_fold_cv_child_run_models.R"

## Fit all univariate models
climate_vars <- c(
  "mean_temperature",
  "total_precipitation",
  "relative_humidity",
  "mean_high_temperature",
  "mean_low_temperature",
  "precipitation_days",
  "days_over_30C",
  "days_over_26C"
)

secondary_climate_vars <- climate_vars[climate_vars!="mean_temperature"]
secondary_climate_vars <- secondary_climate_vars[secondary_climate_vars!="mean_high_temperature"]
secondary_climate_vars <- secondary_climate_vars[secondary_climate_vars!="mean_low_temperature"]

# Update output_log and error_log to personal directory in slurmoutput

job_name_root <- 'child_mortality'


for (var in climate_vars) {
  
  output_log <- paste0('/ihme/temp/slurmoutput/elyeb/output/%x.o%j','_',fold,'_',var)
  error_log <- paste0('/ihme/temp/slurmoutput/elyeb/errors/%x.e%j','_',fold,'_',var)
  
  job_name <- paste0(fold,var,job_name_root)
  
  qsub_str <- paste("sbatch -J",job_name,"--mem=100G -c 6 -A proj_integrated_analytics -t 4-24 -p long.q",
                    "-o ",output_log,"-e",error_log, 
                    "/ihme/singularity-images/rstudio/shells/execR.sh",
                    "-s ", child_script, fold_file,var,sep=" ")
  
  system(qsub_str)
  
  Sys.sleep(5.0)
}

## Fit all models with mean_temperature + another variable
for (var in secondary_climate_vars) {
  
  output_log <- paste0('/ihme/temp/slurmoutput/elyeb/output/%x.o%j','_',fold,'_',"mean_temperature","_",var)
  error_log <- paste0('/ihme/temp/slurmoutput/elyeb/errors/%x.e%j','_',fold,'_',"mean_temperature","_",var)
  
  
  job_name <- paste0(fold,"mean_temperature",var,job_name_root)
  
  qsub_str <- paste("sbatch -J",job_name,"--mem=100G -c 6 -A proj_integrated_analytics -t 4-24 -p long.q",
                    "-o ",output_log,"-e",error_log, 
                    "/ihme/singularity-images/rstudio/shells/execR.sh",
                    "-s ", child_script, fold_file,"mean_temperature",var,sep=" ")
  
  system(qsub_str)
  
  Sys.sleep(5.0)
}