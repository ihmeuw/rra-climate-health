################################################################################
# DESCRIPTION: Call child scripts with zone number as command-line argument.
# PROJECT: Climate nutrition
# DATE: 2025-12-10
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

time_periods <- c(1,3,6,9)
thresholds <- c(9,95)
path_root <- "/ihme/homes/elyeb/repos/rra-climate-health/notebooks/neonatal/quantiles_simplified/"

for (t in time_periods){
  for(th in thresholds){
    
    child_script <- paste0(path_root,"neo_",t,"_mo_q",th,".R")
    job_name <- paste0("q",th,"_mo",t)
    
    output_log <- paste0('/ihme/temp/slurmoutput/elyeb/output/%x.o%j','_',job_name)
    error_log <- paste0('/ihme/temp/slurmoutput/elyeb/errors/%x.e%j','_',job_name)
    
    qsub_str <- paste("sbatch -J",job_name,"--mem=600G -c 6 -A proj_rapidresponse -t 5-24 -p long.q",
                      "-o ",output_log," -e",error_log, 
                      "/ihme/singularity-images/rstudio/shells/execR.sh -i /ihme/singularity-images/rstudio/ihme_rstudio_4423.img",
                      "-s ", child_script,sep=" ")

    system(qsub_str)
    
    Sys.sleep(1.0)
  }
}
