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


zone_nos <- seq(6,29)

child_scripts <- c("/ihme/homes/elyeb/repos/rra-climate-health/notebooks/neonatal/zones_neo_1_mo.R",
                   "/ihme/homes/elyeb/repos/rra-climate-health/notebooks/neonatal/zones_neo_3_mo.R",
                   "/ihme/homes/elyeb/repos/rra-climate-health/notebooks/neonatal/zones_neo_6_mo.R",
                   "/ihme/homes/elyeb/repos/rra-climate-health/notebooks/neonatal/zones_neo_9_mo.R")

for (c in child_scripts){
  for (z in zone_nos) {
    
    script_name <- strsplit(c,"/")[[1]][9]
    script_name <- strsplit(script_name,"\\.")[[1]][[1]]

    job_name <- paste0(script_name,'_z_',z)
    output_log <- paste0('/ihme/temp/slurmoutput/elyeb/output/%x.o%j','_',job_name)
    error_log <- paste0('/ihme/temp/slurmoutput/elyeb/errors/%x.e%j','_',job_name)
    
    qsub_str <- paste("sbatch -J",job_name,"--mem=80G -c 6 -A proj_rapidresponse -t 06:00:00 -p long.q",
                      "-o ",output_log," -e",error_log, 
                      "/ihme/singularity-images/rstudio/shells/execR.sh -i /ihme/singularity-images/rstudio/ihme_rstudio_4423.img",
                      "-s ", c, z,sep=" ")
    
    system(qsub_str)
    
    Sys.sleep(1.0)
  }
}