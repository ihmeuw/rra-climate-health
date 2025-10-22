#' @Title: [1_compile_mortality_sources.R]
#' @Purpose: Compile mortality sources.


#SET-UP
rm(list=ls())

if (Sys.info()[1] == "Linux"){
  j <- "/home/j/"
  h <- paste0("/ihme/homes/", Sys.info()[7], "/")
  k <- "/ihme/cc_resources/"
} else if (Sys.info()[1] == "Windows"){
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
}

library(haven)
library(dplyr)
library(readr)
library(stringr)
library(fs)

#Set_up
###############################################
#Enter filepath with old date
today_date <- "2025_10_14" # ENTER TODAY'S DATE
compiled_old <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/dem_br/dem_br_combined.parquet")
parent_folder <- "/mnt/team/demographics/priv/data/winnower/8/"  
output_folder <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/"  
exclude_folders <- c("archived", "source_lists","dem_hhd", "dem_hhm", "dem_mn", "dem_wn", "dem_sibs", "dem_vr", "dem_br_vr", "lbw", "lbwsg", "anemia")

if (!dir_exists(output_folder)) dir_create(output_folder)

all_folders <- dir_ls(parent_folder, type = "directory")
folders_to_process <- setdiff(basename(all_folders), exclude_folders)

for (folder_name in folders_to_process) {
  folder_path <- file.path(parent_folder, folder_name)
  
  # Output folder, same name as input folder
  output_folder <- file.path(output_folder, folder_name)
  if (!dir_exists(output_folder)) dir_create(output_folder)
  
  dta_files <- dir_ls(folder_path, glob = "*.dta")
  
  df_list <- list()
  
  for (f in dta_files) {
    df <- tryCatch({
      read_dta(f) %>% mutate(across(everything(), as.character))
    }, error = function(e) NULL)
    if (!is.null(df) && nrow(df) > 0) {
      cat(sprintf("Read %s rows from %s\n", nrow(df), f))
      df_list[[length(df_list) + 1]] <- df
    }
  }
  if (length(df_list) == 0) next
  
  # Combine
  combined_df <- bind_rows(df_list)
 
# Quality checks
###############################################
  
#Were any new nids added to the demographics team folder?
nid_add <- setdiff(combined_df$nid, compiled_old$nid)
cat("NIDS ADDED:", unique(nid_add), "\n")

#Were any nids dropped from the last dataset?
nid_drop <- setdiff(compiled_oldf$nid, combined_df$nid)
cat("NIDS DROPPED:", unique(nid_drop), "\n")
  
#Save combined output as Parquet
  combined_file <- file.path(output_folder, paste0(folder_name, "_combined_",today_date,".parquet")) # CHANGE DATE
  write_parquet(combined_df, combined_file)
  cat(sprintf("Saved combined Parquet: %s\n", combined_file))
}