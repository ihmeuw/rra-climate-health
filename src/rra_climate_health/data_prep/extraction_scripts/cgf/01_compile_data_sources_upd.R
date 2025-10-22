#####################################################################################
######### Script updated for RRA climate malnutrition analysis ######################
### Date: 1/24/2025
### User: khong1 (Kristin Hong) and mayaol (Maya Oleynikova)
### Function: Compile extracted data from multiple sources.
### Input: /mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/extracted_00_DHS.csv
### Output: /mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/extracted_ALL_compiled_all_polygon.csv 
###         /mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/extracted_ALL_compiled_point_and_polygon.csv 
### Notes: Original comments left below. Adapted from original script found here: https://github.com/ihmeuw/rra-climate-health/blob/feature/raw-extractions/src/rra_climate_health/data_prep/extraction_scripts/archive_unused/wealth/01_compile_data_sources.R
######################################################################################

######################################################################################
#' @Title: [01_compile_data_sources.R]  
#' @Authors: Kayleigh Bhangdia and Bianca Zlavog 
#' @contact: bhangdia@uw.edu; zlavogb@uw.edu
#' @Date_code_last_updated: 10/2023
#' @Date_data_last_updated: 10/2023
#' @Purpose: Script to compile extracted data across multiple data sources.
#' Runs once on all data where points have been matched to polygons for SAE polygon modeling (point_to_polygon = TRUE), 
#' and once with separate point and polygon data for MBG/point modeling (point_to_polygon = FALSE).
#' When running this script to update the compiled data, change the `change_note` variable
#' to describe the updates made since last round of compiling.
#' @File_Input(s)
#' Source-specific extracted and formatted files at 
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_*.csv`
#' @File_Output(s)
#' All compiled extracted data in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_ALL_compiled_all_polygon.csv `
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_ALL_compiled_point_and_polygon.csv`
#' and identical archive files at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_ALL_compiled_all_polygon_[datestamp].csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_ALL_compiled_point_and_polygon_[datestamp].csv`
#' and changelogs tracking changes at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/changelogs/changelog_compiled_all_polygon_[datestamp].csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/changelogs/changelog_compiled_point_and_polygon_[datestamp].csv`
######################################################################################

##### Setup

rm(list = ls())
require(pacman)
p_load(data.table, dplyr, haven)

user <- Sys.getenv("USER")
if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", user, "/")
  i <- "/ihme/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "I:/"
}

source(paste0(h, "repos/indicators/1_retrospective/1_GDP_LDI/02_subnational_LSAE/helper_functions.R"))
root_fold <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/"

modeling_location_version <- get_LSAE_location_versions()
modeling_version_id <- max(modeling_location_version$location_set_version_ids)
modeling_shapefile_version <- modeling_location_version[
  location_set_version_ids == modeling_version_id, shapefile_dates]

# Modify the note here to explain what changed between compiles
change_note <- "Ran script on new 00 output file"

save_extraction <- function(dt, compiled = F, point_to_polygon){
  #' @description Saves out extracted data to filesystem, creating both a datestamped and stable filepath
  #' @param dt [data.frame/data.table] The extracted data to save out, in extraction template format
  #' @param compiled [logical] Whether or not the data to save out is compiled data from all sources, or individual sources. Default is FALSE, to be used with individual data sources.
  #' @param point_to_polygon [logical] Whether or not the data has aggregated points to polygons, to create a separate dataset for point vs polygon modeling. Typically only applicable for survey sources such as DHS, MICS, and LSMS, as well as for the compiled dataset.
  #' @return None
  #' @output files /ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_<source>_<date>.csv and /ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_<source>.csv
  
  date <- gsub(" ", "", format(Sys.time(), "%m %d %Y"))
  if("source" %in% colnames(dt)) {
    # Save out 
    if(compiled) {
      source <- "ALL_compiled"
    } else {
      source <- unique(dt$source)
    }
    if(source == "" | is.na(source)) {
      message(paste0("Error: Column \'source\' must contain nonmissing values"))
    } else if(length(source) > 1) {
      message(paste0("Error: Multiple values present in column \'source\': ", paste(source, collapse = ", ")))
    } else {
      path <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/"
      
      if(source %in% c("DHS", "MICS", "LSMS", "COUNTRY_SPECIFIC", "ALL_compiled")) {
        if(point_to_polygon) {
          poly_string <- "all_polygon"
        } else {
          poly_string <- "point_and_polygon"
        }
        out_path <- paste0(path, "extracted_", source, "_", poly_string, ".csv")
      } else {
        poly_string <- ""
        out_path <- paste0(path, "extracted_", source, ".csv")
      }
      archive_path <- paste0(path, "extracted_", source, "_", poly_string, "_", date, ".csv")
      fwrite(dt, archive_path)
      fwrite(dt, out_path)
      cat("Files saved out to:\n", out_path, "\n", archive_path)
    }
  } else {
    message(paste0("Error: Column \'source\' missing in data"))
  }
}


for(point_to_polygon in c(TRUE, FALSE)) {
  
  if(point_to_polygon) {
    point_to_polygon_string <- "_all_polygon"
  } else {
    point_to_polygon_string <- "_point_and_polygon"
  }
  
  # Read in previous compiled file for versioning purposes
  previous_extraction <- paste0(root_fold, "extracted_ALL_compiled", point_to_polygon_string, ".csv")
  previous_extraction_timestamp <- gsub(" ", "", format(file.info(previous_extraction)$ctime, "%m %d %Y"))
  previous_data <- fread(previous_extraction)
  
  # Read in all extracted wealth files (latest best versions)
  extracted_files <- list.files(path = root_fold, pattern = "extracted_.*.csv",  recursive = F, ignore.case = T)
  extracted_files <- extracted_files[!(extracted_files %like% "ALL_compiled")]
  if(point_to_polygon) {
    extracted_files <- extracted_files[!(extracted_files %like% "_point_and_polygon")]
  } else {
    extracted_files <- extracted_files[!(extracted_files %like% "_all_polygon")]
  }
  extracted_files <- paste0(root_fold, extracted_files)

  # Bind together all extracted data
  extracted_files_dt <- lapply(extracted_files, fread, sep=",")
  dt <- rbindlist(extracted_files_dt, fill = TRUE)
  
  # Save out data
  save_extraction(dt, compiled = T, point_to_polygon = point_to_polygon)
  
  # Create changelog
  date <- gsub(" ", "", format(Sys.time(), "%m %d %Y"))
  fileConn <- file(paste0(root_fold, "changelog_compiled", point_to_polygon_string, "_", date, ".txt"))
  
  changelog <- (paste0("Changelog for extracted_ALL_compiled", point_to_polygon_string, "_", date, ".csv:\n"))
  changelog <- paste0(changelog, "Input source files included:\n")
  # changelog <- paste0(changelog, paste("   ", extracted_files, ", timestamp ", gsub(" ", "", format(file.info(extracted_files)$ctime, "%m %d %Y")), collapse = "\n"), "\n")
  changelog <- paste0(changelog, "Run by user: ", user, "\n")
  changelog <- paste0(changelog, "Number of NIDs included: ", length(unique(dt$nid)), "\n")
  changelog <- paste0(changelog, "Number of years included: ", length(unique(dt$year)), "\n")
  changelog <- paste0(changelog, "Number of observations: ", nrow(dt), "\n")
  changelog <- paste0(changelog, "Location set info: location_set_id = 125, location_set_version_id = ", 
                      modeling_version_id, ", shapefile_date = ", 
                      modeling_shapefile_version, "\n")
  changelog <- paste0(changelog, "\nPrevious version: extracted_ALL_compiled_", previous_extraction_timestamp, ".csv\n")
  changelog <- paste0(changelog, "Differences to previous extraction:\n")
  changelog <- paste0(changelog, "Change in number of sources included: ", length(unique(dt$source)) - length(unique(previous_data$source)), "\n")
  changelog <- paste0(changelog, "   Sources added: ", paste(setdiff(unique(dt$source), unique(previous_data$source)), collapse = ", "), "\n")
  changelog <- paste0(changelog, "   Sources removed: ", paste(setdiff(unique(previous_data$source), unique(dt$source)), collapse = ", "), "\n")
  changelog <- paste0(changelog, "Change in number of NIDs included: ", length(unique(dt$nid)) - length(unique(previous_data$nid)), "\n")
  changelog <- paste0(changelog, "   NIDs added: ", paste(setdiff(unique(dt$nid), unique(previous_data$nid)), collapse = ", "), "\n")
  changelog <- paste0(changelog, "   NIDs removed: ", paste(setdiff(unique(previous_data$nid), unique(dt$nid)), collapse = ", "), "\n")
  changelog <- paste0(changelog, "Change in number of years included: ", length(unique(dt$year)) - length(unique(previous_data$year)), "\n")
  changelog <- paste0(changelog, "   Years added: ", paste(setdiff(unique(dt$year), unique(previous_data$year)), collapse = ", "), "\n")
  changelog <- paste0(changelog, "   Years removed: ", paste(setdiff(unique(previous_data$year), unique(dt$year)), collapse = ", "), "\n")
  changelog <- paste0(changelog, "Change in number of observations: ", nrow(dt) - nrow(previous_data), "\n")
  changelog <- paste0(changelog, "\nOther notes on changes: ", change_note)
  
  writeLines(changelog, fileConn)
  close(fileConn)
  cat(changelog)
}
