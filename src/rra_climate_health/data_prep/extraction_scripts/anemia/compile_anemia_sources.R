#' @Title: [rra_compile_anemia_sources.R]
#' @Purpose: Compile anemia sources


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



library(readxl)
library(dplyr)
library(haven)
library(data.table)
library(arrow)

#Wealth Data
dhs_path <- "/mnt/team/anemia/pub/anemia_envelope/rapid_response/"
dhs_files <- list.files(path = dhs_path, pattern = "*.dta", recursive = T, ignore.case = T)
dhs_hh_files <- paste0(dhs_path, dhs_files)

#read in all extracted surveys and combine###
dhs_extracts <- lapply(dhs_hh_files,function(i){
  read_dta(i)
})

dhs_extracts <- rbindlist(dhs_extracts, fill=TRUE)

write_parquet(dhs_extracts,"/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/extractions/anemia/anemia_extracts_compiled_09_02_2025.parquet")