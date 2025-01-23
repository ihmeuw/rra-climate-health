#' @Title: [00_a_DHS_preprocess.R]  
#' @Authors: Bianca Zlavog, Audrey Serfes
#' @contact: zlavogb@uw.edu, aserfes@uw.edu
#' @Purpose: Preprocess DHS surveys for UbCov extraction
#' A DSS note about outputs: If it's an 'L:/IDENT" file it can be saved with J:/ drive files. If it's "limited_use" use the seperate L:/ folder. 

rm(list = ls())
library(data.table)
library(haven)
library(dplyr)
library(readstata13)

if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", Sys.getenv("USER"), "/")
  k <- "/ihme/cc_resources/"
  l <- "/ihme/limited_use/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
  l <- "L:/"
}

out_dir <- "/ihme/resource_tracking/LSAE_income/0_preprocessing/DHS/"


#UZB DHS 1996 21033
uzb_96_hh <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/UZB/1996/UZB_DHS3_1996_HH_UZHR31FL_Y2019M04D01.DTA")))[, .(hhid, hv024, hv025, hv001, hv005, hv002, hv003)]
uzb_96_w <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/UZB/1996/UZB_DHS3_1996_W_UZWI31FL_Y2019M04D01.DTA")))
uzb_96_hh <- merge(uzb_96_hh,uzb_96_w, by.x = c("hhid"), by.y = c("whhid"))
fwrite(uzb_96_hh, paste0(out_dir, "UZB_DHS_1996_preprocess.csv"))


#UGA DHS 1995 21033
uga_95_hh <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/UGA/1995/UGA_DHS3_1995_HH_UGHR33FL_Y2019M04D02.DTA")))[, .(hhid, hv024, hv025, hv001, hv005, hv002, hv003)]
uga_95_w <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/UGA/1995/UGA_DHS3_1995_W_UGWI33FL_Y2019M04D02.DTA")))
uga_95_hh <- merge(uga_95_hh, uga_95_w, by.x = c("hhid"), by.y = c("whhid"))
fwrite(uga_95_hh, paste0(out_dir, "UGA_DHS_1995_preprocess.csv"))


#COL DHS 1990 19341
col_90_hh <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/COL/1990/COL_DHS2_1990_HH_COHR22FL_Y2018M05D15.DTA")))[, .(hhid, hv024, hv025, hv001, hv005, hv002, hv003)]
col_90_w <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/COL/1990/COL_DHS2_1990_W_COWI22FL_Y2018M05D15.DTA")))
col_90_hh <- merge(col_90_hh, col_90_w, by.x = c("hhid"), by.y = c("whhid"))
fwrite(col_90_hh, paste0(out_dir, "COL_DHS_1990_preprocess.csv"))


#PER DHS 2013 210231
per_2013_hh <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/PER/2013/PER_DHS6_2013_HH_Y2014M05D21.DTA")))
per_2013_hv <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/PER/2013/PER_DHS6_2013_RECH23_Y2015M04D16.DTA")))
per_2013_hh <- merge(per_2013_hh, per_2013_hv, by = "hhid")
fwrite(per_2013_hh, paste0(out_dir, "PER_DHS_2013_preprocess.csv"))


#PER DHS 2014 210182
per_2014_hh <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/PER/2014/PER_DHS7_2014_RECH23_Y2015M06D29.DTA")))
per_2014_hv <- data.table(read_dta(paste0(j,"/DATA/DHS_PROG_DHS/PER/2014/PER_DHS7_2014_RECH0_Y2015M06D29.DTA")))
per_2014_hh <- merge(per_2014_hh, per_2014_hv, by = "hhid")
fwrite(per_2014_hh, paste0(out_dir, "PER_DHS_2014_preprocess.csv"))
