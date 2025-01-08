#' @Title: [00_IPUMS_a_preprocess.R]  
#' @Authors: Paul Nam
#' @contact: sgy0003@uw.edu
#' @Purpose: Preprocess IPUMS surveys for UbCov extraction
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

out_dir <- "/share/resource_tracking/LSAE_income/0_preprocessing/IPUMS/"

#IND 1983 - needed to merge on geomatching info
IND_hh <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1983/IND_SOCIOECONOMIC_SURVEY_1983_HH_WEALTH.DTA")))
IND_av <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1983/IND_SOCIOECONOMIC_SURVEY_1983_ALL_VARIABLES_Y2018M08D02.DTA")))
IND_geo <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1983/IND_SOCIOECONOMIC_SURVEY_1983_EXT_GEO_Y2016M12D09.DTA")))
IND_geo <- IND_geo[, c('geo1_inx', 'geo2_inx', 'pernum', 'serial')]
IND_merged <- merge(IND_hh,IND_av, by = c("serial", "pernum", "urban", "year"))
IND_merged <- subset(IND_merged, select = -c(relate.y, related.y, age.y,          
                                             sex.y, empstat.y, empstatd.y, incwage.y,
                                             sample.y))
IND_merged <- setnames(IND_merged, old = c("sample.x", "relate.x", "related.x",
                                           "age.x", "sex.x", "empstat.x", "empstatd.x", 
                                           "incwage.x"),
                       new = c("sample", "relate", "related",
                               "age", "sex", "empstat", "empstatd", 
                               "incwage"))
IND_merged <- merge(IND_merged, IND_geo, by = c('serial', 'pernum'), all.x = T)
write_dta(IND_merged, paste0(out_dir, "IND_SOCIOECONOMIC_SURVEY_1983_Merged.dta"))

#IND 1987-1988
IND_hh <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1987_1988/IND_SOCIOECONOMIC_SURVEY_1987_1988_HH_WEALTH.DTA")))
IND_av <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1987_1988/IND_SOCIOECONOMIC_SURVEY_1987_1988_ALL_VARIABLES_Y2018M08D02.DTA")))
IND_merged <- merge(IND_hh,IND_av, by = c("year", "serial", "pernum", "urban"))
IND_merged <- subset(IND_merged, select = -c(relate.y, related.y, age.y,          
                                             sex.y, empstat.y, empstatd.y, incwage.y,
                                             sample.y, educin.y))
IND_merged <- setnames(IND_merged, old = c("sample.x", "relate.x", "related.x",
                                           "age.x", "sex.x", "empstat.x", "empstatd.x", 
                                           "incwage.x", "educin.x"),
                       new = c("sample", "relate", "related",
                               "age", "sex", "empstat", "empstatd", 
                               "incwage", "educin"))
write_dta(IND_merged, paste0(out_dir, "IND_SOCIOECONOMIC_SURVEY_1987_1988_Merged.dta"))

#IND 1993-1994
IND_hh <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1993_1994/IND_SOCIOECONOMIC_SURVEY_1993_1994_HH_WEALTH.DTA")))
IND_av <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1993_1994/IND_SOCIOECONOMIC_SURVEY_1993_1994_ALL_VARIABLES_Y2018M08D02.DTA")))
IND_merged <- merge(IND_hh,IND_av, by = c("year", "serial", "pernum", "urban"))
IND_merged <- subset(IND_merged, select = -c(relate.y, related.y, age.y,          
                                             sex.y, empstat.y, empstatd.y, incwage.y,
                                             sample.y, educin.y))
IND_merged <- setnames(IND_merged, old = c("sample.x", "relate.x", "related.x",
                                           "age.x", "sex.x", "empstat.x", "empstatd.x", 
                                           "incwage.x", "educin.x"),
                       new = c("sample", "relate", "related",
                               "age", "sex", "empstat", "empstatd", 
                               "incwage", "educin"))
write_dta(IND_merged, paste0(out_dir, "IND_SOCIOECONOMIC_SURVEY_1993_1994_Merged.dta"))

#IND 1999-2000
IND_hh <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1999_2000/IND_SOCIOECONOMIC_SURVEY_1999_2000_HH_WEALTH.DTA")))
IND_av <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/1999_2000/IND_SOCIOECONOMIC_SURVEY_1999_2000_ALL_VARIABLES_Y2018M08D02.DTA")))
IND_merged <- merge(IND_hh,IND_av, by = c("year", "serial", "pernum", "urban"))
IND_merged <- subset(IND_merged, select = -c(relate.y, related.y, age.y,          
                                             sex.y, empstat.y, empstatd.y, incwage.y,
                                             sample.y, educin.y))
IND_merged <- setnames(IND_merged, old = c("sample.x", "relate.x", "related.x",
                                           "age.x", "sex.x", "empstat.x", "empstatd.x", 
                                           "incwage.x", "educin.x"),
                                   new = c("sample", "relate", "related",
                                           "age", "sex", "empstat", "empstatd", 
                                           "incwage", "educin"))
write_dta(IND_merged, paste0(out_dir, "IND_SOCIOECONOMIC_SURVEY_1999_2000_Merged.dta"))


#IND 2004-2005
IND_hh <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/2004_2005/IND_SOCIOECONOMIC_SURVEY_2004_2005_HH_WEALTH.DTA")))
IND_av <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/IND/2004_2005/IND_SOCIOECONOMIC_SURVEY_2004_2005_ALL_VARIABLES_Y2018M08D02.DTA")))
IND_merged <- merge(IND_hh,IND_av, by = c("year", "serial", "pernum", "urban"))
IND_merged <- subset(IND_merged, select = -c(relate.y, related.y, age.y,          
                                             sex.y, empstat.y, empstatd.y, incwage.y,
                                             sample.y, educin.y))
IND_merged <- setnames(IND_merged, old = c("sample.x", "relate.x", "related.x",
                                           "age.x", "sex.x", "empstat.x", "empstatd.x", 
                                           "incwage.x", "educin.x"),
                       new = c("sample", "relate", "related",
                               "age", "sex", "empstat", "empstatd", 
                               "incwage", "educin"))
write_dta(IND_merged, paste0(out_dir, "IND_SOCIOECONOMIC_SURVEY_2004_2005_Merged.dta"))


# DOM 1981
DOM_1981_HH <- data.table(read_dta(paste0(j,"/DATA/IPUMS_CENSUS/DOM/1981/DOM_CENSUS_1981_ALL_VARIABLES_Y2018M07D27.DTA")))
DOM_1981_GEO <- data.table(haven::read_stata(paste0(j,"DATA/IPUMS_CENSUS/DOM/1981/DOM_CENSUS_1981_EXT_GEO_Y2016M12D09.DTA"), encoding = "latin1"))
DOM_1981_GEO <- DOM_1981_GEO[, c('geo1_dox', 'geo2_dox', 'pernum', 'serial')]
DOM_1981_merged <- merge(DOM_1981_HH, DOM_1981_GEO, by = c('pernum', 'serial'), all.x = T)
write_dta(DOM_1981_merged, paste0(out_dir, "DOM_CENSUS_1981_ALL_VARIABLES_AND_GEO_MERGED.DTA"))


