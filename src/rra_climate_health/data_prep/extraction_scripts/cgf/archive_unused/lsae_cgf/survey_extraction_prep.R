###################################################################################################################################
# PREP SURVEY REPORT EXTRACTIONS FOR CGF AND DBM
#
# By: Alice Lazzar-Atwood
# Date: 1/23/2019
#
# The purpose of this script is to take the master extraction sheet, seperate indicators, drop columns that aren't needed for the 
# model, and save specifc indicator extractions in the appropriate folders.
###################################################################################################################################

library(data.table)

master_list <- fread("/home/j/temp/alicela/report_extraction/cgfdbm_report_extractions.csv")
master_list <- master_list[,type:= 1]

indicators <- c('overweight_who_b', 'wasting_mod_cond', 'wasting_mod_who', 'normal_mod_who', 'overweight_mod_who', 'stunting_mod_b', 'underweight_mod_b', 'wasting_mod_b')

for (ind in indicators){
  if (ind == "overweight_who_b"){
    overweight_who_b <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'overweight_who_b', 'WHZN', 'point', 'type')]
    overweight_who_b <- na.omit(overweight_who_b, col = "overweight_who_b")
    setnames(overweight_who_b, old = "WHZN", new = "N")
    fwrite(overweight_who_b, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
  if(ind == 'wasting_mod_who'){
    wasting_mod_who <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'wasting_mod_who', 'WHZN', 'point', 'type')]
    wasting_mod_who <- na.omit(wasting_mod_who, col = "wasting_mod_who")
    setnames(wasting_mod_who, old = "WHZN", new = "N")
    fwrite(wasting_mod_who, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
  if(ind == 'normal_mod_who'){
    normal_mod_who <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'normal_mod_who', 'WHZN', 'point', 'type')]
    normal_mod_who <- na.omit(normal_mod_who, col = "normal_mod_who")
    setnames(normal_mod_who, old = "WHZN", new = "N")
    fwrite(normal_mod_who, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
  if(ind == 'overweight_mod_who'){
    overweight_mod_who <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'overweight_mod_who', 'WHZN', 'point', 'type')]
    overweight_mod_who <- na.omit(overweight_mod_who, col = "overweight_mod_who")
    setnames(overweight_mod_who, old = "WHZN", new = "N")
    fwrite(overweight_mod_who, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
  if(ind == 'stunting_mod_b'){
    stunting_mod_b <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'stunting_mod_b', 'HAZN', 'point', 'type')]
    stunting_mod_b <- na.omit(stunting_mod_b, col = "stunting_mod_b")
    setnames(stunting_mod_b, old = "HAZN", new = "N")
    fwrite(stunting_mod_b, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
  if(ind == 'wasting_mod_b'){
    wasting_mod_b <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'wasting_mod_b', 'WHZN', 'point', 'type')]
    wasting_mod_b <- na.omit(wasting_mod_b, col = "wasting_mod_b")
    setnames(wasting_mod_b, old = "WHZN", new = "N")
    fwrite(wasting_mod_b, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
  if(ind == 'underweight_mod_b'){
    underweight_mod_b <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'underweight_mod_b', 'WAZN', 'point', 'type')]
    underweight_mod_b <- na.omit(underweight_mod_b, col = "underweight_mod_b")
    setnames(underweight_mod_b, old = "WAZN", new = "N")
    fwrite(underweight_mod_b, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
  if(ind == 'wasting_mod_cond'){
    wasting_mod_cond <- master_list[,c('country', 'location_code', 'shapefile', 'source', 'svy_id', 'pweight_sum', 'start_year', 'wasting_mod_b', "wasting_mod_cond", "overweight_who_b", 'WHZN', 'point', 'type')]
    wasting_mod_cond <- na.omit(wasting_mod_cond, col = c("wasting_mod_b", "overweight_who_b"))
    setnames(wasting_mod_cond, old = "WHZN", new = "N")
    wasting_mod_cond <- wasting_mod_cond[, N:=N-overweight_who_b][,wasting_mod_cond := wasting_mod_b][,c("wasting_mod_b", "overweight_who_b"):=NULL]
    fwrite(wasting_mod_cond, paste0('/ihme/limited_use/LIMITED_USE/LU_GEOSPATIAL/collapsed/cgf/', ind, '/extractions/collapsed_polys_extractions_mf.csv'))
  }
}