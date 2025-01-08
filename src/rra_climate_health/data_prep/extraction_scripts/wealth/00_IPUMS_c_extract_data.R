#' @Title: [00_process_ipums_data.R]  
#' @Authors: Bianca Zlavog, Paul Nam
#' @contact: zlavogb@uw.edu, sgy0003@uw.edu
#' @Date_code_last_updated: 08/2023
#' @Date_data_last_updated: 08/2023
#' @Purpose: Process IPUMS wealth data into extraction template format
#' Includes cleaning, geomatching, point-to-polygon matching, aggregation, and validation
#' @File_Input(s)
#' ipums survey files extracted by UbCov at 
#' `/ihme/resource_tracking/LSAE_income/0_preprocessing/IPUMS/extractions/`
#' @File_Output(s)
#' Compiled ipums surveys in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_IPUMS.csv`
#' and identical archive file at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/archive/extracted_IPUMS_[datestamp].csv`
#' @Documentation:
#' Paul's handoff files have been uploaded to:
#' https://uwnetid-my.sharepoint.com/personal/hashimig_uw_edu/_layouts/15/onedrive.aspx?csf=1&web=1&e=ofnO2c&cid=d4e0b862%2D4808%2D4470%2Dbf8a%2D73d8bcf3e54c&FolderCTID=0x012000B7102EDAEA4DC84DB95A7654F0F05F5D&isAscending=true&id=%2Fpersonal%2Fhashimig%5Fuw%5Fedu%2FDocuments%2FShared%20with%20Everyone%2FResource%20Tracking%20Team%2F3%5FEconomic%20Indicators%2FDocumentation%2FLSAE%20income%2FHandoffs%2FIPUMS&sortField=LinkFilename&view=0
#' TODO: Need to finish incomplete geomatching particularly for MEX, BRA, USA surveys, and investigate some buggy geomatching as printed in the geomatching section validations

##### Setup
rm(list = ls())
require(pacman)
p_load(data.table, dplyr, readxl, haven, testthat)

lbd_loader_dir <- paste0(
  "/share/geospatial/code/geospatial-libraries/lbd.loader-", R.version$major, ".",
  strsplit(R.version$minor, '.', fixed = TRUE)[[1]][[1]]
)
library(lbd.loader, lib.loc = lbd_loader_dir)
suppressWarnings(suppressMessages(
  library(lbd.mbg, lib.loc = lbd.loader::pkg_loc("lbd.mbg"))
))


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


source(paste0(h, "/indicators/1_retrospective/1_GDP_LDI/LSAE/helper_functions.R"))

modeling_shapefile_version <- get_LSAE_location_versions()
modeling_shapefile_version <- modeling_shapefile_version[
  location_set_version_ids == max(modeling_shapefile_version$location_set_version_ids), shapefile_dates]
global_admin2_shapefile_fp <- get_admin_shapefile(
  admin_level = 2,
  version = modeling_shapefile_version
)
global_admin2_sf <- sf::st_read(global_admin2_shapefile_fp)
global_admin1_shapefile_fp <- get_admin_shapefile(
  admin_level = 1,
  version = modeling_shapefile_version
)
global_admin1_sf <- sf::st_read(global_admin1_shapefile_fp)

#############################
## Load and read all files ##
#############################

ipums_path <- "/ihme/resource_tracking/LSAE_income/0_preprocessing/IPUMS/"
ipums_countries <- c("BRA", "CAN", "DOM", "IND", "ISR", "JAM", "MEX", 
                     "PAN", "PRI", "VEN", "MUS", "ZAF", "USA") 

# This section compiles all the winnower-extracted DTA files by country-year into one CSV file by country
# This can take a long time to run, but creating the CSV files allows us to subsequently read in the data more quickly
# You only need to rerun this section when adding a new survey for a country, or updating a winnower survey extraction for a country
# If updating the extractions for just one or a subset of IPUMS countries, you don't have to run the loop below for all countries, 
# but you can subset the `ipums_countries` list to just the countries you need to update
# Use the `compile_country_csvs` variable to toggle running this section off or on, by default it is turned off
compile_country_csvs <- FALSE
if(compile_country_csvs) {
  for(country in ipums_countries) {
    ipums_extracts <- list.files(path = paste0(ipums_path, '/extractions/'), pattern = paste0("_", country, "_"), recursive = T, ignore.case = T)
    ipums_extracts <- paste0(ipums_path, "/extractions/", ipums_extracts)
    ipums_country_extracts <- lapply(ipums_extracts,function(i){
      read_dta(i, encoding="latin1")
      if(country %in% c("MEX", "BRA", "VEN")) {
        read_dta(i, encoding="latin1")
      } else {
        read_dta(i)
      }
    })
    ipums_country_extracts <- rbindlist(ipums_country_extracts, fill=TRUE)
    fwrite(ipums_country_extracts, paste0(ipums_path, "/appended_extractions/IPUMS_", country, "_appended.csv"))
  }
}

# This section processes the appended files by country to remove special codes in the data 
# and aggregate from individual to household level, and saves out one CSV file by country.
# This can take a long time to run, but creating the CSV files allows us to subsequently read in the data more quickly.
# You only need to rerun this section when adding a new survey for a country, or updating a winnower survey extraction for a country
# If updating the extractions for just one or a subset of IPUMS countries, you don't have to run the loop below for all countries, 
# but you can subset the `ipums_countries` list to just the countries you need to update
# Use the `aggregate_country_csvs` variable to toggle running this section off or on, by default it is turned off
aggregate_country_csvs <- FALSE
if(aggregate_country_csvs) {
  for(country in ipums_countries) {
    ipums_extracts_csv <- fread(paste0(ipums_path, "/appended_extractions/IPUMS_", country, "_appended.csv"))
    
    #Aggregating individual-level values to household-level values. Sum up income, 
    #consumption, or expenditure across all individuals within a household to get 
    #the total amount for the household, and then divide by the number of 
    #individuals in the household to get to total amount per person in the household.
    
    print(paste0("Calculating average household values per capita for country ", country))
    
    # Handle special NA income codes in particular surveys as listed below:
    # income: [9999999 (CAN, BRA, DOM, JAM, MUS, PAN, PRI, ZAF, USA), 9999998 (BRA, DOM, JAM, MUS, PAN, ZAF), 
    # 9999997/9999996/9999991/9999988/9999987 (BRA), 999999/999998/999997 (ISR), 99999999/99999998 (MEX, PAN, VEN)]
    # consumption/expenditure: [99999999 (IND)]
    if(country == "USA") {
      ipums_extracts_csv[income_total_lcu == "BBBBBBBBB", income_total_lcu := NA]
      ipums_extracts_csv[, income_total_lcu := as.numeric(income_total_lcu)]
      ipums_extracts_csv[household_size == "BB", household_size := 1]
      ipums_extracts_csv[, household_size := as.numeric(household_size)]
    }
    ipums_extracts_csv[(country %in% c("CAN", "BRA", "DOM", "JAM", "MUS", "PAN", "PRI", "ZAF", "USA") & income_total_lcu == 9999999) | 
                              (country %in% c("BRA", "DOM", "JAM", "MUS", "PAN", "ZAF") & income_total_lcu == 9999998) | 
                              (country %in% c("BRA") & income_total_lcu %in% c(9999997, 9999996, 9999991, 9999988, 9999987)) | 
                              (country %in% c("ISR") & income_total_lcu %in% c(999999, 999998, 999997)) | 
                              (country %in% c("BRA") & income_total_lcu %in% c(9999000, 9990000)) | 
                              (country %in% c("MEX", "PAN", "VEN") & income_total_lcu %in% c(99999999, 99999998)),
                            income_total_lcu := NA]
    
    # Handle differently IND surveys where income is measured per person, and expenditure/consumption is the average per person value in the household
    ind_income_pc_surveys <- c(152601, 5292, 5293, 5294, 152602)
    
    ipums_extracts_csv[, income_total_lcu := as.numeric(income_total_lcu)]
    ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per capita") | (unit %in% c("per household") & nid %in% ind_income_pc_surveys),
                                                       inc_avg := sum(as.numeric(income_total_lcu), na.rm = T), by = .(ihme_loc_id, int_year, hh_id)]
    ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per capita") | (unit %in% c("per household") & nid %in% ind_income_pc_surveys), 
                                                       inc_avg:= inc_avg / household_size]
    ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per household") & !(nid %in% ind_income_pc_surveys), inc_avg := income_total_lcu]
    
    if("tot_exp" %in% colnames(ipums_extracts_csv)) {
      ipums_extracts_csv[(unique(ipums_extracts_csv$ihme_loc_id) %in% c("IND") & tot_exp == 99999999),
                              tot_exp := NA]
      ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per capita"),
                                                         exp_avg := sum(as.numeric(tot_exp), na.rm = T), by = .(ihme_loc_id, int_year, hh_id)]
      ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per capita"), exp_avg:= exp_avg / household_size]
      ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per household"), exp_avg := tot_exp]
    }
    
    if("tot_conexp" %in% colnames(ipums_extracts_csv)) {
      ipums_extracts_csv[(unique(ipums_extracts_csv$ihme_loc_id) %in% c("IND") & tot_conexp == 99999999),
                              tot_conexp := NA]
      ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per capita"),
                                                         conexp_avg := sum(as.numeric(tot_conexp), na.rm = T), by = .(ihme_loc_id, int_year, hh_id)]
      ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per capita"), conexp_avg:= conexp_avg / household_size]
      ipums_extracts_csv <- ipums_extracts_csv[unit %in% c("per household"), conexp_avg := tot_conexp]
    }
    ipums_extracts_csv <- ipums_extracts_csv[line_id == 1]
    
    fwrite(ipums_extracts_csv, paste0(ipums_path, "/aggregated_extractions/IPUMS_", country, "_aggregated.csv"))
  }
}

# Read in all aggregated surveys that have been saved out as CSV
ipums_csvs <- list.files(path = paste0(ipums_path, 'aggregated_extractions/'), pattern = "*.csv", recursive = T, ignore.case = T)
ipums_hh_csvs <- paste0(ipums_path, "aggregated_extractions/", ipums_csvs)
ipums_extracts_csv <- lapply(ipums_hh_csvs,function(i){
  fread(i)[, geospatial_id := as.character(geospatial_id)]
})
ipums_extracts_csv_all <- rbindlist(ipums_extracts_csv, fill=TRUE)

fwrite(ipums_extracts_csv_all, paste0(ipums_path, '/IPUMS_all_aggregated.csv'))

#####################
###NID Checks########
#####################

ipums_extracts_csv <- fread(paste0(ipums_path, '/IPUMS_all_aggregated.csv'))

#Check on NID count share folder vs. codebook
# used comma seperating tool for econ_indicators codebook with total variables here: https://delim.co/#
x <- c(105322, 38225, 30309, 38230, 2101, 43359, 43362, 294250, 151289, 151296, 
       152601, 5292, 5293, 5294, 152602, 39416, 39420, 39438, 39444, 39450, 
       43761, 294574, 43766, 43771, 56480, 40897, 40902, 40907, 106529, 41456,
       41460, 41463, 41467, 227224, 43146, 43152, 43158, 227194, 412194, 412202,
       412265, 412313, 412322, 412389, 412414, 412423, 412455, 412461, 106543,
       127946, 127951, 412499, 412531, 367726, 412538, 412541, 438795, 520621,
       520775, 520820, 482857, 482856, 482855, 482854, 482853, 482852, 482850,
       482849, 482848, 482847, 482846, 504635, 43299, 43274, 43298, 
       43402, 43407, 43412, 520879)


#list of nids from share drive
y <- unique(ipums_extracts_csv$nid)

#missing NIDS
nid_diff <- setdiff(x,y)

if(length(nid_diff) > 0){
  print(paste("Check why survey data is missing: ", paste(nid_diff, collapse = ", ")))
}

#We only want these specific columns for total consumption and total income if available. Add sum columns at later date.
ipums_extracts_csv <- ipums_extracts_csv[, c("currency", "base_year", "currency_detail", "unit", 
                                             "nid", "ihme_loc_id", "int_year", "file_path", "hhweight", 
                                             "household_size", "inc_avg", "exp_avg", "conexp_avg", "geospatial_id")]

###############################################
## Read in, check NIDS, prepare geocodebooks ##
###############################################

# pull all available MBG and SAE geocodebooks
source("/mnt/team/sae/pub/geocodebook_db/geocodebook_functions.R")
geo_sae <- get_geocodebooks(nids = unique(ipums_extracts_csv$nid), data_type = "sae")
geo_mbg <- get_geocodebooks(nids = unique(ipums_extracts_csv$nid), data_type = "mbg")

# Identify surveys with both MBG and SAE geocodebooks, and keep the preferred geomatching info
geocodebooks_overlap <- intersect(unique(geo_sae$nid), unique(geo_mbg$nid))

# Prefer MBG codebooks if they have only point data, 
# OR if they have lbd_standard_admin_1/2 geomatching AND the SAE codebook does not have lbd_standard_admin_1/2 geomatching (may have stable shapefile matching)
# Otherwise, prefer SAE codebooks for all other surveys
preferred_mbg <- c()
preferred_sae <- c()
geo_mbg_subset <- geo_mbg[!(is.na(point) & is.na(lat) & is.na(long) & is.na(shapefile))]
geo_sae_subset <- geo_sae[!(is.na(point) & is.na(lat) & is.na(long) & is.na(shapefile))]
for(nid_check in geocodebooks_overlap) {
  if(all(unique(geo_mbg_subset[nid == nid_check]$point) == 1)) {
    preferred_mbg <- c(preferred_mbg, nid_check)
  } else {
    if(all(unique(geo_mbg_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_") & 
       all(!unique(geo_sae_subset[nid == nid_check]$shapefile) %like% "lbd_standard_admin_")) {
      preferred_mbg <- c(preferred_mbg, nid_check)
    } else {
      preferred_sae <- c(preferred_sae, nid_check)
    }
  }
}

geo_mbg <- geo_mbg[!nid %in% preferred_sae]
geo_sae <- geo_sae[!nid %in% preferred_mbg]
geocodebooks_overlap <- intersect(unique(geo_sae$nid), unique(geo_mbg$nid)) # should now be an empty list
sae_mbg_bind <- rbind(geo_sae, geo_mbg)
sae_mbg_bind <- unique(sae_mbg_bind) # Removing duplicate rows from buggy geocodebooks

# Identify any surveys missing geocodebooks
surveys_missing_geocodebooks <- c(setdiff(y, unique(sae_mbg_bind$nid))) # 482846 482847 482848 482849 482850 482852 482853 482854 482855 482856 482857 504635 520621 520775 520820 127946 127951 367726 412194 412202 412265 412313 412322 412389 412414 412423 412455 412461 412499 412531 412538 412541 438795

#Removing some characters for 520879 to match what's in the sae_mbg_bind later down the line
ipums_extracts_csv$geospatial_id <- ifelse(ipums_extracts_csv$nid == 520879,
                                           substring(ipums_extracts_csv$geospatial_id, 1, nchar(ipums_extracts_csv$geospatial_id) - 3),
                                           ipums_extracts_csv$geospatial_id)
#Removing extra rows in NID 43152 and 43362 for the purpose of matching sae_mbg_bind
ipums_extracts_csv <- ipums_extracts_csv[!(ipums_extracts_csv$geospatial_id == 710999999 & ipums_extracts_csv$nid == 43362)]
ipums_extracts_csv <- ipums_extracts_csv[!(ipums_extracts_csv$geospatial_id %in% c("10", "11", "12", "13", 
                                                                                   "24", "35", "46", "47",
                                                                                   "48", "59", "60")
                                           & ipums_extracts_csv$nid == 43362)]
ipums_extracts_csv <- ipums_extracts_csv[!(ipums_extracts_csv$geospatial_id %in%
                                             c("356002099", "356008099", "356015099",
                                               "356028099", "356034099") & ipums_extracts_csv$nid == 5292)]

# Merge extracted dataset and geography dataset together
sae_mbg_bind[, location_code := as.character(location_code)]
ipums_part1 <- merge(sae_mbg_bind, ipums_extracts_csv[nid %in% unique(geo_sae$nid)], 
                     by.x = c("nid","iso3","geospatial_id"), by.y = c("nid","ihme_loc_id","geospatial_id"), 
                     all.y = T, allow.cartesian=TRUE)
match_mbg <- unique(geo_mbg$nid) # 5293  30309  38225  38230  39416  39420  43761  43766  43771  56480 105322
match_mbg_location_code <- c(5293, 38225, 38230, 43761, 43766, 43771, 56480, 105322)
match_mbg_geospatial_id <- c(39420, 39416, 30309)
ipums_part2 <- merge(sae_mbg_bind, ipums_extracts_csv[nid %in% match_mbg_location_code], 
                     by.x = c("nid","iso3","location_code"), by.y = c("nid","ihme_loc_id", "geospatial_id"), 
                     all.y = T)
ipums_part3 <- merge(sae_mbg_bind, ipums_extracts_csv[nid %in% match_mbg_geospatial_id], 
                     by.x = c("nid","iso3","geospatial_id"), by.y = c("nid","ihme_loc_id", "geospatial_id"), 
                     all.y = T)
ipums_all <- rbind(ipums_part1,ipums_part2)
ipums_all <- rbind(ipums_all,ipums_part3)

##################################
## Check survey information ##
#################################

##Finds nas in  all location columns
geo_missing <- ipums_all %>% filter_at(vars('lat','long','location_code','shapefile'), all_vars(is.na(.)))
geo_missing <- data.table(geo_missing)

#INSERT CHECKED NIDS:
#These NIDS have been checked by an analyst, and they are surveys known to have missingness in their geocodebooks, or it is not possible to create geocodebooks for these
# See documentation at https://uwnetid-my.sharepoint.com/:x:/r/personal/hashimig_uw_edu/_layouts/15/Doc.aspx?sourcedoc=%7BD9D8B39B-DE1F-44B8-B753-1068A47764FF%7D&file=ipums_geomatch_hhweight_notes.xlsx&action=default&mobileredirect=true
geo_checked <- c(151289, 151296, 43146, 152601, 43152, 30309, 39416, 39420)
# for(n in geo_checked) {
#   print(paste0("NID: ", n, ", number of missing geospatial_ids: ", length(unique(geo_missing[nid==n]$geospatial_id))))
# }
geo_missing <- geo_missing[!geo_missing$nid %in% geo_checked,]
geo_missing <- geo_missing[!geo_missing$nid %in% surveys_missing_geocodebooks,]

if(nrow(geo_missing) > 0){
  print(paste("Check why there is missing location information from survey data: ", paste(unique(geo_missing$nid), collapse = ", ")))
}
# TODO: this section prints NIDs: 43274, 43298, 43299, 106543 (USA surveys). need to revisit the geospatial_id variables used to extract the surveys.


#Check also for missing hhweight 
test_hhweight <- ipums_all[(is.na(hhweight))]


#Checked by an analyst. See documentation.
hhw_checked <- c()
test_hhweight <- test_hhweight[!test_hhweight$nid %in% hhw_checked,]

if(length(unique(test_hhweight$nid)) > 0){
  print(paste("Check why there are missing hhweight from survey data: ", paste(unique(test_hhweight$nid), collapse = ", ")))
}

ipums_all <- ipums_all[!(nid %in% hhw_checked & is.na(hhweight))]


#Make sure housesize is available for all surveys
missing_hs <- ipums_all[is.na(ipums_all$hhsize) & unit == "per household" & 
                         !(is.na(ipums_all$conexp_avg) & is.na(ipums_all$exp_avg) & is.na(ipums_all$inc_avg))]

if(length(unique(missing_hs$nid)) > 0){
  print(paste("Check why there are missing housesizes from survey data: ", paste(unique(missing_hs$nid), collapse = ", ")))
}


## drop all rows with per household data missing hhsize
ipums_all <- ipums_all[!(is.na(ipums_all$household_size) & unit == 'per household')]


#########################
## Column Adjustments ##
########################

#Adjust for recall period to get to annual data 
consump_list <- c("conexp_avg", "exp_avg", "inc_avg")

#If classified under "per household", divide by householdsize to get to per capita
ipums_all <- ipums_all[!(ipums_all$household_size == 0),]
for (vars in consump_list){
 ipums_all[unit == "per household" & !(nid %in% ind_income_pc_surveys), paste0(vars) := get(vars)/household_size ,]
}


# Renaming exp. classifications
setnames(ipums_all, "conexp_avg", "consumption_expenditure")
setnames(ipums_all, "exp_avg", "expenditure")
setnames(ipums_all, "inc_avg", "income")


#Confirm  unit variable values are correct
variables <- c("unit")
expected_unit <- c("per household", "per capita")
                   
for(var in variables) {
  if(var %in% colnames(ipums_all)) {
    observed_var <- unique(ipums_all[, c(get(var))])
    for(i in observed_var) {
      if(!(i %in% get(paste0("expected_", var)))) {
        message("Error: Value \'", i, "\' in column \'", var, "\' outside of expected range: \'", paste(get(paste0("expected_", var)), collapse = ', '), "\'")
      }
    }
  }
}


#########################
## Duplicate Removal ##
#########################
#Find duplicated rows from surveys where multiple people in HH were interviewed and income and expenditure stayed the same for multiple rows.
#Checked by an analyst. See documentation.
duplicate_check <- c()
find_dup <- ipums_all[duplicated(ipums_all) | duplicated(ipums_all, fromLast=TRUE), ]
find_dup <- find_dup[!nid %in% duplicate_check]

if(length(unique(find_dup$nid)) > 0){
  print(paste0("These NIDS have duplicate data, check why: ", paste(unique(find_dup$nid), collapse = ", ")))
}

#duplicate removal      
ipums_all <- ipums_all[!duplicated(ipums_all),]

####################################
## Reshape data from wide to long ##
####################################

setnames(ipums_all, "consumption_expenditure", "consumption expenditure")

ipums_all <- ipums_all %>%
  pivot_longer(
    cols = c("income":"consumption expenditure"),
    names_to = "measure",
    values_to = "value",
    values_drop_na = TRUE   # removing NAs here 
  )

ipums_all <- ipums_all %>% filter(measure %in% c("consumption expenditure", "expenditure", "income"))
ipums_all <- data.table(ipums_all)

######################################
##Rename and Format Columns##
#####################################

# Replace year (the year the survey started in) with int_year (the median interview year for each location)
setnames(ipums_all, "int_year", "year")
setnames(ipums_all, "survey_series", "source")
setnames(ipums_all, "hhweight", "weight")
setnames(ipums_all, "admin_level", "location_type")
#matching taulated data
ipums_all[location_type=="1", location_type := "admin1"]
ipums_all[location_type=="2", location_type := "admin2"]
ipums_all[location_type=="3", location_type := "admin3"]
ipums_all[location_type %in% c(">1", "<1", "<3", ">2"), location_type := NA]

#Add column data based on extraction template
ipums_all$source <- "IPUMS"
ipums_all$data_type <- "survey"
ipums_all$location_id <- NA
ipums_all$source_location_id <- NA
ipums_all$source_location_type <- NA 
#measure column classifications
ipums_all$multiplier <- 1
ipums_all$denominator <- "per capita"
ipums_all$value_type <- "aggregated"
ipums_all$currency_detail <- NA #This comes from econ_indicators codebook
ipums_all$currency <- "LCU"
ipums_all$notes <- NA
ipums_all$geomatching_notes <- NA
ipums_all$initials <- "PN"


#Check for negative values and NAs in value columns and why:
ipums_debt <- ipums_all[(ipums_all$value < 0),]
ipums_debt_checked <- c(2101, 41456, 41460, 41463, 
                        41467, 43359, 43362, 227224, 294250)
ipums_debt <- ipums_debt[!ipums_debt$nid %in% ipums_debt_checked,]

if(length(ipums_debt$value) > 0) {
  print(paste("Check these nids for negative values:", paste(unique(ipums_debt$nid), collapse = ", ")))
}

# Pull location_ids for data that has been geomatched to LSAE standard location set, based on location_codes
ipums_all <- data.table(ipums_all)
ad1_locs_info <- data.table(global_admin1_sf)[, .(location_code = ADM1_CODE, location_id_fill_ad1 = loc_id)]
ad2_locs_info <- data.table(global_admin2_sf)[, .(location_code = ADM2_CODE, location_id_fill_ad2 = loc_id)]
 
ipums_all_ad1_fill <- ipums_all[grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)]
ad1_locs_info$location_code <- sapply(ad1_locs_info$location_code, as.character)
ipums_all_ad1_fill <- merge(ipums_all_ad1_fill, ad1_locs_info, by = c("location_code"), all.x = TRUE)
nrow(ipums_all_ad1_fill[is.na(location_id_fill_ad1)]) # should be 0
ipums_all_ad1_fill[, location_id := location_id_fill_ad1][, location_id_fill_ad1 := NULL]
ipums_all_ad2_fill <- ipums_all[grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code)]
ipums_all_ad2_fill$location_code <- sapply(ipums_all_ad2_fill$location_code, as.numeric)
ipums_all_ad2_fill <- merge(ipums_all_ad2_fill, ad2_locs_info, by = c("location_code"), all.x = TRUE)
nrow(ipums_all_ad2_fill[is.na(location_id_fill_ad2)]) # should be 0
ipums_all_ad2_fill[, location_id := location_id_fill_ad2][, location_id_fill_ad2 := NULL]
 
ipums_fill <- rbind(ipums_all_ad1_fill, ipums_all_ad2_fill)
ipums_all_before <- copy(ipums_all)
ipums_all <- ipums_all[!(grepl("lbd_standard_admin_1", shapefile) & is.na(location_id) & !is.na(location_code)) & 
                    !(grepl("lbd_standard_admin_2", shapefile) & is.na(location_id) & !is.na(location_code))]
ipums_all <- rbind(ipums_all, ipums_fill)
validate_merge_nrows(ipums_all_before, ipums_all)

# Fill location_ids based on standard shapefiles from stable shapefiles
# TODO - revisit this section and see if there is a central function (lsae::translate_stable_poly_ids) to translate stable shapefile location_ids to standard shapefile location_ids
stable_shapefile_nids <- unique(ipums_all[shapefile_type=="stable"]$nid) # 294574, does not map perfectly to LSAE admin2 standard locations

###########################################
## Collapse data ##
###########################################

# split the data into point and polygon datasets
#dpoint <- subset(ipums_all, point == 1) # not needed since there are no point-geomatched IPUMS data
dpoly  <- subset(ipums_all, point == 0) # TODO: this is dropping some data, need to see why shapefile and point info is missing for those surveys

#cast weight to numeric values
dpoly[, weight := as.numeric(weight)]

# take the weighted mean of the indicator
dpoly <- dpoly[, .(value = weighted.mean(value, w=weight), sd_weighted = sqrt(wtd.var(value, weights = weight, na.rm = TRUE)), sd_unweighted = sd(value, na.rm = TRUE), N_households = .N),
               by = .(iso3, nid, source, year, shapefile, location_code, data_type, location_id, file_path, location_name, location_type,source_location_type, source_location_id, measure, denominator, multiplier, value_type, currency, base_year, currency_detail, geomatching_notes, notes, initials)]


# Combine dpoint and dpoly back together
ipums_all <- rbindlist(list(dpoly), fill=T)


# Process SD
# Drop some observations where only one household was surveyed, since this is not representative of the region, 
# and because the very small SD throws off the SAE models
ipums_all <- ipums_all[!(N_households == 1 & sd_weighted < 1)]
ipums_all[is.na(sd_weighted), sd_weighted := sd_unweighted] # Correct a few clusters where SD goes missing
nrow(ipums_all[is.na(sd_unweighted)]) + nrow(ipums_all[is.na(sd_weighted)]) # should be 0

#Select columns for final dataset
ipums_all[, c('lat','long') := NA]
ipums_all <- ipums_all[, c("nid", "source", "data_type", "file_path", "year", "iso3", "location_id", "location_type", "location_name", 
                         "source_location_type",  "source_location_id", "lat",  "long", "location_code",  "shapefile", "measure", "denominator",
                         "multiplier", "value", "sd_weighted", "sd_unweighted", "value_type", "currency", "base_year", "N_households",
                         "currency_detail", "notes", "geomatching_notes", "initials")]

# Hotfixes for a few surveys with 0 values
# This location had only 3 households surveyed, all with 0 household income, so we drop it as an outlier
ipums_all <- ipums_all[!(nid == 43771 & location_name == "Nicolas Ruiz")]
# These surveys had a handful of locations where all income values were one of the special NA codes within the raw survey file, 
# which causes 0 values for those locations, so we drop them here
ipums_all <- ipums_all[!(nid == 294574 & location_code %in% c(20057, 20140, 20407, 20427, 21138, 26070, 8010, 8012, 8024, 8063, 8065))]
ipums_all <- ipums_all[!(nid %in% c(43761) & location_code %in% c(20074, 20120, 20191, 20196, 20218, 20321, 20352,
                                                                  20383, 20471, 20480, 20501, 20524, 20536))]
ipums_all <- ipums_all[!(nid %in% c(5292) & measure == "income" & location_code %in% c(13008, 22033, 25009, 4017, 4038))]
ipums_all <- ipums_all[!(nid %in% c(5294) & measure == "income" & location_code %in% c(15002, 15004))]

###########################################
## Validate and Save ##
###########################################
validate_extractions(ipums_all)

#list of nids from share drive
y <- unique(ipums_all$nid)

#Check NID count, x is count of codebook nids from top of script
nid_diff <- setdiff(x,y)
nid_diff <- nid_diff[!nid_diff %in% surveys_missing_geocodebooks]

if(length(nid_diff) > 0){
  print(paste("Check that these NIDS have been dropped correctly: ", paste(unique(nid_diff), collapse = ", ")))
}
# TODO: this is printing the following NIDs, need to check why: 30309, 151289, 151296, 152601, 5293, 39416, 39420, 43146 

save_extraction(ipums_all)
