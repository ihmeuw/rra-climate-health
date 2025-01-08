#####################################################################
# POST UBCOV EXTRACTION DATA CLEANING FOR GEOSPATIAL DATA EXTRACTIONS & GEOGRAPHY MATCHING
# PIONEERED BY ANNIE BROWNE
# UPDATED & OVERHAULED BY MANNY GARCIA
# STANDARDIZED BY SCOTT SWARTZ
# REWORKED FOR CGF BY BRANDON PICKERING
# EDITED BY ALICE LAZZAR-ATWOOD
# EMAIL ABROWNE@WELL.OX.AC.UK
# EMAIL GMANNY@UW.EDU
# EMAIL SSWARTZ@UW.EDU
# EMAIL BVP@UW.EDU
# EMAIL ALICELA@UW.EDU

# INSTRUCTIONS: 
# Steps:
#
# 1. Set-up
#
# 2. Define functions
#
# 3. Bind ubcov extracts together
#
# 4. Pull in geo codebooks
#
# 5. Prep data for merge
#
# 6. Merge ubcov extracts and geo codebooks
#
# 7. Clean up and save 
#
# 8. Generate diagnostics
#
# UBCOV OUTPUTS MUST BE SAVED IN LIMITED USE DIRECTORY
#####################################################################

#####################################################################
## 1.) SETUP 
#####################################################################

#clear the workspace
rm(list=ls())

#Define values
topic <- "cgf"
priority <- NULL #list '1', '2a', '2b', or '3' to output missing geogs. from a given priority level
nid_vec <- c() #list the NIDs of surveys (if any) to merge using admin_1
cluster <- TRUE #running on cluster true/false
cores <- 4

#FOR THE CLUSTER:
#to log in
#qlogin -P proj_geo_nodes -l geos_node=TRUE  -pe multi_slot 5 -now no
#to run this script:
#source('/snfs2/HOME/bvp/child_growth_failure_h/data_prep/undernutrition_post_extraction_processing.R')

#Setup
j <- ifelse(Sys.info()[1]=="Windows", "J:/", "/home/j/")
l <- ifelse(Sys.info()[1]=="Windows", "L:/", "/ihme/limited_use/")
#The cgf extractions are stored here:
folder_in <- paste0(l, "LIMITED_USE/LU_GEOSPATIAL/ubCov_extractions/", topic, "/all") 
#The geo-matched extract will be saved here:
folder_out <- paste0(l,"/LIMITED_USE/LU_GEOSPATIAL/geo_matched/", topic, "/test") 
#Ignore all warnings (change if desired)
options(warn=-1)

#Create a date variable and replace hyphens with asterisks
module_date <- Sys.Date()
module_date <- gsub("-", "_", module_date)


#.libPaths(package_lib)    
package_list <- c('survey', 'magrittr', 'foreign', 'rgeos', 'data.table',
                  'raster','rgdal','INLA','seegSDM','seegMBG','plyr','dplyr', 
                  'foreach', 'snow', 'parallel', 'doParallel', 'stringr','haven', 'gdata')
for(package in package_list) {
  library(package, character.only=TRUE)
}


#####################################################################
## 2.) DEFINE FUNCTIONS
#####################################################################

#Read in geo codebook, add column with corresponding survey series
read_add_name_col <- function(file){
  #FOR GEOGRAPHY CODEBOOKS. READS THEM IN AND ADDS A COLUMN WITH THEIR CORRESPONDING SURVEY_SERIES
  message(file)
  rn <- gsub(".csv", "", file, ignore.case=T)
  spl <- strsplit(rn, "/") %>% unlist()
  svy <- spl[length(spl)]
  df <- read.csv(file, encoding="utf8", stringsAsFactors = F) #this encoding scheme plays nice with the default excel format
  df <- as.data.table(df)
  #df <- fread(file)
  df[, survey_series := svy]
  df <- lapply(df, as.character, stringsAsFactors = FALSE)
  return(df)
}

#Convert all values of a data frame to characters
all_to_char_df <- function(df){
  df <- data.frame(lapply(df, as.character), stringsAsFactors = FALSE)
  return(df)
}

#This function forces all year_start values to int_year whenever int_year is populated. At the collapse phase, we take a weighted int/start year per geography as the data's modeling year
#This ensures that the data is representative of the geography.
adjust_year_start_to_int_year <- function(df){
  nids_missing_int_year <- df[!is.na(int_year), nid]
  #  nids_with_big_differences <- df[!is.na(int_year)&!is.na(year_start),  list(nid, ihme_loc_id, year_start, year_end, survey_name, change := year_start - int_year)]
  #  write.csv(nids_with_big_differences, paste0(folder_out, "/changed_years.csv"), row.names=F)
  #  updated <- df[!is.na(int_year) & int_year <= year_start+5 & int_year >= year_start, year_start := int_year]
  updated <- df[, year_svy := year_start]
  updated <- updated[!is.na(int_year) & int_year >= year_start, year_svy := int_year]
  return(updated)
}

#####################################################################
## 3.) BIND UBCOV EXTRACTS 
#####################################################################

#Generate list of extraction filepaths
extractions <- list.files(folder_in, full.names=T, pattern = ".dta$", ignore.case=T, recursive = T)
#CGF does not use IPUMS data, so this line is unnecessary but does no harm
extractions <- grep("IPUMS_CENSUS", extractions, invert=T, value = T)

#read all ubcov extracts into a list
if(cluster == TRUE) {
  message("Make cluster")
  cl <- makeCluster(cores)
  message("Register cluster")
  registerDoParallel(cl)
  #suppressMessages(clusterCall(cl, function(x) .libPaths(x), .libPaths()))
  message("Start foreach")
  #Read in each .dta file in parallel - returns a list of data frames
  top <- foreach(i=1:length(extractions), .packages = c('haven')) %dopar% {
    dta <- read_dta(extractions[i], encoding = "latin1")
    return(dta)
  }
  message("Foreach finished")
  message("Closing cluster")
  stopCluster(cl)
} else if(cluster == FALSE) {
  top <- foreach(i=1:length(extractions)) %do% {
    message(paste0("Reading in: ", extractions[i]))
    dta <- read_dta(extractions[i], encoding = "latin1")
    return(dta)
  }
}

#Bind the extracts in that list into a single data table
message("rbindlist all extractions together")
topics <- rbindlist(top, fill=T, use.names=T)
#Overwrite the start year with interview year.
topics <- adjust_year_start_to_int_year(topics) %>% as.data.table()
rm(top)

topics[, year := NULL]
topics[, end_year := NULL]

## .CSV with exclusion types and notes
exclusions <- fread(paste0("/share/code/geospatial/", 'alicela', "/cgf/data_prep/exclusions.csv"))

#Merge on exclude reasons

topics <- merge(topics, exclusions, by.x="nid", by.y="NID", all.x =TRUE, all.y=F)

#Remove extra variables and make exclude variables uniform
topics[, NID := NULL]
topics[, Name := NULL]
topics[, Note := NULL]
topics[is.na(exclude_weights), exclude_weights := 0]
topics[is.na(exclude_age_range), exclude_age_range := 0] 
topics[is.na(exclude_age_granularity), exclude_age_granularity := 0]
topics[is.na(exclude_data_lbw), exclude_data_lbw := 0]
topics[is.na(exclude_data_cgf), exclude_data_cgf := 0]
topics[is.na(exclude_representation), exclude_representation := 0]
topics[is.na(exclude_geography), exclude_geography := 0]
topics[is.na(exclude_interview), exclude_interview := 0]
topics[is.na(exclude_longitudinal), exclude_longitudinal := 0]
topics[is.na(exclude_BMIZ), exclude_BMIZ := 0]
topics[is.na(exclude_duplicative), exclude_duplicative := 0]

##Save raw nid table, if desired
raw_nid_table <- table(topics$nid)
#write.csv(raw_nid_table, file=paste0(folder_out, "/topics_no_geogs_", module_date, ".csv")) 
rm(raw_nid_table)

#####################################################################
## 4.) PULL IN GEO CODEBOOKS
#####################################################################

#Read table with country priority stages
iso <- fread(paste0(j, "/WORK/11_geospatial/10_mbg/stage_master_list.csv"), stringsAsFactors=F)

#Get all geog codebooks and package them together
message("Retrieve geo codebook filepaths")
files <- list.files(paste0(j, "WORK/11_geospatial/05_survey shapefile library/codebooks"), pattern=".csv$", ignore.case = T, full.names = T)
files <- grep("IPUMS|special|20190402", files, value = T, invert = T) # list any strings from geo codebooks you want excluded here 

message("Read geo codebooks into list")
geogs <- lapply(files, read_add_name_col)

message("Bind geo codebooks together")
geo <- rbindlist(geogs, fill=T, use.names=T)
rm(geogs)

keyvars <- c("nid", "iso3", "geospatial_id")
#Dedupe the geography codebook by geospatial_id and nid
setkeyv(geo, keyvars)
geo <- unique(geo, use.key=T)

#coerce lat/longs to numeric, convert iso3s to standard format if desired. We currently need to use the full ihme_loc_id/iso3 for certain India surveys--
geo[, lat := as.numeric(lat)]
geo[, long := as.numeric(long)]
#geo <- geo[, iso3 := substr(iso3, 1, 3)]

#for the sake of some old extracts, overwriting geospatial_id with psu
topics[is.na(geospatial_id), geospatial_id := psu]
topics[, geospatial_id := as.character(geospatial_id)]

#####################################################################
## 5.) PREP DATA FOR MERGE 
#####################################################################

#Reconcile ubCov & geo codebook data types
message("make types between merging datasets match")
if (class(topics$nid) == "numeric"){
  geo[, nid := as.numeric(nid)]
} else if (class(topics$nid) == "character"){
  geo[, nid := as.character(nid)]
} else{
  message("update code to accomodate topics nid as")
  message(class(topics$nid))
}

#Drop unnecessary geo codebook columns
geo[,nid:=as.numeric(nid)]
geo_keep <- c("nid", "iso3", "geospatial_id", "point", "lat", "long", "shapefile", "location_code", "survey_series")
geo_k <- geo[, geo_keep, with=F]


#####################################################################
## 6.) MERGE 
#####################################################################

message("Merge ubCov outputs & geo codebooks together")
all <- merge(geo_k, topics, by.x=c("nid", "iso3", "geospatial_id"), by.y=c("nid", "ihme_loc_id", "geospatial_id"), all.x=F, all.y=T)


##################### OPTIONAL - SPECIFY SURVEYS TO MERGE ON admin_1 NAME RATHER THAN GEOSPATIAL_ID #######################

#if(length(nid_vec) > 0) {
#  message("SPECIFIC RECODES")
#  match_on_admin1_string <- function(nid_vec){
#    nid_vec <- as.character(nid_vec)
#    for (nid_i in nid_vec){
#      if(nid_i %in% all$nid){
#        message(paste("fixing", nid_i))
#        nid_geo <- subset(geo, nid == nid_i)
#        nid_dta <- subset(topics, nid == nid_i)
#        nid_geo[, "svy_area1"] <- as.character(nid_geo[, "svy_area1"])
#        nid_dta[, "admin_1"] <- str_trim(nid_dta[, "admin_1"])
#        nid_all <- merge(nid_geo, nid_dta, by.x = c("nid", "svy_area1"), by.y=c("nid", "admin_1"), all.x=F, all.y = T)
#        all <<- all[all$nid != nid_i, ] #clear from all
#        all <<- rbind.fill(all, nid_all) #rbind to all
#      }
#    }
#  }
#  match_on_admin1_string(nid_vec)
#}

#####################################################################
## 7.) CLEAN UP & SAVE 
#####################################################################

##Fill in lat & long from ubCov extracts, if present

#Overwriting individual India DHS surveys with parent NID
#nids <- c(19794,19803,19806,19812,19818,19824,19830,19836,19842,19848,19854,19860,19866,
#	19872,19878,19884,19890,19896,19902,19908,19914,19920,19926,19932,19938,19944)

#all[(nid %in% nids), nid:=19950]

#Adding hhweights for presumably self-weighting Kwa-Zulu Natal surveys, also 2 Iran surveys, CHNS, the entire Yemen Nutrition Mortality Survey (16 NIDs), and 6 India NNMB/PHFI surveys
all[(nid %in% c(12013, 31142, 159868, 159873, 200838, 246145, 244463, 244464, 244465, 
  244467, 244468, 244471, 244469, 244472, 246249, 244473, 246254, 246250, 246209, 
  246246, 246248, 334953, 129770, 129905, 129913, 130486, 129783)), hhweight := 1]

#We also used a child NID for this source
all[nid == 23172, nid := 23183]

#Overwriting age in months with age in years*12 for LSMS survey where they only took age in years above age 2
all[(nid == 9370 & age_year >2 & is.na(age_month)), age_month := age_year*12]

#Recoding years on some surveys to get them in-scope. Must change int year also so that seasonality will be run on these. # Just a note - this actually does nothing now because data needs to be 2000-present to be in model - Discuss whether we want to do this or not
#IFLS 1997
all[(nid == 5827), year_svy := 1998]
all[(nid == 5827), int_year := 1998]
#LSMS PNG 1996--our only PNG data.
all[(nid == 46563), year_svy := 1998]
all[(nid == 46563), int_year := 1998]
#DHS BRA 1996--padding out Brazil data per Damaris' recommendation 
all[(nid == 19046), year_svy := 1998]
all[(nid == 19046), int_year := 1998]

#swap out subnational ID's for IHME loc ID
all[,ihme_loc_id:=iso3]
all[,iso3:=substr(ihme_loc_id, 1, 3)]

all[admin_1_id == "", admin_1_id := NA]
all[admin_2_id == "", admin_2_id := NA]
all[admin_1_urban_id == "", admin_1_urban_id := NA]

all[!is.na(all$admin_1_id), ihme_loc_id := admin_1_id]
all[!is.na(all$admin_2_id), ihme_loc_id := admin_2_id]
all[!is.na(all$admin_1_urban_id), ihme_loc_id := admin_1_urban_id]
#all$ihme_loc_id[all$ihme_loc_id == "KOSOVO"] <- "SRB"


ubcov_names<-c("survey_series","iso3",
               "lat","long", "year_svy",
               "sex_id", "year_end",
               "metab_height","metab_weight")
new_names<-c("source","country",
             "latitude","longitude", "start_year",
             "sex", "end_year",
             "child_height","child_weight")

setnames(all,ubcov_names,new_names)  ; rm(new_names,ubcov_names)

all[sex==2,sex:=0]

#Save
message("Saving as .Rds")
#saveRDS(all, file=paste0(folder_out, "/", module_date, ".Rds"))
message("Saving as .csv")
#write.csv(all, file=paste0(folder_out, "/", module_date, ".csv"))

#######################################################################################################################################
## 8.) GENERATE DIAGNOSTICS
#######################################################################################################################################

#Create & export a list of all surveys that have not yet been matched & added to the geo codebooks
message("Exporting a list of surveys that need to be geo matched")
gnid <- unique(geo$nid)
fix <- subset(all, !(all$nid %in% gnid))
fix <- as.data.frame(fix)
  fix_collapse <- unique(fix[c("nid", "country", "year_start", "survey_name")])
#write.csv(fix_collapse, paste0(folder_out, "/geographies_to_match", module_date, ".csv"), row.names=F)

#Same as above, only for countries of a designated priority level
if(!is.null(priority)) {
  message(paste0("writing csv of unmatched extractions from priority ", priority, " countries"))
  stage = iso[Stage == priority,]$iso3
  fix_collapse$iso3 <- strsplit(fix_collapse$iso3, 1, 3)
  fix_stage <- subset(fix_collapse, iso3 %in% stage)
  #write.csv(fix_stage, file=paste0(folder_out, "/priority", priority, "_geography_matching_to_do", module_date, ".csv"), row.names = F)
}

merge_diag <- copy(all)

#Diagnostic for completeness of the geography match for all surveys
message("Generate diagnostic table of geo matching completeness")
merge_diag[, n_row := .N, by=nid]
merge_diag[, n_matched := length(na.omit(point)), by=nid]
merge_diag[, age_missing := sum(is.na(age_month))/prod(dim(age_month)), by=nid]
thin = merge_diag[, c("nid", "country", "year_start", "file_path", "source", "n_row", "n_matched")]
short = unique(thin)
short[,n_in_merge := NA]
short[,n_in_geocb := NA]
short_list <- split(short, seq(nrow(short)))
short_list <- setNames(split(short, seq(nrow(short))), rownames(short))

compare_original_file <- function(row){
	n <- row$nid
	row[,n_in_merge := all[nid == n, uniqueN(geospatial_id)]]
  	row[,n_in_geocb := geo_k[nid == n,uniqueN(geospatial_id)]]
	this_filepath <- paste0("/home/j/", str_sub(as.character(row$file_path[[1]]), 4))
  tryCatch({
  	if (length(grep(".sav", this_filepath, ignore.case = T)) != 0) {
		    	orig.dta <- read_spss(this_filepath)
			  } else if (length(grep(".csv", this_filepath, ignore.case = T)) != 0) {
			    orig.dta <- read.csv(this_filepath, fileEncoding="latin1")
			  } else if (length(grep(".xlsx", this_filepath, ignore.case = T)) != 0) {
			    orig.wb = loadWorkbook(this_filepath)
			    orig.dta = readWorksheet(orig.wb, sheet = 1, header = TRUE)
			  } else if (length(grep(".xls", this_filepath, ignore.case = T)) != 0) {
			    orig.dta <- read.xls(this_filepath)
			  }else {
			    orig.dta <- read_stata(this_filepath)
			  }
  }, error = function(e){ NULL})
  print(this_filepath)
  if(exists("orig.dta")){
  row[,n_orig := nrow(orig.dta)]
  } 
  return(row)
}
z <- mclapply(short_list[1:length(short_list)], compare_original_file, mc.cores=30)
short <- rbind.fill(z) %>% as.data.table()

short[, pct_present_codebook := (n_in_merge/n_in_geocb)]
short[pct_present_codebook == Inf, pct_present_codebook := 0]

short[, pct_present_orig := (n_row/n_orig)]

#write.csv(short, paste0(folder_out, "/merge_diagnostic_", module_date, ".csv"), row.names = F)

# Diagnostic for missingness
missingness <- copy(all)

missingness[,y0 := 0]
missingness[,y1 := 0]
missingness[,y2 := 0]
missingness[,y3 := 0]
missingness[,y4 := 0]
missingness[,y5 := 0]
missingness[,y6 := 0]


missingness[age_month< 12, y0 := 1]
missingness[age_month>= 12 & age_month < 24, y1 := 1]
missingness[age_month>= 24 & age_month < 36, y2 := 1]
missingness[age_month>= 36 & age_month < 48, y3 := 1]
missingness[age_month>= 48 & age_month < 60, y4 := 1]
missingness[age_month>= 60 & age_month < 72, y5 := 1]
missingness[age_month>= 72 & age_month < 84, y6 := 1]

cgf.missingness <- aggregate(cbind(is.na(child_weight),
                                   is.na(child_height),
                                   is.na(birth_weight),
                                   is.na(birth_weight_card),
                                   is.na(age_year),
                                   is.na(age_month),
                                   is.na(age_day), 
                                   is.na(int_month),
                                   is.na(int_year),                             
                                   is.na(pweight),
                                   is.na(hhweight),
                                   is.na(psu),
                                   is.na(strata),
                                   is.na(admin_1),
                                   is.na(ihme_loc_id),
                                   y0,
                                   y1,
                                   y2,
                                   y3,
                                   y4,
                                   y5,
                                   y6) ~ nid + country + source + year_start,
                             data = missingness, FUN = mean)

colnames(cgf.missingness) <- c('nid', 'country','source','year_start','na.weight', 'na.height', 'na.birth_weight', 'na.birth_card', 'na.age_year', 
                             'na.age_month', 'na.age_day', 'na.interview_month', 'na.interview_year', 'na.pweight', 'na.hhwweight', 'na.psu','na.strata','na.admin1', 'na.ihme_loc_id',
                             'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6')

#write.csv(cgf.missingness, paste0(folder_out, "/missingness_diagnostic_", module_date, ".csv"), row.names = T)

rm(missingness)
