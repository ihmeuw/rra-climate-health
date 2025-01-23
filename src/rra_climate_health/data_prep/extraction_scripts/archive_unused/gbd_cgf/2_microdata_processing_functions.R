library(data.table)
library(plyr)
library(dplyr)
library(haven)
library(anthro)
library(cdcanthro)

"%ni%" <- Negate("%in%")
"%notlike%" <- Negate("%like%")

library(mrbrt003, lib.loc = "/ihme/code/mscm/Rv4/dev_packages/")
# reticulate::use_python("/ihme/code/mscm/miniconda3/envs/mrtool_0.0.1/bin/python")
mr <- import("mrtool")


"%ni%" <- Negate("%in%")
"%notlike%" <- Negate("%like%")


"%ni%" <- Negate("%in%")
"%notlike%" <- Negate("%like%")



create_data_processing_folders <- function(data.date){
  
  
  j.root <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/raw_extractions/"
  new.microdata.j.path <<- paste0(j.root, "microdata/") 
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date)))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/2_initial_processing/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/3_after_mrbrt_outliering/")))
  
  
  
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/4_collapsed/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/4_collapsed/config/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/4_collapsed/only_age_year_surveys/")))
  
  
  
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/J_nids/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/J_nids/small_sample_size/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/J_nids/no_height_weight/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/J_nids/no_sex_info/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/J_nids/suspicious_values/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/L_nids/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/L_nids/small_sample_size/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/L_nids/no_height_weight/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/L_nids/no_sex_info/")))
  suppressWarnings(dir.create(paste0(j.root, "data_", data.date, "/potential_issue_sources/L_nids/suspicious_values/")))
  
  
  
  
  
  
  # L.root <- "/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/"
  # new.microdata.L.path <<- paste0(L.root, "data_", data.date, "/1_raw_extractions/") 
  # suppressWarnings(dir.create(paste0(L.root, "data_", data.date, "/2_initial_processing/")))
  # suppressWarnings(dir.create(paste0(L.root, "data_", data.date, "/3_after_mrbrt_outliering/")))
  # 
  # 
  # 
  # suppressWarnings(dir.create(paste0(L.root, "data_", data.date, "/4_collapsed/")))
  # 
  # suppressWarnings(dir.create(paste0(L.root, "data_", data.date, "/4_collapsed/config/")))
  # suppressWarnings(dir.create(paste0(L.root, "data_", data.date, "/4_collapsed/only_age_year_surveys/")))
  
  
  
  #suppressWarnings(dir.create(paste0(L.root, "data_", data.date, "/potential_issue_sources/")))
  
  
  
  
}

# This function subsets raw microdata down to just rows that have the required sex, age, height/weight information that we need to be able to use it
subset_to_usable_rows <- function(one.file){
  
  # goalkeepers 2024: start with 2,312,282 rows
  # only keep rows that have sex information
  one.file <- one.file[!is.na(sex_id)] 
  
  
  # only keep rows that have at least some degree of age information
  # In preparation for this, add a column full of NA's for any age categorization that the dataframe doesn't have
  
  age.cols <- names(one.file)[names(one.file) %like% "age" & names(one.file) %notlike% "mother"]
  missing.age.cols <- setdiff(c("age_day","age_month", "age_year"), age.cols)
  
  if(length(missing.age.cols) == 1){
    
    one.file[, eval(missing.age.cols) := NA]
    
  }
  
  if(length(missing.age.cols) == 2){
    
    one.file[, eval(missing.age.cols[1]) := NA]
    one.file[, eval(missing.age.cols[2]) := NA]
    
  }
  
  # Then subset to rows where we have at least one age measurement
  one.file <- one.file[!is.na(age_day) | !is.na(age_month) | !is.na(age_year)] 
  
  # Then delete the age columns we added in exclusively for this subset
  if(length(missing.age.cols) == 1){
    
    one.file[, eval(missing.age.cols) := NULL]
    
  }
  
  if(length(missing.age.cols) == 2){
    
    one.file[, eval(missing.age.cols[1]) := NULL]
    one.file[, eval(missing.age.cols[2]) := NULL]
    
  }
  
  
  
  # must have either height or weight
  
  height.weight.cols <- names(one.file)[names(one.file) %in% c("metab_height", "metab_weight")]
  
  # if we only have height, subset out the rows that don't have height
  if(length(height.weight.cols) == 1){
    
    if(height.weight.cols == "metab_height"){
      
      one.file <- one.file[!is.na(metab_height)]
      
      one.file <- one.file[metab_height > 0]
      
      
    }
    
  }
  
  # if we only have weight, subset out the rows that don't have weight
  if(length(height.weight.cols) == 1){
    
    if(height.weight.cols == "metab_weight"){
      
      one.file <- one.file[!is.na(metab_weight)]
      
      one.file <- one.file[metab_weight > 0]
      
      
      
    }
    
  }
  
  # if we have both height and weight, subset out the rows that don't have either
  if(length(height.weight.cols) == 2){
    
    one.file <- one.file[!is.na(metab_weight) | !is.na(metab_height)]
    
    one.file[metab_weight <= 0, metab_weight := NA]
    one.file[metab_height <= 0, metab_height := NA]
    
    
    
  }
  
  
  
  # here we need to fix the age month column for surveys that have age_month coded WRONG
  # wrong coding means a kid who is 1.5 years old got coded as age_year 1, age_month == 6. This age_month should be 18.
  
  
  # Only doing this if we have age month AND age_year but NOT age_day
  if("age_month" %in% names(one.file) & "age_year" %in% names(one.file)){
    
    one.file[age_year > 1 & age_month < 12, age_month := (age_year * 12) + age_month]
    
    
    survey.ihme.loc.id <- unique(one.file$ihme_loc_id)[1]
    
    # GBR surveys are coding every one over 1 year as 12 months
    if(survey.ihme.loc.id %like% "GBR"){
      
      one.file[age_year > 1 & age_month == 12, age_month := (age_year * 12)]
      
    }
    
    
    
    
    
    if("age_day" %in% names(one.file)){
      
      one.file[age_month > 1 & age_day < 31, age_day := round(((age_month * 30.41667) + age_day), 0)]
      
      
    }
    
    
    
  }
  
  
  
  
  
  # Now also subset down to just kids under 5 years
  
  
  # If we have age_day, we'll use that
  if("age_day" %in% names(one.file)){
    
    one.file[, subset.age := age_day]
  }
  
  # If we don't have age_day, we'll use age_month
  if("age_month" %in% names(one.file) & "age_day" %ni% names(one.file)){
    
    one.file[, subset.age := age_month * 30.4375]
  }
  
  # If we don't have age_day or age_month, we'll use age_year
  if("age_year" %in% names(one.file) & "age_day" %ni% names(one.file) & "age_month" %ni% names(one.file)){
    
    one.file[, subset.age := age_year * 365]
  }
  
  
  # If we have NA values, set them equal to age month
  if("age_month" %in% names(one.file)){
    
    one.file[is.na(subset.age), subset.age := age_month * 30.4375]
  }
  
  
  # If we still have NA values, set them equal to age year
  if("age_month" %in% names(one.file)){
    
    one.file[is.na(subset.age), subset.age := age_year * 365]
  }
  
  
  one.file <- one.file[subset.age < 1825]
  one.file[, subset.age := NULL]
  
  
  
  
  return(one.file)
  
  
  
  
}







extraction_or_unit_issue_check <- function(one.file, file.loc = NULL){
  
  
  ######################################################
  # GETTING THE PLOTTING AGES PREPARED
  ######################################################
  
  
  
  # If we have age_day, we'll use that for plotting
  if("age_day" %in% names(one.file)){
    
    one.file[, plot.age := age_day]
  }
  
  # If we don't have age_day, we'll use age_month
  if("age_month" %in% names(one.file) & "age_day" %ni% names(one.file)){
    
    one.file[, plot.age := age_month * 30.4375]
  }
  
  # If we don't have age_day or age_month, we'll use age_year
  if("age_year" %in% names(one.file) & "age_day" %ni% names(one.file) & "age_month" %ni% names(one.file)){
    
    one.file[, plot.age := age_year * 365]
  }
  
  
  # If we have NA values, set them equal to age month
  if("age_month" %in% names(one.file)){
    
    one.file[is.na(plot.age), plot.age := age_month * 30.4375]
  }
  
  
  # If we still have NA values, set them equal to age year
  if("age_month" %in% names(one.file)){
    
    one.file[is.na(plot.age), plot.age := age_year * 365]
  }
  
  
  
  
  ######################################################
  # CHECKING HEIGHT (CENTIMETERS)
  ######################################################
  
  if("metab_height" %in% names(one.file)){
    
    
    
    # Set a wide band of expected values for height in centimeters over our converted age_days
    one.file[, lower.height := (0.016 * plot.age) + 40]
    one.file[, upper.height := (0.02739726 * plot.age) + 75]
    
    
    
    
    # Flag data points that are inside and outside of this band
    one.file[metab_height > lower.height & metab_height < upper.height, inside.expected := 1]
    one.file[is.na(inside.expected), inside.expected := 0]
    
    inside.prop <- nrow(one.file[inside.expected == 1 & !is.na(metab_height)])/nrow(one.file[!is.na(metab_height)])
    
    
    
    
    
    # Write out a csv flagging the NID of any survey that fails to pass this check
    if(inside.prop < .8){
      
      
      
      
      
      survey.nid <- unique(one.file$nid)
      
      survey.ihme.loc.id <- unique(one.file$ihme_loc_id)[1]
      
      
      height.plot = ggplot() +
        geom_point(data = one.file, aes(x = plot.age/30.4375, y = as.numeric(metab_height)*0.0328084, color = inside.expected)) +
        theme_bw() +
        labs(x = "Age (Months)", y = "Height (Feet)", title = paste0("Survey NID: ", survey.nid), 
             subtitle = paste0(unique(one.file$ihme_loc_id), " ", unique(one.file$year_start), " to ", unique(one.file$year_end))) 
      
      if(file.loc == "j"){
        
        pdf(paste0("/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/data_", data.date, "/potential_issue_sources/",survey.ihme.loc.id, "_", survey.nid, "_height.pdf"), height = 8, width = 12)
        suppressWarnings(print(height.plot))
        dev.off()
        
      }
      
      
      
      print(paste0("Source NID ", survey.nid, " has some suspicious height values. Diagnostic written to ", paste0("/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/data_", data.date, "/potential_issue_sources/",survey.ihme.loc.id, "_", survey.nid, "_height.pdf")))
      
      sus.heights = TRUE
      
      
    }
    
    if(inside.prop >= .8){
      sus.heights = FALSE
    }
    
    # Remove the columns we temporarily added
    one.file[,lower.height := NULL]
    one.file[,upper.height := NULL]
    one.file[,inside.expected := NULL]
    
    
    
  }else{
    sus.heights = FALSE
  }
  
  ######################################################
  # CHECKING WEIGHT (kilograms)
  ######################################################
  
  
  
  if("metab_weight" %in% names(one.file)){
    
    
    # Set a wide band of expected values for height in centimeters over age_months
    
    one.file[, lower.weight := (0.003839824 * plot.age) + 1]
    one.file[, upper.weight := ( 0.008196721 * plot.age) + 9]
    
    # Flag data points that are inside and outside of this band
    one.file[metab_weight > lower.weight & metab_weight < upper.weight, inside.expected := 1]
    one.file[is.na(inside.expected), inside.expected := 0]
    
    
    inside.prop <- nrow(one.file[inside.expected == 1 & !is.na(metab_weight)])/nrow(one.file[!is.na(metab_weight)])
    
    # Write out a csv flagging the NID of any survey that fails to pass this check
    if(inside.prop < .8){
      
      
      
      survey.nid <- unique(one.file$nid)
      
      survey.ihme.loc.id <- unique(one.file$ihme_loc_id)[1]
      
      
      
      
      weight.plot = ggplot() +
        geom_point(data = one.file, aes(x = plot.age/30.4375, y = as.numeric(metab_weight)*2.20462, color = inside.expected)) +
        theme_bw() +
        labs(x = "Age (Months)", y = "Weight (Pounds)", title = paste0("Survey NID: ", survey.nid), 
             subtitle = paste0(unique(one.file$ihme_loc_id), " ", unique(one.file$year_start), " to ", unique(one.file$year_end))) 
      
      if(file.loc == "j"){
        
        pdf(paste0("/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/data_", data.date, "/potential_issue_sources/",survey.ihme.loc.id, "_", survey.nid, "_weight.pdf"), height = 8, width = 12)
        suppressWarnings(print(weight.plot))
        dev.off()
        
      }
      
      
      print(paste0("Source NID ", survey.nid, " has some suspicious weight values. Diagnostic written to ", paste0("/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/data_", data.date, "/potential_issue_sources/",survey.ihme.loc.id, "_", survey.nid, "_weight.pdf")))
      
      sus.weights = TRUE
      
      
    }
    
    if(inside.prop >= .8){
      sus.weights = FALSE
    }
    
    # Remove the columns we temporarily added
    one.file[,lower.weight := NULL]
    one.file[,upper.weight := NULL]
    one.file[,inside.expected := NULL]
    
    
  }else{
    sus.weights = FALSE
  }
  
  
  
  # Flagging if we had any suspicious heights or weights in this source
  one.file[, suspicious.heights := sus.heights]
  one.file[, suspicious.weights := sus.weights]
  
  
  # removing the plot age column
  one.file[, plot.age := NULL]
  
  
  return(one.file)
  
  
}




# This calculates Z scores depending on what level of age granularity we have
# Uses days if we have it
# Uses Months if we don't, converting years to months before doing the calculation
calculate_z_scores <- function(one.file){
  
  
  # making them all the same so that this code can run. Then if they're all the same afterwards, we'll delete the HAZ/WAZ/WHZ scores
  if("metab_weight" %ni% names(one.file)){
    
    one.file[, metab_weight := 50]
  }
  
  if("metab_height" %ni% names(one.file)){
    
    one.file[, metab_height := 50]
  }
  
  
  
  
  
  
  # Using Age Days where we have it
  if("age_day" %in% names(one.file)){
    
    zscores = setnames(data.table(anthro_zscores(sex = one.file$sex_id,
                                                 age = one.file$age_day,
                                                 weight = one.file$metab_weight,
                                                 lenhei = one.file$metab_height))[, c("zwei", "zlen", "zwfl"), with = F], c("zwei", "zlen", "zwfl"), c("WAZ", "HAZ", "WHZ"))
    
    # Also calculating for month in case there's some rows without days
    if("age_month" %in% names(one.file)){
      
      
      zscores.month = setnames(data.table(anthro_zscores(sex = one.file$sex_id,
                                                         age = one.file$age_month,
                                                         is_age_in_month = TRUE,
                                                         weight = one.file$metab_weight,
                                                         lenhei = one.file$metab_height))[, c("zwei", "zlen", "zwfl"), with = F], c("zwei", "zlen", "zwfl"), c("WAZ.month", "HAZ.month", "WHZ.month"))
      
      zscores <- cbind(zscores, zscores.month)
      
      
    }
    
    # Also calculating for years in case there's some rows without days and months
    if("age_year" %in% names(one.file)){
      
      one.file[, months.from.year := age_year/12]
      
      zscores.year = setnames(data.table(anthro_zscores(sex = one.file$sex_id,
                                                        age = one.file$months.from.year,
                                                        is_age_in_month = TRUE,
                                                        weight = one.file$metab_weight,
                                                        lenhei = one.file$metab_height))[, c("zwei", "zlen", "zwfl"), with = F], c("zwei", "zlen", "zwfl"), c("WAZ.year", "HAZ.year", "WHZ.year"))
      
      one.file[, months.from.year := NULL]
      
      zscores <- cbind(zscores, zscores.year)
      
      
    }
    
    
    # Replace NA values with the months values where we have it
    zscores[is.na(HAZ), HAZ := HAZ.month]
    zscores[is.na(WAZ), WAZ := WAZ.month]
    zscores[is.na(WHZ), WHZ := WHZ.month]
    
    # Replace NA values with the years values where we have it 
    zscores[is.na(HAZ), HAZ := HAZ.year]
    zscores[is.na(WAZ), WAZ := WAZ.year]
    zscores[is.na(WHZ), WHZ := WHZ.year]
    
    # keep only the most specific Z scores
    zscores <- zscores[, c("HAZ", "WAZ", "WHZ")]
    
    
    # Merge onto the data
    one.file <- cbind(one.file, zscores)
    
    
    
    
    
    
  }
  
  # If we don't have age_day, we'll use age_month
  if("age_month" %in% names(one.file) & "age_day" %ni% names(one.file)){
    
    
    zscores = setnames(data.table(anthro_zscores(sex = one.file$sex_id,
                                                 age = one.file$age_month,
                                                 is_age_in_month = TRUE,
                                                 weight = one.file$metab_weight,
                                                 lenhei = one.file$metab_height))[, c("zwei", "zlen", "zwfl"), with = F], c("zwei", "zlen", "zwfl"), c("WAZ", "HAZ", "WHZ"))
    
    
    
    
    # Also calculating for years in case there's some rows without days and months
    if("age_year" %in% names(one.file)){
      
      one.file[, months.from.year := age_year/12]
      
      zscores.year = setnames(data.table(anthro_zscores(sex = one.file$sex_id,
                                                        age = one.file$months.from.year,
                                                        is_age_in_month = TRUE,
                                                        weight = one.file$metab_weight,
                                                        lenhei = one.file$metab_height))[, c("zwei", "zlen", "zwfl"), with = F], c("zwei", "zlen", "zwfl"), c("WAZ.year", "HAZ.year", "WHZ.year"))
      
      one.file[, months.from.year := NULL]
      
      zscores <- cbind(zscores, zscores.year)
      
      
    }
    
    
    
    # Replace NA values with the years values where we have it 
    zscores[is.na(HAZ), HAZ := HAZ.year]
    zscores[is.na(WAZ), WAZ := WAZ.year]
    zscores[is.na(WHZ), WHZ := WHZ.year]
    
    # keep only the most specific Z scores
    zscores <- zscores[, c("HAZ", "WAZ", "WHZ")]
    
    # Merge onto the data
    one.file <- cbind(one.file, zscores)
    
    
    
    
  }
  
  
  # If we don't have age_day or age_month, we'll use age_year
  if("age_year" %in% names(one.file) & "age_day" %ni% names(one.file) & "age_month" %ni% names(one.file)){
    
    
    one.file[, age_month_imputed_from_year := age_year*12]
    
    zscores = setnames(data.table(anthro_zscores(sex = one.file$sex_id,
                                                 age = one.file$age_month_imputed_from_year,
                                                 is_age_in_month = TRUE,
                                                 weight = one.file$metab_weight,
                                                 lenhei = one.file$metab_height))[, c("zwei", "zlen", "zwfl"), with = F], c("zwei", "zlen", "zwfl"), c("WAZ", "HAZ", "WHZ"))
    
    one.file <- cbind(one.file, zscores)
    
    # Flagging that we had to use age_year, since this isn't ideal
    one.file[, microdata_using_age_year := 1]
    
  }
  
  
  
  #create mild/moderate/severe/extreme flags 
  one.file[, `:=`(HAZ_b1=0, HAZ_b2=0, HAZ_b3=0, HAZ_b4=0,
                  WAZ_b1=0, WAZ_b2=0, WAZ_b3=0, WAZ_b4=0,
                  WHZ_b1=0, WHZ_b2=0, WHZ_b3=0, WHZ_b4=0)]
  
  one.file[ HAZ<(-1) , HAZ_b1:=1]
  one.file[ WAZ<(-1) , WAZ_b1:=1]
  one.file[ WHZ<(-1) , WHZ_b1:=1]
  
  one.file[ HAZ<(-2) , HAZ_b2:=1]
  one.file[ WAZ<(-2) , WAZ_b2:=1]
  one.file[ WHZ<(-2) , WHZ_b2:=1]
  
  one.file[ HAZ<(-3) , HAZ_b3:=1]
  one.file[ WAZ<(-3) , WAZ_b3:=1]
  one.file[ WHZ<(-3) , WHZ_b3:=1]
  
  one.file[ HAZ<(-4) , HAZ_b4:=1]
  one.file[ WAZ<(-4) , WAZ_b4:=1]
  one.file[ WHZ<(-4) , WHZ_b4:=1]
  
  
  
  # removing HAZ/WAZ/WHZ scores from surveys that didn't have height and weight
  if(length(unique(one.file$metab_weight)) == 1 & unique(one.file$metab_weight)[1] == 50){
    
    one.file[, metab_weight := NULL]
    one.file[, WAZ := NA]
    one.file[, WAZ_b1 := NULL]
    one.file[, WAZ_b2 := NULL]
    one.file[, WAZ_b3 := NULL]
    one.file[, WAZ_b4 := NULL]
    one.file[, WHZ := NA]
    one.file[, WHZ_b1 := NULL]
    one.file[, WHZ_b2 := NULL]
    one.file[, WHZ_b3 := NULL]
    one.file[, WHZ_b4 := NULL]
    
    
  }
  
  if(length(unique(one.file$metab_height)) == 1 & unique(one.file$metab_height)[1] == 50){
    
    one.file[, metab_height := NULL]
    one.file[, HAZ := NA]
    one.file[, HAZ_b1 := NULL]
    one.file[, HAZ_b2 := NULL]
    one.file[, HAZ_b3 := NULL]
    one.file[, HAZ_b4 := NULL]
    one.file[, WHZ := NA]
    one.file[, WHZ_b1 := NULL]
    one.file[, WHZ_b2 := NULL]
    one.file[, WHZ_b3 := NULL]
    one.file[, WHZ_b4 := NULL]
    
    
  }
  
  
  
  
  
  
  return(one.file)
  
  
}



save_preprocessed_j_file <- function(file, one.file){
  
  
  file = substr(file, 1, nchar(file)-3)
  
  file = paste0(file, "rds")
  
  saveRDS(one.file, paste0( "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/raw_extractions/data_", data.date, "/2_initial_processing/", file))
  
  
}



log_small_or_excluded_j_sources <- function(file, data.date, one.file, issue.type = NULL){
  
  one.f <- read.csv(paste0("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/raw_extractions/data_",data.date ,"/1_raw_extractions/", file))
  
  i_loc <- unique(one.f$ihme_loc_id)[1]
  
  original.file <- as.data.table(read.csv(paste0(new.microdata.j.path, file)))
  
  source.nid = unique(original.file$nid)
  
  print(paste0("NID ", source.nid, " not processed. Reason: ", issue.type))
  
  write.csv(one.file, paste0("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/raw_extractions/data_", data.date, "/potential_issue_sources/J_nids/", issue.type, "/", i_loc, "_", source.nid, ".csv"), row.names = F)
  
}





get_all_cleaned_data_filepaths <- function(all.microdata.data.dates){
  
  
  
  source.df <- lapply(all.microdata.data.dates, function(ddate){
    
    
    j.cleaned.sources <- list.files(paste0("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/raw_extractions/data_", ddate, "/2_initial_processing/"))
    j.cleaned.sources <- paste0("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/raw_extractions/data_", ddate, "/2_initial_processing/", j.cleaned.sources)
    
    all.sources <- data.table(fp = j.cleaned.sources)
    
    return(all.sources)
    
    
  }) %>% rbindlist()
  
  return(source.df)
  
  
  
}






get_sample_in_each_bin <- function(source.df){
  
  
  
  
  
  mrbrt.microdata <- lapply(source.df$fp, function(one.fp){
    
    df <- readRDS(one.fp)[, c("HAZ", "WAZ", "WHZ"), with = F]
    
    df <- df[, c("HAZ", "WAZ", "WHZ"), with = F]
    df <- df[!is.na(HAZ) & !is.na(WAZ) & !is.na(WHZ)]
    
    df <- df[HAZ >= -6 & HAZ <= 6]
    df <- df[WAZ >= -6 & WAZ <= 6]
    df <- df[WHZ >= -6 & WHZ <= 6]
    
    
    return(df)
    
  }) %>% rbindlist()
  
  
  
  # create a grid from -6 to +6 and assign each bin a number
  small.grid <- data.table(expand.grid(HAZ.grid.vals = seq(-6, 5, by = 1),
                                       WAZ.grid.vals = seq(-6, 5, by = 1)))
  small.grid[, bucket.index := 1:.N]
  
  
  mrbrt.microdata[, HAZ.floor := floor(HAZ)]
  mrbrt.microdata[, WAZ.floor := floor(WAZ)]
  
  for (row in small.grid$bucket.index) {
    
    haz.bucket = small.grid[bucket.index == row]$HAZ.grid.vals
    
    waz.bucket = small.grid[bucket.index == row]$WAZ.grid.vals
    
    mrbrt.microdata[HAZ.floor == haz.bucket & WAZ.floor == waz.bucket, bucket.index := row]
    
    
  }
  
  
  
  # sample 1,000 points from each bin if there are 1,000. If there's less than 1,000 take all points from that bin
  mrbrt.microdata.sample <- lapply(small.grid$bucket.index, function(bucket){
    
    hold.df <- mrbrt.microdata[bucket.index == bucket]
    
    if(nrow(hold.df) > 100){
      
      hold.df.sample <- hold.df[sample(.N, 100)]
      
    }
    
    if(nrow(hold.df) <= 100){
      
      hold.df.sample <- hold.df[sample(.N, nrow(hold.df))]
      
    }
    
    return(hold.df.sample)
    
    
  }) %>% rbindlist()
  
  
  return(mrbrt.microdata.sample)
  
  
  
  
  
}








create_haz_waz_whz_grid <- function(mrbrt.microdata.sample){
  
  
  
  mrbrt.microdata.sample[, HAZ.se := .25]
  mrbrt.microdata.sample[, WAZ.se := .25]
  mrbrt.microdata.sample[, WHZ.se := .25]
  
  
  
  dat1 <- mr$MRData()
  
  
  # predicting WHZ using HAZ and WAZ
  dat1$load_df(
    data = mrbrt.microdata.sample,  col_obs = "WHZ", col_obs_se = "WHZ.se",
    col_covs = list("HAZ", "WAZ"))
  
  mod1 <- mr$MRBRT(
    data = dat1,
    cov_models = list(
      mr$LinearCovModel("intercept", use_re = FALSE),
      mr$LinearCovModel(alt_cov = "HAZ",
                     use_spline = T,
                     spline_degree = 3L) ,
      mr$LinearCovModel(alt_cov = "WAZ",
                     use_spline = T,
                     spline_degree = 3L) ) )
  
  mod1$fit_model(inner_print_level = 2L, inner_max_iter = 2000L)
  
  
  
  # predict out on a grid from -10 to +10 in increments of 0.05
  df_pred1 <- expand.grid(data.frame(HAZ = seq(-10, 10, by = .01),
                                     WAZ = seq(-10, 10, by = .01)))
  
  dat_pred1 <- mr$MRData()
  
  dat_pred1$load_df(
    data = df_pred1, 
    col_covs=list("HAZ", "WAZ")
  )
  
  df_pred1$pred0 <- mod1$predict(data = dat_pred1)
  
  # now we have the predictions for that grid
  names(df_pred1)[names(df_pred1) == 'pred0'] <- 'WHZ'
  
  grid.mean.vector <- as.numeric(df_pred1$WHZ)
  plane.df <- data.table(HAZ = df_pred1$HAZ,
                         WAZ = df_pred1$WAZ,
                         WHZ = grid.mean.vector)
  
  
  plane.df[, HAZ := signif(HAZ, 3)]
  plane.df[, WAZ := signif(WAZ, 3)]
  
  
  # merge on the plane.df to the input data used in this model to determine the 95% on the residuals
  
  
  mrbrt.microdata.sample <- mrbrt.microdata.sample[, c("HAZ", "WAZ", "WHZ"), with = FALSE]
  setnames(mrbrt.microdata.sample, "WHZ", "WHZ.data")
  
  
  combined.df <- merge(mrbrt.microdata.sample, plane.df, by = c("HAZ", "WAZ"), all.x = T)
  
  # find the difference between the WHZ data and the plane
  combined.df[, absolute.diff := abs(WHZ - WHZ.data)]
  
  allowable.diff <- as.numeric(quantile(combined.df$absolute.diff, probs = .95, na.rm = TRUE))
  
  output.list <- list(plane.df, allowable.diff)
  
  
  
  return(output.list)
  
  
  
}








get_average_of_ten_grids <- function(ten.grids){
  
  
  sample.1 <- as.data.table(ten.grids[[1]][1])
  sample.2 <- as.data.table(ten.grids[[2]][1])
  sample.3 <- as.data.table(ten.grids[[3]][1])
  sample.4 <- as.data.table(ten.grids[[4]][1])
  sample.5 <- as.data.table(ten.grids[[5]][1])
  sample.6 <- as.data.table(ten.grids[[6]][1])
  sample.7 <- as.data.table(ten.grids[[7]][1])
  sample.8 <- as.data.table(ten.grids[[8]][1])
  sample.9 <- as.data.table(ten.grids[[9]][1])
  sample.10 <- as.data.table(ten.grids[[10]][1])
  
  
  
  setnames(sample.1, "WHZ", "WHZ.1")
  setnames(sample.2, "WHZ", "WHZ.2")
  setnames(sample.3, "WHZ", "WHZ.3")
  setnames(sample.4, "WHZ", "WHZ.4")
  setnames(sample.5, "WHZ", "WHZ.5")
  setnames(sample.6, "WHZ", "WHZ.6")
  setnames(sample.7, "WHZ", "WHZ.7")
  setnames(sample.8, "WHZ", "WHZ.8")
  setnames(sample.9, "WHZ", "WHZ.9")
  setnames(sample.10, "WHZ", "WHZ.10")
  
  
  
  sample.2 <- sample.2[, c("WHZ.2"), with = FALSE]
  sample.3 <- sample.3[, c("WHZ.3"), with = FALSE]
  sample.4 <- sample.4[, c("WHZ.4"), with = FALSE]
  sample.5 <- sample.5[, c("WHZ.5"), with = FALSE]
  sample.6 <- sample.6[, c("WHZ.6"), with = FALSE]
  sample.7 <- sample.7[, c("WHZ.7"), with = FALSE]
  sample.8 <- sample.8[, c("WHZ.8"), with = FALSE]
  sample.9 <- sample.9[, c("WHZ.9"), with = FALSE]
  sample.10 <- sample.10[, c("WHZ.10"), with = FALSE]
  
  
  all.samples <- cbind(sample.1, sample.2, sample.3, sample.4, sample.5, sample.6, sample.7, sample.8, sample.9, sample.10)
  all.samples$WHZ.grid <- rowMeans(all.samples[, 3:12])
  all.samples <- all.samples[, -c(3:12) ]
  
  return(all.samples)
  
  
}











get_average_residual_allowed <- function(ten.grids){
  
  
  
  sample.1 <- as.numeric(ten.grids[[1]][2])
  sample.2 <- as.numeric(ten.grids[[2]][2])
  sample.3 <- as.numeric(ten.grids[[3]][2])
  sample.4 <- as.numeric(ten.grids[[4]][2])
  sample.5 <- as.numeric(ten.grids[[5]][2])
  sample.6 <- as.numeric(ten.grids[[6]][2])
  sample.7 <- as.numeric(ten.grids[[7]][2])
  sample.8 <- as.numeric(ten.grids[[8]][2])
  sample.9 <- as.numeric(ten.grids[[9]][2])
  sample.10 <- as.numeric(ten.grids[[10]][2])
  
  
  average.diff <- (sample.1 + sample.2 + sample.3 + sample.4 + sample.5 + sample.6 + sample.7 + sample.8 + sample.9 + sample.10) /10
  
  return(average.diff)
  
  
}
















outlier_based_on_mrbrt <- function(fp, allowable.diff, final.grid){
  
  
  
  print(paste0("Starting ", fp))
  
  
  one.file <- readRDS(fp)
  
  one.file[is.na(HAZ), HAZ_b1 := NA]
  one.file[is.na(HAZ), HAZ_b2 := NA]
  one.file[is.na(HAZ), HAZ_b3 := NA]
  one.file[is.na(HAZ), HAZ_b4 := NA]
  
  one.file[is.na(WAZ), WAZ_b1 := NA]
  one.file[is.na(WAZ), WAZ_b2 := NA]
  one.file[is.na(WAZ), WAZ_b3 := NA]
  one.file[is.na(WAZ), WAZ_b4 := NA]
  
  one.file[is.na(WHZ), WHZ_b1 := NA]
  one.file[is.na(WHZ), WHZ_b2 := NA]
  one.file[is.na(WHZ), WHZ_b3 := NA]
  one.file[is.na(WHZ), WHZ_b4 := NA]
  
  
  # we're going to keep all of these rows
  one.file.inside <- one.file[(HAZ >= -6 & HAZ <= 6) & (WAZ >= -6 & WAZ <= 6) & (WHZ >= -6 & WHZ <= 6)]
  
  # get everything not included in this internal component
  one.file.outside <- setdiff(one.file, one.file.inside)
  
  # subset out rows where we have an NA value, can't use these
  one.file.outside.nas <- one.file.outside[is.na(HAZ) | is.na(WAZ) | is.na(WHZ)]
  
  # rows with outside values and no NA
  onefile.outside.no.nas <- setdiff(one.file.outside, one.file.outside.nas)
  
  # subset this df to have no values beyond -10 or +10
  onefile.outside.no.nas <- onefile.outside.no.nas[(HAZ >= -10 & HAZ <= 10) & (WAZ >= -10 & WAZ <= 10) & (WHZ >= -10 & WHZ <= 10)]
  
  # now merge the final grid onto this df
  onefile.outside.no.nas <- merge(onefile.outside.no.nas, final.grid, by = c("HAZ", "WAZ"), all.x = TRUE)
  
  # figure out the absolute difference and only keep rows less than that
  onefile.outside.no.nas[, diff.between.grid := abs(WHZ.grid - WHZ)]
  onefile.outside.no.nas <- onefile.outside.no.nas[diff.between.grid <= allowable.diff]
  
  onefile.outside.no.nas <- onefile.outside.no.nas[, -c("WHZ.grid", "diff.between.grid")]
  
  
  
  
  
  
  output <- rbind(one.file.inside, onefile.outside.no.nas)
  
  
  
  fp.components <- strsplit(fp, split = "/")
  
  
  
  # need to convert the age_year column if age_year is 0
  
  
  # adjust the age_year value if we have either of these
  if("age_day" %in% names(output) | "age_month" %in% names(output)){
    
    
    # ONLY AGE MONTH
    if("age_month" %in% names(output) & "age_day" %ni% names(output)){
      
      output[!is.na(age_month), age_year := age_month/12]
      
    }
    
    # ONLY AGE DAY
    if("age_day" %in% names(output) & "age_month" %ni% names(output)){
      
      output[!is.na(age_day), age_year := age_day/365]
      
    }
    
    
    # HAVE BOTH, ADJUST WHAT WE CAN
    if("age_day" %in% names(output) & "age_month" %in% names(output)){
      
      output[!is.na(age_month), age_year := age_month/12]
      output[!is.na(age_day), age_year := age_day/365]
      
    }
    
    
    # if we're using microdata that truly only has age_year... flag that it needs to be split in kids under 1
  }else{
    
    empty <- data.table()
    
    
    if(fp.components[[1]][4] == "nch"){
      
      
      J.path = copy(fp.components)
      
      source.nid = unique(output$nid)
      
      J.path[[1]][11] <- "4_collapsed"
      J.path[[1]][12] <- "only_age_year_surveys"
      J.path[[1]][13] <- paste0(source.nid, ".csv")
      
      J.path <- paste0("/",J.path[[1]][2],
                       "/", J.path[[1]][3],
                       "/", J.path[[1]][4],
                       "/", J.path[[1]][5],
                       "/", J.path[[1]][6],
                       "/", J.path[[1]][7],
                       "/", J.path[[1]][8],
                       "/", J.path[[1]][9],
                       "/", J.path[[1]][10],
                       "/", J.path[[1]][11],
                       "/", J.path[[1]][12],
                       "/", J.path[[1]][13])
      
      
      
      write.csv(empty, J.path, row.names = FALSE)
    }
    
    
    if(fp.components[[1]][4] == "limited_use"){
      
      L.path <- copy(fp.components)
      
      source.nid = unique(output$nid)
      
      L.path[[1]][10] <- "4_collapsed"
      L.path[[1]][12] <-  paste0(source.nid, ".csv")
      L.path[[1]][11] <-  "only_age_year_surveys"
      
      
      L.path <- paste0("/",L.path[[1]][2],
                       "/", L.path[[1]][3],
                       "/", L.path[[1]][4],
                       "/", L.path[[1]][5],
                       "/", L.path[[1]][6],
                       "/", L.path[[1]][7],
                       "/", L.path[[1]][8],
                       "/", L.path[[1]][9],
                       "/", L.path[[1]][10],
                       "/", L.path[[1]][11],
                       "/", L.path[[1]][12])
      
      write.csv(empty, L.path, row.names = FALSE)
    }
    
    
  }
  
  
  
  
  
  
  
  
  
  # need to get the output filepath
  
  
  
  
  # getting the save filepaths for data on the J drive
  if(fp.components[[1]][4] == "integrated_analytics"){
    
    
    
    fp.components[[1]][11] <- "3_after_mrbrt_outliering"
    
    
    
    
    
    
    output.fp <- paste0("/",fp.components[[1]][2],
                        "/", fp.components[[1]][3],
                        "/", fp.components[[1]][4],
                        "/", fp.components[[1]][5],
                        "/", fp.components[[1]][6],
                        "/", fp.components[[1]][7],
                        "/", fp.components[[1]][8],
                        "/", fp.components[[1]][9],
                        "/", fp.components[[1]][10],
                        "/", fp.components[[1]][11],
                        "/", fp.components[[1]][12])
    
    
    
    print(output.fp)
    
  }
  
  
  
  # change to be csv files
  output.fp <- paste0(substr(output.fp, 1, nchar(output.fp)-4), ".csv")
  
  
  
  
  write.csv(output, output.fp, row.names = FALSE)
  
  
  
  
  
}






fix_any_potential_duplicates <- function(data.dates.to.collapse){
  
  
  
  
  for (data.date in data.dates.to.collapse) {
    
    
    
    # DOING ALL J DRIVE ONES
    
    print(paste0("Checking J drive data for data date ", data.date))
    
    all.data.fps <- list.files(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/", data.date,"/3_after_mrbrt_outliering"))
    
    for (fp in all.data.fps) {
      
      one.file <- fread(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/", data.date,"/3_after_mrbrt_outliering/", fp))
      
      # if it even has subnationals
      if("admin_1_id" %in% names(one.file)){
        
        # if any of those subnationals are the same as the national
        if(nrow(one.file[admin_1_id == ihme_loc_id]) > 0){
          
          print(paste0("Fixing NID ", unique(one.file$nid), "."))
          
          
          columns.to.drop <- intersect(names(one.file), c("admin_1",	"admin_2",	"admin_3",	"urban", "admin_1_mapped",	"admin_1_id", "admin_2_id", "admin_2_mapped", "admin_3_id", "admin_3_mapped", "rural"))
          
          
          for (col in columns.to.drop) {
            
            one.file[, eval(col) := NA]
            
          }
          
          first.type.of.fix <- TRUE
          
          
        }
        
      }else{first.type.of.fix <- FALSE}
      
      if("admin_1_id"  %in% names(one.file) & "admin_2_id" %in% names(one.file)){
        
        print(paste0("Fixing NID ", unique(one.file$nid), "."))
        
        
        second.type.of.fix <- TRUE
        
        
        one.file[admin_1_id == admin_2_id, admin_2_id := NA]
        
        
        
      }else{second.type.of.fix <- FALSE}
      
      if("admin_1"  %in% names(one.file) & "admin_2" %in% names(one.file)){
        
        print(paste0("Fixing NID ", unique(one.file$nid), "."))
        
        
        third.type.of.fix <- TRUE
        
        one.file[admin_1 == admin_2, admin_2 := NA]
        
        
        
      }else{third.type.of.fix <- FALSE}
      
      
      if(first.type.of.fix == TRUE | second.type.of.fix == TRUE | third.type.of.fix == TRUE){
        
        write.csv(one.file, paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/", data.date,"/3_after_mrbrt_outliering/", fp), row.names = FALSE)
        
      }
      
      
      
      
    }
    
    
    # DOING ALL L DRIVE ONES
    
    print(paste0("Checking L drive data for data date ", data.date))
    
    all.data.fps <- list.files(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/", data.date,"/3_after_mrbrt_outliering"))
    
    # for (fp in all.data.fps) {
    #   
    #   one.file <- fread(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/", data.date,"/3_after_mrbrt_outliering/", fp))
    #   
    #   # if it even has subnationals
    #   if("admin_1_id" %in% names(one.file)){
    #     
    #     # if any of those subnationals are the same as the national
    #     if(nrow(one.file[admin_1_id == ihme_loc_id]) > 0){
    #       
    #       print(paste0("Fixing NID ", unique(one.file$nid), "."))
    #       
    #       
    #       columns.to.drop <- intersect(names(one.file), c("admin_1",	"admin_2",	"admin_3",	"urban", "admin_1_mapped",	"admin_1_id", "admin_2_id", "admin_2_mapped", "admin_3_id", "admin_3_mapped", "rural"))
    #       
    #       
    #       for (col in columns.to.drop) {
    #         
    #         one.file[, eval(col) := NA]
    #         
    #       }
    #       
    #       first.type.of.fix <- TRUE
    #       
    #       
    #     }
    #     
    #   }else{first.type.of.fix <- FALSE}
    #   
    #   
    #   if("admin_1_id"  %in% names(one.file) & "admin_2_id" %in% names(one.file)){
    #     
    #     print(paste0("Fixing NID ", unique(one.file$nid), "."))
    #     
    #     
    #     second.type.of.fix <- TRUE
    #     
    #     one.file[admin_1_id, admin_2_id := NA]
    #     
    #     
    #     
    #   }else{second.type.of.fix <- FALSE}
    #   
    #   
    #   if("admin_1"  %in% names(one.file) & "admin_2" %in% names(one.file)){
    #     
    #     print(paste0("Fixing NID ", unique(one.file$nid), "."))
    #     
    #     
    #     third.type.of.fix <- TRUE
    #     
    #     one.file[admin_1 == admin_2, admin_2 := NA]
    #     
    #     
    #     
    #   }else{third.type.of.fix <- FALSE}
    #   
    #   
    #   
    # }
    
    
    # if(first.type.of.fix == TRUE | second.type.of.fix == TRUE | third.type.of.fix == TRUE){
    #   
    #   write.csv(one.file, paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/", data.date,"/3_after_mrbrt_outliering/", fp), row.names = FALSE)
    #   
    # }
    
    
    
    
  }
  
  
}






create_collapse_code_configs <- function(data.dates.to.collapse){
  
  
  
  
  # creating the collapse config files for data on the J drive
  for (ddate in data.dates.to.collapse) {
    
    
    
    if(length(list.files(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/", ddate,"/3_after_mrbrt_outliering/"))) > 0){
      
      
      config.df.j <- data.table(topic = "anthropometrics",
                                input.root = paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",ddate ,"/3_after_mrbrt_outliering/"),
                                output.root = paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",ddate ,"/4_collapsed/"),
                                vars = "HAZ, HAZ_b1, HAZ_b2, HAZ_b3, HAZ_b4, WAZ, WAZ_b1, WAZ_b2, WAZ_b3, WAZ_b4, WHZ, WHZ_b1, WHZ_b2, WHZ_b3, WHZ_b4",
                                vars.categ = "",
                                calc.sd = FALSE,
                                cv.manual = NULL,
                                cv.detect = FALSE,
                                by_sex = 1,
                                by_age = 1,
                                gbd_age_cuts = 1,
                                aggregate_under1 = FALSE,
                                custom_age_cuts = NULL,
                                cond_age_cuts = NULL,
                                sample_threshold = NA,
                                census_data = FALSE)
      
      write.csv(config.df.j, paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",ddate ,"/4_collapsed/config/collapse_config.csv"), row.names = FALSE)
      
      
      
      
    }else{
      
      empty <- data.table()
      
      write.csv(empty, paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",ddate ,"/4_collapsed/config/no_data_to_collapse.csv"), row.names = FALSE)
      
      
    } 
    
    
    
    
    
  }
  
  
  
  # # creating the collapse config files for data on the L drive
  # for (ddate in data.dates.to.collapse) {
  # 
  # 
  # 
  # 
  #   if(length(list.files(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/", ddate,"/3_after_mrbrt_outliering/"))) > 0){
  # 
  # 
  #     config.df.L <- data.table(topic = "anthropometrics",
  #                               input.root = paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",ddate ,"/3_after_mrbrt_outliering/"),
  #                               output.root = paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",ddate ,"/4_collapsed/"),
  #                               vars = "HAZ, HAZ_b1, HAZ_b2, HAZ_b3, HAZ_b4, WAZ, WAZ_b1, WAZ_b2, WAZ_b3, WAZ_b4, WHZ, WHZ_b1, WHZ_b2, WHZ_b3, WHZ_b4",
  #                               vars.categ = "",
  #                               calc.sd = FALSE,
  #                               cv.manual = NULL,
  #                               cv.detect = FALSE,
  #                               by_sex = 1,
  #                               by_age = 1,
  #                               gbd_age_cuts = 1,
  #                               aggregate_under1 = FALSE,
  #                               custom_age_cuts = NULL,
  #                               cond_age_cuts = NULL,
  #                               sample_threshold = NA,
  #                               census_data = FALSE)
  # 
  #     write.csv(config.df.L, paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",ddate ,"/4_collapsed/config/collapse_config.csv"), row.names = FALSE)
  # 
  # 
  # 
  # 
  #   }else{
  # 
  #     empty <- data.table()
  # 
  #     write.csv(empty, paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",ddate ,"/4_collapsed/config/no_data_to_collapse.csv"), row.names = FALSE)
  # 
  # 
  #   }
  
  
  
  
  #}
  
  
  
  
  
  
}







update_file_with_all_cleaned_microdata <- function(all.data.dates){
  
  
  
  
  # read in all J drive microdata
  
  print("Starting to compile J drive data.")
  
  j.data <- lapply(all.data.dates, function(ddate){
    
    date.sources <- list.files(paste0("/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/",ddate ,"/3_after_mrbrt_outliering/"))
    
    j.data.one.date <- lapply(date.sources, function(file.name){
      
      
      one.file.data <- fread(paste0("/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/",ddate ,"/3_after_mrbrt_outliering/", file.name))
      
      
      keep.cols <- c("ihme_loc_id", "nid", "year_start", "year_end", "age_month", "age_year", "age_day", "HAZ", "WAZ", "WHZ", "sex_id")
      
      keep.cols.present <- intersect(names(one.file.data), keep.cols)
      
      one.file.data <- one.file.data[, keep.cols.present, with = FALSE]
      
      
      return(one.file.data)
      
    }) %>% rbindlist(fill = TRUE)
    
    
    return(j.data.one.date)
    
  }) %>% rbindlist(fill = TRUE)
  
  print("Saving J drive data.")
  
  
  saveRDS(j.data, paste0("/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data/wasting_stunting/all_data_outliered_",all.data.dates,".rds"))
  
  
  
  
}







collapse.launch.wrapper.function <- function(data.dates.to.collapse, mem.amount, launch.queue){
  
  
  
  for (data.date.x in data.dates.to.collapse) {
    
    
    
    # SETTINGS FOR J DRIVE COLLAPSE
    
    topic <- "anthropometrics" ## Subset config.csv
    config.path <- paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/config/collapse_config.csv") ## Path to config.csv. Note: Parallel runs will not work if there are spaces in this file path.
    parallel <- TRUE ## Run in parallel?
    cluster_project <- 'proj_nch' ## You must enter a cluster project in order to run in parallel
    fthreads <- 2 ## How many threads per job (used in mclapply) | Set to 1 if running on desktop
    m_mem_free <- mem.amount ## How many GBs of RAM to use per job
    h_rt <- "02:00:00"  ## How much run time to request per job | format is "HH:MM:SS"
    logs <- paste0("/ihme/temp/slurmoutput/",Sys.getenv("USER") ,"/") ## Path to logs (must specify a valid filepath if running in parallel)
    
    ## Launch collapse
    collapse.launch(topic=topic, config.path=config.path, parallel=parallel, cluster_project=cluster_project, fthreads=fthreads, m_mem_free=m_mem_free, h_rt=h_rt, logs=logs, central.root=ubcov_central, queue = launch.queue)
    
    
    
    
    
    # SETTINGS FOR L DRIVE COLLAPSE
    
    topic <- "anthropometrics" ## Subset config.csv
    config.path <- paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/config/collapse_config.csv") ## Path to config.csv. Note: Parallel runs will not work if there are spaces in this file path.
    parallel <- TRUE ## Run in parallel?
    cluster_project <- 'proj_nch' ## You must enter a cluster project in order to run in parallel
    fthreads <- 2 ## How many threads per job (used in mclapply) | Set to 1 if running on desktop
    m_mem_free <- mem.amount ## How many GBs of RAM to use per job
    h_rt <- "00:15:00"  ## How much run time to request per job | format is "HH:MM:SS"
    logs <- paste0("/ihme/temp/slurmoutput/",Sys.getenv("USER") ,"/") ## Path to logs (must specify a valid filepath if running in parallel)
    
    ## Launch collapse
    collapse.launch(topic=topic, config.path=config.path, parallel=parallel, cluster_project=cluster_project, fthreads=fthreads, m_mem_free=m_mem_free, h_rt=h_rt, logs=logs, central.root=ubcov_central, queue = launch.queue)
    
    
    
    
    
    
  }
  
  
  print("############################################################")
  print("############################################################")
  print("#############   DONE COLLAPSING ALL SURVEYS!   #############")
  print("############################################################")
  print("############################################################")
  
  
  
}







check_for_collapse_failures <- function(data.dates.to.collapse){
  
  
  
  failure.dt <- data.table(data.date = data.dates.to.collapse)
  
  for (ddate in data.dates.to.collapse) {
    
    
    
    # Check to see if the J drive collapse worked
    
    J.failure <- file.exists(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",ddate ,"/4_collapsed/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))
    
    if(!J.failure){
      
      print(paste0("For ", ddate, " all the surveys on the J drive successfully collapsed."))
    }
    
    if(J.failure){
      
      print(paste0("For ", ddate, " there was at least 1 survey on the J drive that didn't collapse right."))
    }
    
    # Check to see if the L drive collapse worked
    
    L.failure <- file.exists(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",ddate ,"/4_collapsed/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))
    
    
    if(!L.failure){
      
      print(paste0("For ", ddate, " all the surveys on the L drive successfully collapsed."))
    }
    
    if(L.failure){
      
      print(paste0("For ", ddate, " there was at least 1 survey on the L drive that didn't collapse right."))
    }
    
    
    
    if(J.failure & L.failure){
      collapse.fail.status = "J_L_failures"
    }
    
    if(J.failure & !L.failure){
      collapse.fail.status = "J_failures"
    }
    
    if(!J.failure & L.failure){
      collapse.fail.status = "L_failures"
    }
    
    if(!J.failure & !L.failure){
      collapse.fail.status = "no_failures"
    }
    
    
    
    failure.dt[data.date == ddate, collapse.failure.status := collapse.fail.status]
    
    
    
  }
  
  
  
  return(failure.dt)
  
  
}









retry.collapse.launch.wrapper.function <- function(data.dates.to.collapse, mem.amount, collapse.failures.dt, launch.queue){
  
  
  
  for (data.date.x in data.dates.to.collapse) {
    
    
    # only relaunch the J collapse for this date if there were J failures
    if(data.date.x %in% collapse.failures.dt[collapse.failure.status == "J_failures" | collapse.failure.status == "J_L_failures"]$data.date){
      
      
      # read in the failure csv
      failed.surveys.csv <- fread(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      
      # move all the failed surveys to a new folder
      dir.create(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/temp_redo/"))
      
      for (fs in failed.surveys.csv$failed_collapse) {
        
        failed.old.file = fread(paste0(fs, ".csv"))
        
        
        old.parts <- strsplit(fs, split = "/")
        old.name = old.parts[[1]][length(old.parts[[1]])]
        
        
        write.csv(failed.old.file, paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/temp_redo/", old.name, ".csv"), row.names = FALSE)
        
        
      }
      
      # read in the old config
      old.config <- fread(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/config/collapse_config.csv"))
      
      old.config[, input.root := paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/temp_redo/")]
      old.config[, output.root := paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/output_redo/")]
      
      # create a temporary folder for the redone output
      dir.create(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/output_redo/"))
      
      # Resave the config
      write.csv(old.config, paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/", data.date.x, "/4_collapsed/config/redo_collapse_config.csv"), row.names = FALSE)
      
      
      
      
      
      # SETTINGS FOR J DRIVE COLLAPSE
      
      topic <- "anthropometrics" ## Subset config.csv
      config.path <- paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/config/redo_collapse_config.csv") ## Path to config.csv. Note: Parallel runs will not work if there are spaces in this file path.
      parallel <- TRUE ## Run in parallel?
      cluster_project <- 'proj_nch' ## You must enter a cluster project in order to run in parallel
      fthreads <- 2 ## How many threads per job (used in mclapply) | Set to 1 if running on desktop
      m_mem_free <- mem.amount ## How many GBs of RAM to use per job
      h_rt <- "00:15:00"  ## How much run time to request per job | format is "HH:MM:SS"
      logs <- paste0("/ihme/temp/slurmoutput/",Sys.getenv("USER") ,"/") ## Path to logs (must specify a valid filepath if running in parallel)
      
      ## Launch collapse
      collapse.launch(topic=topic, config.path=config.path, parallel=parallel, cluster_project=cluster_project, fthreads=fthreads, m_mem_free=m_mem_free, h_rt=h_rt, logs=logs, central.root=ubcov_central, queue = launch.queue)
      
      
      
      # now we need to take the new collapse and rbind it onto the existing collapse
      
      new.collapse <- fread(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/output_redo/collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      old.collapse <- fread(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      combined.collapse <- rbind(old.collapse, new.collapse)
      
      #overwriting the old collapse
      write.csv(combined.collapse, paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/", data.date.x, "/4_collapsed/collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      
      
      
      
      
    }
    
    
    # only relaunch the L collapse for this date if there were L failures
    if(data.date.x %in% collapse.failures.dt[collapse.failure.status == "L_failures" | collapse.failure.status == "J_L_failures"]$data.date){
      
      
      # read in the failure csv
      failed.surveys.csv <- fread(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      
      # move all the failed surveys to a new folder
      dir.create(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/temp_redo/"))
      
      for (fs in failed.surveys.csv$failed_collapse) {
        
        failed.old.file = fread(paste0(fs, ".csv"))
        
        
        old.parts <- strsplit(fs, split = "/")
        old.name = old.parts[[1]][length(old.parts[[1]])]
        
        
        write.csv(failed.old.file, paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/temp_redo/", old.name, ".csv"), row.names = FALSE)
        
        
      }
      
      # read in the old config
      old.config <- fread(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/config/collapse_config.csv"))
      
      old.config[, input.root := paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/temp_redo/")]
      old.config[, output.root := paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/output_redo/")]
      
      # create a temporary folder for the redone output
      dir.create(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/output_redo/"))
      
      # Resave the config
      write.csv(old.config, paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/", data.date.x, "/4_collapsed/config/redo_collapse_config.csv"), row.names = FALSE)
      
      
      
      
      
      # SETTINGS FOR L DRIVE COLLAPSE
      
      topic <- "anthropometrics" ## Subset config.csv
      config.path <- paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/config/redo_collapse_config.csv") ## Path to config.csv. Note: Parallel runs will not work if there are spaces in this file path.
      parallel <- TRUE ## Run in parallel?
      cluster_project <- 'proj_nch' ## You must enter a cluster project in order to run in parallel
      fthreads <- 2 ## How many threads per job (used in mclapply) | Set to 1 if running on desktop
      m_mem_free <- mem.amount ## How many GBs of RAM to use per job
      h_rt <- "00:15:00"  ## How much run time to request per job | format is "HH:MM:SS"
      logs <- paste0("/ihme/temp/slurmoutput/",Sys.getenv("USER") ,"/") ## Path to logs (must specify a valid filepath if running in parallel)
      
      ## Launch collapse
      collapse.launch(topic=topic, config.path=config.path, parallel=parallel, cluster_project=cluster_project, fthreads=fthreads, m_mem_free=m_mem_free, h_rt=h_rt, logs=logs, central.root=ubcov_central, queue = launch.queue)
      
      
      
      
      # now we need to take the new collapse and rbind it onto the existing collapse
      
      new.collapse <- fread(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/output_redo/collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      old.collapse <- fread(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      combined.collapse <- rbind(old.collapse, new.collapse)
      
      #overwriting the old collapse
      write.csv(combined.collapse, paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/", data.date.x, "/4_collapsed/collapse_anthropometrics_", Sys.Date(), ".csv"))
      
      
      
      
      
      
    }
    
    
    
  }
  
  
  print("############################################################")
  print("############################################################")
  print("#############   DONE COLLAPSING ALL SURVEYS!   #############")
  print("############################################################")
  print("############################################################")
  
  
  
  
  for (data.date.x in data.dates.to.collapse) {
    
    
    # Checking for additional failures and deleting files if we needed to rerun J
    if(data.date.x %in% collapse.failures.dt[collapse.failure.status == "J_failures" | collapse.failure.status == "J_L_failures"]$data.date){
      
      
      if(file.exists(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/output_redo/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))){
        
        print(paste0("Sources on the J drive for ", data.date.x, " failed again with more memory. Look into them more closely." ))
        
      }else{
        
        print(paste0("All sources on the J drive for ", data.date.x, " that originally failed during collapse succeeded this time with more memory." ))
        
        
        # can remove the output if we've already rbinded it to the existing output
        unlink(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/output_redo/"), recursive = TRUE)
        
        # we can also remove the initial failed_collapse csv
        unlink(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))
        
      }
      
      # Delete where we copied over the files for this redo
      unlink(paste0("/mnt/team/nch/priv/cgf/gbd2022/data/winnower_extraction_process/",data.date.x ,"/4_collapsed/temp_redo/"), recursive = TRUE)
      
      
      
      
      
      
      
      
    }
    
    
    
    if(data.date.x %in% collapse.failures.dt[collapse.failure.status == "L_failures" | collapse.failure.status == "J_L_failures"]$data.date){
      
      
      
      
      if(file.exists(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/output_redo/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))){
        
        print(paste0("Sources on the L drive for ", data.date.x, " failed again with more memory. Look into these more closely" ))
        
      }else{
        
        print(paste0("All sources on the L drive for ", data.date.x, " that originally failed during collapse succeeded this time with more memory." ))
        
        
        # can remove the output if we've already rbinded it to the existing output
        unlink(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/output_redo/"), recursive = TRUE)
        
        
        # we can also remove the initial failed_collapse csv
        unlink(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/failed_collapse_anthropometrics_", Sys.Date(), ".csv"))
        
      }
      
      # Delete where we copied over the files for this redo
      unlink(paste0("/mnt/share/limited_use/LIMITED_USE/LU_GBD/ubcov_extractions/cgf/",data.date.x ,"/4_collapsed/temp_redo/"), recursive = TRUE)
      
      
      
      
      
      
      
      
      
      
      
    }
    
    
    
  }
  
  
  
  
}

