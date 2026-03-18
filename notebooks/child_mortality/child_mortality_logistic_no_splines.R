################################################################################
# DESCRIPTION: Script to run baseline model on child mortality on subset of data.
# model <- emfrail(Surv(age_month, child_mortality) ~ consumption_pd + 
#                    days_over_30C + 
#                    total_precipitation +
#                    sex_id + 
#                    birth_year + 
#                    survival::cluster(ihme_loc_id), 
#                  data = df_model,
#                  verbose = TRUE)
# PROJECT: Climate nutrition
# DATE: 2026-03-13
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


library(scam)
library(data.table)
library(dplyr) # for anti_join function
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- "cm_splines_no_splines"

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_02_27.01/data.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_02_27.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

## Read and format data
df <- read_parquet(data_version)
df <- data.table(df)

df[,location_id := as.integer(location_id)]

climate_vars <- c(
  "mean_temperature",
  "total_precipitation",
  "relative_humidity",
  "mean_high_temperature",
  "mean_low_temperature",
  "precipitation_days",
  "days_over_30C"
  # "days_over_26C",
  # "any_days_over_30C"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year","int_birth_year_diff_months", climate_vars)
df_model <- df[, ..cols]
df_model <- data.table(df_model)
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,birth_year:=as.factor(birth_year)]
df_model[,days_over_30C:=as.integer(days_over_30C)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]

# make time interval as per Ryan: alive at 1 month, 3 months, 6 months, 1 yr, etc
get_time_var <- function(x){
  # use age_month to find time bin of child's age
  if (x==1){
    return("1 mo")
  }
  else if(x <= 3){
    return("1-3 mo")
  }
  else if(x <= 6){
    return("3-6 mo")
  }
  else if(x<=12){
    return("6-12 mo")
  }
  else if(x<=24){
    return("1-2 yr")
  }
  else if(x<=36){
    return("2-3 yr")    
  }
  else if(x<=48){
    return("3-4 yr")
  }
  else{
    return("4-5 yr")
  }
}
time_var_levels <- c("1 mo","1-3 mo","3-6 mo","6-12 mo","1-2 yr","2-3 yr","3-4 yr","4-5 yr")
# this will revert to 1 through 8 if as.numeric(time_var)

df_model$time_var <- sapply(df_model$age_month,get_time_var)
df_model$time_var <- factor(df_model$time_var,levels=time_var_levels,ordered = TRUE)

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================


# fit model
model <- scam(child_mortality ~ time_var + 
                     sex_id + 
                     consumption_pd + 
                     days_over_30C+
                     total_precipitation+
                     birth_year+
                     s(ihme_loc_id, bs = "re"),
                   family = binomial(link = "logit"),
                   data = df_model)

# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, summary_file,".rds"))