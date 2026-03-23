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

library(mgcv)
library(data.table)
library(dplyr) # for anti_join function
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- "cm_mgcv_cubic_knots"

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
cols <- c("indv_id","child_mortality", "age_month","age_month_at_year_end", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year","int_birth_year_diff_months", climate_vars)
df_model <- df[, ..cols]
df_model <- data.table(df_model)
# limit observations 
# df_model <- df_model[int_birth_year_diff_months<=60]
# temp mini dataset
# df_model <- head(df_model,5000)

df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,birth_year:=as.factor(birth_year)] # trying to comment out for memory test
df_model[,days_over_30C:=as.integer(days_over_30C)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================


# fit model
model <- mgcv::gam(age_month_at_year_end ~ 
                     s(consumption_pd,k=4,bs = "cr") +
                     s(days_over_30C,k=4,bs = "cr") +
                     total_precipitation +
                     sex_id +
                     birth_year +
                     s(ihme_loc_id, bs = "re"),
                   data = df_model,
                   family = mgcv::cox.ph(),                 
                   weights = child_mortality)

# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, summary_file,".rds"))


# model = readRDS(file = paste0(results_dir, summary_file,".rds"))
# save model summary:
summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
capture.output(summary(model), file = summary_file_path)
