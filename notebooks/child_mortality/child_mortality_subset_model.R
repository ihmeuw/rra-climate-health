################################################################################
# DESCRIPTION: Script to run baseline model on child mortality on subset of data.
# PROJECT: Climate nutrition
# DATE: 2025-10-07
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

# install.packages('frailtyEM',lib = "/homes/elyeb/rlibs") # able to handle mixed effects and predict on new data
library(frailtyEM,lib.loc = "/homes/elyeb/rlibs")
library(data.table)
library(caret) # for createFolds function
library(dplyr) # for anti_join function
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
sample_percent <- 0.25
summary_file <- "subset_25pct_model_do30"

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_13.01/data_avg_climate.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_13.01/"
# cov_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/covariates/"
model_summary_dir <- paste0(results_dir,"model_summaries/")
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_13.01/neonatal.parquet"


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

neonatal_dir <- paste0(results_dir,"neonatal/")
dir.create(neonatal_dir, recursive = TRUE, showWarnings = FALSE)

## Read and format data
df <- read_parquet(data_version)
df <- data.table(df)

df[,location_id := as.integer(location_id)]

# load SDI estimates
# sdi <- fread(paste0(cov_dir,"sdi.csv"))
# setnames(sdi,old=c("mean_value","year_id"),new=c("sdi","int_year"))
# 
# sdi <- unique(sdi[,.(location_id,int_year,sdi)])
# df <- merge(df,sdi,by=c("location_id","int_year"),all.x=TRUE)

climate_vars <- c(
  "mean_temperature",
  "total_precipitation",
  "relative_humidity",
  "mean_high_temperature",
  "mean_low_temperature",
  "precipitation_days",
  "days_over_30C",
  "days_over_26C"
)
cols <- c("indv_id","child_mortality", "age_year_at_year_end", "sex_id", "ihme_loc_id", "consumption", climate_vars)
df_model <- df[, ..cols]
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]

# Sample data, keeping all observations for any sampled individual child, and 
# balancing countries
df_model <- data.table(df_model)

## Cut off age at 48 months
# df_model <- df_model[age_year_at_year_end<4]

# get sample
indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
indv_counts <- indv_dt[, .N, by = ihme_loc_id]
indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
indv_dt[, n_sample := floor(sample_percent * N)]

set.seed(42)
sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
df_sample <- df_model[indv_id %in% sampled_indv]

# remove some countries to test if fe different from me
# unique_countries <- unique(df_sample$ihme_loc_id)
# df_sample <- df_sample[ihme_loc_id %in% unique_countries[1:40]]
# df_sample[,ihme_loc_id:=as.factor(ihme_loc_id)]

# Read in neonatal df (must be made from full dataset)
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)

neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
neo_df[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]


#==============================================================================
# SECTION 2: FIT MODEL
#==============================================================================

# fit baseline model with days_over_30C
model <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                   # mean_temperature + 
                   days_over_30C + 
                   sex_id + 
                   survival::cluster(ihme_loc_id), 
                 data = df_sample,
                 verbose = TRUE)

summary(model)
capture.output(summary(model), file = paste0(model_summary_dir,summary_file,".txt"))

# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, summary_file,".rds"))

# get predictions of model over same data set - with random effects
pred_surv <- predict(model, df_sample, quantity = "survival",type="conditional")

surv_at_obs_time <- sapply(seq_len(nrow(df_sample)), function(i) {
  surv_df <- pred_surv[[i]]
  obs_time <- df_sample$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

df_sample$model_predictions_me <- 1-surv_at_obs_time

# get predictions of model over same data set - without random effects
pred_surv <- predict(model, df_sample, quantity = "survival",type="marginal")

surv_at_obs_time <- sapply(seq_len(nrow(df_sample)), function(i) {
  surv_df <- pred_surv[[i]]
  obs_time <- df_sample$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

df_sample$model_predictions_fe <- 1-surv_at_obs_time

print("model predictions done")

subset_str <- as.character(sample_percent)
subset_str <- gsub("0.","",subset_str, fixed = TRUE)
write.csv(df_sample,paste0(results_dir,"predictions_",summary_file,".csv"),row.names = FALSE)

print(paste0("results saved to ",results_dir))


# Get 1 month predictions
print("Getting 1 month predictions")
# # get predictions of model over data set with me model
pred_surv <- predict(model, neo_df, quantity = "survival",type="conditional")

# get predictions at 1 month
surv_at_1_mo <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_surv, 1/12)

neo_df$mortality_1_mo_me <- 1-surv_at_1_mo

# # get predictions of model over data set with fe model
pred_surv <- predict(model, neo_df, quantity = "survival",type="marginal")

# get predictions at 1 month
surv_at_1_mo <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_surv, 1/12)

neo_df$mortality_1_mo_fe <- 1-surv_at_1_mo

# save out for heat maps
write_parquet(neo_df,paste0(neonatal_dir,"neonatal_mortality_",model_name,".parquet"))
