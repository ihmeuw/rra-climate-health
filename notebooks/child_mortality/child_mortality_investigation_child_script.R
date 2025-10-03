################################################################################
# DESCRIPTION: Child script to run a single fold in a k-fold cross-validation
# task
# PROJECT: Climate nutrition
# DATE: 2025-09-17
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

# install.packages('coxme',lib = "/homes/elyeb/rlibs") # for survival analysis with mixed effects
# install.packages('frailtyEM',lib = "/homes/elyeb/rlibs") # able to handle mixed effects and predict on new data
library(frailtyEM,lib.loc = "/homes/elyeb/rlibs")
library(coxme,lib.loc = "/homes/elyeb/rlibs") 
library(data.table)
library(caret) # for createFolds function
library(dplyr) # for anti_join function
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

fold_file <- commandArgs()[4]

print(paste0("running on fold file ",fold_file))
#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_09_15.01/data.parquet"
plot_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_09_15.01/"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_09_15.01/"
folds_dir <- paste0(results_dir,"folds/")

fold_indices <- readRDS(paste0(folds_dir,fold_file))[[1]]
fold_number <- as.integer(gsub(".*_(\\d+)\\.rds$", "\\1", fold_file))

df <- read_parquet(data_version)
df <- data.table(df)

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df[,child_mortality := 1-child_alive]

# need to create annual version of age_month_at_year_end to match annual climate 
# vars. Note however that this does not appear to change model results.
df[,age_year_at_year_end := age_month_at_year_end/12]

setnames(df,old="ldipc_weighted_no_match",new="consumption")

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
cols <- c("child_mortality", "age_year_at_year_end", "sex_id", "ihme_loc_id", "consumption", climate_vars)
df_model <- df[, ..cols]

#==============================================================================
# SECTION 2: FIT MODELS
#==============================================================================

## Test secondary climate variables with k-fold cross-validation

test <- df_model[fold_indices]
train <- anti_join(df_model, test)

# fit model with total_precipitation
model1 <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                    mean_temperature + 
                    total_precipitation + 
                    sex_id + 
                    survival::cluster(ihme_loc_id), 
                  data = train,
                  verbose = TRUE)

# get predictions on test set
pred_out <- predict(model1, test, re.form = ~0, quantity="survival")

surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, test$age_year_at_year_end)

test$model1_total_precipitation_pred <- 1-surv_at_obs_time
print("model1_total_precipitation_pred done")

# fit model with relative_humidity
model2 <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                    mean_temperature + 
                    relative_humidity + 
                    sex_id + 
                    survival::cluster(ihme_loc_id), 
                  data = train,
                  verbose = TRUE)

# get predictions on test set
pred_out <- predict(model2, test, re.form = ~0, quantity="survival")

surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, test$age_year_at_year_end)

test$model2_relative_humidity_pred <- 1-surv_at_obs_time
print("model2_relative_humidity_pred done")

# fit model with precipitation_days
model3 <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                    mean_temperature + 
                    precipitation_days + 
                    sex_id + 
                    survival::cluster(ihme_loc_id), 
                  data = train,
                  verbose = TRUE)

# get predictions on test set
pred_out <- predict(model1, test, re.form = ~0, quantity="survival")

surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, test$age_year_at_year_end)

test$model3_precipitation_days_pred <- 1-surv_at_obs_time
print("model3_precipitation_days_pred done")

# fit model with days_over_30C
model4 <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                    mean_temperature + 
                    days_over_30C + 
                    sex_id + 
                    survival::cluster(ihme_loc_id), 
                  data = train,
                  verbose = TRUE)

# get predictions on test set
pred_out <- predict(model4, test, re.form = ~0, quantity="survival")

surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, test$age_year_at_year_end)

test$model4_days_over_30C_pred <- 1-surv_at_obs_time
print("model4_days_over_30C_pred done")

# fit model with days_over_26C
model5 <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                    mean_temperature + 
                    days_over_30C + 
                    sex_id + 
                    survival::cluster(ihme_loc_id), 
                  data = train,
                  verbose = TRUE)

# get predictions on test set
pred_out <- predict(model5, test, re.form = ~0, quantity="survival")

surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, test$age_year_at_year_end)

test$model5_days_over_26C_pred <- 1-surv_at_obs_time
print("model5_days_over_26C_pred done")

# fit model with only mean_temperature
model6 <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                    mean_temperature + 
                    sex_id + 
                    survival::cluster(ihme_loc_id), 
                  data = train,
                  verbose = TRUE)

# get predictions on test set
pred_out <- predict(model6, test, re.form = ~0, quantity="survival")

surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, test$age_year_at_year_end)

test$model6_mean_temperature_only_pred <- 1-surv_at_obs_time
print("model6_mean_temperature_only_pred done")

# Summarize results 
results <- data.table(test)[,.(child_mortality, 
                               model1_total_precipitation_pred,
                               model2_relative_humidity_pred,
                               model3_precipitation_days_pred, 
                               model4_days_over_30C_pred,
                               model5_days_over_26C_pred,
                               model6_mean_temperature_only_pred)]

results <- melt(results,id.vars='child_mortality')
setnames(results,old=c('variable','value'),new=c('model','predictions'))
results <- results[,.(MSE = mean((predictions - child_mortality)^2),
                      RMSE = sqrt(mean((predictions - child_mortality)^2)),
                      MAE = mean(abs(predictions - child_mortality))),
                   by=model]

results$fold <- fold_number

write.csv(results,paste0(results_dir,"manual_CV_results_fold_",fold_number,".csv"),row.names = FALSE)