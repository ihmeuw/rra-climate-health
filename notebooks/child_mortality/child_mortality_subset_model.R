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
sample_percent <- 0.5
summary_file <- "subset_50pct_model_do30"

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_18.01/data.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_18.01/"
# cov_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/covariates/"
model_summary_dir <- paste0(results_dir,"model_summaries/")
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_18.01/neonatal_data.parquet"


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

neonatal_dir <- paste0(results_dir,"neonatal/")
dir.create(neonatal_dir, recursive = TRUE, showWarnings = FALSE)

## Read and format data
df <- read_parquet(data_version)
df <- data.table(df)

df[,location_id := as.integer(location_id)]
setnames(df,old="ldipc_weighted_no_match",new="consumption")
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
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption", climate_vars)
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
setnames(neo_df,old="ldipc_weighted_no_match",new="consumption")

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================

# tmp override: 
# df_sample <- df_model

# fit baseline model with days_over_30C
model <- emfrail(Surv(age_month, child_mortality) ~ consumption + 
                   # mean_temperature + 
                   days_over_30C + 
                   sex_id + 
                   survival::cluster(ihme_loc_id), 
                 data = df_sample,
                 verbose = TRUE)

# Extract frailty estimates for each cluster (ihme_loc_id)
frailty_effects <- model$frail

# Create frailty lookup data frame
frailty_df <- data.frame(
  ihme_loc_id = names(frailty_effects),
  frailty = as.numeric(frailty_effects)
)

# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, summary_file,".rds"))

# save model summary:
summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
capture.output(summary(model), file = summary_file_path)
# Append frailty estimates
cat("\n\n", file = summary_file_path, append = TRUE)
cat("================================================================================\n", 
    file = summary_file_path, append = TRUE)
cat("CLUSTER-SPECIFIC FRAILTY ESTIMATES (RANDOM EFFECTS)\n", 
    file = summary_file_path, append = TRUE)
cat("================================================================================\n\n", 
    file = summary_file_path, append = TRUE)
frailty_output <- capture.output(print(frailty_df, row.names = FALSE))
cat(paste(frailty_output, collapse = "\n"), file = summary_file_path, append = TRUE)
# Also save frailty estimates as a separate CSV for easier access
write.csv(frailty_df, paste0(model_summary_dir, "frailty_estimates_", summary_file, ".csv"), 
          row.names = FALSE)

#==============================================================================
# SECTION 3: PREDICT MODEL ON ALL AGES
#==============================================================================

## Predict mixed effects and fixed effects manually

# Extract fixed effect coefficients
coefs <- coef(model)
beta_consumption <- coefs["consumption"]
beta_days_over_30C <- coefs["days_over_30C"]
beta_sex_female <- coefs["sex_idFemale"]

# Extract baseline hazard - Note this is only as long as unique months in which
# someone died.
baseline_hazard <- model$hazard  # This contains time and cumulative baseline hazard
baseline_hazard <- data.frame(
  time = model$tev,
  hazard = baseline_hazard
)

# Merge frailty estimates on data
df_sample <- merge(df_sample, frailty_df, by = "ihme_loc_id", all.x = TRUE)

# Calculate linear predictor (fixed effects only)
df_sample$linear_pred <- (
  beta_consumption * df_sample$consumption +
    beta_days_over_30C * df_sample$days_over_30C +
    beta_sex_female * (df_sample$sex_id == "Female")
)

# Function to get cumulative baseline hazard at a given time
get_cumhaz_baseline <- function(time, basehaz_df) {
  if (time <= 0) return(0)
  idx <- max(which(basehaz_df$time <= time))
  if (length(idx) == 0 || idx == 0) return(0)
  return(basehaz_df$hazard[idx])
}

# Calculate cumulative baseline hazard at each observation time
df_sample$cumhaz_baseline <- sapply(df_sample$age_month, function(t) {
  get_cumhaz_baseline(t, baseline_hazard)
})

# MANUAL PREDICTION WITH RANDOM EFFECTS (Mixed Effects)
# Formula: H(t|X,Z) = Z * H0(t) * exp(X'β)
# where Z is the frailty for that cluster
df_sample$cumhaz_me <- df_sample$frailty * 
  df_sample$cumhaz_baseline * 
  exp(df_sample$linear_pred)

# Survival probability = exp(-cumulative hazard)
df_sample$survival_me <- exp(-df_sample$cumhaz_me)

# Mortality probability = 1 - survival
df_sample$mortality_me <- 1 - df_sample$survival_me


# MANUAL PREDICTION WITHOUT RANDOM EFFECTS (Fixed Effects Only)
# Formula: H(t|X) = H0(t) * exp(X'β)
# Equivalent to setting frailty Z = 1 (or E[Z] = 1)
df_sample$cumhaz_fe <- df_sample$cumhaz_baseline * 
  exp(df_sample$linear_pred)

# Survival probability = exp(-cumulative hazard)
df_sample$survival_fe <- exp(-df_sample$cumhaz_fe)

# Mortality probability = 1 - survival
df_sample$mortality_fe <- 1 - df_sample$survival_fe

## Predict mixed effects and fixed effects using package predict function

# # get predictions of model over same data set - with random effects
# pred_surv <- predict(model, df_sample, quantity = "survival",type="conditional")
# 
# surv_at_obs_time <- sapply(seq_len(nrow(df_sample)), function(i) {
#   surv_df <- pred_surv[[i]]
#   obs_time <- df_sample$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# df_sample$mortality_me_auto <- 1-surv_at_obs_time
# 
# # get predictions of model over same data set - without random effects
# pred_surv <- predict(model, df_sample, quantity = "survival",type="marginal")
# 
# surv_at_obs_time <- sapply(seq_len(nrow(df_sample)), function(i) {
#   surv_df <- pred_surv[[i]]
#   obs_time <- df_sample$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# df_sample$mortality_fe_auto <- 1-surv_at_obs_time

print("model predictions done")


# save results
write_parquet(df_sample,paste0(results_dir,"predictions_",summary_file,".parquet"))
print(paste0("child mortality predictions saved to ",paste0(results_dir,"predictions_",summary_file,".parquet")))

#==============================================================================
# SECTION 4: PREDICT MODEL FOR NEONATAL
#==============================================================================

# Get 1 month predictions
print("Getting 1 month predictions")

## Predict mixed effects and fixed effects manually

# Merge frailty estimates with neonatal data
neo_df <- merge(neo_df, frailty_df, by = "ihme_loc_id", all.x = TRUE)

# Calculate linear predictor for neonatal data
neo_df$linear_pred <- (
  beta_consumption * neo_df$consumption +
    beta_days_over_30C * neo_df$days_over_30C +
    beta_sex_female * (neo_df$sex_id == "Female")
)

# Get baseline cumulative hazard at 1 month
time_1mo <- 1
cumhaz_baseline_1mo <- get_cumhaz_baseline(time_1mo, baseline_hazard)

# MANUAL PREDICTION WITH RANDOM EFFECTS (Mixed Effects)
neo_df$cumhaz_me_1mo <- neo_df$frailty * cumhaz_baseline_1mo * exp(neo_df$linear_pred)
neo_df$survival_me_1mo <- exp(-neo_df$cumhaz_me_1mo)
neo_df$mortality_me <- 1 - neo_df$survival_me_1mo

# MANUAL PREDICTION WITHOUT RANDOM EFFECTS (Fixed Effects Only)
neo_df$cumhaz_fe_1mo <- cumhaz_baseline_1mo * exp(neo_df$linear_pred)
neo_df$survival_fe_1mo <- exp(-neo_df$cumhaz_fe_1mo)
neo_df$mortality_fe <- 1 - neo_df$survival_fe_1mo


## Predict mixed effects and fixed effects using package predict function

# # # get predictions of model over data set with me model
# pred_surv <- predict(model, neo_df, quantity = "survival",type="conditional")
# 
# # get predictions at 1 month
# surv_at_1_mo <- mapply(function(df, t) {
#   idx <- max(which(df$time <= t))
#   df$survival[idx]
# }, pred_surv, 1/12)
# 
# neo_df$mortality_me_auto <- 1-surv_at_1_mo
# 
# # # get predictions of model over data set with fe model
# pred_surv <- predict(model, neo_df, quantity = "survival",type="marginal")
# 
# # get predictions at 1 month
# surv_at_1_mo <- mapply(function(df, t) {
#   idx <- max(which(df$time <= t))
#   df$survival[idx]
# }, pred_surv, 1/12)
# 
# neo_df$mortality_fe_auto <- 1-surv_at_1_mo

# save out for heat maps
write_parquet(neo_df,paste0(neonatal_dir,"neonatal_mortality_",summary_file,".parquet"))
print(paste0("neonatal mortality predictions saved to ",paste0(neonatal_dir,"neonatal_mortality_",summary_file,".parquet")))
