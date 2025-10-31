################################################################################
# DESCRIPTION: Read in fitted models and explore summaries and predictions
# PROJECT: Climate nutrition
# DATE: 2025-10-10
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
library(lme4)
library(data.table)
library(arrow) # to read parquet
library(ggplot2)
library(scales)


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

# data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data_filtered.parquet"
data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data.parquet"
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/neonatal_data.parquet"

# data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.csv"
plot_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_24.01/"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")

neonatal_dir <- paste0(results_dir,"neonatal/")
model_objects_dir <- paste0(neonatal_dir,"model_objects/")
dir.create(neonatal_dir, recursive = TRUE, showWarnings = FALSE)


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
  "days_over_30C",
  "days_over_26C",
  "any_days_over_30C"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year","int_birth_year_diff_months", climate_vars)
df_model <- df[, ..cols]
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]


df_model <- data.table(df_model)

# Read in neonatal df (must be made from full dataset)
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)

# only keep age_month 1
neo_df <- neo_df[age_month==1]

neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
neo_df[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]


## Read and format data

climate_vars <- c(
  "mean_temperature",
  "total_precipitation",
  "relative_humidity",
  "mean_high_temperature",
  "mean_low_temperature",
  "precipitation_days",
  "days_over_30C",
  "days_over_26C",
  "any_days_over_30C"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year","int_birth_year_diff_months", climate_vars)
df_model_neo <- neo_df[, ..cols]

#==============================================================================
# SECTION 2: READ MODELS
#==============================================================================


## Read in and print model summaries from successful runs:

# 10/30 - predict models on mean values for birth year, precipitation, sex_id
summary_file <- "cm_v7"
model = readRDS(file = paste0(results_dir, summary_file,".rds"))

# Extract frailty estimates for each cluster (ihme_loc_id)
frailty_effects <- model$frail

# Create frailty lookup data frame
frailty_df <- data.frame(
  ihme_loc_id = names(frailty_effects),
  frailty = as.numeric(frailty_effects)
)

# Extract fixed effect coefficients
coefs <- coef(model)
beta_consumption <- coefs["consumption_pd"]
beta_total_precipitation <- coefs["total_precipitation"]
beta_days_over_30C <- coefs["days_over_30C"]
beta_sex_female <- coefs["sex_idFemale"]
beta_birth_year <- coefs["birth_year"]


# Extract baseline hazard - Note this is only as long as unique months in which
# someone died.
baseline_hazard <- model$hazard  
baseline_hazard <- data.frame(
  time = model$tev,
  hazard = baseline_hazard
)

baseline_hazard <- baseline_hazard[order(baseline_hazard$time), ]
baseline_hazard$cumhazard <- cumsum(baseline_hazard$hazard)

# Merge frailty estimates on data
df_model <- merge(df_model, frailty_df, by = "ihme_loc_id", all.x = TRUE)

# create mean variables
df_model[,mean_precipitation:= mean(df_model$total_precipitation)]
df_model[,mean_birth_year:= mean(df_model$birth_year)]
# note that data is coded as 1=male, 2=female
df_model[,mean_fraction_female:= mean(as.numeric(df_model$sex_id))-1]

# Calculate linear predictor (fixed effects only)
df_model$linear_pred <- (
  beta_consumption * df_model$consumption_pd +
    beta_days_over_30C * df_model$days_over_30C +
    beta_total_precipitation * df_model$mean_precipitation + 
    beta_sex_female * df_model$mean_fraction_female+
    beta_birth_year * df_model$mean_birth_year
)

# Function to get cumulative baseline hazard at a given time
get_cumhaz_baseline <- function(time, basehaz_df) {
  if (time <= 0) return(0)
  idx <- max(which(basehaz_df$time <= time))
  if (length(idx) == 0 || idx == 0) return(0)
  return(basehaz_df$cumhazard[idx])
}

# Calculate cumulative baseline hazard at each observation time
df_model$cumhaz_baseline <- sapply(df_model$age_month, function(t) {
  get_cumhaz_baseline(t, baseline_hazard)
})

# MANUAL PREDICTION WITH RANDOM EFFECTS (Mixed Effects)
# Formula: H(t|X,Z) = Z * H0(t) * exp(X'β)
# where Z is the frailty for that cluster
df_model$cumhaz_me <- df_model$frailty * 
  df_model$cumhaz_baseline * 
  exp(df_model$linear_pred)


# Survival probability = exp(-cumulative hazard)
df_model$survival_me <- exp(-df_model$cumhaz_me)

# Mortality probability = 1 - survival
df_model$mortality_me <- 1 - df_model$survival_me


# MANUAL PREDICTION WITHOUT RANDOM EFFECTS (Fixed Effects Only)
# Formula: H(t|X) = H0(t) * exp(X'β)
# Equivalent to setting frailty Z = 1 (or E[Z] = 1)
df_model$cumhaz_fe <- df_model$cumhaz_baseline * 
  exp(df_model$linear_pred)

# Survival probability = exp(-cumulative hazard)
df_model$survival_fe <- exp(-df_model$cumhaz_fe)

# Mortality probability = 1 - survival
df_model$mortality_fe <- 1 - df_model$survival_fe


# Also get point probability estimates
df_model <- merge(df_model, baseline_hazard[, c("time", "hazard")], 
                  by.x = "age_month", by.y = "time", all.x = TRUE, suffixes = c("", "_point"))

# Calculate point hazard for each observation
df_model$hazard_point_me <- df_model$frailty * df_model$hazard * exp(df_model$linear_pred)
df_model$hazard_point_fe <- df_model$hazard * exp(df_model$linear_pred)

# Convert to point mortality probability (probability of dying in that month)
df_model$mortality_point_me <- 1 - exp(-df_model$hazard_point_me)
df_model$mortality_point_fe <- 1 - exp(-df_model$hazard_point_fe)
print("model predictions done")


# save results
write_parquet(df_model,paste0(results_dir,"predictions_",summary_file,"_means.parquet"))
print(paste0("child mortality predictions saved to ",paste0(results_dir,"predictions_",summary_file,"_means.parquet")))

# Do the same for the latest neonatal data
summary_file <- "nm_v7"
model = readRDS(file = paste0(model_objects_dir, summary_file,".rds"))
df_model <- copy(df_model_neo)
setDT(df_model)
# override existing variables to be able to use predict function from package
df_model[,birth_year:=as.numeric(birth_year)]
df_model[,sex_id:=as.numeric(sex_id)]

df_model[,total_precipitation:= mean(df_model$total_precipitation)]
df_model[,birth_year:= mean(df_model$birth_year)]
# note that data is coded as 1=male, 2=female
df_model[,sex_id:= mean(as.numeric(df_model$sex_id))-1]

# get coefficients from model
coefs <- lme4::fixef(model)
beta_intercept <- coefs["(Intercept)"]
beta_consumption <- coefs["consumption_pd"]
beta_days_over_30C <- coefs["days_over_30C"]
beta_total_precipitation <- coefs["total_precipitation"]
beta_sex_female <- coefs["sex_idFemale"]
beta_birth_year <- coefs["birth_year"]

ranef_df <- as.data.frame(lme4::ranef(model)$ihme_loc_id)
ranef_df$ihme_loc_id <- rownames(lme4::ranef(model)$ihme_loc_id)
colnames(ranef_df)[1] <- "random_intercept"

df_model <- merge(df_model, ranef_df, by = "ihme_loc_id", all.x = TRUE)

df_model$linear_pred_fe <- (
  beta_intercept +
    beta_consumption * df_model$consumption_pd +
    beta_days_over_30C * df_model$days_over_30C +
    beta_total_precipitation * df_model$total_precipitation +
    beta_sex_female * df_model$sex_id +  
    beta_birth_year * df_model$birth_year
)

df_model$linear_pred_me <- (
  beta_intercept +
    beta_consumption * df_model$consumption_pd +
    beta_days_over_30C * df_model$days_over_30C +
    beta_total_precipitation * df_model$total_precipitation +
    beta_sex_female * df_model$sex_id + 
    beta_birth_year * df_model$birth_year +
    df_model$random_intercept
)


df_model$pred_fe <- 1 / (1 + exp(-df_model$linear_pred_fe))
df_model$pred_me <- 1 / (1 + exp(-df_model$linear_pred_me))

# Save predictions to parquet
write_parquet(df_model, paste0(neonatal_dir, "predictions_", summary_file, "_means.parquet"))
paste0(neonatal_dir, "predictions_", summary_file, "_means.parquet")

# 10/29 - data with censored survivors filtered as well as 7 year cutoff
# normal subset
# summary_file <- "cm_v7_subset"
# 
# model <- readRDS(file = paste0(results_dir, summary_file,".rds"))
# 
# df_model <- read_parquet(paste0(results_dir,"predictions_",summary_file,".parquet"))
# 
# baseline_hazard <- model$hazard  
# baseline_hazard <- data.frame(
#   time = model$tev,
#   hazard = baseline_hazard
# )
# 
# baseline_hazard <- baseline_hazard[order(baseline_hazard$time), ]
# baseline_hazard$cumhazard <- cumsum(baseline_hazard$hazard)
# 
# # Extract frailty estimates for each cluster (ihme_loc_id)
# frailty_effects <- model$frail
# 
# # Create frailty lookup data frame
# frailty_df <- data.frame(
#   ihme_loc_id = names(frailty_effects),
#   frailty = as.numeric(frailty_effects)
# )
# 
# setorder(frailty_df,frailty)
# 
# summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
# options(width=1000)
# capture.output(summary(model, width=1000), file = summary_file_path)
# 
# # Append frailty estimates
# cat("\n\n", file = summary_file_path, append = TRUE)
# cat("================================================================================\n", 
#     file = summary_file_path, append = TRUE)
# cat("CLUSTER-SPECIFIC FRAILTY ESTIMATES (RANDOM EFFECTS)\n", 
#     file = summary_file_path, append = TRUE)
# cat("================================================================================\n\n", 
#     file = summary_file_path, append = TRUE)
# frailty_output <- capture.output(print(frailty_df, row.names = FALSE))
# cat(paste(frailty_output, collapse = "\n"), file = summary_file_path, append = TRUE)
# 
# baseline_hazard_normal <- copy(baseline_hazard)
# 
# # Function to get cumulative baseline hazard at a given time
# get_cumhaz_baseline <- function(time, basehaz_df) {
#   if (time <= 0) return(0)
#   idx <- max(which(basehaz_df$time <= time))
#   if (length(idx) == 0 || idx == 0) return(0)
#   return(basehaz_df$cumhazard[idx])
# }
# 
# # store results in data table with age_month, avg actual mortality,
# # avg cum mortality probability, and avg point mortality probability
# probs_dt <- data.table(
#   age_month = seq(1,60,1),
#   avg_mortality = rep(NA_real_,60),
#   avg_probs_me = rep(NA_real_,60),
#   avg_probs_fe = rep(NA_real_,60),
#   avg_cum_probs_me = rep(NA_real_,60),
#   avg_cum_probs_fe = rep(NA_real_,60),
#   avg_mortality_alt = rep(NA_real_,60)
# )
# 
# 
# for (i in seq_along(probs_dt$age_month)){
#   month <- probs_dt$age_month[i]
#   
#   # get avg mortality
#   numerator <- nrow(df_model[(age_month==month)&(child_mortality==1)])
#   denominator <- nrow(df_model[age_month>=month])
#   
#   denominator_alt <- nrow(df_model[(age_month>=month)|(child_mortality==0)])
#   
#   probs_dt[age_month==month,avg_mortality:=numerator/denominator]
#   probs_dt[age_month==month,avg_mortality_alt:=numerator/denominator_alt]
#   
#   # get avg prob of mortality for that point in time
#   df_tmp <- copy(df_model)
#   df_tmp$hazard <- NULL
#   df_tmp[,age_month := month]
#   setDT(baseline_hazard)
#   baseline_merge <- baseline_hazard[, .(time, hazard)]
#   setnames(baseline_merge,old="time",new="age_month")
#   df_tmp <- merge(df_tmp, baseline_merge,
#                   by = "age_month", all.x = TRUE)
#   
#   # Calculate point hazard for each observation
#   df_tmp$hazard_point_me <- df_tmp$frailty * df_tmp$hazard * exp(df_tmp$linear_pred)
#   df_tmp$hazard_point_fe <- df_tmp$hazard * exp(df_tmp$linear_pred)
#   
#   # Convert to point mortality probability (probability of dying in that month)
#   df_tmp$mortality_point_me <- 1 - exp(-df_tmp$hazard_point_me)
#   df_tmp$mortality_point_fe <- 1 - exp(-df_tmp$hazard_point_fe)
#   
#   probs_dt[age_month==month,avg_probs_me:=mean(df_tmp$mortality_point_me)]
#   probs_dt[age_month==month,avg_probs_fe:=mean(df_tmp$mortality_point_fe)]
#   
#   # get avg cumulative prob of mortality
#   df_tmp$cumhaz_baseline <- sapply(df_tmp$age_month, function(t) {
#     get_cumhaz_baseline(t, baseline_hazard)
#   })
#   
#   df_tmp$cumhaz_me <- df_tmp$frailty *
#     df_tmp$cumhaz_baseline *
#     exp(df_tmp$linear_pred)
#   
#   df_tmp$cumhaz_fe <- df_tmp$cumhaz_baseline *
#     exp(df_tmp$linear_pred)
#   
#   # Survival probability = exp(-cumulative hazard)
#   df_tmp$survival_me <- exp(-df_tmp$cumhaz_me)
#   
#   # Mortality probability = 1 - survival
#   df_tmp$mortality_me <- 1 - df_tmp$survival_me
#   
#   # Survival probability = exp(-cumulative hazard)
#   df_tmp$survival_fe <- exp(-df_tmp$cumhaz_fe)
#   
#   # Mortality probability = 1 - survival
#   df_tmp$mortality_fe <- 1 - df_tmp$survival_fe
#   
#   probs_dt[age_month==month,avg_cum_probs_me:=mean(df_tmp$mortality_me)]
#   probs_dt[age_month==month,avg_cum_probs_fe:=mean(df_tmp$mortality_fe)]
#   
# }
# 
# write.csv(probs_dt,paste0(model_summary_dir,"avg_prob_table_subset.csv"),row.names = FALSE)
# 
# 
# probs_dt_normal <- fread(paste0(model_summary_dir,"avg_prob_table_subset.csv"))
# 
# 
# # filtered data
# summary_file <- "cm_v7_filtered"
# 
# model <- readRDS(file = paste0(results_dir, summary_file,".rds"))
# 
# df_model <- read_parquet(paste0(results_dir,"predictions_",summary_file,".parquet"))
# 
# # what is the avg effect of the random effects?
# mean(df_model$frailty) # 1.070184... so me >fe
# 
# baseline_hazard <- model$hazard  
# baseline_hazard <- data.frame(
#   time = model$tev,
#   hazard = baseline_hazard
# )
# 
# baseline_hazard <- baseline_hazard[order(baseline_hazard$time), ]
# baseline_hazard$cumhazard <- cumsum(baseline_hazard$hazard)
# 
# baseline_hazard_filtered <- copy(baseline_hazard)

################################################################################
# Plot survival curve for an individual that includes linear effects. Separate
# one that also includes frailty
# Extract fixed effect coefficients
coefs <- coef(model)
beta_consumption <- coefs["consumption_pd"]
beta_total_precipitation <- coefs["total_precipitation"]
beta_days_over_30C <- coefs["days_over_30C"]
beta_sex_female <- coefs["sex_idFemale"]
beta_birth_year <- coefs["birth_year"]

# Create frailty lookup data frame
frailty_effects <- model$frail
frailty_df <- data.frame(
  ihme_loc_id = names(frailty_effects),
  frailty = as.numeric(frailty_effects)
)
baseline_hazard <- model$hazard  
baseline_hazard <- data.frame(
  time = model$tev,
  hazard = baseline_hazard
)
baseline_hazard <- baseline_hazard[order(baseline_hazard$time), ]
baseline_hazard$cumhazard <- cumsum(baseline_hazard$hazard)

df_test <- copy(df_model)
df_test$cumhaz_me
df_test$frailty
df_test$linear_pred
df_test$
df_test <- merge(df_model, frailty_df, by = "ihme_loc_id", all.x = TRUE)

# Calculate linear predictor (fixed effects only)
df_model$linear_pred <- (
  beta_consumption * df_model$consumption_pd +
    beta_days_over_30C * df_model$days_over_30C +
    beta_total_precipitation * df$total_precipitation + 
    beta_sex_female * (df_model$sex_id == "Female")+
    beta_birth_year * (df_model$birth_year)
)
df_model$cumhaz_me <- df_model$frailty * 
  df_model$cumhaz_baseline * 
  exp(df_model$linear_pred)


# Survival probability = exp(-cumulative hazard)
df_model$survival_me <- exp(-df_model$cumhaz_me)

# Mortality probability = 1 - survival
df_model$mortality_me <- 1 - df_model$survival_me

################################################################################

# Function to get cumulative baseline hazard at a given time
get_cumhaz_baseline <- function(time, basehaz_df) {
  if (time <= 0) return(0)
  idx <- max(which(basehaz_df$time <= time))
  if (length(idx) == 0 || idx == 0) return(0)
  return(basehaz_df$cumhazard[idx])
}

# store results in data table with age_month, avg actual mortality,
# avg cum mortality probability, and avg point mortality probability
probs_dt <- data.table(
  age_month = seq(1,60,1),
  avg_mortality = rep(NA_real_,60),
  avg_probs_me = rep(NA_real_,60),
  avg_probs_fe = rep(NA_real_,60),
  avg_cum_probs_me = rep(NA_real_,60),
  avg_cum_probs_fe = rep(NA_real_,60),
  avg_mortality_alt = rep(NA_real_,60)
)


for (i in seq_along(probs_dt$age_month)){
  month <- probs_dt$age_month[i]
  
  # get avg mortality
  numerator <- nrow(df_model[(age_month==month)&(child_mortality==1)])
  denominator <- nrow(df_model[age_month>=month])
  
  denominator_alt <- nrow(df_model[(age_month>=month)|(child_mortality==0)])
  
  probs_dt[age_month==month,avg_mortality:=numerator/denominator]
  probs_dt[age_month==month,avg_mortality_alt:=numerator/denominator_alt]
  
  # get avg prob of mortality for that point in time
  df_tmp <- copy(df_model)
  df_tmp$hazard <- NULL
  df_tmp[,age_month := month]
  setDT(baseline_hazard)
  baseline_merge <- baseline_hazard[, .(time, hazard)]
  setnames(baseline_merge,old="time",new="age_month")
  df_tmp <- merge(df_tmp, baseline_merge,
                  by = "age_month", all.x = TRUE)
  
  # Calculate point hazard for each observation
  df_tmp$hazard_point_me <- df_tmp$frailty * df_tmp$hazard * exp(df_tmp$linear_pred)
  df_tmp$hazard_point_fe <- df_tmp$hazard * exp(df_tmp$linear_pred)
  
  # Convert to point mortality probability (probability of dying in that month)
  df_tmp$mortality_point_me <- 1 - exp(-df_tmp$hazard_point_me)
  df_tmp$mortality_point_fe <- 1 - exp(-df_tmp$hazard_point_fe)
  
  probs_dt[age_month==month,avg_probs_me:=mean(df_tmp$mortality_point_me)]
  probs_dt[age_month==month,avg_probs_fe:=mean(df_tmp$mortality_point_fe)]
  
  # get avg cumulative prob of mortality
  df_tmp$cumhaz_baseline <- sapply(df_tmp$age_month, function(t) {
    get_cumhaz_baseline(t, baseline_hazard)
  })
  
  df_tmp$cumhaz_me <- df_tmp$frailty *
    df_tmp$cumhaz_baseline *
    exp(df_tmp$linear_pred)
  
  df_tmp$cumhaz_fe <- df_tmp$cumhaz_baseline *
    exp(df_tmp$linear_pred)
  
  # Survival probability = exp(-cumulative hazard)
  df_tmp$survival_me <- exp(-df_tmp$cumhaz_me)
  
  # Mortality probability = 1 - survival
  df_tmp$mortality_me <- 1 - df_tmp$survival_me
  
  # Survival probability = exp(-cumulative hazard)
  df_tmp$survival_fe <- exp(-df_tmp$cumhaz_fe)
  
  # Mortality probability = 1 - survival
  df_tmp$mortality_fe <- 1 - df_tmp$survival_fe
  
  probs_dt[age_month==month,avg_cum_probs_me:=mean(df_tmp$mortality_me)]
  probs_dt[age_month==month,avg_cum_probs_fe:=mean(df_tmp$mortality_fe)]
  
}

write.csv(probs_dt,paste0(model_summary_dir,"avg_prob_table_filtered.csv"),row.names = FALSE)

probs_dt <- fread(paste0(model_summary_dir,"avg_prob_table_filtered.csv"))
probs_dt_filtered <- copy(probs_dt)

# observation: on filtered data avg predicted prob is more often higher than 
# actual mortality rate, whereas the opposite is the case for unfiltered
probs_dt_filtered[avg_probs_me>avg_mortality,.N] # 57
probs_dt_normal[avg_probs_me>avg_mortality,.N] # 21

probs_dt_filtered[avg_probs_fe>avg_mortality,.N] # 0
probs_dt_normal[avg_probs_fe>avg_mortality,.N] # 11


# Look at results from yearly model
model_yr <- readRDS(file = paste0(results_dir, "cm_v7_subset_yearly",".rds"))
probs_yr <- fread(paste0(model_summary_dir,"avg_prob_table_yearly.csv"))



# 10/17 - using manual me and fe predictions

# 10/16 10% model on latest data update with all survivors coded at 60 months
# model_name <- "subset_10pct_model_do30"
# model <- readRDS(paste0(results_dir,model_name,".rds"))
# model_pred <- fread(paste0(results_dir,"predictions_",model_name,".csv"))
# 
# frailty_effects <- model$frail
# print(frailty_effects)


# 10/15 - Collapsing data to have average climate vars per child, 25%
# model_name <- "25pct_model_do30"

# 25% data model
# model <- readRDS(paste0(results_dir,"subset_",model_name,".rds"))
# model_pred <- fread(paste0(results_dir,"predictions_subset_",model_name,".csv"))

# Get 1 month predictions

# # # get predictions of model over data set
# pred_surv <- predict(model, neo_df, quantity = "survival")
# 
# # get predictions at 1 month
# surv_at_1_mo <- mapply(function(df, t) {
#   idx <- max(which(df$time <= t))
#   df$survival[idx]
# }, pred_surv, 1/12)
# 
# neo_df$mortality_1_mo <- 1-surv_at_1_mo
# 
# # save out for heat maps
# write_parquet(neo_df,paste0(neonatal_dir,"neonatal_mortality_",model_name,".parquet"))
# 
# # compare re vs fe for unseen data
# # Predict model on non-included data
# # test_df <- df[!(indv_id %in% unique(model_pred$indv_id))]
# test_df <- df
# 
# # mixed effects predictions
# pred_surv_me <- predict(model, test_df, quantity = "survival",type="conditional")  # make mixed effects predictions
# surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
#   surv_df <- pred_surv_me[[i]]
#   obs_time <- test_df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# test_df$model_predictions_me <- 1-surv_at_obs_time
# 
# # fixed effects predictions
# # pred_surv_fe <- predict(model, test_df,re.form = ~0, quantity = "survival") # make fixed effects predictions
# pred_surv_fe <- predict(model, test_df,quantity = "survival",type="marginal")
# surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
#   surv_df <- pred_surv_fe[[i]]
#   obs_time <- test_df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# test_df$model_predictions_fe <- 1-surv_at_obs_time
# 
# write_parquet(test_df,paste0(results_dir,"test_set_predictions_me_fe_",model_name,".parquet"))
# 
# # plot survival curves in next section
# pred_surv <-pred_surv_me


# # 10/14 - Collapsing data to have average climate vars per child
# model_name <- "05pct_model_do30"
# 
# # 5% data model
# model <- readRDS(paste0(results_dir,"subset_",model_name,".rds"))
# model_pred <- fread(paste0(results_dir,"predictions_subset_",model_name,".csv"))
# 
# # compare re vs fe for unseen data
# # Predict model on non-included data
# test_df <- df[!(indv_id %in% unique(model_pred$indv_id))]
# 
# # mixed effects predictions
# pred_surv_me <- predict(model, test_df, quantity = "survival")
# surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
#   surv_df <- pred_surv_me[[i]]
#   obs_time <- test_df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# test_df$model_predictions_me <- 1-surv_at_obs_time
# 
# # fixed effects predictions
# pred_surv_fe <- predict(model, test_df,re.form = ~0, quantity = "survival") # make fixed effects predictions
# surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
#   surv_df <- pred_surv_fe[[i]]
#   obs_time <- test_df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# test_df$model_predictions_fe <- 1-surv_at_obs_time
# 
# write_parquet(test_df,paste0(results_dir,"test_set_predictions_me_fe_",model_name,".parquet"))
# 
# 
# # plot survival curves in next section
# pred_surv <-pred_surv_me

# ## First successful run of full data
# model_baseline <- readRDS(paste0(results_dir,"baseline_model_object.rds"))
# summary(model_baseline)
# 
# 
# ## 50% data with days_over_30
# model_50_pc <- readRDS(paste0(results_dir,"subset_5pct_model_do30_object.rds"))
# summary(model_50_pc)
# 
# # Predict model on non-included data
# model_50_pred <- fread(paste0(results_dir,"subset_5pct_model_do30_results.csv"))
# test_df <- df[!(indv_id %in% unique(model_50_pred$indv_id))]
# 
# # mixed effects predictions
# pred_surv_me <- predict(model_50_pc, test_df, quantity = "survival")
# surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
#   surv_df <- pred_surv_me[[i]]
#   obs_time <- test_df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# test_df$model_predictions_me <- 1-surv_at_obs_time
# 
# # fixed effects predictions
# pred_surv_fe <- predict(model_50_pc, test_df,re.form = ~0, quantity = "survival") # make fixed effects predictions
# surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
#   surv_df <- pred_surv_fe[[i]]
#   obs_time <- test_df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# test_df$model_predictions_fe <- 1-surv_at_obs_time
# 
# write_parquet(test_df,paste0(results_dir,"test_set_predictions_50pc_do30_fe.parquet"))
# 
# # get predictions of model over data set
# pred_surv <- predict(model_50_pc, df_min_age, quantity = "survival")
# # pred_surv <- predict(model_50_pc, df,re.form = ~0, quantity = "survival") # make fixed effects predictions
# 
# # get predictions at 1 month
# surv_at_1_mo <- mapply(function(df, t) {
#   idx <- max(which(df$time <= t))
#   df$survival[idx]
# }, pred_surv, 1/12)
# 
# surv_under_1_mo <- mapply(function(df, t) {
#   idx <- min(which(df$time <= t))
#   df$survival[idx]
# }, pred_surv, 1/12)
# 
# df_min_age$mortality_1_mo <- 1-surv_at_1_mo
# df_min_age$mortality_under_1_mo <- 1-surv_under_1_mo
# 
# # save out for heat maps
# write_parquet(df_min_age,paste0(neonatal_dir,"neonatal_mortality_1_mo.parquet"))
# 
# ## 50% data with days_over_30 only
# model_do30_50pc <- readRDS(paste0(results_dir,"subset_5pct_model_do30_object.rds"))
# summary(model_do30_50pc)
# summary_file <- "subset_50pct_model_do30_object.txt"
# capture.output(summary(model_do30_50pc), file = paste0(model_summary_dir,summary_file))
# 
# # get mixed effects and fixed effects predictions over raw data
# pred_surv_me <- predict(model_50_pc, df, quantity = "survival")
# surv_at_obs_time <- sapply(seq_len(nrow(df)), function(i) {
#   surv_df <- pred_surv_me[[i]]
#   obs_time <- df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# df$model_predictions_me <- 1-surv_at_obs_time
# 
# write_parquet(df,paste0(results_dir,"predictions_50pc_do30_me.parquet"))
# 
# pred_surv_fe <- predict(model_50_pc, df,re.form = ~0, quantity = "survival") # make fixed effects predictions
# surv_at_obs_time <- sapply(seq_len(nrow(df)), function(i) {
#   surv_df <- pred_surv_fe[[i]]
#   obs_time <- df$age_year_at_year_end[i]
#   idx <- max(which(surv_df$time <= obs_time))
#   surv_df$survival[idx]
# })
# 
# df$model_predictions_fe <- 1-surv_at_obs_time
# 
# write_parquet(df,paste0(results_dir,"predictions_50pc_do30_fe.parquet"))
# 
# # Model diagnostics
# mean_observed <- mean(df$child_mortality)
# mean_predicted <- mean(df$model_predictions_me)
# print(paste("Observed mean mortality:", round(mean_observed, 4)))
# print(paste("Predicted mean mortality:", round(mean_predicted, 4)))
# hist(df$child_mortality, breaks=20, main="Observed Mortality", xlab="Mortality")
# hist(df$model_predictions_me, breaks=20, main="Predicted Mortality", xlab="Predicted")
# df$pred_bin <- cut(df$model_predictions_me, breaks=seq(0,1,by=0.05))
# calib <- df[, .(obs_rate = mean(child_mortality), pred_rate = mean(model_predictions_me)), by=pred_bin]
# ggplot(calib, aes(x=pred_rate, y=obs_rate)) +
#   geom_point() +
#   geom_abline(slope=1, intercept=0, linetype="dashed", color="red") +
#   labs(x="Predicted Rate", y="Observed Rate", title="Calibration Plot")
# 
# by_country <- df[, .(obs_rate = mean(child_mortality), pred_rate = mean(model_predictions_me)), by=ihme_loc_id]
# ggplot(by_country, aes(x = obs_rate, y = pred_rate)) +
#   geom_point() +
#   geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
#   labs(x = "Observed Rate", y = "Predicted Rate", title = "Country-level Calibration")
# 
# ## 25% data with days_over_30 only
# model_do30_25pc <- readRDS(paste0(results_dir,"subset_25pct_model_do30_object.rds"))
# summary(model_do30_25pc)
# summary_file <- "subset_25pct_model_do30_object.txt"
# capture.output(summary(model_do30_25pc), file = paste0(model_summary_dir,summary_file))
# 
# # predictions
# pred <- fread(paste0(results_dir,"subset_25pct_model_do30_results.csv"))
# 
# # get predictions of model over data set
# pred_surv <- predict(model_do30_25pc, df_min_age, quantity = "survival")
# # pred_surv_fe <- predict(model_50_pc, df,re.form = ~0, quantity = "survival") # make fixed effects predictions
# 
# # get predictions at 1 month
# surv_at_1_mo <- mapply(function(df, t) {
#   idx <- max(which(df$time <= t))
#   df$survival[idx]
# }, pred_surv, 1/12)
# 
# df_min_age$mortality_1_mo_do30 <- 1-surv_at_1_mo
# 
# # save out for heat maps
# write_parquet(df_min_age,paste0(neonatal_dir,"neonatal_mortality_1_mo_do30.parquet"))



#==============================================================================
# SECTION 3: MAKE PLOTS
#==============================================================================

library(survival)
coxph_model <- coxph(Surv(age_month, child_mortality) ~ 1, data = df_model)
basehaz <- basehaz(coxph_model, centered = FALSE)
plot(basehaz$time, basehaz$hazard, type = "l")

# hist(exp(df_model$linear_pred), breaks = 50, main = "Distribution of exp(linear_pred)", xlab = "exp(linear_pred)")
hist(df_model$linear_pred, breaks = 50, main = "Distribution of linear_pred", xlab = "linear_pred")
summary(exp(df_model$linear_pred))

indiv_row <- df_model[1, ]
pred_surv <- predict(model, newdata = indiv_row, quantity = "survival")
ggplot(pred_surv, aes(x = time, y = survival)) +
  geom_line() +
  labs(title = "Predicted Survival Curve for Individual 1",
       x = "Time",
       y = "Survival Probability") +
  theme_minimal()

# Plot random effects
country_effects_plot <- autoplot(model, type = "frail")
country_effects_plot <- country_effects_plot +
  labs(title="Z-scores of country effects")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))
ggsave(paste0(plot_dir, "country_effects_z_scores",model_name,".png"), plot = country_effects_plot, width = 8, height = 5)

# # Plot survival curves together
# dt_list <- lapply(seq_along(pred_surv), function(i) {
#   dt <- as.data.table(pred_surv[[i]])
#   dt[, obs_id := i]  # Add observation ID
#   dt
# })
# 
# # Combine all into one data.table
# all_surv <- rbindlist(dt_list)
# 
# p <- ggplot(all_surv, aes(x = time, y = survival)) +
#   geom_point(alpha = 0.3) +
#   labs(title = "Predicted Survival Curves from Model",
#        x = "Age in Years",
#        y = "Survival Probability") +
#   theme_minimal() +
#   theme(plot.background = element_rect(fill = "white", color = NA),
#         panel.background = element_rect(fill = "white", color = NA))
# 
# ggsave(paste0(plot_dir, "survival_curves_",model_name,".png"), plot = p, width = 8, height = 5)
