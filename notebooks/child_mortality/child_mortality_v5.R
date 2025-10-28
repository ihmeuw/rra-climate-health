################################################################################
# DESCRIPTION: Script to run baseline model on child mortality on subset of data.
# model <- emfrail(Surv(age_month, child_mortality) ~ consumption_pd + 
#                    days_over_30C + 
#                    any_days_over_30C*consumption_pd +
#                    sex_id + 
#                    birth_year + 
#                    survival::cluster(ihme_loc_id), 
#                  data = df_model,
#                  verbose = TRUE)
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
summary_file <- "cm_v5"

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/data.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/"
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
  "days_over_30C",
  "days_over_26C",
  "any_days_over_30C"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year","int_birth_year_diff_months", climate_vars)
df_model <- df[, ..cols]
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]


df_model <- data.table(df_model)

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================

# tmp override: 
# df_model <- df_model

# fit model with days_over_30C, days_over_30C*consumption, and birth_year
model <- emfrail(Surv(age_month, child_mortality) ~ consumption_pd + 
                   days_over_30C + 
                   any_days_over_30C:consumption_pd +
                   sex_id + 
                   birth_year + 
                   survival::cluster(ihme_loc_id), 
                 data = df_model,
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

# model = readRDS(file = paste0(results_dir, summary_file,".rds"))
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
beta_consumption <- coefs["consumption_pd"]
beta_days_over_30C <- coefs["days_over_30C"]
beta_sex_female <- coefs["sex_idFemale"]
beta_birth_year <- coefs["birth_year"]
beta_interaction <- coefs["consumption_pd:any_days_over_30C"]


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

# Calculate linear predictor (fixed effects only)
df_model$linear_pred <- (
  beta_consumption * df_model$consumption_pd +
    beta_days_over_30C * df_model$days_over_30C +
    beta_interaction * (df_model$consumption_pd*df_model$any_days_over_30C) +
    beta_sex_female * (df_model$sex_id == "Female")+
    beta_birth_year * (df_model$birth_year)
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
write_parquet(df_model,paste0(results_dir,"predictions_",summary_file,".parquet"))
print(paste0("child mortality predictions saved to ",paste0(results_dir,"predictions_",summary_file,".parquet")))

