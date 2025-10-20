################################################################################
# DESCRIPTION: Child script to run a single fold in a k-fold cross-validation
# task, for a specific model specification
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

# install.packages('frailtyEM',lib = "/homes/elyeb/rlibs") # able to handle mixed effects and predict on new data
library(frailtyEM,lib.loc = "/homes/elyeb/rlibs")
library(data.table)
library(caret) # for createFolds function
library(dplyr) # for anti_join function
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

# Get fold to run model on, and get variables for model specification
args <- commandArgs(trailingOnly = TRUE)

fold_file <- args[1]
climate_var_1 <- args[2]
print(climate_var_1)
climate_var_2 <- if (length(args) >= 3) args[3] else "" # optional secondary var
print(climate_var_2)

# fold_file <- commandArgs()[4]
# climate_var_1 <- commandArgs()[5]
# climate_var_2 <- if (length(args) >= 6) args[6] else "" 

# fold_file <- "fold_indices_1.rds"
# climate_var_1 <- "mean_temperature"
# climate_var_2 <- "days_over_30C"

print(paste0("running on fold file ",fold_file))

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_16.01/data.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_16.01/"



folds_dir <- paste0(results_dir,"folds/")
folds_data_subdir <- paste0(folds_dir,"data_subsets/")
folds_results_subdir <- paste0(folds_dir,"folds_results/")
model_summary_dir <- paste0(folds_dir,"model_summaries/")
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

fold_indices <- readRDS(paste0(folds_dir,fold_file))
fold_number <- as.integer(gsub(".*_(\\d+)\\.rds$", "\\1", fold_file))

df <- read_parquet(paste0(folds_data_subdir,"df_with_folds.parquet"))
df <- data.table(df)

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df[,child_mortality := 1-child_alive]
df[,ihme_loc_id:=as.factor(ihme_loc_id)]
df[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]

setnames(df,old="ldipc_weighted_no_match",new="consumption")

#==============================================================================
# SECTION 2: FIT MODELS
#==============================================================================

## Test secondary climate variables with k-fold cross-validation

test <- df[fold_indices]
train <- anti_join(df, test)

# fit model 
base_formula <- "Surv(age_year_at_year_end, child_mortality) ~ consumption + sex_id + survival::cluster(ihme_loc_id) + "
full_formula_string <- paste0(base_formula,climate_var_1)
if (climate_var_2!=""){
  full_formula_string <- paste0(full_formula_string," + ",climate_var_2)
}

print("model specification:")
print(full_formula_string)

full_formula <- as.formula(full_formula_string)

# free up space:
rm(df)

model <- emfrail(formula = full_formula, data = train, verbose = TRUE)

# save model 
summary(model)
summary_file <- paste0("model_summary_vars_",climate_var_1)
if (climate_var_2!=""){
  summary_file <- paste0(summary_file,"_",climate_var_2)
}
summary_file <- paste0(summary_file,"_fold_",fold_number,".txt")

# save model parameters for future use:
saveRDS(model, file = paste0(model_summary_dir, summary_file,".rds"))

# save model summary

# Extract frailty estimates for each cluster (ihme_loc_id)
frailty_effects <- model$frail
# Create frailty lookup data frame
frailty_df <- data.frame(
  ihme_loc_id = names(frailty_effects),
  frailty = as.numeric(frailty_effects)
)

setorder(frailty_df,frailty)

# save model summary with random effects:
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

#==============================================================================
# SECTION 3: PREDICT MODEL ON TEST SET
#==============================================================================

# Extract fixed effect coefficients
coefs <- coef(model)
beta_consumption <- coefs["consumption"]
beta_sex_female <- coefs["sex_idFemale"]

# Dynamically extract climate variable coefficients
climate_vars <- c(climate_var_1, climate_var_2)
climate_vars <- climate_vars[climate_vars != ""] # Remove empty string if no secondary var
# Get coefficients for climate variables
beta_climate <- coefs[climate_vars]

# Calculate linear predictor (fixed effects only)
test$linear_pred <- beta_consumption * test$consumption +
  beta_sex_female * (test$sex_id == "Female")

# Add climate variables to linear predictor
for (i in seq_along(climate_vars)) {
  var <- climate_vars[i]
  beta <- beta_climate[i]
  test$linear_pred <- test$linear_pred + beta * test[[var]]
}

# Extract baseline hazard - Note this is only as long as unique months in which
# someone died.
baseline_hazard <- model$hazard  # This contains time and cumulative baseline hazard
baseline_hazard <- data.frame(
  time = model$tev,
  hazard = baseline_hazard
)

baseline_hazard <- baseline_hazard[order(baseline_hazard$time), ]
baseline_hazard$cumhazard <- cumsum(baseline_hazard$hazard)

# Function to get cumulative baseline hazard at a given time
get_cumhaz_baseline <- function(time, basehaz_df) {
  if (time <= 0) return(0)
  idx <- max(which(basehaz_df$time <= time))
  if (length(idx) == 0 || idx == 0) return(0)
  return(basehaz_df$cumhazard[idx])
}

# Calculate cumulative baseline hazard at each observation time
test$cumhaz_baseline <- sapply(test$age_month, function(t) {
  get_cumhaz_baseline(t, baseline_hazard)
})

# Merge frailty estimates on data
test <- merge(test, frailty_df, by = "ihme_loc_id", all.x = TRUE)

# MANUAL PREDICTION WITH RANDOM EFFECTS (Mixed Effects)
# Formula: H(t|X,Z) = Z * H0(t) * exp(X'β)
# where Z is the frailty for that cluster
test$cumhaz_me <- test$frailty * 
  test$cumhaz_baseline * 
  exp(test$linear_pred)

# Survival probability = exp(-cumulative hazard)
test$survival_me <- exp(-test$cumhaz_me)

# Mortality probability = 1 - survival
test$mortality_me <- 1 - test$survival_me


# MANUAL PREDICTION WITHOUT RANDOM EFFECTS (Fixed Effects Only)
# Formula: H(t|X) = H0(t) * exp(X'β)
# Equivalent to setting frailty Z = 1 (or E[Z] = 1)
test$cumhaz_fe <- test$cumhaz_baseline * 
  exp(test$linear_pred)

# Survival probability = exp(-cumulative hazard)
test$survival_fe <- exp(-test$cumhaz_fe)

# Mortality probability = 1 - survival
test$mortality_fe <- 1 - test$survival_fe

# Legacy predictions on test set using package pred function.
# pred_out <- predict(model, test, re.form = ~0, quantity="survival")
# 
# surv_at_obs_time <- mapply(function(df, t) {
#   idx <- max(which(df$time <= t))
#   df$survival[idx]
# }, pred_out, test$age_year_at_year_end)
# 
# test$model_predictions <- 1-surv_at_obs_time

print("predictions done")

outfile <- paste0("predictions_fold_",fold_number,"_vars_",climate_var_1)
if (climate_var_2!=""){
  outfile <- paste0(outfile,"_",climate_var_2)
}
outfile <- paste0(outfile,".parquet")

write_parquet(test,paste0(folds_results_subdir,outfile))