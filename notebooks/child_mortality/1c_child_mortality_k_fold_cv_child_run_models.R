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

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_08.01/data.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"



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

# save model summary
summary(model)
summary_file <- paste0("model_summary_vars_",climate_var_1)
if (climate_var_2!=""){
  summary_file <- paste0(summary_file,"_",climate_var_2)
}
summary_file <- paste0(summary_file,"_fold_",fold_number,".txt")

capture.output(summary(model), file = paste0(model_summary_dir,summary_file))

# get predictions on test set
pred_out <- predict(model, test, re.form = ~0, quantity="survival")

surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, test$age_year_at_year_end)

test$model_predictions <- 1-surv_at_obs_time
print("predictions done")

outfile <- paste0("predictions_fold_",fold_number,"_vars_",climate_var_1)
if (climate_var_2!=""){
  outfile <- paste0(outfile,"_",climate_var_2)
}
outfile <- paste0(outfile,".parquet")

write_parquet(test,paste0(folds_results_subdir,outfile))