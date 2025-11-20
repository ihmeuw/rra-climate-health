################################################################################
# DESCRIPTION: Script to run baseline model on neonatal mortality data, first 
# using a logistic regression
# model <- glmer(
#   child_mortality ~ consumption_pd +
#     days_over_30C +
#     total_precipitation + 
#     sex_id +
#     birth_year +
#     (1 | ihme_loc_id),
#   data = df_model,
#   family = binomial(link = "logit")
# )
# PROJECT: Climate nutrition
# DATE: 2025-10-23
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
# library(frailtyEM,lib.loc = "/homes/elyeb/rlibs")
library(data.table)
library(lme4)
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- "nm_v7_factored_yr"


results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_24.01/"

neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_24.01/neonatal_data.parquet"


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)


neonatal_dir <- paste0(results_dir,"neonatal/")
dir.create(neonatal_dir, recursive = TRUE, showWarnings = FALSE)

model_summary_dir <- paste0(neonatal_dir,"model_summaries/")
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

model_objects_dir <- paste0(neonatal_dir,"model_objects/")
dir.create(model_objects_dir, recursive = TRUE, showWarnings = FALSE)

# Read in neonatal df (must be made from full dataset)
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)

# only keep age_month 1
neo_df <- neo_df[age_month==1]

neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
neo_df[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]
neo_df[,birth_year:=as.factor(birth_year)]

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
df_model <- neo_df[, ..cols]

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================


model <- glmer(
  child_mortality ~ consumption_pd +
    days_over_30C +
    total_precipitation +
    sex_id +
    birth_year +
    (1 | ihme_loc_id),
  data = df_model,
  family = binomial(link = "logit")
)


# save model parameters for future use:
saveRDS(model, file = paste0(model_objects_dir, summary_file,".rds"))

# Read model back in
# model = readRDS(file = paste0(model_objects_dir, summary_file,".rds"))

summary(model)

# Extract random effects 
re_df <- as.data.frame(ranef(model)$ihme_loc_id)
re_df$ihme_loc_id <- rownames(ranef(model)$ihme_loc_id)
colnames(re_df)[1] <- "random_effects"
setorder(re_df,random_effects)

# Save model summary and random effects to text file
summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
capture.output(summary(model), file = summary_file_path)
cat("\n\n", file = summary_file_path, append = TRUE)
cat("================================================================================\n", file = summary_file_path, append = TRUE)
cat("CLUSTER-SPECIFIC RANDOM EFFECTS ESTIMATES\n", file = summary_file_path, append = TRUE)
cat("================================================================================\n\n", file = summary_file_path, append = TRUE)
re_output <- capture.output(print(re_df, row.names = FALSE))
cat(paste(re_output, collapse = "\n"), file = summary_file_path, append = TRUE)

# Also save frailty estimates as a separate CSV for easier access
write.csv(re_df, paste0(model_summary_dir, "re_estimates_", summary_file, ".csv"), row.names = FALSE)

#==============================================================================
# SECTION 3: PREDICT MODEL FOR NEONATAL
#==============================================================================

# # Predict WITHOUT random effects (fixed effects only)
# df_model$pred_fe <- predict(model, newdata = df_model, type = "response", re.form = NA)
# 
# # Predict WITH random effects (mixed effects)
# df_model$pred_me <- predict(model, newdata = df_model, type = "response", re.form = NULL)
# 
# # Save predictions to parquet
# write_parquet(df_model, paste0(neonatal_dir, "predictions_", summary_file, ".parquet"))
