################################################################################
# DESCRIPTION: Script to run baseline model on neonatal mortality data, first 
# using a logistic regression
# PROJECT: Climate nutrition
# DATE: 2025-12-10
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
zone_no <- commandArgs()[4]
summary_file <- paste0("nnm_6_mo_zone_",zone_no,"_simplified_summary")

results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_09.01/zones/simplified/mo_6/"
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/training_data/2025_12_09.01/neonatal/neonatal_data.parquet"


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)


model_summary_dir <- paste0(results_dir,"model_summaries/")
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

model_objects_dir <- paste0(results_dir,"model_objects/")
dir.create(model_objects_dir, recursive = TRUE, showWarnings = FALSE)

# Read in neonatal df (must be made from full dataset)
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)

neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
# convert sex_id to int between 0 and 1, where 0 is male and 1 is female
neo_df[,sex_id := as.integer(sex_id)]
neo_df[,sex_id := sex_id-1]
neo_df <- neo_df[zone==zone_no]

# simplification steps
neo_df[,birth_year:=as.integer(birth_year)] # simplified
neo_df[,total_precipitation_prev_0_mo := scale(total_precipitation_prev_0_mo)]
neo_df[,total_precipitation_prev_3_mo_avg := scale(total_precipitation_prev_3_mo_avg)]
neo_df[,total_precipitation_prev_6_mo_avg := scale(total_precipitation_prev_6_mo_avg)]
neo_df[,total_precipitation_prev_9_mo_avg := scale(total_precipitation_prev_9_mo_avg)]

climate_vars <- c(
  # "mean_temperature",
  # "total_precipitation",
  # "relative_humidity",
  # "mean_high_temperature",
  # "mean_low_temperature",
  # "precipitation_days",
  # "days_over_30C",
  # "days_over_26C",
  # "any_days_over_30C",
  "days_over_30C_prev_0_mo",
  "days_over_30C_prev_3_mo_avg",
  "days_over_30C_prev_6_mo_avg",
  "days_over_30C_prev_9_mo_avg",
  "total_precipitation_prev_0_mo",
  "total_precipitation_prev_3_mo_avg",
  "total_precipitation_prev_6_mo_avg",
  "total_precipitation_prev_9_mo_avg"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year","int_birth_year_diff_months", climate_vars)
df_model <- neo_df[, ..cols]

# get sample
# indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
# indv_counts <- indv_dt[, .N, by = ihme_loc_id]
# indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
# indv_dt[, n_sample := floor(sample_percent * N)]

# set.seed(42)
# sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
# df_sample <- df_model[indv_id %in% sampled_indv]

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================
## Options
# "days_over_30C_prev_0_mo",
# "days_over_30C_prev_3_mo_avg",
# "days_over_30C_prev_6_mo_avg",
# "days_over_30C_prev_9_mo_avg",
# "total_precipitation_prev_0_mo",
# "total_precipitation_prev_3_mo_avg",
# "total_precipitation_prev_6_mo_avg",
# "total_precipitation_prev_9_mo_avg"

model <- glmer(
  child_mortality ~ consumption_pd +
    days_over_30C_prev_6_mo_avg +
    total_precipitation_prev_6_mo_avg +
    sex_id +
    birth_year +
    (1 | ihme_loc_id),
  data = df_model,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
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
# SECTION 3: PREDICT MODEL FOR NEONATAL ON AVG BIRTH YEAR, SEX, PRECIPITATION
#==============================================================================

df_avg <- copy(df_model)

# Extract levels for a specific factor
# birth_year_levels <- levels(model@frame$birth_year)
ihme_loc_id_levels <- levels(model@frame$ihme_loc_id)

# df_avg$birth_year <- factor(df_avg$birth_year, levels = birth_year_levels)
df_avg$ihme_loc_id <- factor(df_avg$ihme_loc_id, levels = ihme_loc_id_levels)

# remove any NAs imposed from above step (could be because some years didn't make it in the subset)
# df_avg <- df_avg[!is.na(birth_year)] # should not be a problem on full data

# # Predict WITH random effects (mixed effects)
df_avg$pred_me <- predict(model, newdata = df_avg, type = "response", re.form = NULL)

# # override existing variables to be able to use predict function from package
# df_avg[, birth_year := factor(round(mean(as.numeric(as.character(birth_year))), 0),
#                               levels = birth_year_levels)]
df_avg[, birth_year := round(mean(birth_year), 0)]

df_avg[,sex_id:= mean(df_avg$sex_id)]

# applying average to all precipitation variables to make code copy-pastable
# across scripts
df_avg[,total_precipitation_prev_0_mo:= mean(df_avg$total_precipitation_prev_0_mo)]
df_avg[,total_precipitation_prev_3_mo_avg:= mean(df_avg$total_precipitation_prev_3_mo_avg)]
df_avg[,total_precipitation_prev_6_mo_avg:= mean(df_avg$total_precipitation_prev_6_mo_avg)]
df_avg[,total_precipitation_prev_9_mo_avg:= mean(df_avg$total_precipitation_prev_9_mo_avg)]

# # Predict WITHOUT random effects (fixed effects only)
df_avg$pred_fe <- predict(model, newdata = df_avg, type = "response", re.form = NA)

# # Save predictions to parquet
write_parquet(df_avg, paste0(results_dir, "predictions_", summary_file, ".parquet"))

