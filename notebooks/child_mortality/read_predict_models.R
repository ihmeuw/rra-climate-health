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
library(data.table)
library(arrow) # to read parquet
library(ggplot2)
library(scales)


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_13.01/data_avg_climate.parquet"
# data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.csv"
plot_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_13.01/"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_13.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")

neonatal_dir <- paste0(results_dir,"neonatal/")
dir.create(neonatal_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_parquet(data_version)
df <- data.table(df)

df[,ihme_loc_id:=as.factor(ihme_loc_id)]
df[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]


#==============================================================================
# SECTION 2: READ MODELS
#==============================================================================


## Read in and print model summaries from successful runs:

# 10/14 - Collapsing data to have average climate vars per child

# 5% data model
model <- readRDS(paste0(results_dir,"subset_05pct_model_do30.rds"))

# plot survival curves
pred_surv <- predict(model, df, quantity = "survival")

# compare re vs fe for unseen data



## First successful run of full data
model_baseline <- readRDS(paste0(results_dir,"baseline_model_object.rds"))
summary(model_baseline)


## 50% data with days_over_30
model_50_pc <- readRDS(paste0(results_dir,"subset_5pct_model_do30_object.rds"))
summary(model_50_pc)

# Predict model on non-included data
model_50_pred <- fread(paste0(results_dir,"subset_5pct_model_do30_results.csv"))
test_df <- df[!(indv_id %in% unique(model_50_pred$indv_id))]

# mixed effects predictions
pred_surv_me <- predict(model_50_pc, test_df, quantity = "survival")
surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
  surv_df <- pred_surv_me[[i]]
  obs_time <- test_df$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

test_df$model_predictions_me <- 1-surv_at_obs_time

# fixed effects predictions
pred_surv_fe <- predict(model_50_pc, test_df,re.form = ~0, quantity = "survival") # make fixed effects predictions
surv_at_obs_time <- sapply(seq_len(nrow(test_df)), function(i) {
  surv_df <- pred_surv_fe[[i]]
  obs_time <- test_df$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

test_df$model_predictions_fe <- 1-surv_at_obs_time

write_parquet(test_df,paste0(results_dir,"test_set_predictions_50pc_do30_fe.parquet"))

# get predictions of model over data set
pred_surv <- predict(model_50_pc, df_min_age, quantity = "survival")
# pred_surv <- predict(model_50_pc, df,re.form = ~0, quantity = "survival") # make fixed effects predictions

# get predictions at 1 month
surv_at_1_mo <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_surv, 1/12)

surv_under_1_mo <- mapply(function(df, t) {
  idx <- min(which(df$time <= t))
  df$survival[idx]
}, pred_surv, 1/12)

df_min_age$mortality_1_mo <- 1-surv_at_1_mo
df_min_age$mortality_under_1_mo <- 1-surv_under_1_mo

# save out for heat maps
write_parquet(df_min_age,paste0(neonatal_dir,"neonatal_mortality_1_mo.parquet"))

## 50% data with days_over_30 only
model_do30_50pc <- readRDS(paste0(results_dir,"subset_5pct_model_do30_object.rds"))
summary(model_do30_50pc)
summary_file <- "subset_50pct_model_do30_object.txt"
capture.output(summary(model_do30_50pc), file = paste0(model_summary_dir,summary_file))

# get mixed effects and fixed effects predictions over raw data
pred_surv_me <- predict(model_50_pc, df, quantity = "survival")
surv_at_obs_time <- sapply(seq_len(nrow(df)), function(i) {
  surv_df <- pred_surv_me[[i]]
  obs_time <- df$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

df$model_predictions_me <- 1-surv_at_obs_time

write_parquet(df,paste0(results_dir,"predictions_50pc_do30_me.parquet"))

pred_surv_fe <- predict(model_50_pc, df,re.form = ~0, quantity = "survival") # make fixed effects predictions
surv_at_obs_time <- sapply(seq_len(nrow(df)), function(i) {
  surv_df <- pred_surv_fe[[i]]
  obs_time <- df$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

df$model_predictions_fe <- 1-surv_at_obs_time

write_parquet(df,paste0(results_dir,"predictions_50pc_do30_fe.parquet"))

# Model diagnostics
mean_observed <- mean(df$child_mortality)
mean_predicted <- mean(df$model_predictions_me)
print(paste("Observed mean mortality:", round(mean_observed, 4)))
print(paste("Predicted mean mortality:", round(mean_predicted, 4)))
hist(df$child_mortality, breaks=20, main="Observed Mortality", xlab="Mortality")
hist(df$model_predictions_me, breaks=20, main="Predicted Mortality", xlab="Predicted")
df$pred_bin <- cut(df$model_predictions_me, breaks=seq(0,1,by=0.05))
calib <- df[, .(obs_rate = mean(child_mortality), pred_rate = mean(model_predictions_me)), by=pred_bin]
ggplot(calib, aes(x=pred_rate, y=obs_rate)) +
  geom_point() +
  geom_abline(slope=1, intercept=0, linetype="dashed", color="red") +
  labs(x="Predicted Rate", y="Observed Rate", title="Calibration Plot")

by_country <- df[, .(obs_rate = mean(child_mortality), pred_rate = mean(model_predictions_me)), by=ihme_loc_id]
ggplot(by_country, aes(x = obs_rate, y = pred_rate)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Observed Rate", y = "Predicted Rate", title = "Country-level Calibration")

## 25% data with days_over_30 only
model_do30_25pc <- readRDS(paste0(results_dir,"subset_25pct_model_do30_object.rds"))
summary(model_do30_25pc)
summary_file <- "subset_25pct_model_do30_object.txt"
capture.output(summary(model_do30_25pc), file = paste0(model_summary_dir,summary_file))

# predictions
pred <- fread(paste0(results_dir,"subset_25pct_model_do30_results.csv"))

# get predictions of model over data set
pred_surv <- predict(model_do30_25pc, df_min_age, quantity = "survival")
# pred_surv_fe <- predict(model_50_pc, df,re.form = ~0, quantity = "survival") # make fixed effects predictions

# get predictions at 1 month
surv_at_1_mo <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_surv, 1/12)

df_min_age$mortality_1_mo_do30 <- 1-surv_at_1_mo

# save out for heat maps
write_parquet(df_min_age,paste0(neonatal_dir,"neonatal_mortality_1_mo_do30.parquet"))



#==============================================================================
# SECTION 3: MAKE PLOTS
#==============================================================================

# Plot survival curves together
dt_list <- lapply(seq_along(pred_surv), function(i) {
  dt <- as.data.table(pred_surv[[i]])
  dt[, obs_id := i]  # Add observation ID
  dt
})

# Combine all into one data.table
all_surv <- rbindlist(dt_list)

p <- ggplot(all_surv, aes(x = time, y = survival)) +
  geom_point(alpha = 0.3) +
  labs(title = "Predicted Survival Curves from Model",
       x = "Age in Years",
       y = "Survival Probability") +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

ggsave(paste0(plot_dir, "survival_curves_10_14.png"), plot = p, width = 8, height = 5)
