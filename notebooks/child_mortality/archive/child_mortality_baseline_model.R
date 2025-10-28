################################################################################
# DESCRIPTION: Script to run baseline model on child mortality on full data set.
# PROJECT: Climate nutrition
# DATE: 2025-10-06
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

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_08.01/data.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_parquet(data_version)
df <- data.table(df)

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df[,child_mortality := 1-child_alive]

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
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]


#==============================================================================
# SECTION 2: FIT MODEL
#==============================================================================

# fit baseline model with mean_temperature and days_over_30C
model <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                    mean_temperature + 
                    days_over_30C + 
                    sex_id + 
                    survival::cluster(ihme_loc_id), 
                  data = df_model,
                  verbose = TRUE)


# get predictions of model over same data set
pred_surv <- predict(model, df_model, quantity = "survival")

surv_at_obs_time <- sapply(seq_len(nrow(df_model)), function(i) {
  surv_df <- pred_surv[[i]]
  obs_time <- df_model$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

df_model$model_predictions <- 1-surv_at_obs_time
print("model predictions done")


write.csv(df_model,paste0(results_dir,"baseline_model_results.csv"),row.names = FALSE)

# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, "baseline_model_object.rds"))

print(paste0("results saved to ",results_dir))
