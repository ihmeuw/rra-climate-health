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

sample_percent <- 0.25

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
cols <- c("nid","psu","hh_id","line_id","child_mortality", "age_year_at_year_end", "sex_id", "ihme_loc_id", "consumption", climate_vars)
df_model <- df[, ..cols]
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]

# Sample data, keeping all observations for any sampled individual child, and 
# balancing countries
df_model <- data.table(df_model)
# make indv ID
df_model[,indv_id := paste(nid,psu,hh_id,line_id,sep="_")]
# get sample
indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
indv_counts <- indv_dt[, .N, by = ihme_loc_id]
indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
indv_dt[, n_sample := floor(sample_percent * N)]

set.seed(42)
sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
df_sample <- df_model[indv_id %in% sampled_indv]



#==============================================================================
# SECTION 2: FIT MODEL
#==============================================================================

# fit baseline model with mean_temperature and days_over_30C
model <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption + 
                   mean_temperature + 
                   days_over_30C + 
                   sex_id + 
                   survival::cluster(ihme_loc_id), 
                 data = df_sample,
                 verbose = TRUE)


# get predictions of model over same data set
pred_surv <- predict(model, df_sample, quantity = "survival")

surv_at_obs_time <- sapply(seq_len(nrow(df_sample)), function(i) {
  surv_df <- pred_surv[[i]]
  obs_time <- df_sample$age_year_at_year_end[i]
  idx <- max(which(surv_df$time <= obs_time))
  surv_df$survival[idx]
})

df_sample$model_predictions <- 1-surv_at_obs_time
print("model predictions done")

subset_str <- as.character(sample_percent)
subset_str <- gsub("0.","",subset_str, fixed = TRUE)
write.csv(df_sample,paste0(results_dir,"subset_",subset_str,"pct_model_results.csv"),row.names = FALSE)

# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, "subset_",subset_str,"pct_model_object.rds"))

print(paste0("results saved to ",results_dir))
