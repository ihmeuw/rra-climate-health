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

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_08.01/data.parquet"
# data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.csv"
plot_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_08.01/"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")

neonatal_dir <- paste0(results_dir,"neonatal/")
dir.create(neonatal_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_parquet(data_version)
df <- data.table(df)

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df[,child_mortality := 1-child_alive]
df[,ihme_loc_id:=as.factor(ihme_loc_id)]
df[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]

setnames(df,old="ldipc_weighted_no_match",new="consumption")

#==============================================================================
# SECTION 2: READ MODELS
#==============================================================================


## Read in and print model summaries from successful runs:

## 50% data with mean_temperature and days_over_30
model_50_pc <- readRDS(paste0(results_dir,"subset_5pct_model_object.rds"))
summary(model_50_pc)

# get smallest age for each individual to reduce computation
df_min_age <- df[, .SD[which.min(age_year_at_year_end)], by = indv_id]
min(df_min_age$age_year_at_year_end) # 0.08333333 i.e. 1 month
table(df_min_age$child_mortality) # about 4%

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

ggsave(paste0(plot_dir, "cv_rmse_results_25pc.png"), plot = p, width = 8, height = 5)
