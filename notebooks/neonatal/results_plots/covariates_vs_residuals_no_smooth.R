################################################################################
# DESCRIPTION: Script to plot model residuals against covariates
# PROJECT: Climate nutrition
# DATE: 2025-12-30
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
library(ggplot2)
library(gridExtra)


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- paste0("nnm_1_mo_q95_no_smooth_summary")

results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/"
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2025_12_16.01/neonatal_data.parquet"

model_objects_dir <- paste0(results_dir,"model_objects/")
plot_dir <- paste0(results_dir,"plots/")

# Read in neonatal df (must be made from full dataset)
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)

neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
# convert sex_id to int between 0 and 1, where 0 is male and 1 is female
neo_df[,sex_id := as.integer(sex_id)]
neo_df[,sex_id := sex_id-1]
neo_df[,birth_year:=as.factor(birth_year)]

## Read and format data

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
  # "days_over_30C_prev_0_mo",
  # "days_over_30C_prev_3_mo_avg",
  # "days_over_30C_prev_6_mo_avg",
  # "days_over_30C_prev_9_mo_avg",
  'q9_prev_0_mo',
  'q95_prev_0_mo',
  'q9_prev_3_mo_avg',
  'q9_prev_6_mo_avg',
  'q9_prev_9_mo_avg',
  'q95_prev_3_mo_avg',
  'q95_prev_6_mo_avg',
  'q95_prev_9_mo_avg',
  'zone',
  "total_precipitation_prev_0_mo",
  "total_precipitation_prev_3_mo_avg",
  "total_precipitation_prev_6_mo_avg",
  "total_precipitation_prev_9_mo_avg"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year", climate_vars)
df_model <- neo_df[, ..cols]

#==============================================================================
# SECTION 2: LOAD MODEL
#==============================================================================

# Read model 
model = readRDS(file = paste0(model_objects_dir, summary_file,".rds"))

summary(model)

# residuals <- residuals(model)

# length of residuals is less than df_model, possibly due to random effects.
# Use model_data instead.
model_data <- model.frame(model)
setDT(model_data)

predicted_probs <- predict(model, type = "response")  
residuals <- model_data$child_mortality - predicted_probs

model_data[,residuals := residuals]

range(model_data$residuals)


#==============================================================================
# SECTION 3: PLOT COVARIATES AGAINST MODEL RESIDUALS
#==============================================================================

outfile <- gsub("summary","residuals_vs_covariates.pdf",summary_file)

# Define the covariates to plot
plot_covs <- c("consumption_pd",
               "q95_prev_0_mo",
               "total_precipitation_prev_0_mo",
               "sex_id")

# Create a function to generate and save plots
generate_and_save_plot <- function(covariate, data, plot_dir, outfile_prefix) {
  p <- ggplot(data, aes_string(x = covariate, y = "residuals")) +
    geom_point(alpha = 0.5) +
    labs(title = paste("Residuals vs", covariate),
         x = covariate,
         y = "Residuals") +
    theme_minimal()
  
  # Save the plot as a PDF
  pdf_filename <- paste0(plot_dir, outfile_prefix, "_", covariate, ".pdf")
  pdf(pdf_filename, width = 8, height = 6)
  print(p)
  dev.off()
}

# Generate and save each plot
outfile_prefix <- gsub("summary", "residuals_vs_covariates", summary_file)
for (covariate in plot_covs) {
  generate_and_save_plot(covariate, model_data, plot_dir, outfile_prefix)
}
