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
summary_file <- paste0("nnm_1_mo_q95_summary")

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

# Create an empty list to store the plots
plot_list <- list()

# "consumption_pd"
p <- ggplot(model_data, aes(x = consumption_pd, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", color = "blue", se = FALSE) +
  labs(title = paste("Residuals vs", "consumption_pd"),
       x = "consumption_pd",
       y = "Residuals") +
  theme_minimal()

# Add the plot to the list
plot_list[["consumption_pd"]] <- p

# "q95_prev_0_mo"
p <- ggplot(model_data, aes(x = q95_prev_0_mo, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", color = "blue", se = FALSE) +
  labs(title = paste("Residuals vs", "q95_prev_0_mo"),
       x = "q95_prev_0_mo",
       y = "Residuals") +
  theme_minimal()

# Add the plot to the list
plot_list[["q95_prev_0_mo"]] <- p

# "total_precipitation_prev_0_mo"
p <- ggplot(model_data, aes(x = total_precipitation_prev_0_mo, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", color = "blue", se = FALSE) +
  labs(title = paste("Residuals vs", "total_precipitation_prev_0_mo"),
       x = "total_precipitation_prev_0_mo",
       y = "Residuals") +
  theme_minimal()

# Add the plot to the list
plot_list[["total_precipitation_prev_0_mo"]] <- p

# "sex_id"
p <- ggplot(model_data, aes(x = sex_id, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", color = "blue", se = FALSE) +
  labs(title = paste("Residuals vs", "sex_id"),
       x = "sex_id",
       y = "Residuals") +
  theme_minimal()

# Add the plot to the list
plot_list[["sex_id"]] <- p

# Save all plots to a 2x2 layout in a PDF
pdf(paste0(plot_dir,outfile), width = 8, height = 8)  
grid.arrange(grobs = plot_list, ncol = 2, nrow = 2)        
dev.off()                                                 

