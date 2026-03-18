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
mean(model_data$residuals) #0.00000000006939527

# Grouping variable(s) - replace "group_var" with your desired grouping column(s)
group_vars <- c("ihme_loc_id", "birth_year")  # Example grouping variables

# Aggregate observed and predicted values by group
grouped_residuals <- model_data[, .(
  observed = mean(child_mortality, na.rm = TRUE),
  predicted = mean(predicted_probs, na.rm = TRUE),
  consumption_pd = mean(consumption_pd,na.rm=TRUE),
  q95_prev_0_mo = mean(q95_prev_0_mo,na.rm=TRUE),
  total_precipitation_prev_0_mo = mean(total_precipitation_prev_0_mo,na.rm=TRUE),
  sex_id = mean(sex_id,na.rm=TRUE)
), by = group_vars]

# Calculate residuals at the group level
grouped_residuals[, residuals := observed - predicted]

# View the range of grouped residuals
range(grouped_residuals$residuals)
range(grouped_residuals$observed)
mean(grouped_residuals$residuals) #0.007781135

#==============================================================================
# SECTION 3: PLOT COVARIATES AGAINST MODEL RESIDUALS
#==============================================================================


# Define the covariates to plot
# plot_covs <- c("consumption_pd",
#                "q95_prev_0_mo",
#                "total_precipitation_prev_0_mo",
#                "sex_id")
# 
# # Save "consumption_pd" plot as PNG
# png(paste0(plot_dir, gsub("summary", "residuals_vs_consumption_pd", summary_file), ".png"), width = 800, height = 600, res = 150)
# ggplot(model_data, aes(x = consumption_pd, y = residuals)) +
#   geom_point(alpha = 0.5) +
#   labs(title = "Residuals vs consumption_pd",
#        x = "consumption_pd",
#        y = "Residuals") +
#   theme_minimal()
# dev.off()
# 
# # Save "q95_prev_0_mo" plot as PNG
# png(paste0(plot_dir, gsub("summary", "residuals_vs_q95_prev_0_mo", summary_file), ".png"), width = 800, height = 600, res = 150)
# ggplot(model_data, aes(x = q95_prev_0_mo, y = residuals)) +
#   geom_point(alpha = 0.5) +
#   labs(title = "Residuals vs q95_prev_0_mo",
#        x = "q95_prev_0_mo",
#        y = "Residuals") +
#   theme_minimal()
# dev.off()
# 
# # Save "total_precipitation_prev_0_mo" plot as PNG
# png(paste0(plot_dir, gsub("summary", "residuals_vs_total_precipitation_prev_0_mo", summary_file), ".png"), width = 800, height = 600, res = 150)
# ggplot(model_data, aes(x = total_precipitation_prev_0_mo, y = residuals)) +
#   geom_point(alpha = 0.5) +
#   labs(title = "Residuals vs total_precipitation_prev_0_mo",
#        x = "total_precipitation_prev_0_mo",
#        y = "Residuals") +
#   theme_minimal()
# dev.off()
# 
# # Save "sex_id" plot as PNG
# png(paste0(plot_dir, gsub("summary", "residuals_vs_sex_id", summary_file), ".png"), width = 800, height = 600, res = 150)
# ggplot(model_data, aes(x = sex_id, y = residuals)) +
#   geom_point(alpha = 0.5) +
#   labs(title = "Residuals vs sex_id",
#        x = "sex_id",
#        y = "Residuals") +
#   theme_minimal()
# dev.off()


# Save "consumption_pd" plot as PNG
png(paste0(plot_dir, gsub("summary", "grouped_residuals_vs_consumption_pd", summary_file), ".png"), width = 800, height = 600, res = 150)
ggplot(grouped_residuals, aes(x = consumption_pd, y = residuals)) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_hline(yintercept = 0, color = "black") +
  labs(title = "Residuals vs consumption_pd,\ngrouped by country and birth year",
       x = "consumption_pd",
       y = "Residuals") +
  theme_minimal()
dev.off()

# Save "q95_prev_0_mo" plot as PNG
png(paste0(plot_dir, gsub("summary", "grouped_residuals_vs_q95_prev_0_mo", summary_file), ".png"), width = 800, height = 600, res = 150)
ggplot(grouped_residuals, aes(x = q95_prev_0_mo, y = residuals)) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_hline(yintercept = 0, color = "black") +
  labs(title = "Residuals vs q95_prev_0_mo,\ngrouped by country and birth year",
       x = "q95_prev_0_mo",
       y = "Residuals") +
  theme_minimal()
dev.off()

# Save "total_precipitation_prev_0_mo" plot as PNG
png(paste0(plot_dir, gsub("summary", "grouped_residuals_vs_total_precipitation_prev_0_mo", summary_file), ".png"), width = 800, height = 600, res = 150)
ggplot(grouped_residuals, aes(x = total_precipitation_prev_0_mo, y = residuals)) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_hline(yintercept = 0, color = "black") +
  labs(title = "Residuals vs total_precipitation_prev_0_mo,\ngrouped by country and birth year",
       x = "total_precipitation_prev_0_mo",
       y = "Residuals") +
  theme_minimal()
dev.off()

# Save "sex_id" plot as PNG
png(paste0(plot_dir, gsub("summary", "grouped_residuals_vs_sex_id", summary_file), ".png"), width = 800, height = 600, res = 150)
ggplot(grouped_residuals, aes(x = sex_id, y = residuals)) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_hline(yintercept = 0, color = "black") +
  labs(title = "Residuals vs sex_id,\ngrouped by country and birth year",
       x = "sex_id",
       y = "Residuals") +
  theme_minimal()
dev.off()

# Save "birth_year" plot as PNG
png(paste0(plot_dir, gsub("summary", "grouped_residuals_vs_birth_year", summary_file), ".png"), width = 800, height = 600, res = 150)
ggplot(grouped_residuals, aes(x = birth_year, y = residuals)) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_hline(yintercept = 0, color = "black") +
  labs(title = "Residuals vs birth_year,\ngrouped by country and birth year",
       x = "birth_year",
       y = "Residuals") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, size = 5)  # Rotate x-axis labels and make font smaller
  )
dev.off()
