################################################################################
# DESCRIPTION: Script to gather required coefficients and basis functions for prediction
# PROJECT: Climate nutrition
# DATE: 2026-01-23
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
library(mgcv)
library(scam)
library(arrow) # to read parquet
library(ggplot2)
library(dplyr)
library(scales)

options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: LOADING MODEL AND GETTING PARAMETERS
#==============================================================================

summary_file <- paste0("nnm_1_mo_do30_scam_summary")

results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/"


model_summary_dir <- paste0(results_dir,"model_summaries/")

model_objects_dir <- paste0(results_dir,"model_objects/")

inference_objects_dir <- paste0(results_dir,"inference_format/")

model = readRDS(file = paste0(model_objects_dir, summary_file,".rds"))

#==============================================================================
# SECTION 2: GET REQUIRED SPLINE TERMS
#==============================================================================


# Extract basis functions and betas for spline terms:
consumption_spline_indices <- model$smooth[[1]]$first.para:model$smooth[[1]]$last.para
climate_spline_indices <- model$smooth[[2]]$first.para:model$smooth[[2]]$last.para
consumption_spline_coefs <- coef(model)[consumption_spline_indices]
basis_functions_consumption <- model$smooth[[1]]$X
climate_spline_coefs <- coef(model)[climate_spline_indices]
basis_functions_climate <- model$smooth[[2]]$X

Xp <- predict(model, type="lpmatrix")

library(mgcv)

# Extract the smooth term for the first spline (e.g., s(consumption_pd))
consumption_spline <- model$smooth[[1]]

# Extract the data used to fit the model
fitted_data <- model$model

# Generate the basis matrix for the first smooth term
basis_functions_consumption <- PredictMat(consumption_spline, fitted_data)

# Extract the coefficients for the first smooth term
coef_indices_consumption <- consumption_spline$first.para:consumption_spline$last.para
spline_coefs_consumption <- coef(model)[coef_indices_consumption]

# Combine basis functions and coefficients
basis_and_coefs_consumption <- data.frame(
  Basis = basis_functions_consumption,
  Coefficient = spline_coefs_consumption
)

#########