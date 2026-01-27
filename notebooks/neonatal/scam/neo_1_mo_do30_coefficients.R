################################################################################
# DESCRIPTION: Script to gather required coefficients and basis functions for prediction
# PROJECT: Climate nutrition
# DATE: 2026-01-23
################################################################################

#==============================================================================
# SECTION 0: PACKAGE LOADING AND ENVIRONMENT SETUP
#==============================================================================
# Clear workspace
dev.off()
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
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2025_12_16.01/neonatal_data.parquet"


model_summary_dir <- paste0(results_dir,"model_summaries/")

model_objects_dir <- paste0(results_dir,"model_objects/")

inference_objects_dir <- paste0(results_dir,"inference_format/")

# load original data for country effects
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)
neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
# convert sex_id to int between 0 and 1, where 0 is male and 1 is female
neo_df[,sex_id := as.integer(sex_id)]
neo_df[,sex_id := sex_id-1]
neo_df[,birth_year:=as.factor(birth_year)]

climate_vars <- c(
  "days_over_30C_prev_0_mo",
  "total_precipitation_prev_0_mo"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year", climate_vars)
df_model <- neo_df[, ..cols]
df_model <- na.omit(df_model)

model = readRDS(file = paste0(model_objects_dir, summary_file,".rds"))
# NOTE: Model specs as below:
# model <- scam(
#   child_mortality ~ s(consumption_pd, bs="mpd") +
#     s(days_over_30C_prev_0_mo, bs="mpi") +
#     total_precipitation_prev_0_mo +
#     sex_id +
#     birth_year +
#     s(ihme_loc_id, bs="re"),
#   data = df_model,
#   family = binomial(link = "logit")
# )


# load test data:
test_df <- fread(paste0(results_dir, "single_var_spline_test_", summary_file, ".parquet"))
test_df$pred_fixed_consumption <- NULL
test_df$linear_predictor <- NULL

# only days_over_30C_prev_0_mo should be varying among explanatory vars
summary(test_df)

# get random effects coefficients

coef_names <- names(coef(model))
coefs <- data.table(
  variable = coef_names,
  coefficient = coef(model)
)


loc_levels <- data.frame(ihme_loc_id = levels(df_model$ihme_loc_id))
# Extract random effect terms from coefs
random_effect_terms <- grep("^s\\(ihme_loc_id\\)", coefs$variable, value = TRUE)

# Create a mapping of random effect terms to ihme_loc_id levels
random_effect_mapping <- data.table(
  variable = random_effect_terms,
  ihme_loc_id = loc_levels$ihme_loc_id
)

# Replace random effect terms in coefs with corresponding ihme_loc_id levels
coefs <- merge(coefs, random_effect_mapping, by = "variable", all.x = TRUE, sort = FALSE)

# which indices are the random effects from coefs?
random_effect_indices <- which(coefs$variable %in% random_effect_terms)
random_effect_coefs <- coefs[random_effect_indices,]

# # If the replacement is successful, update the variable column
# coefs[, variable := ifelse(!is.na(ihme_loc_id), ihme_loc_id, variable)]
# 
# 
# # Drop the temporary ihme_loc_id column
# coefs[, ihme_loc_id := NULL]


#==============================================================================
# SECTION 2: GET REQUIRED SPLINE TERMS
#==============================================================================


# Extract smooth terms for consumption and climate
consumption_smooth <- model$smooth[[1]]  # s(consumption_pd)
climate_smooth <- model$smooth[[2]]      # s(days_over_30C_prev_0_mo)

# Extract coefficients for the smooth terms
consumption_spline_indices <- consumption_smooth$first.para:consumption_smooth$last.para
climate_spline_indices <- climate_smooth$first.para:climate_smooth$last.para
consumption_spline_coefs <- coef(model)[consumption_spline_indices]
climate_spline_coefs <- coef(model)[climate_spline_indices]

# Extract knot locations and spline order for climate smooth
climate_knots <- climate_smooth$knots
climate_spline_order <- climate_smooth$m[1]  # Typically 2 for quadratic

# same for consumption
consumption_knots <- consumption_smooth$knots
consumption_spline_order <- consumption_smooth$m[1]  # Typically 2 for quadratic

# Extract linear coefficients
linear_coefs <- copy(coefs)
linear_coefs <- linear_coefs[-c(consumption_spline_indices,climate_spline_indices,random_effect_indices)]

#==============================================================================
# SECTION 3: MANUAL BASIS FUNCTION CALCULATION
#==============================================================================

# Function to calculate basis functions manually
calculate_basis <- function(x, knots, degree) {
  # Recursive calculation of B-spline basis functions
  n <- length(knots) - 1
  B <- matrix(0, nrow = length(x), ncol = n)
  
  # Degree 0 basis functions
  for (i in 1:n) {
    B[, i] <- ifelse(x >= knots[i] & x < knots[i + 1], 1, 0)
  }
  
  # Higher degree basis functions
  for (d in 1:degree) {
    for (i in 1:(n - d)) {
      B[, i] <- ((x - knots[i]) / (knots[i + d] - knots[i])) * B[, i] +
        ((knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1])) * B[, i + 1]
    }
  }
  
  return(B[, 1:(n - degree)])  # Return only the valid basis functions
}

# Calculate basis functions for climate smooth
x_climate <- test_df$days_over_30C_prev_0_mo
basis_functions_climate <- calculate_basis(x_climate, climate_knots, climate_spline_order)

#==============================================================================
# SECTION 4: RECREATE PREDICTIONS
#==============================================================================

# Calculate smooth contributions
smooth_contribution_climate <- basis_functions_climate %*% climate_spline_coefs

# Linear contributions
linear_contribution <- test_df$total_precipitation_prev_0_mo * linear_coefs["total_precipitation_prev_0_mo"] +
  test_df$sex_id * linear_coefs["sex_id"] +
  test_df$birth_year * linear_coefs["birth_year"]

# Random effects (if applicable)
random_effects <- coef(model)[model$smooth[[3]]$first.para:model$smooth[[3]]$last.para]
random_effect_contribution <- random_effects[match(test_df$ihme_loc_id, names(random_effects))]

# Combine contributions to calculate the linear predictor
test_df$linear_predictor <- as.vector(smooth_contribution_climate) +
  as.vector(linear_contribution) +
  as.vector(random_effect_contribution)

# Apply the inverse logit function to calculate predicted probabilities
test_df$pred_fixed_consumption <- 1 / (1 + exp(-test_df$linear_predictor))

#==============================================================================
# SECTION 5: VALIDATION
#==============================================================================

# Compare manual predictions with the model's predict function
test_df$pred_prob <- predict(model, newdata = test_df, type = "response")
test_df$pred_linear <- predict(model, newdata = test_df, type = "link")

# Check ranges
range(test_df$pred_fixed_consumption)
range(test_df$pred_prob)

# Validate manually calculated predictions
all.equal(test_df$pred_fixed_consumption, test_df$pred_prob)
all.equal(test_df$linear_predictor, test_df$pred_linear)
