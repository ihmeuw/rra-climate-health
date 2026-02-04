################################################################################
# DESCRIPTION: Script to gather required coefficients and basis functions for prediction
# PROJECT: Climate nutrition
# DATE: 2026-01-23
################################################################################

# TOC:
# SECTION 0: PACKAGE LOADING AND ENVIRONMENT SETUP
# SECTION 1: LOADING MODEL AND DATA
# SECTION 2: GET NON-SPLINE TERMS
# SECTION 4: MANUAL BASIS FUNCTION CALCULATION
# SECTION 5: RECREATE PREDICTIONS
# SECTION 6: VALIDATION AGAINST PACKAGE PREDICTIONS
# SECTION 7: CREATE LOOKUP TABLES FOR DEPLOYMENT
# SECTION 8: TROUBLESHOOTING

# Key issues/notes -
# - manual predictions wildly different from predict() 
# - For both days_over_30C and consumption_pd smooths, there are 14 knots, 
#   order 2 (quadratic), dimension 10, rank 8, and only 9 coefficients from model
# - The scam package can use PredMat to give basis functions for a spline term, 
#   but actually produces a matrix with 10 columns (likely only 9 used, with 1 dropped
#   due to monotonicity constraints)
# - Cannot calculate package's own predict() using basis function matrices from 
#   PredictMat(), due to dimension mismatch
# - Cannot derive calculate_basis() function to create basis function matrices 
#   that result in same predictions

# EXPERIMENTS:
# - VERIFY PREDICT AGAINST LP MATRIX -> get different results, even using the full 
#   raw lpmatrix * coefficients
# - USE Reconstruct FROM predict(type="terms") -> This uses a combination of the 
#   linear terms and the smoothed terms from the pred(...,type = "terms") results,
#   which yields the same results as the overall pred(), telling us that the linear 
#   terms are not the problem, only the spline terms are. 

# Notes from Claude:
# - The issue is that lpmatrix from scam models doesn't work the same way as in 
#   mgcv. The scam package uses constrained optimization that modifies the parameter 
#   space, so the lpmatrix columns don't directly correspond to the model coefficients 
#   in a simple multiplicative way.
# - The fundamental limitation: SCAM's monotonic constraints create a complex non-linear 
#   transformation that cannot be expressed as simple basis functions × coefficients. 
#   You must either use the model object or pre-compute lookup tables.
# - For deployment without the model object, you need to save a lookup table of 
#   the smooth contributions at a fine grid of x-values, since you cannot manually 
#   reconstruct the SCAM basis functions.

#==============================================================================
# SECTION 0: PACKAGE LOADING AND ENVIRONMENT SETUP
#==============================================================================
# Clear workspace
# dev.off()
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
library(stringr)

options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: LOADING MODEL AND DATA
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
test_df <- as.data.frame(test_df)  # avoid data.table predict quirks
test_df$ihme_loc_id <- factor(test_df$ihme_loc_id, levels = levels(df_model$ihme_loc_id))
test_df$birth_year  <- factor(test_df$birth_year,  levels = levels(df_model$birth_year))

test_df$pred_fixed_consumption <- NULL
test_df$linear_predictor <- NULL
test_df$V1 <- NULL
# only days_over_30C_prev_0_mo should be varying among explanatory vars
summary(test_df)

#==============================================================================
# SECTION 2: GET TERMS COEFFICIENTS
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


coef_names <- names(coef(model))
coefs <- data.table(
  variable = coef_names,
  coefficient = coef(model)
)


# get random effects coefficients
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

# Extract linear coefficients
linear_coefs <- copy(coefs)
linear_coefs <- linear_coefs[-c(consumption_spline_indices,climate_spline_indices,random_effect_indices)]

# get birth year as integer
birth_years <- grep("(?<=birth_year).*", coefs$variable, value = TRUE, perl = TRUE)
linear_coefs[,birth_year:=sub("birth_year", "", variable)] 

# get intercept
intercept <- linear_coefs[variable == "(Intercept)", coefficient]

#==============================================================================
# SECTION 4: MANUAL BASIS FUNCTION CALCULATION
#==============================================================================

# Section no longer relevant for predictions

calculate_basis_ispline <- function(x, knots, degree, num_coefs) {
  # Calculate B-spline basis functions first
  n <- length(knots) - 1
  B <- matrix(0, nrow = length(x), ncol = n)
  
  # Degree 0 basis functions (piecewise constant)
  for (i in 1:n) {
    B[, i] <- ifelse(x >= knots[i] & x < knots[i + 1], 1, 0)
  }
  
  # Higher degree basis functions
  if (degree > 0) {
    for (d in 1:degree) {
      for (i in 1:(n - d)) {
        denom1 <- knots[i + d] - knots[i]
        denom2 <- knots[i + d + 1] - knots[i + 1]
        
        term1 <- if (denom1 > 0) ((x - knots[i]) / denom1) * B[, i] else 0
        term2 <- if (denom2 > 0) ((knots[i + d + 1] - x) / denom2) * B[, i + 1] else 0
        
        B[, i] <- term1 + term2
      }
    }
  }
  
  # For monotonic splines, we need I-splines (integrated B-splines)
  # The number of I-splines should match the number of coefficients
  num_basis <- num_coefs  # Use the number of coefficients directly
  
  # Extract valid basis functions
  B_valid <- B[, 1:num_basis, drop = FALSE]
  
  # Compute I-splines by cumulative integration
  # For each basis function, integrate from the left
  I <- matrix(0, nrow = length(x), ncol = num_basis)
  
  for (j in 1:num_basis) {
    # Cumulative sum gives the integral for monotonic splines
    I[, j] <- rowSums(B_valid[, 1:j, drop = FALSE])
  }
  
  return(I)
}

# Recalculate basis functions using I-splines with explicit num_coefs
basis_functions_climate <- calculate_basis_ispline(
  x = test_df$days_over_30C_prev_0_mo,
  knots = climate_knots,
  degree = climate_spline_order,
  num_coefs = length(climate_spline_coefs)  # Pass the number of coefficients
)

basis_functions_consumption <- calculate_basis_ispline(
  x = test_df$consumption_pd,
  knots = consumption_knots,
  degree = consumption_spline_order,
  num_coefs = length(consumption_spline_coefs)  # Pass the number of coefficients
)

# Verify dimensions match
cat("Climate basis dimensions:", dim(basis_functions_climate), "\n")
cat("Climate coefs length:", length(climate_spline_coefs), "\n")
cat("Consumption basis dimensions:", dim(basis_functions_consumption), "\n")
cat("Consumption coefs length:", length(consumption_spline_coefs), "\n")


#==============================================================================
# SECTION 5: RECREATE PREDICTIONS
#==============================================================================

# Calculate smooth contributions
smooth_contribution_climate <- basis_functions_climate %*% climate_spline_coefs
smooth_contribution_consumption <- basis_functions_consumption %*% consumption_spline_coefs

# Linear contributions
# Extract the intercept
intercept <- linear_coefs[variable == "(Intercept)", coefficient]

# Repeat the intercept for all rows in test_df
intercept_vector <- rep(intercept, nrow(test_df))

linear_contribution <- intercept_vector+
  test_df$total_precipitation_prev_0_mo * linear_coefs[variable=="total_precipitation_prev_0_mo",coefficient] +
  test_df$sex_id * linear_coefs[variable=="sex_id",coefficient] +
  sapply(test_df$birth_year, function(year) {
    # Dynamically match the coefficient for the specific birth year
    coef_name <- paste0("birth_year", year)
    linear_coefs[variable == coef_name, coefficient]
  })

random_effect_contribution <- sapply(test_df$ihme_loc_id, function(loc) {
  # Dynamically match the coefficient for the specific birth year
  random_effect_coefs[ihme_loc_id == loc, coefficient]
})

# Combine contributions to calculate the linear predictor
test_df$manual_linear <- as.vector(smooth_contribution_climate) +
  as.vector(smooth_contribution_consumption) +
  as.vector(linear_contribution) +
  as.vector(random_effect_contribution)

# Apply the inverse logit function to calculate predicted probabilities
test_df$manual_prob <- 1 / (1 + exp(-test_df$manual_linear))


#==============================================================================
# SECTION 6: VALIDATION AGAINST PACKAGE PREDICTIONS
#==============================================================================

# Compare manual predictions with the model's predict function
test_df$pred_prob <- predict(model, newdata = test_df, type = "response")
test_df$pred_linear <- predict(model, newdata = test_df, type = "link")

# Check ranges
range(test_df$manual_prob)
range(test_df$pred_prob)
range(test_df$manual_linear)
range(test_df$pred_linear)


# Validate manually calculated predictions
all.equal(test_df$manual_prob, test_df$pred_prob)
all.equal(test_df$manual_linear, test_df$pred_linear)

# extract predicted smooth terms individually
model_smooth_climate <- predict(model, newdata = test_df, type = "terms")[, "s(days_over_30C_prev_0_mo)"]
model_smooth_consumption <- predict(model, newdata = test_df, type = "terms")[, "s(consumption_pd)"]


#==============================================================================
# SECTION 7: CREATE LOOKUP TABLES FOR DEPLOYMENT
#==============================================================================


#------------------------------------------------------------------------------
# 8.1: Create fine grid for climate smooth (days_over_30C_prev_0_mo)
#------------------------------------------------------------------------------
# Determine the range from training data
climate_range <- range(df_model$days_over_30C_prev_0_mo, na.rm = TRUE)

# Create a fine grid (1000 points should be sufficient for interpolation)
climate_grid <- seq(climate_range[1], climate_range[2], length.out = 1000)

# Create dummy dataframe with climate grid
# Use median/mode values for other variables
dummy_climate <- data.frame(
  days_over_30C_prev_0_mo = climate_grid,
  consumption_pd = median(df_model$consumption_pd, na.rm = TRUE),
  total_precipitation_prev_0_mo = median(df_model$total_precipitation_prev_0_mo, na.rm = TRUE),
  sex_id = 0,  # Use baseline
  birth_year = factor("2022", levels = levels(df_model$birth_year)),  # Year 2022
  ihme_loc_id = factor(levels(df_model$ihme_loc_id)[1], levels = levels(df_model$ihme_loc_id))  # Will use 0 effect by excluding from prediction
)

# Get smooth term predictions for this grid
# Use type="terms" and extract only the climate smooth to get centered contribution
pred_terms_climate <- predict(model, newdata = dummy_climate, type = "terms")
climate_smooth_values <- pred_terms_climate[, "s(days_over_30C_prev_0_mo)"]

# Create lookup table
climate_lookup <- data.table(
  days_over_30C_prev_0_mo = climate_grid,
  smooth_contribution = climate_smooth_values
)

#------------------------------------------------------------------------------
# 8.2: Create fine grid for consumption smooth (consumption_pd)
#------------------------------------------------------------------------------

# Determine the range from training data
consumption_range <- range(df_model$consumption_pd, na.rm = TRUE)

# Create a fine grid
consumption_grid <- seq(consumption_range[1], consumption_range[2], length.out = 1000)

# Create dummy dataframe with consumption grid
dummy_consumption <- data.frame(
  days_over_30C_prev_0_mo = median(df_model$days_over_30C_prev_0_mo, na.rm = TRUE),
  consumption_pd = consumption_grid,
  total_precipitation_prev_0_mo = median(df_model$total_precipitation_prev_0_mo, na.rm = TRUE),
  sex_id = 0,
  birth_year = factor("2022", levels = levels(df_model$birth_year)),  # Year 2022
  ihme_loc_id = factor(levels(df_model$ihme_loc_id)[1], levels = levels(df_model$ihme_loc_id))  # Will use 0 effect
)

# Get smooth term predictions
pred_terms_consumption <- predict(model, newdata = dummy_consumption, type = "terms")
consumption_smooth_values <- pred_terms_consumption[, "s(consumption_pd)"]

# Create lookup table
consumption_lookup <- data.table(
  consumption_pd = consumption_grid,
  smooth_contribution = consumption_smooth_values
)


#------------------------------------------------------------------------------
# 8.3: Create lookup function for linear interpolation
#------------------------------------------------------------------------------

# Function to interpolate smooth contributions from lookup table
interpolate_smooth <- function(x_values, lookup_table, x_col = "x", y_col = "smooth_contribution") {
  # Use approx for linear interpolation
  # rule=2 means use nearest value for points outside range
  interpolated <- approx(
    x = lookup_table[[x_col]], 
    y = lookup_table[[y_col]], 
    xout = x_values,
    rule = 2  # Extrapolate using nearest boundary value
  )$y
  
  return(interpolated)
}

#------------------------------------------------------------------------------
# 8.4: Test the lookup table approach on test_df
#------------------------------------------------------------------------------

# Interpolate smooth contributions using lookup tables
test_df$smooth_climate_lookup <- interpolate_smooth(
  x_values = test_df$days_over_30C_prev_0_mo,
  lookup_table = climate_lookup,
  x_col = "days_over_30C_prev_0_mo",
  y_col = "smooth_contribution"
)

test_df$smooth_consumption_lookup <- interpolate_smooth(
  x_values = test_df$consumption_pd,
  lookup_table = consumption_lookup,
  x_col = "consumption_pd",
  y_col = "smooth_contribution"
)

# Calculate predictions using lookup tables
test_df$manual_linear_lookup <- intercept +
  test_df$total_precipitation_prev_0_mo * linear_coefs[variable=="total_precipitation_prev_0_mo", coefficient] +
  test_df$sex_id * linear_coefs[variable=="sex_id", coefficient] +
  sapply(test_df$birth_year, function(y) {
    val <- linear_coefs[variable == paste0("birth_year", y), coefficient]
    if (length(val) == 0) 0 else val
  }) +
  test_df$smooth_climate_lookup +
  test_df$smooth_consumption_lookup +
  sapply(test_df$ihme_loc_id, function(loc) {
    val <- random_effect_coefs[ihme_loc_id == loc, coefficient]
    if (length(val) == 0) 0 else val
  })

test_df$manual_prob_lookup <- 1 / (1 + exp(-test_df$manual_linear_lookup))

# Validate against actual predictions
range(test_df$manual_linear_lookup)
range(test_df$pred_linear)
range(test_df$manual_prob_lookup)
range(test_df$pred_prob)



#------------------------------------------------------------------------------
# 8.5: Visualize lookup tables and interpolation accuracy
#------------------------------------------------------------------------------

# Plot climate smooth lookup
p1 <- ggplot(climate_lookup, aes(x = days_over_30C_prev_0_mo, y = smooth_contribution)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point(data = data.frame(
    days_over_30C_prev_0_mo = test_df$days_over_30C_prev_0_mo,
    smooth_contribution = test_df$smooth_climate_lookup
  ), color = "red", alpha = 0.3, size = 0.5) +
  labs(
    title = "Climate Smooth Term Lookup Table",
    subtitle = "Blue line = lookup table, Red points = interpolated test values",
    x = "Days over 30°C (previous month)",
    y = "Smooth contribution to log-odds"
  ) +
  theme_minimal()

# ggsave(paste0(results_dir, "climate_lookup_table.png"), p1, width = 10, height = 6)

# Plot consumption smooth lookup
p2 <- ggplot(consumption_lookup, aes(x = consumption_pd, y = smooth_contribution)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point(data = data.frame(
    consumption_pd = test_df$consumption_pd,
    smooth_contribution = test_df$smooth_consumption_lookup
  ), color = "red", alpha = 0.3, size = 0.5) +
  labs(
    title = "Consumption Smooth Term Lookup Table",
    subtitle = "Blue line = lookup table, Red points = interpolated test values",
    x = "Consumption per capita",
    y = "Smooth contribution to log-odds"
  ) +
  theme_minimal()

# ggsave(paste0(results_dir, "consumption_lookup_table.png"), p2, width = 10, height = 6)

# Plot prediction accuracy
p3 <- ggplot(test_df, aes(x = pred_prob, y = manual_prob_lookup)) +
  geom_point(alpha = 0.3) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Lookup Table Predictions vs Model Predictions",
    x = "Model predicted probability",
    y = "Lookup table predicted probability"
  ) +
  theme_minimal() +
  coord_fixed(ratio = 1)

# ggsave(paste0(results_dir, "lookup_prediction_accuracy.png"), p3, width = 8, height = 8)

#------------------------------------------------------------------------------
# 8.6: Save all artifacts for deployment
#------------------------------------------------------------------------------

# Save lookup tables
fwrite(climate_lookup, paste0(inference_objects_dir, "climate_smooth_lookup.csv"))
fwrite(consumption_lookup, paste0(inference_objects_dir, "consumption_smooth_lookup.csv"))

# Format and save linear and random effects coefficients

# save coefficients in required inference format:

# example format
# coefficients
ex_coef <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/stunting/models/2025_11_07.05/base_model_coefs.parquet")

# random effects
ex_re <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/stunting/models/2025_11_07.05/base_model_ranef.parquet")

inf_coef <- copy(ex_coef)[.I==0]

coefficients <- fixef(model)

# Convert coefficients to a dataframe
inf_coef <- data.frame(
  index = names(coefficients),
  Estimate = coefficients
)
inf_coef <- setDT(copy(inf_coef))

# rename vars as expected format
required <- c('(Intercept)','consumption_pd','q95_prev_0_mo','total_precipitation_prev_0_mo','C(sex_id)1','C(birth_year)2022')
inf_coef[index=='sex_id',index:='C(sex_id)1']
inf_coef[index=='birth_year2022',index:='C(birth_year)2022']
inf_coef <- inf_coef[index %in% required]
rownames(inf_coef) <- inf_coef$index
inf_coef$index <- NULL

outfile_coef <- gsub("_summary","_coefs.csv",summary_file)
write.csv(inf_coef,paste0(inference_objects_dir,outfile),row.names=TRUE)
print(paste0(inference_objects_dir,outfile_coef))

inf_re <- copy(re_df)
setnames(inf_re,old=c("random_effects","ihme_loc_id"),new=c("X.Intercept.","index"))
rownames(inf_re) <- inf_re$index
inf_re$index <- NULL
outfile_re <- gsub("_summary","_ranef.csv",summary_file)
write.csv(inf_re,paste0(inference_objects_dir,outfile),row.names = TRUE)
print(paste0(inference_objects_dir,outfile_re))


###################################################
# Save in default format
fwrite(linear_coefs, paste0(inference_objects_dir, "linear_coefficients.csv"))


# Save random effects
fwrite(random_effect_coefs, paste0(inference_objects_dir, "random_effects.csv"))


#==============================================================================
# SECTION 7: ARCHIVED TROUBLESHOOTING
#==============================================================================


# monotonicity:
## PredictMat tests:
basis_functions_pred_climate <- PredictMat(climate_smooth, test_df)
basis_functions_pred_consumption <- PredictMat(consumption_smooth, test_df)

# Calculate smooth contributions 
smooth_contribution_pred_climate <- basis_functions_pred_climate %*% climate_spline_coefs
smooth_contribution_pred_consumption <- basis_functions_pred_consumption %*% consumption_spline_coefs
# ERROR: non-conformable arguments. PredictMat
# produces 10 columns, whereas climate_spline_coefs only has 9 terms

# Combine contributions to calculate the linear predictor
test_df$manual_predmat_linear <- as.vector(smooth_contribution_pred_climate) +
  as.vector(smooth_contribution_pred_consumption) +
  as.vector(linear_contribution) +
  as.vector(random_effect_contribution)

# Apply the inverse logit function to calculate predicted probabilities
test_df$manual_predmat_prob <- 1 / (1 + exp(-test_df$manual_predmat_linear))

range(test_df$manual_predmat_linear)
range(test_df$manual_predmat_prob)



#==============================================================================
# EXPERIMENTS
#==============================================================================


#==============================================================================
# EXPERIMENT: VERIFY PREDICT AGAINST LP MATRIX
#==============================================================================



# If lpmatrix works, this is our solution for manual calculation
# Extract the lpmatrix for the full test_df
lpmatrix_full <- predict(model, newdata = test_df, type = "lpmatrix")

# First test the whole matrix against predictions
# Reconstruct link (linear predictor)
manual_link <- as.vector(lpmatrix_full %*% coef(model))

# Reconstruct response (probability)
manual_response <- 1 / (1 + exp(-manual_link))

# Get model predictions for comparison
model_link <- predict(model, newdata = test_df, type = "link")
model_response <- predict(model, newdata = test_df, type = "response")

range(manual_link)
# result: [1] -32.616819  -1.102587
range(model_link)
# result: [1] -3.437606 -3.273758

range(model_response)
# result: [1] 0.03114063 0.03648251
range(manual_response)
# result: [1] 0.000000000000006834323 0.249255449773025911098
 
# FAILS

# diagnosis
pred_terms <- predict(model, newdata = test_df, type = "terms")
intercept <- coef(model)["(Intercept)"]
manual_from_terms <- intercept + rowSums(pred_terms)
range(manual_from_terms)
basis_climate <- PredictMat(climate_smooth, test_df)

range(pred_terms$s.days_over_30C_prev_0_mo.)
# does any combination of columns from basis_climate%*%climate_spline_coefs = pred_terms$s.days_over_30C_prev_0_mo.?
columns = 1:10
for (i in 1:10){
  # drop i column
  test_bases <- basis_climate[,columns[-i]]
  test_spline_contribution <- test_bases%*%climate_spline_coefs
  print(paste0("dropping column ",i))
  print("range = ")
  print(range(test_spline_contribution))
  
  # test centered
  center_test <- test_spline_contribution - mean(test_spline_contribution)+mean(pred_terms[, "s.days_over_30C_prev_0_mo."])
  print("center range=")
  print(range(center_test))
  
}
# FAILS - no obvious combination of columns from PredictMat result in same predictions 

# Get model predictions for comparison
model_link <- predict(model, newdata = test_df_clean, type = "link")
model_response <- predict(model, newdata = test_df_clean, type = "response")


# Extract climate and consumption basis from lpmatrix
basis_climate_lp <- lpmatrix_full[, climate_spline_indices]
basis_consumption_lp <- lpmatrix_full[, consumption_spline_indices]

# Test the corrected basis
smooth_climate_lp <- basis_climate_lp %*% climate_spline_coefs
smooth_consumption_lp <- basis_consumption_lp %*% consumption_spline_coefs

test_df$manual_linear_lp <- as.vector(smooth_climate_lp) +
  as.vector(smooth_consumption_lp) +
  as.vector(linear_contribution) +
  as.vector(random_effect_contribution)

test_df$manual_prob_lp <- 1 / (1 + exp(-test_df$manual_linear_lp))

range(test_df$manual_prob_lp)
range(test_df$manual_linear_lp)

# FAILS

#==============================================================================
# USE Reconstruct FROM predict(type="terms")
#==============================================================================

pred_terms <- predict(model, newdata = test_df, type = "terms")

smooth_climate_correct <- pred_terms[, "s(days_over_30C_prev_0_mo)"]
smooth_consumption_correct <- pred_terms[, "s(consumption_pd)"]

# Rebuild linear predictor
intercept <- coef(model)["(Intercept)"]

test_df$manual_linear_from_pred_terms <- intercept +
  test_df$total_precipitation_prev_0_mo * linear_coefs[variable=="total_precipitation_prev_0_mo", coefficient] +
  test_df$sex_id * linear_coefs[variable=="sex_id", coefficient] +
  sapply(test_df$birth_year, function(y) {
    val <- linear_coefs[variable == paste0("birth_year", y), coefficient]
    if (length(val) == 0) 0 else val
  }) +
  smooth_climate_correct +
  smooth_consumption_correct +
  sapply(test_df$ihme_loc_id, function(loc) {
    val <- random_effect_coefs[ihme_loc_id == loc, coefficient]
    if (length(val) == 0) 0 else val
  })

test_df$manual_prob_from_pred_terms <- 1 / (1 + exp(-test_df$manual_linear_from_pred_terms))

range(test_df$manual_prob_from_pred_terms)
range(test_df$pred_prob)
range(test_df$manual_linear_from_pred_terms)

# PASSES