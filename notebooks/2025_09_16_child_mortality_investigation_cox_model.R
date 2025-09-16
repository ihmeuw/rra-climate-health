################################################################################
# DESCRIPTION: Test R's coxme package to perform a Cox proportional hazard model
# with mixed effects (r.e. on location, not yet available in Python packages)
# PROJECT: Climate nutrition
# DATE: 2025-09-16
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

install.packages('coxme',lib = "/homes/elyeb/rlibs") # for survival analysis with mixed effects
library(coxme,lib.loc = "/homes/elyeb/rlibs") 
library(data.table)
library(arrow) # to read parquet

options(scipen = 999) # turn off scientific notation
 
#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

df <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_09_15.01/data.parquet")

df <- data.table(df)

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df[,child_mortality := 1-child_alive]
  
setnames(df,old="ldipc_weighted_no_match",new="consumption")

df_model <- df[,.(child_mortality,age_month_at_year_end,sex_id,ihme_loc_id,consumption,mean_temperature,days_over_30C)]

#==============================================================================
# SECTION 2: MAIN PROCESSING
#==============================================================================

fit <- coxme(Surv(age_month_at_year_end, child_mortality) ~ consumption + mean_temperature + days_over_30C + sex_id + (1|ihme_loc_id), data = df_model)
summary(fit)
# Mixed effects coxme model
# Formula: Surv(age_month_at_year_end, child_mortality) ~ consumption +      mean_temperature + days_over_30C + sex_id + (1 | ihme_loc_id) 
# Data: df_model 
# 
# events, n = 42130, 4893786
# 
# Random effects:
#   group  variable       sd variance
# 1 ihme_loc_id Intercept 1.135688 1.289787
# Chisq    df p   AIC   BIC
# Integrated loglik 16118  5.00 0 16108 16065
# Penalized loglik 16241 18.78 0 16204 16041
# 
# Fixed effects:
#   coef    exp(coef)     se(coef)      z                   p
# consumption      -0.000229703  0.999770323  0.000004872 -47.15 <0.0000000000000002
# mean_temperature -0.015280408  0.984835745  0.001395616 -10.95 <0.0000000000000002
# days_over_30C     0.003450759  1.003456720  0.000152407  22.64 <0.0000000000000002
# sex_id           -0.017863722  0.982294889  0.009749483  -1.83              0.0669

