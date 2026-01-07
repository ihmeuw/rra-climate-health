################################################################################
# DESCRIPTION: Script to run baseline model on neonatal mortality data, first 
# using a logistic regression

# PROJECT: Climate nutrition
# DATE: 2025-10-23
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

Sys.setenv("RETICULATE_PYTHON" = '/ihme/code/mscm/miniconda3/envs/mrtool_0.0.2/bin/python')
# Sys.setenv("RETICULATE_PYTHON" = "/ihme/code/mscm/miniconda3/envs/mrtool_0.0.1/bin/python") # this line might be necessary on some Singularity images
library(reticulate)
reticulate::use_python("/ihme/code/mscm/miniconda3/envs/mrtool_0.0.2/bin/python")
mr <- reticulate::import("mrtool")

# install.packages('frailtyEM',lib = "/homes/elyeb/rlibs") # able to handle mixed effects and predict on new data
# library(frailtyEM,lib.loc = "/homes/elyeb/rlibs")
library(data.table)
# library(lme4)
library(arrow) # to read parquet
# install.packages("fastDummies",lib = "/homes/elyeb/rlibs")
library(fastDummies,lib.loc = "/homes/elyeb/rlibs")


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- paste0("nnm_1_mo_do30_mrbrt_factor_yr")


results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/"
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2025_12_16.01/neonatal_data.parquet"


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

model_summary_dir <- paste0(results_dir,"model_summaries/")
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

model_objects_dir <- paste0(results_dir,"model_objects/")
dir.create(model_objects_dir, recursive = TRUE, showWarnings = FALSE)

# Read in neonatal df (must be made from full dataset)
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)

neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
# convert sex_id to int between 0 and 1, where 0 is male and 1 is female
neo_df[,sex_id := as.integer(sex_id)]
neo_df[,sex_id := sex_id-1]

# note: mrbrt cannot handle factor variables as dummies
neo_df <- dummy_cols(neo_df, select_columns = "birth_year", remove_first_dummy = FALSE, remove_selected_columns = TRUE)


# get sample
# indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
# indv_counts <- indv_dt[, .N, by = ihme_loc_id]
# indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
# indv_dt[, n_sample := floor(sample_percent * N)]

# set.seed(42)
# sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
# df_sample <- df_model[indv_id %in% sampled_indv]

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================
## Options

df_model <- neo_df[,.(child_mortality,
                        consumption_pd,
                        days_over_30C_prev_0_mo,
                        total_precipitation_prev_0_mo,
                        sex_id,
                        ihme_loc_id,
                        birth_year_1959,
                        birth_year_1960,
                        birth_year_1961,
                        birth_year_1962,
                        birth_year_1963,
                        birth_year_1964,
                        birth_year_1965,
                        birth_year_1966,
                        birth_year_1967,
                        birth_year_1968,
                        birth_year_1969,
                        birth_year_1970,
                        birth_year_1971,
                        birth_year_1972,
                        birth_year_1973,
                        birth_year_1974,
                        birth_year_1975,
                        birth_year_1976,
                        birth_year_1977,
                        birth_year_1978,
                        birth_year_1979,
                        birth_year_1980,
                        birth_year_1981,
                        birth_year_1982,
                        birth_year_1983,
                        birth_year_1984,
                        birth_year_1985,
                        birth_year_1986,
                        birth_year_1987,
                        birth_year_1988,
                        birth_year_1989,
                        birth_year_1990,
                        birth_year_1991,
                        birth_year_1992,
                        birth_year_1993,
                        birth_year_1994,
                        birth_year_1995,
                        birth_year_1996,
                        birth_year_1997,
                        birth_year_1998,
                        birth_year_1999,
                        birth_year_2000,
                        birth_year_2001,
                        birth_year_2002,
                        birth_year_2003,
                        birth_year_2004,
                        birth_year_2005,
                        birth_year_2006,
                        birth_year_2007,
                        birth_year_2008,
                        birth_year_2009,
                        birth_year_2010,
                        birth_year_2011,
                        birth_year_2012,
                        birth_year_2013,
                        birth_year_2014,
                        birth_year_2015,
                        birth_year_2016,
                        birth_year_2017,
                        birth_year_2018,
                        birth_year_2019,
                        birth_year_2020,
                        birth_year_2021,
                        birth_year_2022,
                        birth_year_2023
                      )]

df_model <- na.omit(df_model)

dat <- mr$MRData()

dat$load_df(
  data = df_model,  
  col_obs = "child_mortality", 
  col_covs = list("consumption_pd",
                  "days_over_30C_prev_0_mo",
                  "total_precipitation_prev_0_mo",
                  "sex_id",
                  "birth_year_1959",
                  "birth_year_1960",
                  "birth_year_1961",
                  "birth_year_1962",
                  "birth_year_1963",
                  "birth_year_1964",
                  "birth_year_1965",
                  "birth_year_1966",
                  "birth_year_1967",
                  "birth_year_1968",
                  "birth_year_1969",
                  "birth_year_1970",
                  "birth_year_1971",
                  "birth_year_1972",
                  "birth_year_1973",
                  "birth_year_1974",
                  "birth_year_1975",
                  "birth_year_1976",
                  "birth_year_1977",
                  "birth_year_1978",
                  "birth_year_1979",
                  "birth_year_1980",
                  "birth_year_1981",
                  "birth_year_1982",
                  "birth_year_1983",
                  "birth_year_1984",
                  "birth_year_1985",
                  "birth_year_1986",
                  "birth_year_1987",
                  "birth_year_1988",
                  "birth_year_1989",
                  "birth_year_1990",
                  "birth_year_1991",
                  "birth_year_1992",
                  "birth_year_1993",
                  "birth_year_1994",
                  "birth_year_1995",
                  "birth_year_1996",
                  "birth_year_1997",
                  "birth_year_1998",
                  "birth_year_1999",
                  "birth_year_2000",
                  "birth_year_2001",
                  "birth_year_2002",
                  "birth_year_2003",
                  "birth_year_2004",
                  "birth_year_2005",
                  "birth_year_2006",
                  "birth_year_2007",
                  "birth_year_2008",
                  "birth_year_2009",
                  "birth_year_2010",
                  "birth_year_2011",
                  "birth_year_2012",
                  "birth_year_2013",
                  "birth_year_2014",
                  "birth_year_2015",
                  "birth_year_2016",
                  "birth_year_2017",
                  "birth_year_2018",
                  "birth_year_2019",
                  "birth_year_2020",
                  "birth_year_2021",
                  "birth_year_2022",
                  "birth_year_2023"
                  ),
  col_study_id = "ihme_loc_id"
  )
# inclusion of "birth_year" causes error, possibly because it's factor variable

model <- mr$MRBRT(
  data = dat,
  cov_models = list(
    mr$LinearCovModel("intercept", use_re = FALSE),
    mr$LinearCovModel(
      alt_cov = "consumption_pd",
      use_spline = TRUE,
      # spline_degree = 2L, # 2L is quadratic
      # spline_knots = array(seq(0, 1, length.out = 5)), # length.out=5 means 3 internal knots
      # spline_knots_type = 'frequency', # 'frequency' means spaced according to data density
      # spline_r_linear = TRUE,
      # spline_l_linear = TRUE
      prior_spline_monotonicity = "increasing"
    ),
    mr$LinearCovModel(
      alt_cov = "days_over_30C_prev_0_mo",
      use_spline = TRUE,
      prior_spline_monotonicity = "increasing"
    ),
    mr$LinearCovModel(
      alt_cov = "total_precipitation_prev_0_mo"
    ),
    mr$LinearCovModel(
      alt_cov = "sex_id"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1959"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1960"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1961"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1962"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1963"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1964"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1965"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1966"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1967"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1968"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1969"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1970"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1971"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1972"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1973"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1974"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1975"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1976"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1977"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1978"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1979"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1980"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1981"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1982"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1983"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1984"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1985"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1986"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1987"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1988"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1989"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1990"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1991"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1992"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1993"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1994"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1995"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1996"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1997"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1998"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_1999"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2000"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2001"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2002"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2003"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2004"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2005"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2006"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2007"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2008"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2009"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2010"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2011"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2012"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2013"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2014"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2015"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2016"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2017"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2018"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2019"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2020"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2021"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2022"
    ),
    mr$LinearCovModel(
      alt_cov = "birth_year_2023"
    )
  )
)


model$fit_model()


print(model$summary())

py_save_object(object = model, filename = paste0(model_objects_dir, summary_file,".pkl"), pickle = "dill")

# Read model back in
# model = py_load_object(filename =  paste0(model_summary_dir,model_name,".pkl"), pickle = "dill")

# summary(model)
# 
# # Extract random effects 
# re_df <- as.data.frame(ranef(model)$ihme_loc_id)
# re_df$ihme_loc_id <- rownames(ranef(model)$ihme_loc_id)
# colnames(re_df)[1] <- "random_effects"
# setorder(re_df,random_effects)
# 
# # Save model summary and random effects to text file
# summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
# capture.output(summary(model), file = summary_file_path)
# cat("\n\n", file = summary_file_path, append = TRUE)
# cat("================================================================================\n", file = summary_file_path, append = TRUE)
# cat("CLUSTER-SPECIFIC RANDOM EFFECTS ESTIMATES\n", file = summary_file_path, append = TRUE)
# cat("================================================================================\n\n", file = summary_file_path, append = TRUE)
# re_output <- capture.output(print(re_df, row.names = FALSE))
# cat(paste(re_output, collapse = "\n"), file = summary_file_path, append = TRUE)
# 
# # Also save frailty estimates as a separate CSV for easier access
# write.csv(re_df, paste0(model_summary_dir, "re_estimates_", summary_file, ".csv"), row.names = FALSE)
# 
# #==============================================================================
# # SECTION 3: PREDICT MODEL FOR NEONATAL ON AVG BIRTH YEAR, SEX, PRECIPITATION
# #==============================================================================
# 
# df_avg <- copy(df_model)
# 
# # Extract levels for a specific factor
# birth_year_levels <- levels(model@frame$birth_year)
# ihme_loc_id_levels <- levels(model@frame$ihme_loc_id)
# 
# df_avg$birth_year <- factor(df_avg$birth_year, levels = birth_year_levels)
# df_avg$ihme_loc_id <- factor(df_avg$ihme_loc_id, levels = ihme_loc_id_levels)
# 
# # remove any NAs imposed from above step (could be because some years didn't make it in the subset)
# # df_avg <- df_avg[!is.na(birth_year)] # should not be a problem on full data
# 
# # # Predict WITH random effects (mixed effects)
# df_avg$pred_me <- predict(model, newdata = df_avg, type = "response", re.form = NULL)
# 
# # # override existing variables to be able to use predict function from package
# df_avg[, birth_year := factor(round(mean(as.numeric(as.character(birth_year))), 0),
#                               levels = birth_year_levels)]
# 
# df_avg[,sex_id:= mean(df_avg$sex_id)]
# 
# df_avg[,total_precipitation_prev_0_mo:= mean(df_avg$total_precipitation_prev_0_mo)]
# 
# # # Predict WITHOUT random effects (fixed effects only)
# df_avg$pred_fe <- predict(model, newdata = df_avg, type = "response", re.form = NA)
# 
# # # Save predictions to parquet
# write_parquet(df_avg, paste0(results_dir, "predictions_", summary_file, ".parquet"))
