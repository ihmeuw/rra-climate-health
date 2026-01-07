################################################################################
# DESCRIPTION: 
# PROJECT: Climate nutrition
# DATE: 2025-06-01
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
library(dplyr)


results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2025_12_16.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")
model_name <- "test_mrbrt"

set.seed(2)
k_studies <- 10
n_per_study <- 5
tau_1 <- 7
sigma_1 <- 1

df_sim_study <- data.frame(study_id = as.factor(1:k_studies)) %>%
  mutate(study_effect1 = rnorm(n = k_studies, mean = 0, sd = tau_1) )

df_sim1 <- do.call("rbind", lapply(1:nrow(df_sim_study), function(i) {
  df_sim_study[rep(i, n_per_study), ] })) %>%
  mutate(
    x1 = runif(n = nrow(.), min = 0, max = 10),
    y1 = 0.9*x1 + study_effect1 + rnorm(nrow(.), mean = 0, sd = sigma_1),
    y1_se = sigma_1,
    is_outlier = FALSE) %>%
  arrange(x1)


dat1 <- mr$MRData()
dat1$load_df(
  data = df_sim1,  col_obs = "y1", col_obs_se = "y1_se",
  col_covs = list("x1"), col_study_id = "study_id" )


mod1 <- mr$MRBRT(
  data = dat1,
  cov_models = list(
    mr$LinearCovModel("intercept", use_re = TRUE),
    mr$LinearCovModel("x1")
  )
)

mod1$fit_model()
summary(mod1)
print(summary(mod1))
print(mod1$summary())


py_save_object(object = mod1, filename = paste0(model_summary_dir,model_name,".pkl"), pickle = "dill")
mod1_back <- py_load_object(filename =  paste0(model_summary_dir,model_name,".pkl"), pickle = "dill")
