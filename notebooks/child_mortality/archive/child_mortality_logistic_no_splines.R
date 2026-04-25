################################################################################
# DESCRIPTION: 
# PROJECT: Climate nutrition
# DATE: 2026-04-08
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


library(scam)
library(data.table)
library(dplyr) # for anti_join function
library(arrow) # to read parquet


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- "cm_logistic_no_splines"

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_03_23.01/child_mortality_exploded_binned_age_month.parquet"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_03_23.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

## Read and format data
df <- read_parquet(data_version)
df <- data.table(df) # 54 m obs

# Impose time cutoff between interview year and birth year of 10 years
df <- df[int_birth_year_diff_months<=120] # 22.8 m obs

climate_vars <- c(
  "mean_temperature",
  "total_precipitation",
  "relative_humidity",
  "mean_high_temperature",
  "mean_low_temperature",
  "precipitation_days",
  "days_over_30C"
)
time_vars <- c("age_1_m",
               "age_3_m",
               "age_6_m",
               "age_12_m",
               "age_24_m",
               "age_36_m",
               "age_48_m",
               "age_60_m")

other_vars <- c("indv_id",
                "child_mortality",
                "age_month", 
                "sex_id", 
                "ihme_loc_id", 
                "consumption_pd",
                "birth_year")

cols <- c(time_vars,other_vars,climate_vars)

df_model <- df[, ..cols]

# format vars
df_model <- data.table(df_model)
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,days_over_30C:=as.numeric(days_over_30C)] # this is now a weighted avg
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]
# new changes
df_model[,birth_year:=as.integer(birth_year)]
df_model[, child_mortality := as.integer(child_mortality)]
df_model[, indv_id := factor(as.character(indv_id))]

# Get data sample ~50k rows
sample_percent <- 50000/nrow(df_model) # 2.2 %
indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
indv_counts <- indv_dt[, .N, by = ihme_loc_id]
indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
indv_dt[, n_sample := floor(sample_percent * N)]

set.seed(42)
sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
df_sample <- df_model[indv_id %in% sampled_indv]

length(unique(df_sample$indv_id)) 

# make time interval as per Ryan: alive at 1 month, 3 months, 6 months, 1 yr, etc
# NOTE: Not using currently in factor of set of dummy variables rather than single
# get_time_var <- function(x){
#   # use age_month to find time bin of child's age
#   if (x==1){
#     return("1 mo")
#   }
#   else if(x <= 3){
#     return("1-3 mo")
#   }
#   else if(x <= 6){
#     return("3-6 mo")
#   }
#   else if(x<=12){
#     return("6-12 mo")
#   }
#   else if(x<=24){
#     return("1-2 yr")
#   }
#   else if(x<=36){
#     return("2-3 yr")
#   }
#   else if(x<=48){
#     return("3-4 yr")
#   }
#   else{
#     return("4-5 yr")
#   }
# }
# time_var_levels <- c("1 mo","1-3 mo","3-6 mo","6-12 mo","1-2 yr","2-3 yr","3-4 yr","4-5 yr")
# # this will revert to 1 through 8 if as.numeric(time_var)
# 
# time_vars <- c("age_until_1m"=1,"age_until_3m"=3,"age_until_6m"=6,"age_until_12m"=12,"age_until_24m"=24,
#                "age_until_36m"=36,"age_until_48m"=48,"age_until_60m"=60)
# 
# for (v in names(time_vars)){
#   upper_lim <- time_vars[[v]]
#   df_sample[[v]] <- ifelse(df_sample$age_month >= upper_lim, 1, 0)
# }



# df_model$time_var <- sapply(df_model$age_month,get_time_var)
# df_model$time_var <- factor(df_model$time_var,levels=time_var_levels,ordered = TRUE)

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================

# Testing on sample data:
model <- scam(child_mortality ~
                age_1_m+
                age_3_m+
                age_6_m+
                age_12_m+
                age_24_m+
                age_36_m+
                age_48_m+
                age_60_m+
                sex_id + 
                consumption_pd +
                days_over_30C+
                total_precipitation+
                birth_year+
                s(ihme_loc_id, bs = "re")+
                s(indv_id, bs = "re"),
              family = binomial(link = "logit"),
              data = df_sample)

summary(model)



# fit model
# model <- scam(child_mortality ~ time_var + 
#                      sex_id + 
#                      consumption_pd + 
#                      days_over_30C+
#                      total_precipitation+
#                      birth_year+
#                      s(ihme_loc_id, bs = "re"),
#                    family = binomial(link = "logit"),
#                    data = df_model)

# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, summary_file,".rds"))


# model = readRDS(file = paste0(results_dir, summary_file,".rds"))
# save model summary:
summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
capture.output(summary(model), file = summary_file_path)