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
summary_file <- "cm_7m_logistic_splines_no_re"

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



# Get data sample ~7m rows
sample_percent <- 7000000/nrow(df_model) # 
indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
indv_counts <- indv_dt[, .N, by = ihme_loc_id]# format vars
df_model <- data.table(df_model)
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,days_over_30C:=as.numeric(days_over_30C)] # this is now a weighted avg
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]
# new changes
df_model[,birth_year:=as.integer(birth_year)]
df_model[, child_mortality := as.integer(child_mortality)]
df_model[, indv_id := factor(as.character(indv_id))]
indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
indv_dt[, n_sample := floor(sample_percent * N)]

set.seed(42)
sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
df_sample <- df_model[indv_id %in% sampled_indv]

print("Number of unique individuals in sample:")
print(length(unique(df_sample$indv_id))) 

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
                s(consumption_pd, bs="mpd") +
                s(days_over_30C, bs="mpi")+
                total_precipitation+
                birth_year+
                s(ihme_loc_id, bs = "re"),
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


#==============================================================================
# SECTION 3: PLOT ISOLATED SPLINE TERM CONTRIBUTIONS
#==============================================================================
library(ggplot2)

plot_dir <- paste0(results_dir, "plots/")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# --- Base-R spline plots via scam's plot method ---
png(paste0(plot_dir, summary_file, "_spline_terms.png"),
    width = 1200, height = 500, res = 150)
par(mfrow = c(1, 2))
plot(model, select = 1, shade = TRUE, shade.col = "lightblue",
     main = "s(consumption_pd, bs='mpd')", ylab = "Partial effect (log-odds)")
plot(model, select = 2, shade = TRUE, shade.col = "lightcoral",
     main = "s(days_over_30C, bs='mpi')", ylab = "Partial effect (log-odds)")
dev.off()

# --- ggplot versions using predict(type="terms") ---
# Build a template row: all age dummies = 1, median for continuous covariates
template <- data.table(
  age_1_m  = 1, age_3_m  = 1, age_6_m  = 1, age_12_m = 1,
  age_24_m = 1, age_36_m = 1, age_48_m = 1, age_60_m = 1,
  sex_id   = factor("Male", levels = c("Male", "Female")),
  total_precipitation = median(df_sample$total_precipitation, na.rm = TRUE),
  birth_year          = median(df_sample$birth_year, na.rm = TRUE),
  consumption_pd      = median(df_sample$consumption_pd, na.rm = TRUE),
  days_over_30C       = median(df_sample$days_over_30C, na.rm = TRUE),
  ihme_loc_id         = df_sample$ihme_loc_id[1],
  indv_id             = df_sample$indv_id[1]
)

# Grid for consumption_pd
cons_seq <- seq(min(df_sample$consumption_pd, na.rm = TRUE),
                max(df_sample$consumption_pd, na.rm = TRUE),
                length.out = 200)
newdata_cons <- template[rep(1, length(cons_seq))]
newdata_cons[, consumption_pd := cons_seq]

# Grid for days_over_30C
days_seq <- seq(min(df_sample$days_over_30C, na.rm = TRUE),
                max(df_sample$days_over_30C, na.rm = TRUE),
                length.out = 200)
newdata_days <- template[rep(1, length(days_seq))]
newdata_days[, days_over_30C := days_seq]

# Extract per-term contributions (linear predictor scale), zeroing out REs
pred_cons <- predict(model, newdata = newdata_cons, type = "terms",
                     exclude = c("s(ihme_loc_id)", "s(indv_id)"))
pred_days <- predict(model, newdata = newdata_days, type = "terms",
                     exclude = c("s(ihme_loc_id)", "s(indv_id)"))

cons_effect <- pred_cons[, "s(consumption_pd)"]
days_effect <- pred_days[, "s(days_over_30C)"]

p1 <- ggplot(data.frame(consumption_pd = cons_seq, effect = cons_effect),
             aes(x = consumption_pd, y = effect)) +
  geom_line(color = "steelblue", linewidth = 1) +
  labs(x = "Consumption per day", y = "Partial effect (log-odds)",
       title = "Monotone decreasing spline: consumption_pd") +
  theme_minimal()
ggsave(paste0(plot_dir, summary_file, "_spline_consumption.png"), p1,
       width = 6, height = 4, dpi = 150)

p2 <- ggplot(data.frame(days_over_30C = days_seq, effect = days_effect),
             aes(x = days_over_30C, y = effect)) +
  geom_line(color = "firebrick", linewidth = 1) +
  labs(x = "Days over 30\u00B0C", y = "Partial effect (log-odds)",
       title = "Monotone increasing spline: days_over_30C") +
  theme_minimal()
ggsave(paste0(plot_dir, summary_file, "_spline_days_over_30C.png"), p2,
       width = 6, height = 4, dpi = 150)

#==============================================================================
# SECTION 4: PREDICT ON NEW DATA — CUMULATIVE MORTALITY THROUGH 60 MONTHS
#==============================================================================
# Setting all age dummies to 1 sums the coefficient contributions from every
# age interval the child passes through, giving a prediction that reflects
# total exposure to mortality risk from birth through 60 months.
# Random effects are excluded so predictions are population-average.

# 2-D grid over both spline variables
pred_grid <- CJ(
  days_over_30C  = seq(min(df_sample$days_over_30C, na.rm = TRUE),
                       max(df_sample$days_over_30C, na.rm = TRUE),
                       length.out = 50),
  consumption_pd = seq(min(df_sample$consumption_pd, na.rm = TRUE),
                       max(df_sample$consumption_pd, na.rm = TRUE),
                       length.out = 50)
)
pred_grid[, `:=`(
  age_1_m  = 1, age_3_m  = 1, age_6_m  = 1, age_12_m = 1,
  age_24_m = 1, age_36_m = 1, age_48_m = 1, age_60_m = 1,
  sex_id   = factor("Male", levels = c("Male", "Female")),
  total_precipitation = median(df_sample$total_precipitation, na.rm = TRUE),
  birth_year          = median(df_sample$birth_year, na.rm = TRUE),
  ihme_loc_id         = df_sample$ihme_loc_id[1],
  indv_id             = df_sample$indv_id[1]
)]

pred_grid[, pred_prob := predict(model, newdata = pred_grid, type = "response",
                                 exclude = c("s(ihme_loc_id)", "s(indv_id)"))]

fwrite(pred_grid, paste0(results_dir, summary_file, "_predictions.csv"))

# --- Marginal effect of days_over_30C at median consumption ---
marginal_days <- data.table(
  days_over_30C = seq(min(df_sample$days_over_30C, na.rm = TRUE),
                      max(df_sample$days_over_30C, na.rm = TRUE),
                      length.out = 200),
  consumption_pd      = median(df_sample$consumption_pd, na.rm = TRUE),
  age_1_m  = 1, age_3_m  = 1, age_6_m  = 1, age_12_m = 1,
  age_24_m = 1, age_36_m = 1, age_48_m = 1, age_60_m = 1,
  sex_id   = factor("Male", levels = c("Male", "Female")),
  total_precipitation = median(df_sample$total_precipitation, na.rm = TRUE),
  birth_year          = median(df_sample$birth_year, na.rm = TRUE),
  ihme_loc_id         = df_sample$ihme_loc_id[1],
  indv_id             = df_sample$indv_id[1]
)
marginal_days[, pred_prob := predict(model, newdata = marginal_days,
                                     type = "response",
                                     exclude = c("s(ihme_loc_id)", "s(indv_id)"))]

p4 <- ggplot(marginal_days, aes(x = days_over_30C, y = pred_prob)) +
  geom_line(color = "firebrick", linewidth = 1) +
  labs(x = "Days over 30\u00B0C", y = "P(mortality before 60 months)",
       title = "Marginal effect of heat days (at median consumption)") +
  theme_minimal()
ggsave(paste0(plot_dir, summary_file, "_marginal_days.png"), p4,
       width = 6, height = 4, dpi = 150)

# --- Marginal effect of consumption_pd at median days_over_30C ---
marginal_cons <- data.table(
  consumption_pd = seq(min(df_sample$consumption_pd, na.rm = TRUE),
                       max(df_sample$consumption_pd, na.rm = TRUE),
                       length.out = 200),
  days_over_30C       = median(df_sample$days_over_30C, na.rm = TRUE),
  age_1_m  = 1, age_3_m  = 1, age_6_m  = 1, age_12_m = 1,
  age_24_m = 1, age_36_m = 1, age_48_m = 1, age_60_m = 1,
  sex_id   = factor("Male", levels = c("Male", "Female")),
  total_precipitation = median(df_sample$total_precipitation, na.rm = TRUE),
  birth_year          = median(df_sample$birth_year, na.rm = TRUE),
  ihme_loc_id         = df_sample$ihme_loc_id[1],
  indv_id             = df_sample$indv_id[1]
)
marginal_cons[, pred_prob := predict(model, newdata = marginal_cons,
                                     type = "response",
                                     exclude = c("s(ihme_loc_id)", "s(indv_id)"))]

p5 <- ggplot(marginal_cons, aes(x = consumption_pd, y = pred_prob)) +
  geom_line(color = "steelblue", linewidth = 1) +
  labs(x = "Consumption per day", y = "P(mortality before 60 months)",
       title = "Marginal effect of consumption (at median heat days)") +
  theme_minimal()
ggsave(paste0(plot_dir, summary_file, "_marginal_consumption.png"), p5,
       width = 6, height = 4, dpi = 150)

message("Prediction and plotting complete. Outputs saved to: ", plot_dir)