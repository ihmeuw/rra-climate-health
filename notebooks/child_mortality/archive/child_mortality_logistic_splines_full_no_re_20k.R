################################################################################
# PROJECT: Climate nutrition
# DATE: 2026-04-13
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
library(ggplot2)


options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- "cm_20k_10yr_cutoff_splines_no_re" 

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_13.01/data_binned.parquet" 
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_04_13.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

## Read and format data
df <- read_parquet(data_version)
df <- data.table(df) # 54 m obs

# Impose time cutoff between interview year and birth year of 10 years
df <- df[int_birth_year_diff_months<=120] # 22.8 m obs

climate_vars <- c(
  "mean_temperature_monthly",
  "total_precipitation_monthly",
  "days_over_30C_monthly"
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

df_model <- data.table(df_model)
df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,days_over_30C_monthly:=as.numeric(days_over_30C_monthly)] # this is now a weighted avg
df_model[,total_precipitation_monthly:=as.numeric(total_precipitation_monthly)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]
# new changes
df_model[,birth_year:=as.integer(birth_year)]
df_model[, child_mortality := as.integer(child_mortality)]
df_model[, indv_id := factor(as.character(indv_id))]

# # Get data sample if needed
sample_percent <- 20000/nrow(df_model) # 
indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
indv_counts <- indv_dt[, .N, by = ihme_loc_id]# format vars

indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
indv_dt[, n_sample := floor(sample_percent * N)]

set.seed(42)
sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
df_sample <- df_model[indv_id %in% sampled_indv]

print("Number of rows after cutoff")
print(length(df_sample$indv_id))
print("Number of unique individuals in sample:")
print(length(unique(df_sample$indv_id)))

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
                s(days_over_30C_monthly, bs="mpi")+
                total_precipitation_monthly+
                birth_year+
                s(ihme_loc_id, bs = "re"),
              family = binomial(link = "logit"),
              data = df_sample)

summary(model)


# save model parameters for future use:
saveRDS(model, file = paste0(results_dir, summary_file,".rds"))

# Read back in if necessary
# model = readRDS(file = paste0(results_dir, summary_file,".rds"))
# save model summary:
summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
capture.output(summary(model), file = summary_file_path)

# Extract random effects and save to CSV
random_effects_file_path <- paste0(model_summary_dir, summary_file, "_random_effects.csv")
coefs <- data.frame(coef(model))
coefs$variable <- names(coef(model))
setDT(coefs)
names(coefs) <- c("coefficient","variable")
coefs <- coefs[,.(variable,coefficient)]

loc_levels <- data.frame(ihme_loc_id = levels(df_model$ihme_loc_id))

random_effect_terms <- grep("^s\\(ihme_loc_id\\)", coefs$variable, value = TRUE)

# Create a mapping of random effect terms to ihme_loc_id levels
random_effect_mapping <- data.table(
  variable = random_effect_terms,
  ihme_loc_id = loc_levels$ihme_loc_id
)

# Replace random effect terms in coefs with corresponding ihme_loc_id levels
coefs <- merge(coefs, random_effect_mapping, by = "variable", all.x = TRUE)

# If the replacement is successful, update the variable column
coefs[, variable := ifelse(!is.na(ihme_loc_id), ihme_loc_id, variable)]

# Drop the temporary ihme_loc_id column
random_effects <- coefs[!is.na(ihme_loc_id)] 
random_effects[, ihme_loc_id := NULL]
coefs[, ihme_loc_id := NULL]


write.csv(
  random_effects,
  file = random_effects_file_path,
  row.names = FALSE
)

#==============================================================================
# SECTION 3: PLOT ISOLATED SPLINE TERM CONTRIBUTIONS
#==============================================================================

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
  total_precipitation_monthly = median(df_model$total_precipitation_monthly, na.rm = TRUE),
  birth_year          = median(df_model$birth_year, na.rm = TRUE),
  consumption_pd      = median(df_model$consumption_pd, na.rm = TRUE),
  days_over_30C_monthly       = median(df_model$days_over_30C_monthly, na.rm = TRUE),
  ihme_loc_id         = df_model$ihme_loc_id[1],
  indv_id             = df_model$indv_id[1]
)

# Grid for consumption_pd
cons_seq <- seq(min(df_model$consumption_pd, na.rm = TRUE),
                max(df_model$consumption_pd, na.rm = TRUE),
                length.out = 200)
newdata_cons <- template[rep(1, length(cons_seq))]
newdata_cons[, consumption_pd := cons_seq]

# Grid for days_over_30C
days_seq <- seq(min(df_model$days_over_30C_monthly, na.rm = TRUE),
                max(df_model$days_over_30C_monthly, na.rm = TRUE),
                length.out = 200)
newdata_days <- template[rep(1, length(days_seq))]
newdata_days[, days_over_30C_monthly := days_seq]

# Extract per-term contributions (linear predictor scale), zeroing out REs
pred_cons <- predict(model, newdata = newdata_cons, type = "terms",
                     exclude = "s(ihme_loc_id)")
pred_days <- predict(model, newdata = newdata_days, type = "terms",
                     exclude = "s(ihme_loc_id)")

cons_effect <- pred_cons[, "s(consumption_pd)"]
days_effect <- pred_days[, "s(days_over_30C_monthly)"]

p1 <- ggplot(data.frame(consumption_pd = cons_seq, effect = cons_effect),
             aes(x = consumption_pd, y = effect)) +
  geom_line(color = "steelblue", linewidth = 1) +
  labs(x = "Consumption per day", y = "Partial effect (log-odds)",
       title = "Monotone decreasing spline: consumption_pd") +
  theme_minimal()
ggsave(paste0(plot_dir, summary_file, "_spline_consumption.png"), p1,
       width = 6, height = 4, dpi = 150)

p2 <- ggplot(data.frame(days_over_30C_monthly = days_seq, effect = days_effect),
             aes(x = days_over_30C_monthly, y = effect)) +
  geom_line(color = "firebrick", linewidth = 1) +
  labs(x = "Days over 30\u00B0C", y = "Partial effect (log-odds)",
       title = "Monotone increasing spline: days_over_30C_monthly") +
  theme_minimal()
ggsave(paste0(plot_dir, summary_file, "_spline_days_over_30C.png"), p2,
       width = 6, height = 4, dpi = 150)

#==============================================================================
# SECTION 3b: PREDICT ON INPUT DATA WITH RANDOM EFFECTS
#==============================================================================
df_model[, pred_prob_re := predict(model, newdata = df_model, type = "response")]
write_parquet(df_model, paste0(results_dir, summary_file, "_input_predictions_with_re.parquet"))
print("Input-data predictions (with RE) saved.")

#==============================================================================
# SECTION 4: PREDICT ON NEW DATA — CUMULATIVE MORTALITY THROUGH 60 MONTHS
#==============================================================================
# Discrete-time survival: for each age interval, activate only that interval's
# dummy (all others = 0) and predict the conditional hazard h_t(x). Then
# combine via the survival product:
#   P(die before 60m) = 1 - prod_t(1 - h_t(x))
# This stays within the training data's feature space, unlike setting all
# age dummies to 1 simultaneously (which is severe extrapolation).

age_vars <- c("age_1_m","age_3_m","age_6_m","age_12_m",
              "age_24_m","age_36_m","age_48_m","age_60_m")

# Helper: given a data.table with all non-age covariates (age dummies will be
# overwritten), returns the cumulative mortality probability for each row.
predict_cumulative_mortality <- function(model, newdata, age_vars) {
  survival <- rep(1, nrow(newdata))
  for (av in age_vars) {
    row_data <- copy(newdata)
    row_data[, (age_vars) := 0L]
    row_data[, (av) := 1L]
    h <- predict(model, newdata = row_data, type = "response",
                 exclude = "s(ihme_loc_id)")
    survival <- survival * (1 - h)
  }
  return(1 - survival)
}

# 2-D grid over both spline variables
pred_grid <- CJ(
  days_over_30C_monthly = seq(min(df_model$days_over_30C_monthly, na.rm = TRUE),
                              max(df_model$days_over_30C_monthly, na.rm = TRUE),
                              length.out = 50),
  consumption_pd        = seq(min(df_model$consumption_pd, na.rm = TRUE),
                              max(df_model$consumption_pd, na.rm = TRUE),
                              length.out = 50)
)
pred_grid[, `:=`(
  age_1_m  = 0L, age_3_m  = 0L, age_6_m  = 0L, age_12_m = 0L,
  age_24_m = 0L, age_36_m = 0L, age_48_m = 0L, age_60_m = 0L,
  sex_id   = factor("Male", levels = c("Male", "Female")),
  total_precipitation_monthly = median(df_model$total_precipitation_monthly, na.rm = TRUE),
  birth_year          = as.integer(median(df_model$birth_year, na.rm = TRUE)),
  ihme_loc_id         = df_model$ihme_loc_id[1],
  indv_id             = df_model$indv_id[1]
)]

pred_grid[, pred_prob := predict_cumulative_mortality(model, pred_grid, age_vars)]

fwrite(pred_grid, paste0(results_dir, summary_file, "_predictions_with_both_splines_ranged.csv"))

# --- Marginal effect of days_over_30C at median consumption ---
marginal_days <- data.table(
  days_over_30C_monthly = seq(min(df_model$days_over_30C_monthly, na.rm = TRUE),
                              max(df_model$days_over_30C_monthly, na.rm = TRUE),
                              length.out = 200),
  consumption_pd      = median(df_model$consumption_pd, na.rm = TRUE),
  age_1_m  = 0L, age_3_m  = 0L, age_6_m  = 0L, age_12_m = 0L,
  age_24_m = 0L, age_36_m = 0L, age_48_m = 0L, age_60_m = 0L,
  sex_id   = factor("Male", levels = c("Male", "Female")),
  total_precipitation_monthly = median(df_model$total_precipitation_monthly, na.rm = TRUE),
  birth_year          = as.integer(median(df_model$birth_year, na.rm = TRUE)),
  ihme_loc_id         = df_model$ihme_loc_id[1],
  indv_id             = df_model$indv_id[1]
)
marginal_days[, pred_prob := predict_cumulative_mortality(model, marginal_days, age_vars)]

p4 <- ggplot(marginal_days, aes(x = days_over_30C_monthly, y = pred_prob)) +
  geom_line(color = "firebrick", linewidth = 1) +
  labs(x = "Days over 30\u00B0C", y = "P(mortality before 60 months)",
       title = "Marginal effect of heat days (at median consumption)") +
  theme_minimal()
ggsave(paste0(plot_dir, summary_file, "_marginal_days.png"), p4,
       width = 6, height = 4, dpi = 150)

# --- Marginal effect of consumption_pd at median days_over_30C ---
marginal_cons <- data.table(
  consumption_pd = seq(min(df_model$consumption_pd, na.rm = TRUE),
                       max(df_model$consumption_pd, na.rm = TRUE),
                       length.out = 200),
  days_over_30C_monthly = median(df_model$days_over_30C_monthly, na.rm = TRUE),
  age_1_m  = 0L, age_3_m  = 0L, age_6_m  = 0L, age_12_m = 0L,
  age_24_m = 0L, age_36_m = 0L, age_48_m = 0L, age_60_m = 0L,
  sex_id   = factor("Male", levels = c("Male", "Female")),
  total_precipitation_monthly = median(df_model$total_precipitation_monthly, na.rm = TRUE),
  birth_year          = as.integer(median(df_model$birth_year, na.rm = TRUE)),
  ihme_loc_id         = df_model$ihme_loc_id[1],
  indv_id             = df_model$indv_id[1]
)
marginal_cons[, pred_prob := predict_cumulative_mortality(model, marginal_cons, age_vars)]

p5 <- ggplot(marginal_cons, aes(x = consumption_pd, y = pred_prob)) +
  geom_line(color = "steelblue", linewidth = 1) +
  labs(x = "Consumption per day", y = "P(mortality before 60 months)",
       title = "Marginal effect of consumption (at median heat days)") +
  theme_minimal()

ggsave(paste0(plot_dir, summary_file, "_marginal_consumption.png"), p5,
       width = 6, height = 4, dpi = 150)

message("Prediction and plotting complete. Outputs saved to: ", plot_dir)