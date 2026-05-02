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
summary_file_root <- "cm_splines_" 


data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2026_04_28.01/data_cumulative_bins.parquet" 
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2026_04_29.01/"
model_summary_dir <- paste0(results_dir,"model_summaries/")


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

## Read and format data
df <- read_parquet(data_version)
df <- data.table(df) 

# Impose time cutoff between interview year and birth year of 10 years
df <- df[int_birth_year_diff_months<=120] 

setnames(df,old=c("days_over_30C_monthly_cumul","consumption_pd_cumul","mean_temperature_monthly_cumul","total_precipitation_monthly_cumul"),
         new=c("days_over_30C_monthly","consumption_pd","mean_temperature_monthly","total_precipitation_monthly"))

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

# Make sure all NAs are removed
df_model <- na.omit(df_model)

df_model[,ihme_loc_id:=as.factor(ihme_loc_id)]
df_model[,days_over_30C_monthly:=as.numeric(days_over_30C_monthly)] # this is now a weighted avg
df_model[,total_precipitation_monthly:=as.numeric(total_precipitation_monthly)]
df_model[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]
# new changes
df_model[,birth_year:=as.integer(birth_year)]
df_model[, child_mortality := as.integer(child_mortality)]
df_model[, indv_id := factor(as.character(indv_id))]

# # Get data sample if needed
# sample_percent <- 7000000/nrow(df_model) # 
# indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
# indv_counts <- indv_dt[, .N, by = ihme_loc_id]# format vars

# indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
# indv_dt[, n_sample := floor(sample_percent * N)]

# set.seed(42)
# sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
# df_sample <- df_model[indv_id %in% sampled_indv]

print("Number of rows after cutoff")
print(length(df_model$indv_id))
print("Number of unique individuals in sample:")
print(length(unique(df_model$indv_id))) #3,149,184

rm(df) # free space
#drop any  unused columns
df_model <- df_model[,.(child_mortality,
                        age_1_m,
                        age_3_m,
                        age_6_m,
                        age_12_m,
                        age_24_m,
                        age_36_m,
                        age_48_m,
                        age_60_m,
                        sex_id,
                        consumption_pd,
                        days_over_30C_monthly,
                        total_precipitation_monthly,
                        birth_year,
                        ihme_loc_id,
                        age_month)]

#==============================================================================
# SECTION 2: FIT MODEL ON ALL AGES
#==============================================================================

# Check data distributions
quantile(df_model[days_over_30C_monthly>0]$days_over_30C_monthly,probs=c(0.25, 0.5, 0.75))
quantile(df_model[days_over_30C_monthly>0]$days_over_30C_monthly,probs=c(0.3,0.6,0.9))

# Define interior breakpoints only — data boundaries are added automatically
thresh_knots      <- c(1.5, 5.25, 9.3,15.5)    # days_over_30C_monthly
consumption_knots <- c(2.0, 5.0, 10.0,20.0, 30.0)   # consumption_pd

# Builds the full augmented knot vector for scam mpi/mpd smooths.
#   inner_knots : interior breakpoints (data boundaries added automatically)
#   x_data      : raw data vector used to set boundary knots
#   m           : I-spline order passed to s(..., m=m); default 2
#                 → underlying B-spline order = m+1; exterior knots per side = m+1
# Returns list(knots, k) for s(x, k=k, bs="mpi") + knots=list(x=knots)
make_scam_knots <- function(inner_knots, x_data, m = 2L) {
  breaks <- sort(unique(c(range(x_data, na.rm = TRUE), inner_knots)))
  n      <- length(breaks)
  
  h_L <- breaks[2]   - breaks[1]      # gap near left boundary
  h_R <- breaks[n]   - breaks[n - 1]  # gap near right boundary
  
  knots <- c(
    breaks[1] - seq(m + 1L, 1L) * h_L,   # m+1 left exterior knots
    breaks,                                # boundary + inner breakpoints
    breaks[n] + seq(1L, m + 1L) * h_R    # m+1 right exterior knots
  )
  # total knots = n + 2*(m+1)
  # k (basis dimension) = total_knots - (m+1) = n + m
  list(knots = knots, k = n + m)
}

res_thresh      <- make_scam_knots(thresh_knots,      df_model$days_over_30C_monthly)
res_consumption <- make_scam_knots(consumption_knots, df_model$consumption_pd)

# Loop over each age group and create single regression
age_groups <- c('age_1_m','age_3_m','age_6_m','age_12_m',
                  'age_24_m','age_36_m','age_48_m','age_60_m')
for (ag in age_groups){
  
  summary_file <- paste0(summary_file_root,ag)
  df_age <- df_model[get(ag)==1]
  
  model <- scam(child_mortality ~
                  sex_id +
                  s(consumption_pd, k = res_consumption$k, bs = "mpd") +
                  s(days_over_30C_monthly, k = res_thresh$k, bs = "mpi") +
                  total_precipitation_monthly +
                  birth_year +
                  s(ihme_loc_id, bs = "re"),
                knots = list(
                  consumption_pd        = res_consumption$knots,
                  days_over_30C_monthly = res_thresh$knots
                ),
                family = binomial(link = "logit"),
                data = df_age)
  
  summary(model)
  
  # save model parameters for future use:
  saveRDS(model, file = paste0(results_dir, summary_file,".rds"))
  
  # Read back in if necessary
  # model = readRDS(file = paste0(results_dir, summary_file,".rds"))
  
  # Verify knots actually used by the fitted model
  # model$smooth[[1]] = consumption_pd, [[2]] = days_over_30C_monthly, [[3]] = ihme_loc_id RE
  m_ord <- 2  # spline order (default)
  cons_full_knots <- model$smooth[[1]]$knots
  days_full_knots <- model$smooth[[2]]$knots
  # Inner knots = full vector minus (m+1) boundary knots on each side
  cons_inner_knots_verified <- cons_full_knots[(m_ord + 2):(length(cons_full_knots) - (m_ord + 1))]
  days_inner_knots_verified <- days_full_knots[(m_ord + 2):(length(days_full_knots) - (m_ord + 1))]
  cat("Consumption inner knots (from model):", cons_inner_knots_verified, "\n")
  cat("Days over 30C inner knots (from model):", days_inner_knots_verified, "\n")
  
  
  
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
  
  
  # write.csv(
  #   random_effects,
  #   file = random_effects_file_path,
  #   row.names = FALSE
  # )
  
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
    geom_vline(xintercept = cons_inner_knots_verified, linetype = "dashed",
               color = "gray40", alpha = 0.7) +
    labs(x = "Consumption per day", y = "Partial effect (log-odds)",
         title = paste0(ag," Monotone decreasing spline: consumption_pd")) +
    # scale_x_continuous(
    #   breaks = sort(unique(c(pretty(cons_seq), cons_inner_knots_verified))),
    #   labels = scales::label_number()
    # ) +
    theme_minimal()
  ggsave(paste0(plot_dir, summary_file, "_spline_consumption.png"), p1,
         width = 6, height = 4, dpi = 150)
  
  p2 <- ggplot(data.frame(days_over_30C_monthly = days_seq, effect = days_effect),
               aes(x = days_over_30C_monthly, y = effect)) +
    geom_line(color = "firebrick", linewidth = 1) +
    geom_vline(xintercept = days_inner_knots_verified, linetype = "dashed",
               color = "gray40", alpha = 0.7) +
    labs(x = "Days over 30°C", y = "Partial effect (log-odds)",
         title = paste0(ag," Monotone increasing spline: days_over_30C_monthly")) +
    # scale_x_continuous(
    #   breaks = sort(unique(c(pretty(days_seq), days_inner_knots_verified))),
    #   labels = scales::label_number()
    # ) +
    theme_minimal()
  ggsave(paste0(plot_dir, summary_file, "_spline_days_over_30C.png"), p2,
         width = 6, height = 4, dpi = 150)
  


}