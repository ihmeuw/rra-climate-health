################################################################################
# DESCRIPTION: Script to run baseline model on neonatal mortality data, 
# using a logistic regression
# PROJECT: Climate nutrition
# DATE: 2025-12-09
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
# library(mgcv)
library(scam)
library(arrow) # to read parquet
library(ggplot2)
library(dplyr)
library(scales)

options(scipen = 999) # turn off scientific notation

#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

## set parameters
summary_file <- paste0("nnm_1_mo_do30_scam_summary")

results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/results/2026_03_19.04/"
neo_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/neonatal_mortality/training_data/2026_03_19.04/neonatal_data_prev_month_vars.parquet"


dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

model_summary_dir <- paste0(results_dir,"model_summaries/")
dir.create(model_summary_dir, recursive = TRUE, showWarnings = FALSE)

model_objects_dir <- paste0(results_dir,"model_objects/")
dir.create(model_objects_dir, recursive = TRUE, showWarnings = FALSE)

inference_objects_dir <- paste0(results_dir,"inference_format/")
dir.create(inference_objects_dir, recursive = TRUE, showWarnings = FALSE)

plot_dir <- paste0(results_dir,"plots/")

# Read in neonatal df (must be made from full dataset)
neo_df <- read_parquet(neo_version)
neo_df <- data.table(neo_df)

neo_df[,ihme_loc_id:=as.factor(ihme_loc_id)]
# convert sex_id to int between 0 and 1, where 0 is male and 1 is female
neo_df[,sex_id := as.integer(sex_id)]
neo_df[,sex_id := sex_id-1]
neo_df[,days_over_30C_prev_0_mo:=as.integer(days_over_30C_prev_0_mo)]
neo_df[,birth_year:=as.factor(birth_year)]

## Read and format data

climate_vars <- c(
  # "mean_temperature",
  # "total_precipitation",
  # "relative_humidity",
  # "mean_high_temperature",
  # "mean_low_temperature",
  # "precipitation_days",
  # "days_over_30C",
  # "days_over_26C",
  # "any_days_over_30C",
  "days_over_30C_prev_0_mo",
  # "days_over_30C_prev_3_mo_avg",
  # "days_over_30C_prev_6_mo_avg",
  # "days_over_30C_prev_9_mo_avg",
  # 'q9_prev_0_mo',
  # 'q95_prev_0_mo',
  # 'q9_prev_3_mo_avg',
  # 'q9_prev_6_mo_avg',
  # 'q9_prev_9_mo_avg',
  # 'q95_prev_3_mo_avg',
  # 'q95_prev_6_mo_avg',
  # 'q95_prev_9_mo_avg',
  # 'zone',
  "total_precipitation_prev_0_mo"
  # "total_precipitation_prev_3_mo_avg",
  # "total_precipitation_prev_6_mo_avg",
  # "total_precipitation_prev_9_mo_avg"
)
cols <- c("indv_id","child_mortality", "age_month", "sex_id", "ihme_loc_id", "consumption","consumption_pd","birth_year", climate_vars)
df_model <- neo_df[, ..cols]

df_model <- na.omit(df_model)

# get sample
# indv_dt <- unique(df_model[, .(indv_id, ihme_loc_id)])
# indv_counts <- indv_dt[, .N, by = ihme_loc_id]
# indv_dt <- merge(indv_dt, indv_counts, by = "ihme_loc_id", suffixes = c("", "_total"))
# indv_dt[, n_sample := floor(sample_percent * N)]

# set.seed(42)
# sampled_indv <- indv_dt[, .SD[sample(.N, n_sample[1])], by = ihme_loc_id]$indv_id
# df_sample <- df_model[indv_id %in% sampled_indv]

#==============================================================================
# SECTION 2: FIT MODEL 
#==============================================================================



model <- scam(
  child_mortality ~ s(consumption_pd, bs="mpd") +
    s(days_over_30C_prev_0_mo, bs="mpi") +
    total_precipitation_prev_0_mo +
    sex_id +
    birth_year +
    s(ihme_loc_id, bs="re"),
  data = df_model,
  family = binomial(link = "logit")
)

# save model parameters for future use:
saveRDS(model, file = paste0(model_objects_dir, summary_file,".rds"))

# Read model back in
# model = readRDS(file = paste0(model_objects_dir, summary_file,".rds"))
summary(model)

# Extract random effects 
# re_df <- as.data.frame(ranef(model)$ihme_loc_id)
# re_df$ihme_loc_id <- rownames(ranef(model)$ihme_loc_id)
# colnames(re_df)[1] <- "random_effects"
# setorder(re_df,random_effects)
# 
# # Save model summary and random effects to text file
summary_file_path <- paste0(model_summary_dir, summary_file, ".txt")
capture.output(summary(model), file = summary_file_path)
cat("\n\n", file = summary_file_path, append = TRUE)

coefs <- data.frame(coef(model))
coefs$variable <- rownames(coefs)
setDT(coefs)
names(coefs) <- c("coefficient","variable")
coefs <- coefs[,.(variable,coefficient)]

loc_levels <- data.frame(ihme_loc_id = levels(df_model$ihme_loc_id))
# Extract random effect terms from coefs
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
coefs[, ihme_loc_id := NULL]

write.table(
  coefs,
  file = summary_file_path,
  sep = "\t",         # or "," for csv, or " " for space
  row.names = FALSE,
  col.names = TRUE,   # or FALSE if you don't want a header
  quote = FALSE,
  append = TRUE       # append to the file (after your summary)
)


# # save coefficients in required inference format:
# 
# # example format
# # coefficients
# ex_coef <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/stunting/models/2025_11_07.05/base_model_coefs.parquet")
# ex_coef <- read_parquet(paste0(inference_objects_dir,"nnm_1_mo_q95_coefs.parquet"))


# # random effects
# ex_re <- read_parquet("/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/stunting/models/2025_11_07.05/base_model_ranef.parquet")
# ex_re <- read_parquet(paste0(inference_objects_dir,"nnm_1_mo_q95_ranef.parquet"))

# inf_coef <- copy(ex_coef)[.I==0]
# 
coefficients <- copy(coefs)
setDT(coefficients)
var_list <- c("(Intercept)","sex_id","total_precipitation_prev_0_mo","birth_year2022")
coefficients <- coefficients[variable %in% var_list]

setnames(coefficients,old=c("variable","coefficient"),new=c("__index_level_0__","Estimate"))
coefficients <- coefficients[,.(Estimate,`__index_level_0__`)]

inf_coef <- copy(coefficients)

# rename vars as expected format
inf_coef[`__index_level_0__`=='sex_id',`__index_level_0__`:='C(sex_id)1']
inf_coef[`__index_level_0__`=='birth_year2022',`__index_level_0__`:='C(birth_year)2022']
rownames(inf_coef) <- inf_coef$`__index_level_0__`
inf_coef$`__index_level_0__` <- NULL

outfile_coef <- gsub("_summary","_coefs.csv",summary_file)
write.csv(inf_coef,paste0(inference_objects_dir,outfile_coef),row.names=TRUE)
print(paste0(inference_objects_dir,outfile_coef))

# test conversion
# inf_coef_parquet <- read_parquet(paste0(inference_objects_dir,gsub("csv","parquet",outfile_coef)))

# save out random effects
re_df <- copy(coefs)
setnames(re_df,old="variable",new="ihme_loc_id")
re_df <- merge(re_df,random_effect_mapping,by=c("ihme_loc_id"),all.y=TRUE)
re_df$variable <- NULL
inf_re <- copy(re_df)
setnames(inf_re,old=c("coefficient","ihme_loc_id"),new=c("X.Intercept.","index"))

rownames(inf_re) <- inf_re$index
inf_re$index <- NULL
outfile_re <- gsub("_summary","_ranef.csv",summary_file)
write.csv(inf_re,paste0(inference_objects_dir,outfile_re),row.names = TRUE)
print(paste0(inference_objects_dir,outfile_re))

# test conversion
# inf_re_parquet <- read_parquet(paste0(inference_objects_dir,gsub("csv","parquet",outfile_re)))

#==============================================================================
# SECTION 3: PREDICT MODEL FOR NEONATAL ON AVG BIRTH YEAR, SEX, PRECIPITATION
#==============================================================================

df_avg <- copy(df_model)

df_avg$pred_me <- predict(model, newdata = df_avg, type = "response", re.form = NULL)

# save df_model as input data:
write_parquet(df_avg,paste0(results_dir, "predictions_me_only_", summary_file, ".parquet"))

# # # override existing variables to be able to use predict function from package
df_avg[, birth_year := factor(round(mean(as.numeric(as.character(birth_year))), 0),
                              levels = levels(df_model$birth_year))]

df_avg[,sex_id:= mean(df_avg$sex_id)]

df_avg[,total_precipitation_prev_0_mo:= mean(df_avg$total_precipitation_prev_0_mo)]

# Predict WITHOUT random effects (fixed effects only)


df_avg_fixed_loc <- copy(df_avg)
df_avg_fixed_loc[,ihme_loc_id:="CAF"]
df_avg_fixed_loc <- df_avg_fixed_loc[,.(indv_id,
                                        consumption_pd,
                                        days_over_30C_prev_0_mo,
                                        total_precipitation_prev_0_mo,
                                        sex_id,
                                        birth_year,
                                        ihme_loc_id)]
summary(df_avg_fixed_loc)
df_avg_fixed_loc$pred_fe <- predict(model, newdata = df_avg_fixed_loc, type = "response")

df_avg <- merge(df_avg,df_avg_fixed_loc[,.(indv_id,pred_fe)],by="indv_id")



pred_with_se <- predict(model, type = "terms", se.fit = TRUE)

pred_with_se_fit <- data.table(pred_with_se$fit)
pred_with_se_se <- data.table(pred_with_se$se.fit)

fit_vars <- paste0(colnames(pred_with_se_fit),"_contribution")
names(pred_with_se_fit) <- fit_vars
se_vars <- paste0(colnames(pred_with_se_se),"_se")
names(pred_with_se_se) <- se_vars

df_avg <- cbind(df_avg,pred_with_se_fit)
df_avg <- cbind(df_avg,pred_with_se_se)

# add psu back on
psu <- unique(neo_df[,.(indv_id,psu)])
df_avg_psu <- merge(df_avg,psu,by="indv_id")

setnames(df_avg_psu,
         old=c("s(consumption_pd)_contribution",
               "s(days_over_30C_prev_0_mo)_contribution",
               "s(consumption_pd)_se",
               "s(days_over_30C_prev_0_mo)_se"),
         new=c("s_consumption_pd_contribution",
               "s_days_over_30C_contribution",
               "s_consumption_pd_se",
               "s_days_over_30C_se"))

df_avg_psu[,s_consumption_pd_contribution_upper:=s_consumption_pd_contribution+1.96*s_consumption_pd_se]
df_avg_psu[,s_consumption_pd_contribution_lower:=s_consumption_pd_contribution-1.96*s_consumption_pd_se]
df_avg_psu[,s_days_over_30C_contribution_upper:=s_days_over_30C_contribution+1.96*s_days_over_30C_se]
df_avg_psu[,s_days_over_30C_contribution_lower:=s_days_over_30C_contribution-1.96*s_days_over_30C_se]


# make additional predictions trying to hold rest of other variables flat

# # Save predictions to parquet
write_parquet(df_avg_psu, paste0(results_dir, "predictions_", summary_file, "with_psu.parquet"))

# make synthetic data sets that are smaller but for the purpose of plotting
# marginal effects for q95 and consumption

df_fixed_consumption <- copy(df_avg)
df_fixed_consumption[,consumption_pd := mean(df_avg$consumption_pd)]
df_fixed_consumption[,ihme_loc_id:="CAF"]
df_fixed_consumption <- df_fixed_consumption[,.(consumption_pd,
                                                days_over_30C_prev_0_mo,
                                                total_precipitation_prev_0_mo,
                                                sex_id,
                                                birth_year,
                                                ihme_loc_id)]
summary(df_fixed_consumption)
range(df_fixed_consumption$days_over_30C_prev_0_mo) #[1]  0 31
days_over_30C_range <- seq(min(df_fixed_consumption$days_over_30C_prev_0_mo),max(df_fixed_consumption$days_over_30C_prev_0_mo),0.1)
df_fixed_consumption$days_over_30C_prev_0_mo <- NULL
df_fixed_consumption <- df_fixed_consumption[0:length(days_over_30C_range)]
df_fixed_consumption[,days_over_30C_prev_0_mo:=days_over_30C_range]
df_fixed_consumption[,statistic := "mean_consumption_pd"]

# repeat for upper and lower consumption_pd
df_fixed_consumption_upper <- copy(df_avg)
df_fixed_consumption_upper[,consumption_pd := min(mean(df_avg$consumption_pd)+1.96*sd(df_avg$consumption_pd),max(df_avg$consumption_pd))]
df_fixed_consumption_upper[,ihme_loc_id:="CAF"]
df_fixed_consumption_upper <- df_fixed_consumption_upper[,.(consumption_pd,
                                                            days_over_30C_prev_0_mo,
                                                total_precipitation_prev_0_mo,
                                                sex_id,
                                                birth_year,
                                                ihme_loc_id)]

df_fixed_consumption_upper$days_over_30C_prev_0_mo <- NULL
df_fixed_consumption_upper <- df_fixed_consumption_upper[0:length(days_over_30C_range)]
df_fixed_consumption_upper[,days_over_30C_prev_0_mo:=days_over_30C_range]
df_fixed_consumption_upper[,statistic := "upper_consumption_pd"]

df_fixed_consumption_lower <- copy(df_avg)
df_fixed_consumption_lower[,consumption_pd := max(mean(df_avg$consumption_pd)-1.96*sd(df_avg$consumption_pd),min(df_avg$consumption_pd))]
df_fixed_consumption_lower[,ihme_loc_id:="CAF"]
df_fixed_consumption_lower <- df_fixed_consumption_lower[,.(consumption_pd,
                                                            days_over_30C_prev_0_mo,
                                                            total_precipitation_prev_0_mo,
                                                            sex_id,
                                                            birth_year,
                                                            ihme_loc_id)]

df_fixed_consumption_lower$days_over_30C_prev_0_mo <- NULL
df_fixed_consumption_lower <- df_fixed_consumption_lower[0:length(days_over_30C_range)]
df_fixed_consumption_lower[,days_over_30C_prev_0_mo:=days_over_30C_range]
df_fixed_consumption_lower[,statistic := "lower_consumption_pd"]

df_fixed_consumption <- rbind(df_fixed_consumption,df_fixed_consumption_lower,df_fixed_consumption_upper)

table(df_fixed_consumption$consumption_pd)

# pred_fixed_consumption <- predict(model, newdata = df_fixed_consumption, type = "response", re.form = NA)
df_fixed_consumption$pred_fixed_consumption <- predict(model, newdata = df_fixed_consumption, type = "response")
df_fixed_consumption$linear_predictor <- predict(model, newdata = df_fixed_consumption, type = "link")
range(df_fixed_consumption$linear_predictor)

write.csv(df_fixed_consumption,paste0(results_dir, "single_var_spline_test_", summary_file, ".csv"))

# sanity check
all(diff(df_fixed_consumption$pred_fixed_consumption) >= 0) # TRUE
write_parquet(df_fixed_consumption, paste0(results_dir, "predictions_fixed_consumption_", summary_file, ".parquet"))
paste0(results_dir, "predictions_fixed_consumption_", summary_file, ".parquet")

# Repeat by fixing days_over_30C
df_fixed_do30 <- copy(df_avg)
df_fixed_do30[,days_over_30C_prev_0_mo := mean(df_avg$days_over_30C_prev_0_mo)]
df_fixed_do30[,ihme_loc_id:="CAF"]
df_fixed_do30 <- df_fixed_do30[,.(consumption_pd,
                                  days_over_30C_prev_0_mo,
                                total_precipitation_prev_0_mo,
                                sex_id,birth_year,
                                ihme_loc_id)]
summary(df_fixed_do30)
range(df_fixed_do30$consumption_pd) #0.0000356825 112.7981388261
consumption_pd_range <- seq(min(df_fixed_do30$consumption_pd),max(df_fixed_do30$consumption_pd),0.1)

df_fixed_do30$consumption_pd <- NULL
df_fixed_do30 <- df_fixed_do30[0:length(consumption_pd_range)]
df_fixed_do30[,consumption_pd:=consumption_pd_range]
df_fixed_do30[,statistic := "mean_do30"]

# repeat for upper/lower
df_fixed_do30_upper <- copy(df_avg)
df_fixed_do30_upper[,days_over_30C_prev_0_mo := min(mean(df_avg$days_over_30C_prev_0_mo)+1.96*sd(df_avg$days_over_30C_prev_0_mo),max(df_avg$days_over_30C_prev_0_mo))]
df_fixed_do30_upper[,ihme_loc_id:="CAF"]
df_fixed_do30_upper <- df_fixed_do30_upper[,.(consumption_pd,
                                              days_over_30C_prev_0_mo,
                                              total_precipitation_prev_0_mo,
                                              sex_id,birth_year,
                                              ihme_loc_id)]

df_fixed_do30_upper$consumption_pd <- NULL
df_fixed_do30_upper <- df_fixed_do30_upper[0:length(consumption_pd_range)]
df_fixed_do30_upper[,consumption_pd:=consumption_pd_range]
df_fixed_do30_upper[,statistic := "upper_do30"]

df_fixed_do30_lower <- copy(df_avg)
df_fixed_do30_lower[,days_over_30C_prev_0_mo := max(mean(df_avg$days_over_30C_prev_0_mo)-1.96*sd(df_avg$days_over_30C_prev_0_mo),min(df_avg$days_over_30C_prev_0_mo))]
df_fixed_do30_lower[,ihme_loc_id:="CAF"]
df_fixed_do30_lower <- df_fixed_do30_lower[,.(consumption_pd,
                                              days_over_30C_prev_0_mo,
                                            total_precipitation_prev_0_mo,
                                            sex_id,birth_year,
                                            ihme_loc_id)]

df_fixed_do30_lower$consumption_pd <- NULL
df_fixed_do30_lower <- df_fixed_do30_lower[0:length(consumption_pd_range)]
df_fixed_do30_lower[,consumption_pd:=consumption_pd_range]
df_fixed_do30_lower[,statistic := "lower_do30"]

df_fixed_do30 <- rbind(df_fixed_do30,df_fixed_do30_lower,df_fixed_do30_upper)

table(df_fixed_do30_lower$days_over_30C_prev_0_mo)
table(df_fixed_do30_upper$days_over_30C_prev_0_mo)
table(df_fixed_do30$days_over_30C_prev_0_mo)

# pred_fixed_consumption <- predict(model, newdata = df_fixed_consumption, type = "response", re.form = NA)
df_fixed_do30$pred_fixed_do30 <- predict(model, newdata = df_fixed_do30, type = "response")

# sanity check
all(diff(df_fixed_do30$days_over_30C_prev_0_mo) <= 0) # TRUE
write_parquet(df_fixed_do30, paste0(results_dir, "predictions_fixed_do30_", summary_file, ".parquet"))
paste0(results_dir, "predictions_fixed_do30_", summary_file, ".parquet")

#==============================================================================
# SECTION 4: CUSTOM PLOTS
#==============================================================================


# 
# # Create a data frame with q95_prev_0_mo and its spline contribution
# fit <- data.table(pred_with_se$fit)
# se_fit <- data.table(pred_with_se$se.fit)
# 
# plot_data <- data.table(
#   consumption_pd = df_model$consumption_pd,
#   days_over_30C = df_model$days_over_30C_prev_0_mo,
#   spline_contribution_consumption_pd = fit[["s(consumption_pd)"]],
#   spline_contribution_do30 = fit[["s(days_over_30C_prev_0_mo)"]],
#   se_do30 = se_fit[["s(days_over_30C_prev_0_mo)"]],
#   se_consumption_pd = se_fit[["s(consumption_pd)"]]
# )
# 
# 
# plot_data <- plot_data %>%
#   mutate(
#     do30_lower_ci = spline_contribution_do30 - 1.96 * se_do30,
#     do30_upper_ci = spline_contribution_do30 + 1.96 * se_do30,
#     c_lower_ci = spline_contribution_consumption_pd - 1.96 * se_consumption_pd,
#     c_upper_ci = spline_contribution_consumption_pd + 1.96 * se_consumption_pd
#   )
# 
# ## Add lines for linear models
# linear_do30 <- 0.0064507
# consumption_linear <- -0.0426716
# 
# # plot do30
# p_do30 <- ggplot(plot_data, aes(x = days_over_30C, y = spline_contribution_do30)) +
#   geom_line(aes(color = "Spline")) +
#   geom_ribbon(aes(ymin = do30_lower_ci, ymax = do30_upper_ci), alpha = 0.2, fill = "blue") +
#   geom_abline(
#     aes(color = "Slope from linear model",
#     slope = linear_do30,
#     intercept = -0.0217828),
#     linetype = "dashed"
#   ) +
#   scale_color_manual(
#     name = "Legend",  # Legend title
#     values = c("Slope from linear model" = "red",
#                "Spline"="blue") 
#   ) +
#   labs(
#     title = "Spline Contribution for days_over_30C",
#     x = "days_over_30C",
#     y = "Spline Contribution"
#   ) +
#   ylim(-0.05, 0.2)+
#   theme_minimal() +
#   theme(
#     plot.title = element_text(size = 30),
#     axis.title.x = element_text(size = 26),
#     axis.title.y = element_text(size = 26),
#     axis.text.x = element_text(size = 20),
#     axis.text.y = element_text(size = 20),
#     legend.title = element_text(size = 20),
#     legend.text = element_text(size = 18),
#     legend.position = c(0.8, 0.2),  # Position legend inside the plot (x, y)
#     legend.background = element_rect(fill = "white", color = "black", size = 0.5),  # Add a background box
#     legend.key = element_rect(fill = "white")  # Ensure legend keys have a white background
#   )
# 
# # Save the plot
# ggsave(
#   filename = paste0(plot_dir, summary_file, "_do30_with_linear.png"),
#   plot = p_do30,
#   bg = "white",
#   width = 10,
#   height = 8,
#   dpi = 300
# )
# 
# # plot consumption
# p_consumption <- ggplot(plot_data, aes(x = consumption_pd, y = spline_contribution_consumption_pd)) +
#   geom_line(aes(color = "Spline")) +
#   geom_ribbon(aes(ymin = c_lower_ci, ymax = c_upper_ci), alpha = 0.2, fill = "blue") +
#   labs(
#     title = "Spline Contribution for consumption_pd",
#     x = "consumption_pd",
#     y = "Spline Contribution"
#   ) +
#   geom_abline(
#     aes(
#     slope = consumption_linear, 
#     intercept = 0.3033357, 
#     color = "Slope from linear model"), 
#     linetype = "dashed"
#   ) +
#   scale_color_manual(
#     name = "Legend",  # Legend title
#     values = c("Slope from linear model" = "red",
#                "Spline"="blue") 
#   ) +
#   scale_x_continuous(breaks = seq(0, max(plot_data$consumption_pd, na.rm = TRUE), by = 30)) +
#   ylim(-1.2, 0.4)+
#   scale_y_continuous(breaks = round(seq(-1.2, 0.4, by = 0.4), 1)) +
#   theme_minimal()+
#   theme(
#     plot.title = element_text(size = 30),
#     axis.title.x = element_text(size = 26),
#     axis.title.y = element_text(size = 26),
#     axis.text.x = element_text(size = 20),
#     axis.text.y = element_text(size = 20),
#     legend.title = element_text(size = 20),
#     legend.text = element_text(size = 18),
#     legend.position = c(0.8, 0.8),  # Position legend inside the plot (x, y)
#     legend.background = element_rect(fill = "white", color = "black", size = 0.5),  # Add a background box
#     legend.key = element_rect(fill = "white")  # Ensure legend keys have a white background
#   )
# 
# 
# ggsave(
#   filename = paste0(plot_dir, summary_file, "_consumption_linear.png"),
#   plot = p_consumption,
#   bg = "white",          # Set background to white
#   width = 10,             # Adjust width (in inches)
#   height = 8,            # Adjust height (in inches)
#   dpi = 300              # Set resolution for better quality
# )
# 
# # Make histograms of data density for q95 and consumption_pd
# 
# 
# p_hist_do30 <- ggplot(plot_data, aes(x = days_over_30C)) +
#   geom_histogram(binwidth = 1, fill = "blue", color = "black", alpha = 0.7) +
#   labs(
#     title = "Data Density by days_over_30C",
#     x = "days_over_30C",
#     y = "Data points"
#   ) +
#   scale_y_continuous(labels = comma) +
#   theme_minimal()+
#   theme(
#     plot.title = element_text(size = 30),
#     axis.title.x = element_text(size = 26),
#     axis.title.y = element_text(size = 26),
#     axis.text.x = element_text(size = 20),
#     axis.text.y = element_text(size = 20)
#   )
# 
# # Save the histogram plot
# ggsave(
#   filename = paste0(plot_dir, summary_file, "_do30_density.png"),
#   plot = p_hist_do30,
#   bg = "white",          # Set background to white
#   width = 10,             # Adjust width (in inches)
#   height = 8,            # Adjust height (in inches)
#   dpi = 300              # Set resolution for better quality
# )
# 
# # same for consumption
# p_hist_consumption <- ggplot(plot_data, aes(x = consumption_pd)) +
#   geom_histogram(binwidth = 2, fill = "blue", color = "black", alpha = 0.7) +
#   labs(
#     title = "Data Density by consumption_pd",
#     x = "consumption_pd",
#     y = "Data points"
#   ) +
#   scale_x_continuous(breaks = seq(0, max(plot_data$consumption_pd, na.rm = TRUE), by = 30)) +
#   scale_y_continuous(labels = comma) +  # Format y-axis with commas
#   theme_minimal()+
#   theme(
#     plot.title = element_text(size = 30),
#     axis.title.x = element_text(size = 26),
#     axis.title.y = element_text(size = 26),
#     axis.text.x = element_text(size = 20),
#     axis.text.y = element_text(size = 20)
#   )
# 
# 
# 
# # Save the histogram plot
# ggsave(
#   filename = paste0(plot_dir, summary_file, "_consumption_density.png"),
#   plot = p_hist_consumption,
#   bg = "white",
#   width = 10,             # Adjust width (in inches)
#   height = 8,            # Adjust height (in inches)
#   dpi = 300
# )



