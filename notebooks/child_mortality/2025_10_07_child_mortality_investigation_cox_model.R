################################################################################
# DESCRIPTION: Test R packages to perform a Cox proportional hazard model
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

# install.packages('coxme',lib = "/homes/elyeb/rlibs") # for survival analysis with mixed effects
# install.packages('frailtyEM',lib = "/homes/elyeb/rlibs") # able to handle mixed effects and predict on new data
library(frailtyEM,lib.loc = "/homes/elyeb/rlibs")
library(coxme,lib.loc = "/homes/elyeb/rlibs") 
library(data.table)
library(caret) # for createFolds function
library(dplyr) # for anti_join function
library(arrow) # to read parquet
library(ggplot2)
library(scales)
source("/home/j/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2023/inset_maps/gbd2023_map.R")


options(scipen = 999) # turn off scientific notation
 
#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/training_data/2025_10_08.01/data.parquet"
# data_version <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/tmp/child_mortality_merged_wealth.csv"
plot_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_08.01/"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

folds_dir <- paste0(results_dir,"folds/")

df <- read_parquet(data_version)
# df <- fread(data_version)
df <- data.table(df)

# flip child_alive so 1 = died, 0 = alive for easier interpretation
df[,child_mortality := 1-child_alive]
df[,ihme_loc_id:=as.factor(ihme_loc_id)]
df[,sex_id:= factor(sex_id,levels = c("1", "2"), labels = c("Male", "Female"))]
  
setnames(df,old="ldipc_weighted_no_match",new="consumption")

climate_vars <- c(
  "mean_temperature",
  "total_precipitation",
  "relative_humidity",
  "mean_high_temperature",
  "mean_low_temperature",
  "precipitation_days",
  "days_over_30C",
  "days_over_26C"
)
cols <- c("indv_id","child_mortality", "age_year_at_year_end", "sex_id", "ihme_loc_id", "consumption", climate_vars)
df_model <- df[, ..cols]

#==============================================================================
# SECTION 2: FIT MODELS
#==============================================================================

## Simple model to test coxme library
fit <- coxme(Surv(age_year_at_year_end, child_mortality) ~ consumption + mean_temperature + days_over_30C + sex_id + (1|ihme_loc_id), data = df_model)
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

## Drawback of coxme library: cannot predict model on new data. 

## Simple model to test frailtyEM library
set.seed(123)
df_model_sample <- df_model[sample(nrow(df), 20000), ] # cannot sample across individuals. Will need to fix
fit <- emfrail(Surv(age_year_at_year_end, child_mortality) ~ consumption +
                 mean_temperature +
                 days_over_30C +
                 sex_id +
                 cluster(ihme_loc_id),
               data = df_model_sample,
               verbose = TRUE)
summary(fit)
# Call: 
#   emfrail(formula = Surv(age_year_at_year_end, child_mortality) ~ 
#             consumption + mean_temperature + days_over_30C + sex_id + 
#             cluster(ihme_loc_id), data = df_model_sample, verbose = TRUE)
# 
# Regression coefficients:
#   coef  exp(coef)   se(coef)    adj. se          z    p
# consumption      -0.0002709  0.9997292  0.0000756  0.0000759 -3.5676157 0.00
# mean_temperature -0.0032714  0.9967340  0.0181376  0.0190829 -0.1714290 0.86
# days_over_30C     0.0024950  1.0024981  0.0022261  0.0022385  1.1145732 0.27
# sex_id            0.3703452  1.4482345  0.1467629  0.1467667  2.5233597 0.01
# Estimated distribution: gamma / left truncation: FALSE 
# 
# Fit summary:
#   Commenges-Andersen test for heterogeneity: p-val  0.00000746 
# no-frailty Log-likelihood: -1800.253 
# Log-likelihood: -1785.934 
# LRT: 1/2 * pchisq(28.6), p-val 0.0000000436
# 
# Frailty summary:
#   estimate lower 95% upper 95%
#   Var[Z]                0.412     0.135     1.225
# Kendall's tau         0.171     0.063     0.380
# Median concordance    0.167     0.061     0.382
# E[logZ]              -0.220    -0.725    -0.069
# Var[logZ]             0.508     0.145     2.230
# theta                 2.426     0.816     7.388
# Confidence intervals based on the likelihood function

## Drawback of frailtyEM library: much slower
pred_out <- predict(fit, df_model_sample, re.form = ~0, quantity="survival")
str(pred_out)
surv_at_obs_time <- mapply(function(df, t) {
  idx <- max(which(df$time <= t))
  df$survival[idx]
}, pred_out, df_model_sample$age_year_at_year_end)

df_model_sample$survival_at_obs_time <- 1-surv_at_obs_time


## Read in and print model summaries from successful runs:
model_50_pc <- readRDS(paste0(results_dir,"subset_5pct_model_object.rds"))
summary(model_50_pc)
# get predictions of model over data set
pred_surv <- predict(model_50_pc, df, quantity = "survival")
# pred_surv <- predict(model_50_pc, df,re.form = ~0, quantity = "survival")

df_sub <- df[indv_id=="19557_1_17_1"]
pred_surv <- predict(model_50_pc, df_sub, quantity = "survival")
desired_time = 1/12
pred_surv <- predict(model_50_pc, df_sub, quantity = "survival", times = desired_time)
#==============================================================================
# SECTION 3: MAKE PLOTS
#==============================================================================

## Make global map of NIDs by country
map_data <- df[,.(location_id,nid)]
map_data <- unique(map_data)
map_data <- map_data[,.(mapvar=length(nid)),by=location_id]

# By default, the bins are currently assigned such that a value equal to a breakpoint 
# is included in the bin where that breakpoint is the lower bound.
limits <- c(-1,1,2,5,9)
labels <- c("0",
            "1",
            "2-5",
            "6-8")

custom_palette <- colorRampPalette(c("white", "purple4"))(10) # generate colors

plot_cols <- c("#F0F0F0","#DFB4FA","#A020F0","purple4")

background_col <- colorRampPalette(c("white", "gray"))(10)


pdf(paste0(plot_dir, "child_mortality_nids_map.pdf"), width = 7.5, height = 4.2, pointsize = 9)

map <- gbd_map(data=map_data,
               limits=limits,
               sub_nat="none",
               legend=TRUE,
               inset=FALSE,
               labels=labels,
               pattern=NULL,
               col=plot_cols,
               na.color = "#F0F0F0",
               title="Number of Surveys by Country for Child Mortality",
               title.cex=1,
               fname=NULL,
               legend.title="No. unique NIDs",
               legend.columns = 1,
               legend.cex=1,
               legend.shift=c(0,0))
dev.off()


## Make histogram of deaths by age_month
deaths <- df[child_alive==0]
p <- ggplot(deaths, aes(x = age_month)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white") +
  labs(title = "Number of Deaths by Age in Months",
       x = "Age (in months)",
       y = "Total") +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(breaks = seq(min(deaths$age_month, na.rm = TRUE),
                                  max(deaths$age_month, na.rm = TRUE), 
                                  by = 12)) +
  theme_minimal(base_family = "sans") +
  theme(plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

ggsave(paste0(plot_dir, "child_mortality_by_age_month.png"), plot = p, width = 8, height = 5)

## Make exploratory scatter plots of all reasonable variables
final_outcome_df <- df_model[, .SD[which.max(age_year_at_year_end)], by = indv_id]

ggplot(final_outcome_df,aes(x=mean_temperature,y=consumption))+
  geom_point(aes(child_mortality))

