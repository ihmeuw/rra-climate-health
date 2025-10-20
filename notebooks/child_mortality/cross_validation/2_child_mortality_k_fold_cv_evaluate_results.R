################################################################################
# DESCRIPTION: Script to evaluate model predictions from cross-validation
# PROJECT: Climate nutrition
# DATE: 2025-09-17
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
library(data.table)
library(arrow) # to read parquet
library(patchwork)
library(ggplot2)

options(scipen = 999) # turn off scientific notation


#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

plot_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_16.01/"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_16.01/"
folds_dir <- paste0(results_dir,"folds/")
folds_data_subdir <- paste0(folds_dir,"data_subsets/")
folds_results_subdir <- paste0(folds_dir,"folds_results/")
model_summary_dir <- paste0(folds_dir,"model_summaries/")

#==============================================================================
# SECTION 2: READ AND COMBINE RESULTS
#==============================================================================

all_files <- list.files(folds_results_subdir)

all_results <- data.table(
  model = character(),
  MSE = double(),
  RMSE = double(),
  MAE = double(),
  fold = integer()
)

for (f in all_files){
  # Get fold number and model variables (besides base variables)
  fname <- sub("^predictions_fold_", "", f)
  fname <- sub("\\.parquet$", "", fname)
  
  # Split by "_vars_"
  parts <- strsplit(fname, "_vars_")[[1]]
  fold <- as.integer(parts[1])
  var_str <- parts[2]
  
  predictions <- read_parquet(paste0(folds_results_subdir,f))
  predictions <- data.table(predictions)
  
  if(predictions$fold[1]!=fold){
    print("Filename does not match fold number in data!")
  }
  
  predictions <- predictions[,.(child_mortality,mortality_fe,mortality_me)]
  
  
  mse <- mean((predictions$child_mortality - predictions$mortality_fe)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(predictions$child_mortality - predictions$mortality_fe))

  all_results <- rbind(
    all_results,
    data.table(
      model = paste0(var_str,"_fe"),
      MSE = mse,
      RMSE = rmse,
      MAE = mae,
      fold = fold
    )
  )

  mse <- mean((predictions$child_mortality - predictions$mortality_me)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(predictions$child_mortality - predictions$mortality_me))
  
  all_results <- rbind(
    all_results,
    data.table(
      model = paste0(var_str,"_me"),
      MSE = mse,
      RMSE = rmse,
      MAE = mae,
      fold = fold
    )
  )
}



# Read back in when complete. 
# Take average of k results:
manual_CV_results <- all_results[, .(
  avg_MSE = mean(MSE),
  avg_RMSE = mean(RMSE),
  avg_MAE = mean(MAE)
), by = model]

manual_CV_results <- manual_CV_results[order(avg_RMSE, decreasing = FALSE), ]
write.csv(manual_CV_results,paste0(results_dir,"manual_CV_results_50pc.csv"),row.names = FALSE)

# Plot results
p <- ggplot(manual_CV_results, aes(x = reorder(model, avg_RMSE), y = avg_RMSE)) +
  geom_point() +
  labs(title = "Average Root Mean Squared Errors by Model Specification\nduring 10-fold Cross-Validation",
       x = "Climate Vars in Model",
       y = "Average RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 10))+
  theme(plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

ggsave(paste0(plot_dir, "cv_rmse_results_50pc.png"), plot = p, width = 15, height = 10)

#==============================================================================
# SECTION 3: READ SIGNS AND P-VALUES OF COEFFICIENTs
#==============================================================================

all_models <- list.files(model_summary_dir)

all_models <- all_models[grepl("rds",all_models)]

coef_dt <- data.table(
  formula = character(),
  coefficient = character(),
  value = numeric()
)

for (i in seq_along(all_models)){
  model <- readRDS(paste0(model_summary_dir,all_models[i]))
  formula <- model$formula
  formula <- as.character(formula[3])
  formula <- sub("consumption + sex_id + survival::cluster(ihme_loc_id) + ","",formula,fixed = TRUE)
  for (c in names(model$coefficients)){
    coef <- c
    val <- as.numeric(model$coefficients[c])
    
    coef_dt <- rbind(coef_dt,
                     data.table(
                       formula = formula,
                       coefficient = c,
                       value = val))
    
  }
}

# plot coefficients of interest
climate_vars <- c(
  "mean_temperature",
  "mean_low_temperature",
  "mean_high_temperature",
  "precipitation_days",
  "total_precipitation",
  "relative_humidity",
  "days_over_26C",
  "days_over_27C",
  "days_over_28C",
  "days_over_29C",
  "days_over_30C",
  "days_over_31C",
  "days_over_32C",
  "days_over_33C",
  "elevation"
)

# plot all coefficients
plot_list <- lapply(climate_vars, function(plot_coef) {
  ggplot(coef_dt[coefficient == plot_coef], aes(x = formula, y = value)) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = paste0("Beta Estimate of ", plot_coef, " during 10-fold Cross-Validation"),
         x = "Climate Vars in Model",
         y = "Estimate") +
    ylim(min(coef_dt$value)-0.02,max(coef_dt$value)+0.02)+
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 10),
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "white", color = NA))
})

combined_plot <- wrap_plots(plot_list, nrow = 5, ncol = 3)

ggsave(filename = paste0(plot_dir, "cv_climate_vars_grid.png"),
       plot = combined_plot,
       width = 20, height = 30)

# plot single coefficient
plot_coef <- "days_over_30C"
coef_plot <- ggplot(coef_dt[coefficient==plot_coef],aes(x=formula,y=value))+
  geom_point()+
  geom_hline(yintercept = 0, color = "red") +   
  labs(title = paste0("Beta Estimate of ",plot_coef," during 10-fold Cross-Validation"),
       x = "Climate Vars in Model",
       y = "Estimate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 10))+
  theme(plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

ggsave(paste0(plot_dir, paste0("cv_",plot_coef,".png")), plot = coef_plot, width = 10, height = 10)


