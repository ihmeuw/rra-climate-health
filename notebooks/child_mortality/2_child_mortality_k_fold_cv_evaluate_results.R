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

options(scipen = 999) # turn off scientific notation


#==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
#==============================================================================

plot_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/plots/2025_10_08.01/"
results_dir <- "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/child_mortality/results/2025_10_08.01/"
folds_dir <- paste0(results_dir,"folds/")
folds_data_subdir <- paste0(folds_dir,"data_subsets/")
folds_results_subdir <- paste0(folds_dir,"folds_results/")

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
  predictions <- predictions[,.(child_mortality,model_predictions)]
  mse <- mean((predictions$child_mortality - predictions$model_predictions)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(predictions$child_mortality - predictions$model_predictions))

  all_results <- rbind(
    all_results,
    data.table(
      model = var_str,
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
write.csv(manual_CV_results,paste0(results_dir,"manual_CV_results_25pc.csv"),row.names = FALSE)

# Plot results
p <- ggplot(manual_CV_results, aes(x = reorder(model, avg_RMSE), y = avg_RMSE)) +
  geom_point() +
  labs(title = "Average Root Mean Squared Errors by Model Specification\nduring 10-fold Cross-Validation",
       x = "Climate Vars in Model",
       y = "Average RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

ggsave(paste0(plot_dir, "cv_rmse_results_25pc.png"), plot = p, width = 8, height = 5)
