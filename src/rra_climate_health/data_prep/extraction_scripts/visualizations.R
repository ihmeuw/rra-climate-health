## Author: Maya Oleynikova 
## Purpose: Diagnostics for wealth/CGF data, based on geospatial granularity
## Date: 1/16/2025

## Set up ------------------------------------------------------
library(dplyr)
library(tidyverse)
library(ggplot2)

## Data --------------------------------------------------------
cgf_wealth <- read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/merged_cgf_wealth.csv')

## Visualization 1 ---------------------------------------------
# For each country, how many data points (rows) are available for that country
# and do they have lat/long coordinates, shapefile info, or neither?

