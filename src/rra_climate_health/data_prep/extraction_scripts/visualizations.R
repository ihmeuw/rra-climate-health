## Author: Maya Oleynikova 
## Purpose: Diagnostics for wealth/CGF data, based on geospatial granularity
## Date: 1/16/2025

## Set up ------------------------------------------------------
library(dplyr)
library(tidyverse)
library(ggplot2)

source("/ihme/cc_resources/libraries/current/r/get_outputs.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")

## Data --------------------------------------------------------
cgf_wealth <- read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/merged_cgf_wealth.csv')

modeling_hierarchy_2021 <- get_location_metadata(location_set_id = 35, release_id=9)
countries <- modeling_hierarchy_2021[level==3,]
country_ids <- unique(countries$location_id)

## Visualization 1 ---------------------------------------------
# For each country, how many data points (rows) are available for that country
# and do they have lat/long coordinates, shapefile info, or neither?

# Creating binary variables for geospatial data availability
cgf_wealth$coordinates_present <- ifelse(
  !is.na(cgf_wealth$lat) & !is.na(cgf_wealth$long) & !is.na(cgf_wealth$iso3), 
  1, 
  0)

cgf_wealth$shapefile_present <- ifelse(
  !is.na(cgf_wealth$shapefile) & !is.na(cgf_wealth$location_code), 
  1, 
  0)

cgf_wealth$neither_present <- ifelse(
  cgf_wealth$coordinates_present == 0 & cgf_wealth$shapefile_present == 0, 
  1, 
  0)

cgf_wealth$both_present <- ifelse(
  cgf_wealth$coordinates_present == 1 & cgf_wealth$shapefile_present == 1, 
  1, 
  0)