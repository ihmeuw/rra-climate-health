## Author: Maya Oleynikova 
## Purpose: Diagnostics for wealth/CGF data, based on geospatial granularity
## Date: 1/16/2025

## Set up ------------------------------------------------------
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggrepel)
library(data.table)

source("/ihme/cc_resources/libraries/current/r/get_outputs.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")

## Data --------------------------------------------------------
cgf_wealth <- read_csv('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/data_01_06_2025/2_initial_processing/merged_cgf_wealth.csv')

modeling_hierarchy_2021 <- get_location_metadata(location_set_id = 35, release_id=9)
countries <- modeling_hierarchy_2021[location_type=='admin0',]

setnames(countries, "ihme_loc_id", "iso3")
countries <- countries[, .(iso3, location_name)]

# Merging with all GBD countries for viz purposes
cgf_wealth <- merge(countries, cgf_wealth, by = "iso3", all.x = TRUE)

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

# # Temporary diagnostics: How many rows from each NID do not have geo data?
# cgf_wealth[, num_rows := .N, by = nid]
# 
# filtered_cgf_wealth <- cgf_wealth[neither_present == 1]
# 
# filtered_cgf_wealth[, num_rows_no_geo := .N, by = nid]
# filtered_cgf_wealth <- filtered_cgf_wealth[!is.na(nid)]
# 
# filtered_cgf_wealth <- filtered_cgf_wealth[, .(nid, num_rows, num_rows_no_geo)]
# 
# filtered_cgf_wealth <- unique(filtered_cgf_wealth)
# 
# filtered_cgf_wealth[, prop_no_geo := num_rows_no_geo / num_rows]

# Mutating based on presence of coordinates, shapefile, both, or neither
cgf_wealth <- cgf_wealth %>%
  mutate(
    presence_category = case_when(
      coordinates_present == 1 & shapefile_present == 1 ~ "both_present",
      coordinates_present == 1 & shapefile_present == 0 ~ "coordinates_present",
      coordinates_present == 0 & shapefile_present == 1 ~ "shapefile_present",
      coordinates_present == 0 & shapefile_present == 0 ~ "neither_present",
      TRUE ~ "unknown"  # In case there are any missing or strange values
    )
  )

# Creating bar chart
ggplot(cgf_wealth, aes(x = location_name, fill = presence_category)) +
  geom_bar(position = "stack") +
  labs(
    title = "Number of CGF/wealth observations by country",
    x = "Location Name",
    y = "Number of rows",
    fill = "Geo data: Coordinates, shapefile or neither?"
  ) +
  theme_minimal() + 
  theme(
    axis.text.x = element_text(size = 5, angle = 90, hjust = 1),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5), 
    axis.title.x = element_text(size = 12, face = "bold"),  
    axis.title.y = element_text(size = 12, face = "bold"),  
    legend.title = element_text(size = 10)
  ) + 
  theme(axis.text.x = element_text(size = 5, angle = 90, hjust = 1)) +
  geom_text_repel(
    data = cgf_wealth %>%
      group_by(location_name) %>%  
      tally() %>%  
      filter(n > 1),  
    aes(x = location_name, y = n, label = location_name),  
    size = 4,      
    box.padding = 0.35,
    max.overlaps = 50,
    inherit.aes = FALSE
  )