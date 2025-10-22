#' @Title: [02_standardize_extractions.R]  
#' @Authors: Bianca Zlavog 
#' @contact: zlavogb@uw.edu
#' @Date_code_last_updated: 10/2023
#' @Date_data_last_updated: 10/2023
#' @Purpose: Standardize extractions for modeling by going through each source and:
#' - Currency converting all financial data to 2010 PPP
#' - Converting financial values per total population to per capita
#' - Multiplying any values by the multiplier
#' Also includes some hotfixes for survey names, SD values, 
#' and aggregation of admin2 data to admin1 level to obtain additional datapoints for modeling
#' @File_Input(s)
#' All compiled extracted data in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_ALL_compiled_all_polygon.csv `
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_ALL_compiled_point_and_polygon.csv`
#' Admin1/2 population data created in `../3_modeling/SAE/2b_compile_covariates.R` at
#' `/ihme/geospatial/lsae/economic_indicators/ldi/input/covariate_data_covariate_data.csv`
#' @File_Output(s)
#' Processed compiled extracted data in extraction template format at
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_ALL_compiled_processed_all_polygon.csv`
#' `/ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_ALL_compiled_processed_point_and_polygon.csv`
#' Processed compiled extracted data in extraction template format for SAE modeling input at
#' `/ihme/geospatial/lsae/economic_indicators/ldi/input/input_data_extracted_ALL_compiled_processed_all_polygon.csv`

##### Setup

rm(list = ls())
require(pacman)
p_load(data.table)

user <- Sys.getenv("USER")
if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", user, "/")
  i <- "/ihme/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "I:/"
}


lbd_loader_dir <- paste0(
  "/share/geospatial/code/geospatial-libraries/lbd.loader-", R.version$major, ".",
  strsplit(R.version$minor, '.', fixed = TRUE)[[1]][[1]]
)
library(lbd.loader, lib.loc = lbd_loader_dir)
suppressWarnings(suppressMessages(
  library(lbd.mbg, lib.loc = lbd.loader::pkg_loc("lbd.mbg"))
))

source(paste0(h, "/fgh/FUNCTIONS/currency_conversion.R"))
source(paste0(h, "/fgh/FUNCTIONS/helper_functions.R"))
source(paste0(h, "/indicators/1_retrospective/1_GDP_LDI/LSAE/helper_functions.R"))

root_fold <- paste0(i, "resource_tracking/LSAE_income/1_data_extractions/")
modeling_fold <- paste0(i, "/geospatial/lsae/economic_indicators/ldi/input/")
covariate_data <- fread(paste0(modeling_fold, "covariate_data_covariate_data.csv"))

for(point_to_polygon in c(TRUE, FALSE)) {
  
  if(point_to_polygon) {
    point_to_polygon_string <- "_all_polygon"
  } else {
    point_to_polygon_string <- "_point_and_polygon"
  }
  
  ##### Process extracted data
  extracted_data <- fread(paste0(root_fold, "extracted_ALL_compiled", point_to_polygon_string, ".csv"))
  extracted_data_processed <- copy(extracted_data)
  
  ##### OECD
  # measures/denominators are GDPpc and income pc, this is fine
  # currency is nominal LCU, will convert to 2010 PPP$
  # multiplier is 1
  extracted_data_processed_OECD <- currency_conversion(extracted_data_processed[source == "OECD"], 
                                                       col.loc = 'iso3',  
                                                       col.value = 'value', 
                                                       currency = 'lcu',
                                                       col.currency.year = 'base_year',
                                                       base.year = 2010, 
                                                       base.unit = 'ppp')
  extracted_data_processed_OECD[, currency := "PPP"][, base_year := 2010]
  
  extracted_data_processed <- rbind(extracted_data_processed[source != "OECD"], extracted_data_processed_OECD)
  
  ##### EUROSTAT
  # measures/denominators are GDP and income total, will divide by population
  # currency is nominal LCU, will convert to 2010 PPP$
  # multiplier is 1e06
  # TODO: ideally we may want to extract per capita values in future
  
  extracted_data_processed_EUROSTAT <- currency_conversion(extracted_data_processed[source == "EUROSTAT"], 
                                                       col.loc = 'iso3',  
                                                       col.value = 'value', 
                                                       currency = 'lcu',
                                                       col.currency.year = 'base_year',
                                                       base.year = 2010, 
                                                       base.unit = 'ppp')
  extracted_data_processed_EUROSTAT[, currency := "PPP"][, base_year := 2010]
  extracted_data_processed_EUROSTAT[, value := value * multiplier][, multiplier := 1]
  
  extracted_data_processed_EUROSTAT <- merge(extracted_data_processed_EUROSTAT, 
                                             covariate_data[, .(location_id = loc_id, year, pop, ADM0_NAME)], 
                                             by = c("location_id", "year"), all.x = TRUE) # note there are some NA pops pre-2000
  extracted_data_processed_EUROSTAT[denominator == "total" & !is.na(pop), value := value / pop][denominator == "total" & !is.na(pop), denominator := "per capita"]
  extracted_data_processed_EUROSTAT[, c("pop", "ADM0_NAME") := NULL]
  
  extracted_data_processed <- rbind(extracted_data_processed[source != "EUROSTAT"], extracted_data_processed_EUROSTAT)
  
  ##### USHD
  # measures/denominators are income per capita
  # currency is 2021 real LCU (USD), will convert to 2010 PPP$
  # multiplier is 1
  
  extracted_data_processed_USHD <- currency_conversion(extracted_data_processed[source == "USHD"], 
                                                           col.loc = 'iso3',  
                                                           col.value = 'value', 
                                                           currency = 'lcu',
                                                           col.currency.year = 'base_year',
                                                           base.year = 2010, 
                                                           base.unit = 'ppp')
  extracted_data_processed_USHD[, currency := "PPP"][, base_year := 2010]
  
  
  extracted_data_processed <- rbind(extracted_data_processed[source != "USHD"], extracted_data_processed_USHD)
  
  
  
  ##### LSMS
  # measures/denominators are annual income/consumption/expenditure/consumption expenditure per capita, will group together consumption and consumption expenditure variables
  # currency is nominal LCU, will convert to 2010 PPP$
  # multiplier is 1 or 1000
  
  extracted_data_processed_LSMS <- currency_conversion(extracted_data_processed[source == "LSMS"], 
                                                       col.loc = 'iso3',  
                                                       col.value = 'value', 
                                                       currency = 'lcu',
                                                       col.currency.year = 'base_year',
                                                       base.year = 2010, 
                                                       base.unit = 'ppp')
  extracted_data_processed_LSMS[, currency := "PPP"][, base_year := 2010]
  extracted_data_processed_LSMS[, value := value * multiplier][, multiplier := 1]
  extracted_data_processed_LSMS[measure == "consumption expenditure", measure := "consumption"]
  
  
  extracted_data_processed <- rbind(extracted_data_processed[source != "LSMS"], extracted_data_processed_LSMS)
  
  
  ##### MICS and DHS
  # measures/denominators are asset score/asset percentile in total space
  # no currency measures applicable
  # multiplier is 1
  # MICS asset scores always range from about -1 to 1
  # DHS asset scores usually range from about -100000 to 100000,
  # but sometimes have values on a different scale such as -1 to 1, -100 to 100, -1000000 to 1000000
  # Want to transform all values to a -100000 to 100000 scale for consistency,
  # because having values on different scales can cause issues when fitting models
  
  extracted_data_processed_MICS_DHS <- extracted_data_processed[source %in% c("MICS", "DHS") & measure == "asset score"]
  
  extracted_data_processed_MICS_DHS <- copy(as.data.table(dhs_all)[source %in% c("MICS", "DHS") & measure == "asset score"])
  
  minmax_assetscore_values_MICS_DHS <- copy(extracted_data_processed_MICS_DHS)[, .(min = min(value), max = max(value)), by = c("nid")]
  minmax_assetscore_values_MICS_DHS[, range := max + abs(min)]
  
  minmax_assetscore_values_MICS_DHS[range < 10, multiplier := 100000]
  minmax_assetscore_values_MICS_DHS[range > 10 & range < 100, multiplier := 10000]
  minmax_assetscore_values_MICS_DHS[range > 100 & range < 1000, multiplier := 1000]
  minmax_assetscore_values_MICS_DHS[range > 1000 & range < 10000, multiplier := 100]
  minmax_assetscore_values_MICS_DHS[range > 10000 & range < 100000, multiplier := 10]
  minmax_assetscore_values_MICS_DHS[range > 100000 & range < 1000000, multiplier := 1]
  minmax_assetscore_values_MICS_DHS[range > 1000000, multiplier := 0.1]
  minmax_assetscore_values_MICS_DHS <- minmax_assetscore_values_MICS_DHS[, .(nid, multiplier)]
  
  extracted_data_processed_MICS_DHS[, multiplier := NULL]
  extracted_data_processed_MICS_DHS <- merge(extracted_data_processed_MICS_DHS, 
                                             minmax_assetscore_values_MICS_DHS, by = "nid")
  # MoloMods
  extracted_data_processed_MICS_DHS[, value := value * multiplier]
  extracted_data_processed_MICS_DHS <- subset(extracted_data_processed_MICS_DHS, point == 1)
  write_parquet(extracted_data_processed_MICS_DHS, '/ihme/scratch/users/victorvt/cgfwealth_spatial/dhs_wealth_uncollapsed_onlypoints_rescaled.parquet')
  # extracted_data_processed_MICS_DHS[, value := value * multiplier][
  #   , sd_weighted := sd_weighted * multiplier][
  #     , sd_unweighted := sd_unweighted * multiplier]
  
  # then merge these back on and multiply
  extracted_data_processed <- rbind(extracted_data_processed[!(source %in% c("MICS", "DHS")  & measure == "asset score")], extracted_data_processed_MICS_DHS)
  
  ##### COUNTRY_SPECIFIC
  # measures/denominators are annual income/consumption/expenditure/consumption expenditure per capita, will group together consumption and consumption expenditure variables
  # currency is nominal LCU, will convert to 2010 PPP$
  # multiplier is 1
  
  extracted_data_processed_CS <- currency_conversion(extracted_data_processed[source == "COUNTRY_SPECIFIC"], 
                                                       col.loc = 'iso3',  
                                                       col.value = 'value', 
                                                       currency = 'lcu',
                                                       col.currency.year = 'base_year',
                                                       base.year = 2010, 
                                                       base.unit = 'ppp')
  extracted_data_processed_CS[, currency := "PPP"][, base_year := 2010]
  extracted_data_processed_CS[, value := value * multiplier][, multiplier := 1]
  extracted_data_processed_CS[measure == "consumption expenditure", measure := "consumption"]
  
  
  extracted_data_processed <- rbind(extracted_data_processed[source != "COUNTRY_SPECIFIC"], extracted_data_processed_CS)
  
  ##### GDL
  # measure is income index
  # no currency measures applicable
  # multiplier is 1
  # no transformations needed
  extracted_data_processed <- rbind(extracted_data_processed[source != "GDL"], extracted_data_processed[source == "GDL"][,base_year := NA])
  
  ##### IPUMS
  # measure is annual income/expenditure/consumption expenditure per capita
  # currency is  nominal LCU, will convert to 2010 PPP$
  # multiplier is 1
  
  extracted_data_processed_IPUMS <- currency_conversion(extracted_data_processed[source == "IPUMS"], 
                                                       col.loc = 'iso3',  
                                                       col.value = 'value', 
                                                       currency = 'lcu',
                                                       col.currency.year = 'base_year',
                                                       base.year = 2010, 
                                                       base.unit = 'ppp')
  extracted_data_processed_IPUMS[, currency := "PPP"][, base_year := 2010]
  extracted_data_processed_IPUMS[measure == "consumption expenditure", measure := "consumption"]
  
  extracted_data_processed <- rbind(extracted_data_processed[source != "IPUMS"], extracted_data_processed_IPUMS)
  
  
  extracted_data_processed[, measure := gsub(" ", "_", measure)]  
  
  # Standardize isocodes of a few subnational surveys
  extracted_data_processed[grepl("_", iso3), iso3 := substring(iso3, 1, 3)]
  
  # Drop a couple of DOM DHS_SPECIAL surveys that were only representative of Batey villages at the GPS level, so will exclude from aggregated admin2 data because not representative
  # Also fill in SD for a LSMS VNM manually extracted survey with tabular data
  if(point_to_polygon) {
    extracted_data_processed <- extracted_data_processed[!(nid %in% c(21198, 165645))]
    extracted_data_processed <- extracted_data_processed[nid == 25927, c("sd_unweighted", "sd_weighted") := 0.0001]
  }

  fwrite(extracted_data_processed, paste0(root_fold, "extracted_ALL_compiled_processed", point_to_polygon_string, ".csv"))

}


# Create a dataset for admin1 SAE modeling from aggregated admin2 data using population-weighted mean,
# if 80% of the population is represented in the available admin2 data. Append this to extracted polygon data.
  # Prep location set
  modeling_shapefile_version <- get_LSAE_location_versions()
  modeling_shapefile_version <- modeling_shapefile_version[
    location_set_version_ids == max(modeling_shapefile_version$location_set_version_ids), shapefile_dates]
  ad1_ad2_mapping_str <- # Load admin shapefile
    global_admin_shapefile_fp <- get_admin_shapefile(
      admin_level = 2,
      version = modeling_shapefile_version
    )
  ad1_ad2_mapping <- sf::st_read(global_admin_shapefile_fp)
  ad1_ad2_mapping$geometry <- NULL
  ad1_ad2_mapping <- data.table(ad1_ad2_mapping)
  ad1_ad2_mapping <- ad1_ad2_mapping[, c("loc_id", "ADM1_NAME", "ADM2_NAME")]
  
  # Prep population data at admin1 and admin2 levels
  ad2_pops <- covariate_data[
    , c("loc_id", "year", "pop", "ADM0_NAME", "ADM1_NAME", "ADM2_NAME", "admin_level")]
  setnames(ad2_pops, "pop", "ADM2_pop")
  ad2_pops <- ad2_pops[admin_level == 2][, admin_level:= NULL]
  ad1_pops <- covariate_data[
    , c("loc_id", "year", "pop", "ADM0_NAME", "ADM1_NAME", "admin_level")]
  setnames(ad1_pops, "pop", "ADM1_pop")
  setnames(ad1_pops, "loc_id", "ADM1_loc_id")
  ad1_pops <- ad1_pops[admin_level == 1][, admin_level:= NULL]
  ad1_ad2_pops <- merge(ad2_pops, ad1_pops, by = c("year", "ADM0_NAME", "ADM1_NAME"), all.x = T)
  
  # Read in processed polygon data and merge on location and population data
  extracted_data_processed <- fread(paste0(root_fold, "extracted_ALL_compiled_processed_all_polygon.csv"))
  
  # Reassign source names for some data that occurred for the same location/year/survey, 
  # because SAE model will error out in draw generation step if there are duplicates by these variables
  ## UGA 2011 had a DHS and DHS_AIS survey with separate sampling of similar locations - will keep the regular DHS as the reference
  ## SOM 2011 had two subnational MICS surveys carried out in similar locations - will keep the data with greater N_households and then higher SD for duplicate locations as the reference
  extracted_data_processed[(location_id == 85046 & nid == 91507) | (location_id == 85057 & nid == 91507) | (location_id == 85083 & nid == 91507), source := "MICS_other"]
  extracted_data_processed[nid == 55973, source := "DHS_other"]
  
  # Merge admin1 mappings onto admin2 data
  extracted_data_processed_ad2 <- extracted_data_processed[location_type %in% c("admin2") & !is.na(location_id) & year >= 2000]
  setnames(extracted_data_processed_ad2, "location_id", "loc_id")
  extracted_data_processed_ad2 <- merge(extracted_data_processed_ad2, ad1_ad2_mapping, by = "loc_id", all.x = T)
  extracted_data_processed_ad2 <- merge(extracted_data_processed_ad2, ad1_ad2_pops, 
                                        by = c("loc_id", "year", "ADM1_NAME", "ADM2_NAME"), all.x = T)
  
  # Check population coverage, and only keep data for aggregation where at least 80% of the admin1 population is represented in the admin2 polygons that we have data for
  extracted_data_processed_ad2[, new_ADM1_pop := sum(ADM2_pop), by = c("year", "ADM1_NAME", "ADM0_NAME", "source", "measure", "nid")]
  extracted_data_processed_ad2[, pop_coverage := new_ADM1_pop / ADM1_pop]
  extracted_data_processed_ad2 <- extracted_data_processed_ad2[!is.na(pop_coverage)]
  eval(nrow(extracted_data_processed_ad2[pop_coverage >= 0.8]) / nrow(extracted_data_processed_ad2)) # ~90% of admin2 data has at least 80% population coverage of the admin1 region
  extracted_data_processed_ad2 <- extracted_data_processed_ad2[pop_coverage >= 0.8]
  extracted_data_processed_ad2$pop_coverage <- NULL
  
  # Perform a population-weighted average of the admin2 regions to get to the estimated admin1 value
  extracted_data_processed_ad2_agg <- extracted_data_processed_ad2[, new_value := value * ADM2_pop]
  extracted_data_processed_ad2$ADM2_pop <- NULL
  extracted_data_processed_ad2_agg <- extracted_data_processed_ad2_agg[, .(new_value = sum(new_value), N_households = sum(N_households),
                                                                           sd_weighted = mean(sd_weighted), sd_unweighted = mean(sd_unweighted)), 
                                                                       by = c("year", "ADM1_NAME", "nid", "source", 
                                                                              "data_type", "file_path", "iso3", "location_type", "shapefile",
                                                                              "measure", "denominator", "multiplier", "value_type", "currency",
                                                                              "base_year", "currency_detail", "notes", "geomatching_notes",
                                                                              "initials", "lat", "long", "ADM0_NAME", "ADM1_pop", "new_ADM1_pop", "ADM1_loc_id")]
  extracted_data_processed_ad2_agg[, new_value := new_value / new_ADM1_pop]
  
  # Re-format columns before saving out
  extracted_data_processed_ad2_agg[, value := new_value][, new_value := NULL]
  extracted_data_processed_ad2_agg[, location_name := ADM1_NAME][, loc_id := ADM1_loc_id]
  setnames(extracted_data_processed_ad2_agg, "loc_id", "location_id")
  extracted_data_processed_ad2_agg[, location_type := "admin1"]
  extracted_data_processed_ad2_agg[, value_type := paste0(value_type, ", aggregated from admin2")]
  extracted_data_processed_ad2_agg[, c("ADM1_NAME", "ADM0_NAME", "ADM1_pop", "new_ADM1_pop", "ADM1_loc_id") := NULL]
  extracted_data_processed_ad2_agg[, source := paste0(source, "_adm2_aggregated")]
  extracted_data_processed_all <- rbind(extracted_data_processed, extracted_data_processed_ad2_agg, fill = T)
  
  # For SAE modeling, create standard error from standard deviation in order to reduce uncertainty
  extracted_data_processed_all[, se_weighted := sd_weighted / sqrt(N_households)]
  
  fwrite(extracted_data_processed_all, paste0(root_fold, "extracted_ALL_compiled_processed_all_polygon.csv"))
  fwrite(extracted_data_processed_all, paste0(modeling_fold, "input_data_extracted_ALL_compiled_processed_all_polygon.csv"))
  