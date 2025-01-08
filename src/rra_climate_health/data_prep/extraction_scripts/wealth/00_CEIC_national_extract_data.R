#' @Title: [wealth_data_ceic.R]  
#' @Authors: Kayleigh Bhangdia, Bianca Zlavog, Mark Moses, Gaorui Guo
#' @contact: bhangdia@uw.edu, zlavogb@uw.edu, mwm6@uw.edu, garyguo@uw.edu
#' @Date_code_last_updated:  09/23/2021
#' @Date_data_last_updated: 09/23/2021
#' @Purpose: Process wealth data sources into extraction template format for GDL
#' 
#' @File_Input(s)
#'  
#' @File_Output(s)
#'

##### Setup

rm(list = ls())
require(pacman)
p_load(data.table, dplyr, haven, colorout, readr, tidyr)

if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", Sys.getenv("USER"), "/")
  k <- "/ihme/cc_resources/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
}

source(file = paste0(k, "libraries/current/r/get_location_metadata.R")) 
locs = get_location_metadata(location_set_version_id = 910, location_set_id = 125, gbd_round_id = 8, release_id = 13) # GBD 2021 release LSAE set and version - no release ID marked as active
# locs0 <- locs[location_type %in% c("admin0", "nonsovereign") | location_id == 60348]
locs_0 = locs[location_type %in% c("admin0", "nonsovereign")]

# change all keys to lowercase for merging 
locs_0$location_name = tolower(locs_0$location_name)
locs_0$location_ascii_name = tolower(locs_0$location_ascii_name)


##### Data extraction
ceic_hh_raw = read_csv("/home/j/DATA/Incoming Data/LSAE_income/CEIC/Income related country level.csv")
ceic_gdp_raw = read_csv("/home/j/DATA/Incoming Data/LSAE_income/CEIC/Real GDP per Capita.csv")
ceic_hh_raw = ceic_hh_raw %>% distinct()
ceic_gdp_raw = ceic_gdp_raw %>% distinct()

##### initial clean
col_drop_list = c("Series ID","SR Code","Mnemonic","Function Description","First Obs. Date",
                  "Last Obs. Date","Last Update Time","Series remarks","Suggestions","Mean",
                  "Variance","Standard Deviation","Skewness","Kurtosis","Coefficient Variation",
                  "Min","Max","Median","Status","No. of Obs","Subnational")

ceic_gdp = ceic_gdp_raw %>% 
  select(-all_of(col_drop_list)) %>%
  rename(Indicator = X1) %>%
  arrange(Region, Indicator, Frequency)

ceic_hh = ceic_hh_raw %>% 
  select(-all_of(col_drop_list)) %>%
  rename(Indicator = X1) %>%
  arrange(Region, Indicator, Frequency)

# indicator names cleaning
names(ceic_gdp) = tolower(names(ceic_gdp))
names(ceic_hh) = tolower(names(ceic_hh))

ceic_gdp$indicator = 'Real GDP per Capita'

'%!in%' = Negate('%in%')
ceic_hh = ceic_hh %>% filter(indicator %!in% c("2. Replacement series ID: 355084102",                                                      
                                               "2. Replacement series ID: 355085902",                                                      
                                               "2. Replacement series ID: 384044167",                                                      
                                               "2. Replacement series ID: 384226477",                                                      
                                               "2. Replacement series ID: 384226487",                                                      
                                               "2. Replacement series ID: 412816957"))


# GDP per capita cleaning
ceic_gdp_long = melt(setDT(ceic_gdp), 
                     id.vars = c("indicator","region","frequency","unit","source"), 
                     variable.name = "year",
                     value.name = "value")

ceic_gdp_long = ceic_gdp_long %>% filter(!is.na(value))

ceic_gdp_long$year = substr(as.character(ceic_gdp_long$year), start = 1, stop = 4)
ceic_gdp_long = ceic_gdp_long %>%
  arrange(indicator, region, frequency, unit, source, year) %>%
  distinct()


# household data cleaning
ceic_hh$`06/1992` = NULL
ceic_hh$`06/1987` = NULL
ceic_hh_long = melt(setDT(ceic_hh), 
                    id.vars = c("indicator","region","frequency","unit","source"), 
                    variable.name = "month",
                    value.name = "value")

ceic_hh_long = ceic_hh_long %>% filter(!is.na(value))
ceic_hh_long = ceic_hh_long %>%
  mutate(month = as.character(month)) %>%
  arrange(indicator, region, frequency, unit, source, month) %>%
  distinct()

# duplicates need to be removed based on status and some other columns. sorry for leave this tedious work to you
# difference number of rows 5111 and 4798
