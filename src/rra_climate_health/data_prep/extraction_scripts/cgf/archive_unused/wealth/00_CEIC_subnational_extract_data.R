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
locs_1_2 = locs[location_type %in% c("admin1", "admin2")]

# change all keys to lowercase for merging 
locs_0$location_name = tolower(locs_0$location_name)
locs_0$location_ascii_name = tolower(locs_0$location_ascii_name)
locs_1_2$location_name = tolower(locs_1_2$location_name)
locs_1_2$location_ascii_name = tolower(locs_1_2$location_ascii_name)

suppressWarnings(locs_1_2[, country_location_id := as.integer(sub("^.*?\\,(.*?)\\,.*", "\\1", path_to_top_parent))])
locs_1_2 = merge(locs_1_2, locs_0[, .(location_id, country_location_name = location_name, country_ISO3 = ihme_loc_id)], 
                 by.x = "country_location_id", by.y = "location_id", all.x = T)
# temp = locs_1_2[(locs_1_2$country_location_id==62) & (locs_1_2$level %in% c(1,2)),] # check russian level 1,2 admin1
# Manual fixes to bugs in location set
setDT(locs_1_2)[country_ISO3 == "JPN" & level == 2, location_type := "admin1"]
setDT(locs_1_2)[country_ISO3 == "GBR" & level == 2, location_type := "admin1"]
setDT(locs_1_2)[country_ISO3 == "GBR" & level == 3, location_type := "admin2"]
setDT(locs_1_2)[country_ISO3 == "RUS" & level == 2, location_type := "admin1"]
setDT(locs_1_2)[country_ISO3 == "RUS" & level == 3, location_type := "admin2"]
setDT(locs_1_2)[location_id == 69491, `:=` (parent_id = 61468, path_to_top_parent = "1,92,61468,69491", country_ISO3 = "ESP")]

# drop the duplicates by country and keep only admin1 and admin2
locs_1_2 = locs_1_2[location_type %in% c("admin1", "admin2"),]
# we need to do something about the metadata table for the situation, we have both admin1 and admin2 loc with the same name
# we drop all the admin2 locations if there is a dup for 'location_ascii_name' and 'country_location_name' and only keep 
# the admin1 one since i don't believe our data can go that deep for location hierarchy.
# in this way, the primary key for our location table is unique (country name and subnational name)
# we are using location_ascii_name for merge since it avoids some language issue
locs_1_2_dup = locs_1_2 %>% 
  filter(!is.na(country_location_name)) %>%
  group_by(country_location_name, location_ascii_name) %>% 
  mutate(dupe = n()>1, dupe_num = n()) %>%
  filter(dupe == TRUE) %>%
  arrange(country_location_name, location_name, location_id) %>%
  select(country_location_id, country_location_name, location_id, location_name, location_ascii_name, path_to_top_parent, 
         location_type, dupe, dupe_num)
nrow(locs_1_2_dup[locs_1_2_dup$location_type == "admin1",])
uniqueN(locs_1_2_dup[locs_1_2_dup$location_type == "admin1", c('location_ascii_name', 'country_location_name')])
nrow(locs_1_2_dup[locs_1_2_dup$location_type == "admin2",])
loc_id_drop = locs_1_2_dup[locs_1_2_dup$location_type == "admin2", "location_id"]$location_id
locs_1_2 = locs_1_2[!(locs_1_2$location_id %in% loc_id_drop), ]


##### Data extraction
CEIC1 = read_csv("/home/j/DATA/Incoming Data/LSAE_income/CEIC/China HK MACAU wealth.csv")
CEIC2 = read_csv("/home/j/DATA/Incoming Data/LSAE_income/CEIC/China subnational admin1 wealth.csv")
CEIC3 = read_csv("/home/j/DATA/Incoming Data/LSAE_income/CEIC/China subnational admin2 wealth.csv")
CEIC4 = read_csv("/home/j/DATA/Incoming Data/LSAE_income/CEIC/India subnationals admin1 wealth.csv")
CEIC5 = read_csv("/home/j/DATA/Incoming Data/LSAE_income/CEIC/Russia subnational admin2 wealth.csv")

# data clean
CEIC1$`1990` = gsub('NA', '', paste0(CEIC1$`1990`, CEIC1$`1990_1`))
CEIC1$`1995` = gsub('NA', '', paste0(CEIC1$`1995`, CEIC1$`1995_1`))
CEIC1$`2000` = gsub('NA', '', paste0(CEIC1$`2000`, CEIC1$`2000_1`))
CEIC1$`2003` = gsub('NA', '', paste0(CEIC1$`2003`, CEIC1$`2003_1`))
CEIC1$`2005` = gsub('NA', '', paste0(CEIC1$`2005`, CEIC1$`2005_1`))
CEIC1$`2008` = gsub('NA', '', paste0(CEIC1$`2008`, CEIC1$`2008_1`))
CEIC1$`2010` = gsub('NA', '', paste0(CEIC1$`2010`, CEIC1$`2010_1`))
CEIC1$`2013` = gsub('NA', '', paste0(CEIC1$`2013`, CEIC1$`2013_1`))
CEIC1$`2015` = gsub('NA', '', paste0(CEIC1$`2015`, CEIC1$`2015_1`))
CEIC1$`2018` = gsub('NA', '', paste0(CEIC1$`2018`, CEIC1$`2018_1`))
setDT(CEIC1)[, c('1990_1', '1995_1', '2000_1', '2003_1', '2005_1', '2008_1', '2010_1', '2013_1', '2015_1', '2018_1') := NULL]
CEIC1$Subnational = CEIC1$Region
CEIC1$Region = 'China'

# there is something to notice here, soome indicators are annual ending at Mar. but we treat them all at Dec.
# for easier matching
colnames(CEIC1)[27:86] = paste0('12/', colnames(CEIC1)[27:86])
colnames(CEIC4)[27:67] = paste0('12/', colnames(CEIC4)[27:67])

# colnames(CEIC1)#[1:27]
# colnames(CEIC2)#[1:27]
# colnames(CEIC3)#[1:27]
# colnames(CEIC4)#[1:27]
# colnames(CEIC5)#[1:27]
CEIC = list(CEIC1, CEIC2, CEIC3, CEIC4, CEIC5)
CEIC = rbindlist(CEIC, fill = T)

col_drop_list = c("Series ID","SR Code","Mnemonic","Function Description","First Obs. Date",
                  "Last Obs. Date","Last Update Time","Series remarks","Suggestions","Mean",
                  "Variance","Standard Deviation","Skewness","Kurtosis","Coefficient Variation",
                  "Min","Max","Median","Status","No. of Obs")
CEIC = CEIC %>% 
  select(-all_of(col_drop_list)) %>%
  rename(Indicator = X1) %>%
  filter(!is.na(Subnational)) %>%
  arrange(Region, Subnational, Indicator, Frequency)

# indicator names cleaning
CEIC$Indicator = gsub(":.*$", "", CEIC$Indicator)
CEIC$Indicator = gsub("\\s*\\([^\\)]+\\)","", CEIC$Indicator)
CEIC$Indicator = gsub("GSDP", "Gross State Domestic Product", CEIC$Indicator)
CEIC$Indicator = gsub("NSDP", "Net State Domestic Product", CEIC$Indicator)
CEIC$Indicator = gsub("SNA08", "Gross Domestic Product", CEIC$Indicator)
CEIC$Indicator = gsub("Expenditure", "Exp", CEIC$Indicator)
CEIC$Indicator = gsub("Exp", "Expenditure", CEIC$Indicator)
CEIC$Indicator = gsub("Annual", "", CEIC$Indicator)
CEIC$Indicator = gsub("Monthly", "", CEIC$Indicator)
CEIC$Indicator = gsub("Average", "", CEIC$Indicator)
CEIC$Indicator = gsub("Avg", "", CEIC$Indicator)
CEIC$Indicator = tolower(CEIC$Indicator)
CEIC$Indicator = trimws(CEIC$Indicator)
CEIC$Indicator = gsub("  ", "_", CEIC$Indicator)
CEIC$Indicator = gsub(" ", "_", CEIC$Indicator)
setDT(CEIC)[Indicator == "expenditure", Indicator := "expenditure_per_capita"]
setDT(CEIC)[Indicator == "income", Indicator := "income_per_capita"]
setDT(CEIC)[Indicator == "per_capita_expenditure", Indicator := "expenditure_per_capita"]
CEIC = CEIC[Indicator != "2._данные_с_2016_года_основаны_на_классификации_оквэд2."]
setDT(CEIC)[Indicator == "household_income", Indicator := "household_income_total"]
setDT(CEIC)[Indicator == "household_expenditure", Indicator := "household_expenditure_total"]
setDT(CEIC)[Indicator == "expenditure_per_capita", Indicator := "household_expenditure_per_capita"]
setDT(CEIC)[Indicator == "income_per_capita", Indicator := "household_income_per_capita"]
setDT(CEIC)[Indicator == "gross_state_domestic_product", Indicator := "gross_domestic_product"]
setDT(CEIC)[Indicator == "net_state_domestic_product", Indicator := "net_domestic_product"]
setDT(CEIC)[Indicator == "net_state_domestic_product_per_capita", Indicator := "net_domestic_product_per_capita"]
setDT(CEIC)[Indicator == "gross_value_added", Indicator := "gross_domestic_product"]
setDT(CEIC)[Indicator == "gross_value_added_per_capita", Indicator := "gross_domestic_product_per_capita"]

# wide format to long format
CEIC = CEIC %>% mutate_at(c(7:374), as.numeric)

ceic_long = melt(setDT(CEIC), 
                 id.vars = c("Region","Subnational","Indicator","Frequency","Unit","Source"), 
                 variable.name = "Date")

ceic_long = ceic_long %>% separate(Date, c('Month', 'Year'))
ceic_long = ceic_long %>% 
  select(Region, Subnational, Indicator, Frequency, Unit, Source, Year, Month, value) %>%
  filter(!is.na(value)) %>%
  arrange(Region, Subnational, Indicator, Frequency, Year, Month) %>%
  mutate(Region = tolower(Region),
         Subnational = tolower(Subnational))

# quick clear for some location names
ceic_long$Subnational = gsub(" region", " oblast", ceic_long$Subnational)
ceic_long$Subnational = gsub(" territory", " kray", ceic_long$Subnational)
ceic_long$Subnational = gsub("tumen", "tyumen", ceic_long$Subnational)
setDT(ceic_long)[Subnational == "hong kong sar (china)", Subnational := "hong kong special administrative region of china"]
setDT(ceic_long)[Subnational == "macau sar (china)", Subnational := "macao special administrative region of china"]
setDT(ceic_long)[Subnational == "chukotka area", Subnational := "chukotka autonomous area"]
setDT(ceic_long)[Subnational == "city of sevastopol", Subnational := "sevastopol'"]
setDT(ceic_long)[Subnational == "city of moscow", Subnational := "moscow city"]
setDT(ceic_long)[Subnational == "city of st petersburg", Subnational := "saint petersburg"]
setDT(ceic_long)[Subnational == "nct of delhi", Subnational := "delhi"]
setDT(ceic_long)[Subnational == "xian", Subnational := "xi'an"]

# merge location metadata 
locs_1_2 = locs_1_2[locs_1_2$country_location_id %in% c(163,6,62),] %>%
  select(country_location_id, country_location_name, country_ISO3, location_id, location_name, 
         level, path_to_top_parent, location_type) %>%
  rename(iso3 = country_ISO3)

ceic_long_merged = merge(x = ceic_long, y = locs_1_2,
                         by.x = c('Region', 'Subnational'),
                         by.y = c('country_location_name', 'location_name'),
                         all.x = T)

setDT(ceic_long_merged)[Subnational == "fuzhou", location_id := 66011]
setDT(ceic_long_merged)[Subnational == "altay kray", location_id := 44971]
setDT(ceic_long_merged)[Subnational == "orel oblast", location_id := 44913]
setDT(ceic_long_merged)[Subnational == "republic of marii el", location_id := 44948]
setDT(ceic_long_merged)[Subnational == "republic of adygea", location_id := 44932]
setDT(ceic_long_merged)[Subnational == "republic of altay", location_id := 44967]
setDT(ceic_long_merged)[Subnational == "republic of komi", location_id := 44922]
setDT(ceic_long_merged)[Subnational == "republic of northern osetia alania", location_id := 44944]
setDT(ceic_long_merged)[Subnational == "sevastopol'", location_id := 62998]
setDT(ceic_long_merged)[Subnational == "urumqi", location_id := 66300]
setDT(ceic_long_merged)[Subnational == "arkhangelsk oblast: arkhangelsk oblast excl nenetsky area", location_id := 44924]
setDT(ceic_long_merged)[Subnational == "arkhangelsk oblast: nenetsky area", location_id := 44923]
setDT(ceic_long_merged)[Subnational == "lipetsk oblast", location_id := 44911]
setDT(ceic_long_merged)[Subnational == "republic of chuvashia", location_id := 44952]
setDT(ceic_long_merged)[Subnational == "republic of kabardino balkaria", location_id := 44942]
setDT(ceic_long_merged)[Subnational == "republic of karachaevo cherkessia", location_id := 44943]
setDT(ceic_long_merged)[Subnational == "republic of udmurtia", location_id := 44951]
setDT(ceic_long_merged)[Subnational == "republic of tyva", location_id := 44969]
setDT(ceic_long_merged)[Subnational == "republic of udmurtia", location_id := 44951]
setDT(ceic_long_merged)[Subnational == "republic of crimea", location_id := 62982]
setDT(ceic_long_merged)[Subnational == "tyumen oblast: khanty mansiysky area", location_id := 44963]
setDT(ceic_long_merged)[Subnational == "tyumen oblast: yamalo nenetsky area", location_id := 44964]
setDT(ceic_long_merged)[Subnational == "tyumen oblast: tyumen oblast excl areas", location_id := 44965]

ceic_long_merged_nomatch = ceic_long_merged %>% filter(is.na(location_id))

# add notes for the non matched locations
setDT(ceic_long_merged)[Subnational %in% c("arkhangelsk oblast",
                                           "andaman and nicobar islands",
                                           "tyumen oblast"), geomatching_notes := 'Multiple GBD regions sum to source region']
setDT(ceic_long_merged)[Subnational == "arkhangelsk oblast", notes := 'Sum of "Arkhangelsk oblast without Nenets autonomous district", "Nenets autonomous district"']
setDT(ceic_long_merged)[Subnational == "andaman and nicobar islands", notes := "Sum of 'Nicobars', 'North & Middle Andaman', 'South Andaman'"]
setDT(ceic_long_merged)[Subnational == "tyumen oblast", notes := 'Sum of "Khanty-Mansi autonomous area", "Yamalo-Nenets autonomous area", "Tyumen oblast without autonomous areas"']

setDT(ceic_long_merged)[Subnational %in% c("(dc)aginsky buryatsky area",
                                           "(dc)evenkia area",
                                           "(dc)komi perm area",              
                                           "(dc)koryaksky area", 
                                           "(dc)taymyr area",
                                           "(dc)ust ordynsky buryatsky area", 
                                           "jammu and kashmir"), geomatching_notes := 'Most granular location not matching GBD']

setDT(ceic_long_merged)[Subnational %in% c("(dc)crimean federal district",
                                           "central federal district",
                                           "far east federal district",       
                                           "north caucasian federal district",
                                           "north western federal district",
                                           "siberian federal district",
                                           "southern federal district",
                                           "ural federal district",
                                           "volga oblast federal district"), geomatching_notes := 'Less granular location not matching GBD']

ceic_long_merged = ceic_long_merged %>%
  select(Subnational,Indicator,Frequency,Unit,Source,Year,Month,value,location_id,geomatching_notes,notes) %>%
  rename(location_name_original = Subnational)

ceic_long_merged = merge(ceic_long_merged, locs_1_2, by = 'location_id', all.x = T)

names(ceic_long_merged) = tolower(names(ceic_long_merged))
ceic_long_merged = ceic_long_merged %>% 
  arrange(country_location_name, location_name, location_name_original, indicator, year, month, level) %>%
  select(country_location_id, location_id, country_location_name, iso3, location_name, location_name_original, 
         indicator, frequency, unit, year, month, value, everything())

# sanity check, now for every row, we have location id and NA at notes, or we have notes and NA at location id
nrow(ceic_long_merged) == sum(!is.na(ceic_long_merged$location_id)) + sum(!is.na(ceic_long_merged$geomatching_notes))

# take a look at the coverage for locations and indicators, drop some useless ones
coverage_gdl_admin1_admin2  = ceic_long_merged %>%
  select(country_location_name,location_name,location_type) %>%
  na.omit() %>% distinct() %>% arrange(country_location_name,location_type,location_name) %>%
  group_by(country_location_name,location_type) %>%
  summarise(num_subnationals = n())

coverage_indicators = ceic_long_merged %>%
  select(country_location_name, location_name, indicator, frequency, year) %>% 
  distinct() %>%
  group_by(country_location_name, indicator, frequency) %>%
  summarise(n_loc_year = n()) %>%
  na.omit() %>%
  arrange(country_location_name, indicator, frequency)

coverage_ihme_admin1 = locs_1_2 %>%
  filter(location_type == 'admin1') %>%
  na.omit() %>% distinct() %>% arrange(country_location_name,location_type,location_name) %>%
  group_by(country_location_name,location_type) %>%
  summarise(num_subnationals = n())

# based on the coverage we might want to drop these indicators for now
# the reason is less universal and too specific for specific locations, i mean macau and hk in china here
# 'household_expenditure_total'
# 'household_income_total'
# 'net_domestic_product'
# 'net_domestic_product_per_capita'
'%!in%'= Negate('%in%')
ceic_long_merged = ceic_long_merged %>% filter(indicator %!in% c('household_expenditure_total',
                                                                 'household_income_total',
                                                                 'net_domestic_product',
                                                                 'net_domestic_product_per_capita'))



# clean frequency, we need to set the no match locations aside since it will mess up our cleaning
ceic_long_merged_match = ceic_long_merged %>% filter(!is.na(location_id))
ceic_long_merged_nomatch = ceic_long_merged %>% filter(is.na(location_id))
ceic_quarterly = ceic_long_merged_match[ceic_long_merged_match$frequency == "Quarterly, ending \"Mar, June, Sep, Dec\"",]
ceic_monthly = ceic_long_merged_match[ceic_long_merged_match$frequency == "Monthly",]

# monthly clean frequency
ceic_monthly_check = ceic_monthly %>%
  group_by(country_location_id, location_id, country_location_name, location_name, location_name_original,
           indicator, frequency, unit, year) %>%
  summarize(month_num = n()) %>%
  arrange(country_location_name, location_name, location_name_original, indicator, frequency, unit, year)
table(ceic_monthly_check$month_num)
# impute for 10 and 11 months holders
ceic_monthly_check_incomplete = ceic_monthly_check %>% filter(month_num %in% c(10,11))
ceic_monthly_clean_incomplete = merge(x = ceic_monthly_check_incomplete, y = ceic_monthly,
                           by.x = c("country_location_id", "location_id", "country_location_name", "location_name",
                                    "location_name_original", "indicator", "frequency", "unit", "year"),
                           by.y = c("country_location_id", "location_id", "country_location_name", "location_name",
                                    "location_name_original", "indicator", "frequency", "unit", "year"),
                           all.x = T)
ceic_monthly_clean_incomplete = ceic_monthly_clean_incomplete %>%
  group_by(country_location_id, location_id, country_location_name, location_name, iso3, location_name_original, indicator,
           frequency, source, unit, year, geomatching_notes, notes, level, path_to_top_parent, location_type) %>%
  summarise(value = mean(value)*12) %>%
  mutate(frequency = 'Annual',
         imputed = 1)

ceic_monthly_check = ceic_monthly_check %>% filter(month_num == 12) %>% select(-month_num)
ceic_monthly_clean = merge(x = ceic_monthly_check, y = ceic_monthly,
                           by.x = c("country_location_id", "location_id", "country_location_name", "location_name",
                                    "location_name_original", "indicator", "frequency", "unit", "year"),
                           by.y = c("country_location_id", "location_id", "country_location_name", "location_name",
                                    "location_name_original", "indicator", "frequency", "unit", "year"),
                           all.x = T)
table(ceic_monthly_clean$month)  # quick check
ceic_monthly_agg = ceic_monthly_clean %>%
  group_by(country_location_id, location_id, country_location_name, location_name, iso3, location_name_original, indicator,
           frequency, source, unit, year, geomatching_notes, notes, level, path_to_top_parent, location_type) %>%
  summarize(value = sum(value)) %>%
  mutate(frequency = "Annual")

ceic_monthly_agg = rbind(ceic_monthly_agg, ceic_monthly_clean_incomplete)
ceic_monthly_agg$imputed[is.na(ceic_monthly_agg$imputed)] = 0

# quarterly clean frequency
ceic_quarterly_check = ceic_quarterly %>%
  group_by(country_location_id, location_id, country_location_name, location_name, location_name_original,
           indicator, frequency, unit, year) %>%
  summarize(quarter_num = n()) %>%
  arrange(country_location_name, location_name, location_name_original, indicator, frequency, unit, year)
table(ceic_quarterly_check$quarter_num)
# impute for 3 quarters holders
ceic_quarterly_check_incomplete = ceic_quarterly_check %>% filter(quarter_num == 3)
ceic_quarterly_clean_incomplete = merge(x = ceic_quarterly_check_incomplete, y = ceic_quarterly,
                                      by.x = c("country_location_id", "location_id", "country_location_name", "location_name",
                                               "location_name_original", "indicator", "frequency", "unit", "year"),
                                      by.y = c("country_location_id", "location_id", "country_location_name", "location_name",
                                               "location_name_original", "indicator", "frequency", "unit", "year"),
                                      all.x = T)
ceic_quarterly_clean_incomplete = ceic_quarterly_clean_incomplete %>%
  group_by(country_location_id, location_id, country_location_name, location_name, iso3, location_name_original, indicator,
           frequency, source, unit, year, geomatching_notes, notes, level, path_to_top_parent, location_type) %>%
  summarise(value = mean(value)*4) %>%
  mutate(frequency = 'Annual',
         imputed = 1)

ceic_quarterly_check = ceic_quarterly_check %>% filter(quarter_num == 4) %>% select(-quarter_num)
ceic_quarterly_clean = merge(x = ceic_quarterly_check, y = ceic_quarterly,
                           by.x = c("country_location_id", "location_id", "country_location_name", "location_name",
                                    "location_name_original", "indicator", "frequency", "unit", "year"),
                           by.y = c("country_location_id", "location_id", "country_location_name", "location_name",
                                    "location_name_original", "indicator", "frequency", "unit", "year"),
                           all.x = T)
table(ceic_quarterly_clean$month)  # quick check
ceic_quarterly_agg = ceic_quarterly_clean %>%
  group_by(country_location_id, location_id, country_location_name, location_name, iso3, location_name_original, indicator,
           frequency, source, unit, year, geomatching_notes, notes, level, path_to_top_parent, location_type) %>%
  summarize(value = sum(value)) %>%
  mutate(frequency = "Annual")

ceic_quarterly_agg = rbind(ceic_quarterly_agg, ceic_quarterly_clean_incomplete)
ceic_quarterly_agg$imputed[is.na(ceic_quarterly_agg$imputed)] = 0

# annual data clean
ceic_annual = ceic_long_merged_match %>% 
  filter(frequency %in% c("Annual, ending \"Dec\" of each year", 
                          "Annual, ending \"Sep\" of each year",
                          "Annual, ending \"Mar\" of each year")) %>%
  select(-month)

ceic_final_matched = rbindlist(list(ceic_annual, ceic_monthly_agg, ceic_quarterly_agg), use.names = T, fill = T)

# final check location metadata
locs = locs %>% select(location_id, location_name, level, path_to_top_parent, location_type) 
ceic_final_matched = ceic_final_matched %>% select(-c('location_name', 'level', 'path_to_top_parent', 'location_type'))
ceic_final_matched = merge(x = ceic_final_matched, y = locs, by = "location_id", all.x = T)
setDT(ceic_final_matched)[location_id %in% c(62982, 62998), country_location_id := 62]
setDT(ceic_final_matched)[location_id %in% c(62982, 62998), country_location_name := "russian federation"]
setDT(ceic_final_matched)[location_id %in% c(62982, 62998), iso3 := "RUS"]
setDT(ceic_final_matched)[location_id == 66011, country_location_id := 6]
setDT(ceic_final_matched)[location_id == 66011, country_location_name := "china"]
setDT(ceic_final_matched)[location_id == 66011, iso3 := "CHN"]

ceic_final_matched = ceic_final_matched %>% 
  select(-frequency) %>%
  arrange(iso3, location_type, location_name, year, indicator) %>%
  rename(measure = indicator) %>%
  mutate(nid = '', 
         file_path = '', 
         source_location_type = '', 
         source = 'CEIC',
         data_type = 'admin tabulated',
         value_type = 'observed', 
         currency = 'LCU', 
         initials = 'GG',
         file_path = '/home/j/DATA/Incoming Data/LSAE_income/CEIC/',
         long = '',
         lat = '',
         location_code = '',
         base_year = '',
         currency_detail = '',
         shapefile = '')
ceic_final_matched$imputed[is.na(ceic_final_matched$imputed)] = 0
# unit and currency clear
setDT(ceic_final_matched)[unit == "RMB bn", unit := "billions"]
setDT(ceic_final_matched)[unit %in% c("RMB", "HKD", "MOP", "RUB"), unit := ""]
setDT(ceic_final_matched)[unit == "USD", currency := "USD"]
setDT(ceic_final_matched)[unit == "USD", unit := ""]
setDT(ceic_final_matched)[unit %in% c("HKD mn", "MOP mn", "INR mn", "RUB mn"), unit := "millions"]

# extraction template
ceic_final_matched = ceic_final_matched %>% 
  select(nid, source, data_type, file_path, year, iso3, location_id, location_type, source_location_type,
         location_name, location_name_original, lat, long, location_code, shapefile, measure, unit, value, 
         value_type, currency,base_year, currency_detail, notes, geomatching_notes, imputed, initials) %>%
  arrange(iso3, location_type, location_id, year)



