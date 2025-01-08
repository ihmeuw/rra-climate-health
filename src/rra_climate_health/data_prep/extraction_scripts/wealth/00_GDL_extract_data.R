#' @Title: [wealth_data_GDL.R]  
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
p_load(data.table, dplyr, readxl, haven, tidyverse, foreign)
#p_load(data.table, dplyr, readxl, haven, colorout, tidyverse, foreign)
if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", Sys.getenv("USER"), "/")
  k <- "/ihme/cc_resources/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
}

source("~/indicators/1_retrospective/1_GDP_LDI/LSAE/helper_functions.R")

locs <- get_LSAE_locations(admin_level = "all")


source(file = paste0(k, "libraries/current/r/get_location_metadata.R")) 
locs <- get_location_metadata(location_set_version_id = 910, location_set_id = 125, gbd_round_id = 8, release_id = 13) # GBD 2021 release LSAE set and version - no release ID marked as active
# locs0 <- locs[location_type %in% c("admin0", "nonsovereign") | location_id == 60348]
locs_0 <- locs[location_type %in% c("admin0", "nonsovereign")]
locs_1_2 <- locs[location_type %in% c("admin1", "admin2")]

# Get LBD admin shapefiles data
current = '2021_08_09'
lbd_admin1 <- data.table(read.dbf(paste0("/home/j/WORK/11_geospatial/admin_shapefiles/", current,"/lbd_standard_admin_1.dbf")))
lbd_admin2 <- data.table(read.dbf(paste0("/home/j/WORK/11_geospatial/admin_shapefiles/", current,"/lbd_standard_admin_2.dbf")))

# change all keys to lowercase for merging 
locs_0$location_name <- tolower(locs_0$location_name)
locs_0$location_ascii_name <- tolower(locs_0$location_ascii_name)
locs_1_2$location_name <- tolower(locs_1_2$location_name)
locs_1_2$location_ascii_name <- tolower(locs_1_2$location_ascii_name)

suppressWarnings(locs_1_2[, country_location_id := as.integer(sub("^.*?\\,(.*?)\\,.*", "\\1", path_to_top_parent))])
locs_1_2 <- merge(locs_1_2, locs_0[, .(location_id, country_location_name = location_name, country_ISO3 = ihme_loc_id)], 
                  by.x = "country_location_id", by.y = "location_id", all.x = T)
# Manual fixes to bugs in location set
setDT(locs_1_2)[country_ISO3 == "JPN" & level == 2, location_type := "admin1"]
setDT(locs_1_2)[country_ISO3 == "GBR" & level == 2, location_type := "admin1"]
setDT(locs_1_2)[country_ISO3 == "GBR" & level == 3, location_type := "admin2"]
setDT(locs_1_2)[country_ISO3 == "RUS" & level == 2, location_type := "admin1"]
setDT(locs_1_2)[country_ISO3 == "RUS" & level == 3, location_type := "admin2"]
setDT(locs_1_2)[location_id == 69491, `:=` (parent_id = 61468, path_to_top_parent = "1,92,61468,69491", country_ISO3 = "ESP")]

# drop the duplicates by country and keep only admin1 and admin2
locs_1_2 <- locs_1_2[location_type %in% c("admin1", "admin2"),]
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

# Global Data Lab - Human Development Indices (5.0)
GDL_file = read_excel("/home/j/DATA/Incoming Data/LSAE_income/GDL_human_development_indices/GDL-Income-index-data.xlsx")

# merge the country level metadata
# GDL_file %>% filter(Level == 'National') %>% nrow()
# GDL_file %>% filter(Region == 'Total') %>% nrow()
# intersect(unique(locs$country_ISO3), unique(GDL_file$ISO_Code))
setDT(GDL_file)[ISO_Code == 'XKO', ISO_Code := 'RKS']   # fix for Kosovo
GDL_file = GDL_file[,-c("Country", "Level","GDLCODE")]

# change all keys to lowercase for merging 
GDL_file$Region <- tolower(GDL_file$Region)

GDL_file = merge(GDL_file, locs_0[, .(country_location_id = location_id, country_location_name = location_name, ihme_loc_id)],
                 by.x = "ISO_Code", by.y = "ihme_loc_id", all.x = T)

GDL_file_nat = GDL_file[GDL_file$Region == 'total',]
GDL_file_subnat = GDL_file[GDL_file$Region != 'total',]

# clean the region names before we do the merge
# options(max.print = 2000)
# sort(unique(GDL_file_subnat$Region))
GDL_file_subnat$Region = gsub('^prov. ', '', GDL_file_subnat$Region)
GDL_file_subnat$Region = gsub('^county of ', '', GDL_file_subnat$Region)
GDL_file_subnat$Region = gsub(' region$', '', GDL_file_subnat$Region)
GDL_file_subnat$Region = gsub(' county$', '', GDL_file_subnat$Region)

# location matching
# 1. fuzzy matching agrep
# gdl.name = data.frame(GDL_file_subnat$Region)
# names(gdl.name)[names(gdl.name)=="GDL_file_subnat.Region"] = "name.gdl"
# gdl.name$name.gdl = as.character(gdl.name$name.gdl)
# gdl.name = unique(gdl.name) # Removing duplicates
# head(gdl.name)
# 
# ihme.name = data.frame(locs_1_2$location_ascii_name)
# names(ihme.name)[names(ihme.name)=="locs_1_2.location_ascii_name"] = "name.ihme"
# ihme.name$name.ihme = as.character(ihme.name$name.ihme)
# ihme.name = unique(ihme.name) # Removing duplicates
# head(ihme.name)
# 
# gdl.name$name.ihme <- "" # Creating an empty column
# for(i in 1:dim(gdl.name)[1]) {
#   x <- agrep(gdl.name$name.gdl[i], ihme.name$name.ihme,
#              ignore.case=TRUE, value=TRUE,
#              max.distance = 0, useBytes = FALSE)
#   x <- paste0(x,"")
#   gdl.name$name.ihme[i] <- x
# }


# location matching
# 2. manual merge
# merge the subnational level metadata
# uniqueN(GDL_file_subnat$Region)
# uniqueN(locs_1_2$location_name)
# x = unique(GDL_file_subnat[,c('ISO_Code','Region')]) %>% arrange(ISO_Code,Region)
# y = unique(locs_1_2[,c('country_ISO3','location_name','path_to_top_parent')]) %>% arrange(country_ISO3,location_name)
# intersect(unique(locs_1_2$location_name), unique(GDL_file_subnat$Region))
#
# GDL_file_subnat_temp = merge(GDL_file_subnat, locs_1_2[, .(country_location_id, country_location_name, Region = location_ascii_name, location_id, path_to_top_parent, location_type)],
#                              by = c('country_location_id', 'country_location_name', 'Region'), all.x = T) 
# GDL_file_subnat_temp = GDL_file_subnat_temp %>% 
#   select(country_location_id,country_location_name,Region,location_id,location_type,path_to_top_parent, everything()) %>%
#   arrange(country_location_name,Region,location_id,location_type,path_to_top_parent)
# GDL_file_subnat_temp %>% filter(is.na(location_id)) %>% nrow() # total num not matching
# not_matching_countries = GDL_file_subnat_temp %>% filter(is.na(location_id)) %>% distinct(country_location_name) # total num not matching
# not_matching_countries = not_matching_countries$country_location_name # not matching countries random distributed
# 
# GDL_file_subnat_test = merge(GDL_file_subnat, locs_1_2[, .(country_location_id, country_location_name, Region = location_ascii_name, location_id, path_to_top_parent, location_type)],
#                              by = c('country_location_id', 'country_location_name', 'Region'), all = T) 
# GDL_file_subnat_test = GDL_file_subnat_test %>% 
#   select(country_location_id,country_location_name,Region,location_id,location_type,path_to_top_parent, everything()) %>%
#   arrange(country_location_name,Region,location_id,location_type,path_to_top_parent)
#
# manual hotfix for some locations
# setDT(GDL_file_subnat)[(Region == 'tirana') & (country_location_name == 'albania'), Region := 'tirane']
# setDT(GDL_file_subnat)[(Region == 'malange') & (country_location_name == 'angola'), Region := 'malanje']
# setDT(GDL_file_subnat)[(Region == 'city of buenos aires') & (country_location_name == 'argentina'), Region := 'ciudad autonoma de buenos aires']
# setDT(GDL_file_subnat)[(Region == 'rest of buenos aires') & (country_location_name == 'argentina'), Region := 'buenos aires']
# setDT(GDL_file_subnat)[(Region == 'baku') & (country_location_name == 'azerbaijan'), Region := 'baku city']
# setDT(GDL_file_subnat)[(Region == 'ganja gazakh') & (country_location_name == 'azerbaijan'), Region := 'ganja-gazakh']
# setDT(GDL_file_subnat)[(Region == 'guba khachmaz') & (country_location_name == 'azerbaijan'), Region := 'guba-khachmaz']
# setDT(GDL_file_subnat)[(Region == 'shaki zaqatala') & (country_location_name == 'azerbaijan'), Region := 'shaki-zagatala']
# setDT(GDL_file_subnat)[(Region == 'st michael') & (country_location_name == 'barbados'), Region := 'saint michael']
# setDT(GDL_file_subnat)[(Region == 'brest region') & (country_location_name == 'belarus'), Region := 'brest']
# setDT(GDL_file_subnat)[(Region == 'minsk region') & (country_location_name == 'belarus'), Region := 'minsk']
# setDT(GDL_file_subnat)[(Region == 'bruxelles - brussel') & (country_location_name == 'belgium'), Region := 'bruxelles']
# stopped at 4013


# location matching
# 3. fuzzy matching stringdist
library(stringdist)
library(fuzzyjoin)
GDL_file_subnat_test = GDL_file_subnat %>% select(Region, ISO_Code) %>% na.omit()
locs_1_2_test = locs_1_2 %>% select(location_id, location_ascii_name, country_ISO3) %>% na.omit()
# you can always change the method and max distince for your own dataset, it is a trade off and judgement call
# but we should use both country id and subnational id for merge
test = GDL_file_subnat_test %>%
  stringdist_left_join(locs_1_2_test, by = c(ISO_Code = 'country_ISO3', Region = 'location_ascii_name'), 
                       method = 'osa', max_dist = 2, ignore_case = TRUE, distance_col = 'distance_col')
test = test[!is.na(test$Region),]
test1 = test %>% filter((ISO_Code.distance_col == 0) | (is.na(ISO_Code.distance_col)))
test1 = test1 %>% select(-c("distance_col", "ISO_Code", "ISO_Code.distance_col")) %>% arrange(country_ISO3, Region, location_ascii_name)
# pick up the rows not 100% matched and vet them
test2 = test1 %>% filter((Region.distance_col != 0) | (is.na(Region.distance_col))) %>% arrange(country_ISO3, Region, location_ascii_name)
# drop the locations which has 100% match from our vetting table, we have the perfect one then why we keep the bad ones, saving time move
temp1 = test1[test1$Region.distance_col==0,] %>% select(country_ISO3, Region) %>% na.omit() %>% distinct()
test2 %>% select(country_ISO3, Region) %>% na.omit() %>% distinct() %>% nrow()
test2 = anti_join(test2, temp1, by = c('country_ISO3', 'Region'))
# drop the rows after our vetting
# the vetting below is done before i add the above part so actually many of them is not necessary
test2 = test2[!(test2$Region=="bie" & test2$location_ascii_name=="uige" & test2$country_ISO3=="AGO"),]
test2 = test2[!(test2$Region=="uige" & test2$location_ascii_name=="bie" & test2$country_ISO3=="AGO"),]
test2 = test2[!(test2$Region=="diber" & test2$location_ascii_name=="fier" & test2$country_ISO3=="ALB"),]
test2 = test2[!(test2$Region=="fier" & test2$location_ascii_name=="diber" & test2$country_ISO3=="ALB"),]
test2 = test2[!(test2$Region=="centre-est" & test2$location_ascii_name=="centre-ouest" & test2$country_ISO3=="BFA"),]
test2 = test2[!(test2$Region=="centre-ouest" & test2$location_ascii_name=="centre-est" & test2$country_ISO3=="BFA"),]
test2 = test2[!(test2$country_ISO3=="BRA"),]
test2 = test2[!(test2$Region=="gasa" & test2$location_ascii_name=="haa" & test2$country_ISO3=="BTN"),]
test2 = test2[!(test2$Region=="haa" & test2$location_ascii_name=="gasa" & test2$country_ISO3=="BTN"),]
test2 = test2[!(test2$country_ISO3=="BWA"),]
test2 = test2[!(test2$country_ISO3=="CHN"),]
test2 = test2[!(test2$Region=="est" & test2$location_ascii_name=="ouest" & test2$country_ISO3=="CMR"),]
test2 = test2[!(test2$Region=="ouest" & test2$location_ascii_name=="est" & test2$country_ISO3=="CMR"),]
test2 = test2[!(test2$Region=="arauca" & test2$location_ascii_name=="cauca" & test2$country_ISO3=="COL"),]
test2 = test2[!(test2$Region=="cauca" & test2$location_ascii_name=="arauca" & test2$country_ISO3=="COL"),]
setDT(test2)[Region == "c. habana", location_ascii_name := "Ciudad de la Habana"]
setDT(test2)[Region == "c. habana", location_id := 61300]
setDT(test2)[Region == "habana", location_ascii_name := "la habana"]
setDT(test2)[Region == "habana", location_id := 61305]
setDT(test2)[Region == "habana", country_ISO3 := "CUB"]
test2 = test2[!(test2$Region=="giza" & test2$location_ascii_name=="qina" & test2$country_ISO3=="EGY"),]
test2 = test2[!(test2$Region=="aland" & test2$location_ascii_name=="lapland" & test2$country_ISO3=="FIN"),]
test2 = test2[!(test2$country_ISO3=="GHA"),]
test2 = test2[!(test2$country_ISO3=="GBR"),]
test2 = test2[!(test2$country_ISO3=="HND"),]
test2 = test2[!(test2$country_ISO3=="IDN"),]
test2 = test2[!(test2$Region=="gilan" & test2$location_ascii_name=="ilam" & test2$country_ISO3=="IRN"),]
test2 = test2[!(test2$Region=="ilam" & test2$location_ascii_name=="gilan" & test2$country_ISO3=="IRN"),]
test2 = test2[!(test2$Region=="golestan" & test2$location_ascii_name=="lorestan" & test2$country_ISO3=="IRN"),]
test2 = test2[!(test2$Region=="kordestan" & test2$location_ascii_name=="lorestan" & test2$country_ISO3=="IRN"),]
test2 = test2[!(test2$Region=="lorestan" & test2$location_ascii_name=="golestan" & test2$country_ISO3=="IRN"),]
test2 = test2[!(test2$location_id==44881),]
test2 = test2[!(test2$location_id==44879),]
test2 = test2[!(test2$location_id==61703),]
test2 = test2[!(test2$Region=="amman" & test2$location_ascii_name=="ma'an" & test2$country_ISO3=="JOR"),]
test2 = test2[!(test2$Region=="aqaba" & test2$location_ascii_name=="madaba" & test2$country_ISO3=="JOR"),]
test2 = test2[!(test2$Region=="balqa" & test2$location_ascii_name=="zarqa" & test2$country_ISO3=="JOR"),]
test2 = test2[!(test2$Region=="maan" & test2$location_ascii_name=="amman" & test2$country_ISO3=="JOR"),]
test2 = test2[!(test2$Region=="madaba" & test2$location_ascii_name=="aqaba" & test2$country_ISO3=="JOR"),]
test2 = test2[!(test2$Region=="zarqa" & test2$location_ascii_name=="balqa" & test2$country_ISO3=="JOR"),]
test2 = test2[!(test2$Region=="kampong thum" & test2$location_ascii_name=="kampong cham" & test2$country_ISO3=="KHM"),]
test2 = test2[!(test2$Region=="bomi" & test2$location_ascii_name=="bong" & test2$country_ISO3=="LBR"),]
test2 = test2[!(test2$Region=="bong" & test2$location_ascii_name=="bomi" & test2$country_ISO3=="LBR"),]
test2 = test2[!(test2$location_id==61955),]
test2 = test2[!(test2$location_id==60143),]
test2 = test2[!(test2$location_id==60146),]
test2 = test2[!(test2$location_id==60145),]
test2 = test2[!(test2$location_id==60144),]
test2 = test2[!(test2$location_id %in% c(25334,25346,25348,25353,25329,25347,25345,25333)),]
test2 = test2[!(test2$location_id %in% c(62236,60158)),]
test2 = test2[!(test2$country_ISO3=="PNG"),]
test2 = test2[!(test2$country_ISO3=="POL"),]
test2 = test2[!(test2$country_ISO3=="PRT"),]
test2 = test2[!(test2$Region=="north darfur" & test2$location_ascii_name=="south darfur" & test2$country_ISO3=="SDN"),]
test2 = test2[!(test2$Region=="south darfur" & test2$location_ascii_name=="north darfur" & test2$country_ISO3=="SDN"),]
test2 = test2[!(test2$Region=="west darfur" & test2$location_ascii_name=="east darfur" & test2$country_ISO3=="SDN"),]
test2 = test2[!(test2$Region=="bari" & test2$location_ascii_name=="bay" & test2$country_ISO3=="SOM"),]
test2 = test2[!(test2$Region=="bay" & test2$location_ascii_name=="bari" & test2$country_ISO3=="SOM"),]
test2 = test2[!(test2$Region=="western equatoria" & test2$location_ascii_name=="eastern equatoria" & test2$country_ISO3=="SSD"),]
test2 = test2[!(test2$country_ISO3=="SVN"),]
test2 = test2[!(test2$location_id %in% c(62963,62960,62967,62966,62976)),]
test2 = test2[!(test2$country_ISO3=="USA"),]
test2 = test2[!(test2$location_id %in% c(490,482)),]
test2 = test2[!(test2$location_id %in% c(35539,35538,35537,35532,44844,44843)),]

test2 = test2[!(test2$location_id %in% c(63553,63648,60437,69408,69283,69763,69765,70577,91742,71580,72457,72554,72536,72514,75461,75386,75446,75447,75427,
                                         75577,75648,84449,84499,84489,85308,86000,86020,86026,86026)),]
test2 = test2[!(test2$Region %in% c('katanga','praha','coste','nyanza','armm','centru','sud','vest','lome')),]
test2 = test2[!(test2$location_ascii_name %in% c('point-noire')),]
setDT(test2)[Region == "gangwon", location_ascii_name := "gangwon-do"]
setDT(test2)[Region == "gangwon", location_id := 61807]
setDT(test2)[Region == "gyeongbuk", location_ascii_name := "gyeongsangbuk-do"]
setDT(test2)[Region == "gyeongbuk", location_id := 61810]
setDT(test2)[Region == "gyeongnam", location_ascii_name := "gyeongsangnam-do"]
setDT(test2)[Region == "gyeongnam", location_id := 61811]
setDT(test2)[Region == "ajk", location_ascii_name := "azad jammu & kashmir"]
setDT(test2)[Region == "ajk", location_id := 53615]
setDT(test2)[Region == "bangkok", location_ascii_name := "bangkok metropolis"]
setDT(test2)[Region == "bangkok", location_id := 62706]

# rbind the vetted table with the table 100% merged
test3 = test1 %>% filter(Region.distance_col == 0)
test4 = rbind(test3, test2) %>% arrange(country_ISO3, Region, location_ascii_name) %>% select(-Region.distance_col)
final_loc_match = copy(test4)

rm(test, test1, test2, test3, test4, temp1)
final_loc_match[duplicated(final_loc_match[,c("country_ISO3","Region")]),]  # check the duplicates for the final matching table

GDL_file_subnat = merge(x = GDL_file_subnat, y = final_loc_match, by.x = c("ISO_Code","Region"), by.y = c("country_ISO3","Region"), all.x = T)

GDL_file_subnat = GDL_file_subnat %>% 
  select("country_location_id", "country_location_name", "ISO_Code", "location_id", "Region", "location_ascii_name", everything()) %>%
  arrange(country_location_id, country_location_name, ISO_Code, Region, location_ascii_name) %>%
  dplyr::rename(country_iso_code = ISO_Code,
         location_name_original = Region,
         location_name = location_ascii_name) 

# sanity check for some locations exceed the max distance
setDT(GDL_file_subnat)[location_name_original == "bangka belitung", location_id := 4717]
setDT(GDL_file_subnat)[location_name_original == "di aceh", location_id := 4709]
setDT(GDL_file_subnat)[location_name_original == "di yogyakarta", location_id := 4723]
setDT(GDL_file_subnat)[location_name_original == "dki jakarta", location_id := 4720]
setDT(GDL_file_subnat)[location_name_original == "riau (incl. riau islands)", location_id := 4712]
setDT(GDL_file_subnat)[location_name_original == "south sulawesi (incl sulawesi barat)", location_id := 4735]
setDT(GDL_file_subnat)[location_name_original == "vientiane municipality", location_id := 61838]
setDT(GDL_file_subnat)[location_name_original == "vientiane province", location_id := 61837]
setDT(GDL_file_subnat)[location_name_original == "kuala lumpur federal territory", location_id := 55406]
setDT(GDL_file_subnat)[location_name_original == "labuan federal territory", location_id := 55407]
setDT(GDL_file_subnat)[location_name_original == "ii-cagayan valley", location_id := 53544]
setDT(GDL_file_subnat)[location_name_original == "bangkok", location_id := 62706]
setDT(GDL_file_subnat)[location_name_original == "oecussi", location_id := 62797]
setDT(GDL_file_subnat)[location_name_original == "autonomous region of bougainville", location_id := 62266]
setDT(GDL_file_subnat)[location_name_original == "chimbu, simbu", location_id := 62268]
setDT(GDL_file_subnat)[location_name_original == "northern, oro", location_id := 62282]
setDT(GDL_file_subnat)[location_name_original == "west sepik, sandaun", location_id := 62283]
setDT(GDL_file_subnat)[location_name_original == "ongo niua", location_id := 62810]
setDT(GDL_file_subnat)[location_name_original == "baku", location_id := 60418]
setDT(GDL_file_subnat)[location_name_original == "imereti racha-lochkhumi kvemo svaneti", location_id := 61515]
setDT(GDL_file_subnat)[location_name_original == "almaty city", location_id := 61740]
setDT(GDL_file_subnat)[location_name_original == "gbao", location_id := 62782]
setDT(GDL_file_subnat)[location_name_original == "sughd (formerly leninabad)", location_id := 62784]
setDT(GDL_file_subnat)[location_name_original == "ashgabat city", location_id := 62790]
setDT(GDL_file_subnat)[location_name_original == "bucuresti", location_id := 62445]
setDT(GDL_file_subnat)[location_name_original == "national capital", location_id := 53614]
setDT(GDL_file_subnat)[location_name_original == "luganville", location_id := 90295]
setDT(GDL_file_subnat)[location_name_original == "port vila", location_id := 90314]
setDT(GDL_file_subnat)[location_name_original == "baku", location_id := 60418]
setDT(GDL_file_subnat)[location_name_original == "almaty city", location_id := 61740]
setDT(GDL_file_subnat)[location_name_original == "gbao", location_id := 62782]
setDT(GDL_file_subnat)[location_name_original == "sughd (formerly leninabad)", location_id := 62784]
setDT(GDL_file_subnat)[location_name_original == "belgrade", location_id := 62566]
setDT(GDL_file_subnat)[location_name_original == "bratislavsky kraj", location_id := 62612]
setDT(GDL_file_subnat)[location_name_original == "posavska", location_id := 62629]
setDT(GDL_file_subnat)[location_name_original == "capital", location_id := 61817]



# vetting after geomatching
# In this extraction, we will only focus on these countries since based on geomatching with shapefiles
# the concordance for them are high. But this doesn't mean that we don't want the data from the rest locations. 
GDL_file_subnat$notes = NA_character_
GDL_file_subnat$geomatching_notes = NA_character_
# 76	Belgium	BEL		admin2	high
setDT(GDL_file_subnat)[location_name_original == "bruxelles - brussel", location_id := 64665]
#setDT(GDL_file_subnat)[location_name_original == "bruxelles - brussel", location_id := 61056]
# 98	Chile	CHL		admin1	medium
setDT(GDL_file_subnat)[location_name_original == "antofagasta", location_id := 59690]
setDT(GDL_file_subnat)[location_name_original == "arbucania", location_id := 59694]
setDT(GDL_file_subnat)[location_name_original == "atacama", location_id := 59692]
setDT(GDL_file_subnat)[location_name_original == "coquimbo", location_id := 59693]
setDT(GDL_file_subnat)[location_name_original == "los lagos (incl los rios)", notes := 'location id 59695 + 59696']
setDT(GDL_file_subnat)[location_name_original == "los lagos (incl los rios)", geomatching_notes := 'Multiple GBD regions sum to source region']
setDT(GDL_file_subnat)[location_name_original == "magallanes and la antartica chilena", location_id := 59697]
setDT(GDL_file_subnat)[location_name_original == "maule", location_id := 59702]
setDT(GDL_file_subnat)[location_name_original == "ohiggins", location_id := 59701]
setDT(GDL_file_subnat)[location_name_original == "region metropolitana", location_id := 59703]
setDT(GDL_file_subnat)[location_name_original == "tarapaca (incl arica and parinacota)", notes := 'location id 59698 + 59691']
setDT(GDL_file_subnat)[location_name_original == "tarapaca (incl arica and parinacota)", geomatching_notes := 'Multiple GBD regions sum to source region']
setDT(GDL_file_subnat)[location_name_original == "valparaiso (former aconcagua)", location_id := 59699]
setDT(GDL_file_subnat)[location_name_original == "bio bio", location_id := "NA"]
setDT(GDL_file_subnat)[location_name_original == "bio bio", notes := 'location id 61221 + 59700']
setDT(GDL_file_subnat)[location_name_original == "bio bio", geomatching_notes := 'Multiple GBD regions sum to source region']
setDT(GDL_file_subnat)[location_name_original == "aisen", location_id := 59689]

# 92	Spain	ESP		admin1	medium
setDT(GDL_file_subnat)[location_name_original == "ciudad autonoma de ceuta", notes := 'location id 61460/2']
setDT(GDL_file_subnat)[location_name_original == "ciudad autonoma de ceuta", geomatching_notes := 'Multiple source regions sum to GBD region']
setDT(GDL_file_subnat)[location_name_original == "ciudad autonoma de melilla", notes := 'location id 61460/2']
setDT(GDL_file_subnat)[location_name_original == "ciudad autonoma de melilla", geomatching_notes := 'Multiple source regions sum to GBD region']
setDT(GDL_file_subnat)[location_name_original == "illes balears", location_id := 61466]
setDT(GDL_file_subnat)[location_name_original == 'canarias', geomatching_notes := 'Location set issue']
setDT(GDL_file_subnat)[location_name_original == 'canarias', notes := 'location id 311 but not under ESP']

# 82	Greece	GRC		admin1	medium
#setDT(GDL_file_subnat)[location_name_original == "dytiki makedonia", location_id := 61572]
#setDT(GDL_file_subnat)[location_name_original == "dytiki ellada", location_id := 61573]
#setDT(GDL_file_subnat)[location_name_original == "sterea ellada", location_id := 61574]
#setDT(GDL_file_subnat)[location_name_original == "anatoliki makedonia, thraki", location_id := 61572]
#setDT(GDL_file_subnat)[location_name_original == "ipeiros", location_id := 61571]
#setDT(GDL_file_subnat)[location_name_original == "kriti", location_id := 70175]
#setDT(GDL_file_subnat)[location_name_original == "ionia nisia", location_id := 70180]
#setDT(GDL_file_subnat)[location_name_original == "kentriki makedonia", location_id := 70178]
#setDT(GDL_file_subnat)[location_name_original == "notio aigaio", location_id := 70172]
#setDT(GDL_file_subnat)[location_name_original == "peloponnisos", location_id := 70181]
#setDT(GDL_file_subnat)[location_name_original == "voreio aigaio", location_id := 70171]

# updated Greece 
setDT(GDL_file_subnat)[location_name_original == "dytiki makedonia", location_id := 70177]
setDT(GDL_file_subnat)[location_name_original == "dytiki ellada", location_id := 70182]
setDT(GDL_file_subnat)[location_name_original == "sterea ellada", location_id := 70183]
setDT(GDL_file_subnat)[location_name_original == "anatoliki makedonia, thraki", location_id := 70179]
setDT(GDL_file_subnat)[location_name_original == "ipeiros", location_id := 70176]
setDT(GDL_file_subnat)[location_name_original == "kriti", location_id := 70175]
setDT(GDL_file_subnat)[location_name_original == "ionia nisia", location_id := 70180]
setDT(GDL_file_subnat)[location_name_original == "kentriki makedonia", location_id := 70178]
setDT(GDL_file_subnat)[location_name_original == "notio aigaio", location_id := 70171]
setDT(GDL_file_subnat)[location_name_original == "peloponnisos", location_id := 70181]
setDT(GDL_file_subnat)[location_name_original == "voreio aigaio", location_id := 70172]
setDT(GDL_file_subnat)[location_name_original == "attiki", location_id := 70174]
setDT(GDL_file_subnat)[location_name_original == "thessalia", location_id := 70184]

# 90	Norway	NOR		admin1 	medium
setDT(GDL_file_subnat)[location_name_original == "agder og rogaland", location_id := 4920]
setDT(GDL_file_subnat)[location_name_original == "hedmark og oppland", location_id := 60135]
setDT(GDL_file_subnat)[location_name_original == "nord-norge", notes := 'location id 4926 + 60137']
setDT(GDL_file_subnat)[location_name_original == "nord-norge", geomatching_notes := 'Multiple GBD regions sum to source region']
# 72	New Zealand	NZL		admin1 	medium
setDT(GDL_file_subnat)[location_name_original == "tasman - nelson", notes := 'location id 62237 + 62232']
setDT(GDL_file_subnat)[location_name_original == "tasman - nelson", geomatching_notes := 'Multiple GBD regions sum to source region']


# adding notes and geomatching_notes for subnationals
# below the list is high/medium countries from shapefile geo-matching and all of them are high income countries
### high
# Australia
# Austria
# Belgium - 
# Denmark
# Germany
# Iceland
# Italy
# Luxembourg
# Netherlands
### medium
# Chile - 
# Greece - 
# New Zealand
# Spain - 

# add notes for the non matched locations
setDT(GDL_file_subnat)[is.na(location_id), geomatching_notes := 'Not priority based on geomatching/model-selection']
setDT(GDL_file_subnat)[is.na(location_id), notes := 'Not priority based on geomatching/model-selection']

# sanity check, now for every row, we have location id and NA at notes, or we have notes and NA at location id
nrow(GDL_file_subnat) == sum(!is.na(GDL_file_subnat$location_id)) + sum(!is.na(GDL_file_subnat$geomatching_notes))

# merge national and subnational data
GDL_file_nat = GDL_file_nat %>%
  dplyr::rename(country_iso_code = ISO_Code, location_name_original = Region) %>%
  mutate(location_id = country_location_id)

GDL_final = rbind(GDL_file_nat, GDL_file_subnat, fill = T) %>% 
  select(country_location_id, country_location_name, country_iso_code, location_id, location_name_original, 
         geomatching_notes, notes, everything()) %>%
  select(-location_name)

GDL_final = merge(x = GDL_final, y = locs[,c('location_id', 'location_name', 'location_type')], by = "location_id", all.x = T)

GDL_final = GDL_final %>%
  select(country_location_id, country_location_name, country_iso_code, location_id, location_name, location_type,
         location_name_original, geomatching_notes, notes, everything()) %>%
  arrange(country_location_id, country_location_name, country_iso_code, location_type) %>%
  distinct()

GDL_final_long = melt(setDT(GDL_final), 
                      id.vars = c("country_location_id","country_location_name","country_iso_code","location_id",
                                  "location_name","location_type","location_name_original","geomatching_notes","notes"), 
                      variable.name = "year",
                      value.name = "value")


# Merge on geolocation info from shapefiles
GDL_final_long_before_merge <- copy(GDL_final_long)
GDL_final_long <- merge(GDL_final_long, lbd_admin1[, c("loc_id", "ADM1_CODE")], by.x = "location_id", by.y = "loc_id", all.x = T)
GDL_final_long <- merge(GDL_final_long, lbd_admin2[, c("loc_id", "ADM2_CODE")], by.x = "location_id", by.y = "loc_id", all.x = T)
validate_merge_nrows(GDL_final_long_before_merge, GDL_final_long)
GDL_final_long <- data.table(GDL_final_long)
GDL_final_long[!is.na(ADM1_CODE), location_code := ADM1_CODE] 
GDL_final_long[!is.na(ADM1_CODE), shapefile := "lbd_standard_admin_1"]
GDL_final_long[!is.na(ADM2_CODE), location_code := ADM2_CODE]
GDL_final_long[!is.na(ADM2_CODE), shapefile := "lbd_standard_admin_2"]
GDL_final_long[, c("ADM1_CODE", "ADM2_CODE") := NULL]

# extraction template
GDL_final_long$nid = 498423
GDL_final_long$source = 'GDL'
GDL_final_long$data_type = 'admin_tabulated'
GDL_final_long$file_path = '/home/j/DATA/Incoming Data/LSAE_income/GDL_human_development_indices/GDL-Income-index-data.xlsx'
GDL_final_long$lat = NA
GDL_final_long$long = NA
#GDL_final_long$shapefile = '/home/j/DATA/Incoming Data/LSAE_income/GDL_human_development_indices/GDL_lbd_concordance_mapping.mxd'
GDL_final_long$measure = 'income index'
GDL_final_long$denominator = 'index'
GDL_final_long$multiplier = 1
GDL_final_long$value_type = 'aggregated'
GDL_final_long$source_location_type = NA
GDL_final_long$source_location_id = NA
GDL_final_long$currency = NA
#GDL_final_long$base_year = 'TBD'
GDL_final_long$currency_detail = NA
GDL_final_long$initials = 'GG'

GDL_final_long = GDL_final_long %>% 
  dplyr::rename(iso3 = country_iso_code) %>%
  select(nid, source, data_type, file_path, year = variable, iso3, location_id, location_type, source_location_type, source_location_id, 
         location_name, lat, long, location_code, shapefile, measure, denominator, multiplier, value, 
         value_type, currency,base_year = variable, currency_detail, notes, geomatching_notes, initials) %>%
  arrange(iso3, location_type, location_id, year)

# Drop NA values 
GDL_final_long <- as.data.table(GDL_final_long)
GDL_final_long <- GDL_final_long[!is.na(value)]

#not sure why this isnt working 
#GDL_final_long <- GDL_final_long[!(geomatching_notes == "Not priority based on geomatching/model-selection")]
#subset(GDL_final_long, geomatching_notes!='not_priority')
GDL_final_long <- GDL_final_long[is.na(geomatching_notes)]


# sanity check, now for every row, we have location id and NA at notes, or we have notes and NA at location id
nrow(GDL_final_long) == sum(!is.na(GDL_final_long$location_id)) + sum(!is.na(GDL_final_long$geomatching_notes))
# number of rows matches the raw data in wide format
nrow(GDL_file) == nrow(GDL_final)


# Run some checks to validate data processing
validate_extractions(GDL_final_long)
# For each country, check number of locations against what is expected from the LSAE location set
validate_geomatched_locations(GDL_final_long, 2010)

table(GDL_final_long$geomatching_notes)

gdl <- data.table(GDL_final_long)
save_extraction(gdl, compiled = F)

