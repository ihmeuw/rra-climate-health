#' @Title: [00_extract_eurostat_data.R]  
#' @Authors: Kayleigh Bhangdia, Bianca Zlavog
#' @contact: bhangdia@uw.edu, zlavogb@uw.edu
#' @Date_code_last_updated:  10/24/2022
#' @Purpose: Process wealth data (GDP and income) sources into extraction template format
#' 
#' @File_Input(s)
#'  
#' @File_Output(s)
#'

##### Setup

rm(list = ls())
require(pacman)
p_load(data.table, dplyr, readxl, haven, tidyverse, foreign)

################ EUROSTAT GDP #################
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

##### Data extraction

#EUROSTAT
#Load in NUTS2 location data keeping only relevant columns 
eurostat_nuts2_mapping <- data.table(read_excel(paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/NUTS2_table.xlsx")))
eurostat_nuts2_mapping[ , (6:26) := NULL][ , (2) := NULL]
setnames(eurostat_nuts2_mapping, c("NUTS", "NAME", "NUTS0"), c("NUTS", "location", "country_code"))
eurostat_nuts2_mapping[, `:=` (nuts_level=2)]

#Load in NUTS3 location data keeping only relevant columns 
eurostat_nuts3_mapping <- data.table(read_excel(paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/NUTS3_table.xlsx")))
eurostat_nuts3_mapping[ , (6:26) := NULL][ , (2) := NULL]
setnames(eurostat_nuts3_mapping, c("NUTS", "NAME", "NUTS0"), c("NUTS", "location", "country_code"))
eurostat_nuts3_mapping[, `:=` (nuts_level=3)]

#bind together NUTS2 and NUTS3
eurostat_all <- rbind(eurostat_nuts2_mapping, eurostat_nuts3_mapping)

#load in sheet that contains NUTS and their associated iso3 codes and merge with NUTS location data 
nuts0_iso3 <- data.table(read_excel(paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/NUTS0_to_iso3.xlsx")))
eurostat_all <- merge(eurostat_all, nuts0_iso3, on = 'country_code', all.x=TRUE)

#subset of countries with medium or high concordance with admin1 or admin2 (based on 'Admin level merging by country' sheet)
#also keep low concordance when 'Most granular location not matching GBD' - for this, Denmark, Estonia, Ireland and Portugal have low concordance 
#across all NUTS (so keeping NUTS3) and Austria, Cyprus, Netherlands, Poland and Serbia have medium/high concordance at NUTS2 but low at NUTS3 (so keeping both) 
#Keep countries where NUTS2 match with admin0: Cyprus, Serbia (**add in admin0 for these countries)
#Keep countries where NUTS3 match with admin0: Luxembourg, Malta, Montenegro, North Macedonia (**add in admin0 for these countries)
#Keeping Finland, France, and Germany at NUTS2 with note that 'Less granular location not matching GBD' since they have medium/high concordance at NUTS3 but low at NUTS2 
#Keeping Italy and Spain at NUTS2 since they have high concordance with admin1 (in addition to NUTS3 and admin2 having high concordance) (**add in admin1 for these countries)
eurostat_high <- eurostat_all[country_code=="AT" & nuts_level==2 | country_code=="BG" & nuts_level==3 | country_code=="CZ" & nuts_level==3 
                              | country_code=="DE" & nuts_level==3 | country_code=="GR" & nuts_level==2 |country_code=="ES" & nuts_level==3
                              | country_code=="FI" & nuts_level==3 | country_code=="FR" & nuts_level==3 | country_code=="HR" & nuts_level==3
                              | country_code=="HU" & nuts_level==3| country_code=="IT" & nuts_level==3 | country_code=="LT" & nuts_level==3
                              | country_code=="LV" & nuts_level==3 | country_code=="NL" & nuts_level==2| country_code=="NO" & nuts_level==3
                              | country_code=="PL" & nuts_level==2 | country_code=="RO" & nuts_level==3 | country_code=="SE" & nuts_level==3
                              | country_code=="SI" & nuts_level==3 | country_code=="SK" & nuts_level==3| country_code=="TR" & nuts_level==3
                              | country_code=="DK" & nuts_level==3 | country_code=="EE" & nuts_level==3| country_code=="IE" & nuts_level==3
                              | country_code=="PT" & nuts_level==3 | country_code=="AT" & nuts_level==3| country_code=="CY" & nuts_level==3
                              | country_code=="NL" & nuts_level==3 | country_code=="PL" & nuts_level==3| country_code=="RS" & nuts_level==3
                              | country_code=="CY" & nuts_level==2 | country_code=="RS" & nuts_level==2| country_code=="LU" & nuts_level==3
                              | country_code=="MT" & nuts_level==3 | country_code=="ME" & nuts_level==3| country_code=="MK" & nuts_level==3
                              | country_code=="FI" & nuts_level==2 | country_code=="FR" & nuts_level==2| country_code=="DE" & nuts_level==2
                              | country_code=="IT" & nuts_level==2 | country_code=="ES" & nuts_level==2| country_code=="BE" & nuts_level==2] 

#clean up LBD locs file
locs_eurostat <- copy(locs)[, c("location_name", "location_id", "location_type", "country_ISO3")]
locs_eurostat <- locs_eurostat[country_ISO3 %in% unique(eurostat_high$iso3)]
#locs_eurostat <- locs_eurostat[!(country_ISO3 %in% c("TUR", "SWE", "SVN", "SVK", "ROU", "POL", "PER", "NZL", "NOR", "NLD", "MEX", "LVA", "MLT", "LUX", "USA", "LTU", "KOR", 
#                                                     "IRL", "IDN", "HUN", "HRV") & location_type == "admin2")]
#locs_eurostat <- locs_eurostat[!(country_ISO3 %in% c("GRC", "FIN", "DEU") & location_type == "admin1")]

#keep only countries that had medium to high concordance at admin1 or admin2 level and were labeled HICs according to GBD (total of 21 countries) 
locs_eurostat <- locs_eurostat[country_ISO3=="AUT" & location_type=="admin1" | country_ISO3=="BGR" & location_type=="admin1" | country_ISO3=="CZE" & location_type=="admin1" 
                               | country_ISO3=="DEU" & location_type=="admin2" | country_ISO3=="GRC" & location_type=="admin2" |country_ISO3=="ESP" & location_type=="admin2"
                               | country_ISO3=="FIN" & location_type=="admin1" | country_ISO3=="FRA" & location_type=="admin2" | country_ISO3=="HRV" & location_type=="admin1"
                               | country_ISO3=="HUN" & location_type=="admin1"| country_ISO3=="ITA" & location_type=="admin2" | country_ISO3=="LTU" & location_type=="admin1"
                               | country_ISO3=="LVA" & location_type=="admin1" | country_ISO3=="NLD" & location_type=="admin1"| country_ISO3=="NOR" & location_type=="admin1"
                               | country_ISO3=="POL" & location_type=="admin1" | country_ISO3=="ROU" & location_type=="admin1" | country_ISO3=="SWE" & location_type=="admin1"
                               | country_ISO3=="SVN" & location_type=="admin1" | country_ISO3=="SVK" & location_type=="admin1"| country_ISO3=="TUR" & location_type=="admin1"
                               | country_ISO3=="ITA" & location_type=="admin1" | country_ISO3=="ESP" & location_type=="admin1"| country_ISO3=="CYP" & location_type=="admin0"
                               | country_ISO3=="LUX" & location_type=="admin0" | country_ISO3=="MLT" & location_type=="admin0"| country_ISO3=="MNE" & location_type=="admin0"
                               | country_ISO3=="MKD" & location_type=="admin0" | country_ISO3=="SRB" & location_type=="admin0" |country_ISO3=="BEL" & location_type=="admin2"] 

#clean up naming by country

#Bulgaria
eurostat_high[NUTS == "BG411" & location == "Sofia (stolitsa)", location := "Grad Sofiya"]

#Czechia
eurostat_high[NUTS == "CZ064" & location == "Jihomoravský kraj", location := "Jihomoravský"]
eurostat_high[NUTS == "CZ031" & location == "Jihočeský kraj", location := "Jihočeský"]
eurostat_high[NUTS == "CZ041" & location == "Karlovarský kraj", location := "Karlovarský"]
eurostat_high[NUTS == "CZ052" & location == "Královéhradecký kraj", location := "Královéhradecký"]
eurostat_high[NUTS == "CZ051" & location == "Liberecký kraj", location := "Liberecký"]
eurostat_high[NUTS == "CZ080" & location == "Moravskoslezský kraj", location := "Moravskoslezský"]
eurostat_high[NUTS == "CZ071" & location == "Olomoucký kraj", location := "Olomoucký"]
eurostat_high[NUTS == "CZ053" & location == "Pardubický kraj", location := "Pardubický"]
eurostat_high[NUTS == "CZ032" & location == "Plzeňský kraj", location := "Plzeňský"]
eurostat_high[NUTS == "CZ020" & location == "Středočeský kraj", location := "Středočeský"]
eurostat_high[NUTS == "CZ072" & location == "Zlínský kraj", location := "Zlínský"]
eurostat_high[NUTS == "CZ042" & location == "Ústecký kraj", location := "Ústecký"]
eurostat_high[NUTS == "CZ010" & location == "Hlavní město Praha", location := "Prague"]

#Spain
eurostat_high[NUTS == "ES213" & location == "Bizkaia", location := "Vizcaya"]
eurostat_high[NUTS == "ES630" & location == "Ceuta", location := "Ceuta"]
eurostat_high[NUTS == "ES613" & location == "Córdoba", location := "Córdoba"]
eurostat_high[NUTS == "ES512" & location == "Girona", location := "Girona"]
eurostat_high[NUTS == "ES212" & location == "Gipuzkoa", location := "Guipúzcoa"]
eurostat_high[NUTS == "ES111" & location == "A Coruña", location := "A Coruña"]
eurostat_high[NUTS == "ES513" & location == "Lleida", location := "Lleida"]
eurostat_high[NUTS == "ES220" & location == "Navarra", location := "Navarra"]
eurostat_high[NUTS == "ES113" & location == "Ourense", location := "Ourense"]
eurostat_high[NUTS == "ES243" & location == "Zaragoza", location := "Zaragoza"]
eurostat_high[NUTS == "ES618" & location == "Sevilla", location := "Sevilla"]
eurostat_high[NUTS == "ES211" & location == "Araba/Álava", location := "Álava"]
eurostat_high[NUTS == "ES523" & location == "Valencia/València", location := "Valencia"]
eurostat_high[NUTS == "ES522" & location == "Castellón/Castelló", location := "Castellón"]
eurostat_high[NUTS == "ES521" & location == "Alicante/Alacant", location := "Alicante"]
eurostat_high[NUTS == "ES532" & location == "Mallorca", location := "Baleares"]

eurostat_high[NUTS == "ES63" & location == "Ciudad de Ceuta", location := "Ceuta"]
eurostat_high[NUTS == "ES64" & location == "Ciudad de Melilla", location := "Melilla"]
eurostat_high[NUTS == "ES52" & location == "Comunitat Valenciana", location := "Comunidad Valenciana"]
eurostat_high[NUTS == "ES53" & location == "Illes Balears", location := "Islas Baleares"]


#Sweden
#eurostat_high[NUTS == "SE124" & location == "Örebro", location := "Orebro"]
eurostat_high[NUTS == "SE221" & location == "Blekinge län", location := "Blekinge"]
eurostat_high[NUTS == "SE312" & location == "Dalarnas län", location := "Dalarna"]
eurostat_high[NUTS == "SE214" & location == "Gotlands län", location := "Gotland"]
eurostat_high[NUTS == "SE313" & location == "Gävleborgs län", location := "Gävleborg"]
eurostat_high[NUTS == "SE231" & location == "Hallands län", location := "Halland"]
eurostat_high[NUTS == "SE322" & location == "Jämtlands län", location := "Jämtland"]
eurostat_high[NUTS == "SE211" & location == "Jönköpings län", location := "Jönköping"]
eurostat_high[NUTS == "SE213" & location == "Kalmar län", location := "Kalmar"]
eurostat_high[NUTS == "SE212" & location == "Kronobergs län", location := "Kronoberg"]
eurostat_high[NUTS == "SE332" & location == "Norrbottens län", location := "Norrbotten"]
eurostat_high[NUTS == "SE224" & location == "Skåne län", location := "Skåne"]
eurostat_high[NUTS == "SE110" & location == "Stockholms län", location := "Stockholm"]
eurostat_high[NUTS == "SE122" & location == "Södermanlands län", location := "Södermanland"]
eurostat_high[NUTS == "SE121" & location == "Uppsala län", location := "Uppsala"]
eurostat_high[NUTS == "SE311" & location == "Värmlands län", location := "Värmland"]
eurostat_high[NUTS == "SE331" & location == "Västerbottens län", location := "Västerbotten"]
eurostat_high[NUTS == "SE321" & location == "Västernorrlands län", location := "Västernorrland"]
eurostat_high[NUTS == "SE125" & location == "Västmanlands län", location := "Västmanland"]
eurostat_high[NUTS == "SE232" & location == "Västra Götalands län", location := "Västra Götaland"]
eurostat_high[NUTS == "SE124" & location == "Örebro län", location := "Orebro"]
eurostat_high[NUTS == "SE123" & location == "Östergötlands län", location := "Östergötland"]

#Turkey
eurostat_high[NUTS == "TRC12" & location == "Adıyaman", location := "Adiyaman"]
eurostat_high[NUTS == "TR332" & location == "Afyonkarahisar", location := "Afyon"]
eurostat_high[NUTS == "TR321" & location == "Aydın", location := "Aydin"]
eurostat_high[NUTS == "TRA21" & location == "Ağrı", location := "Agri"]
eurostat_high[NUTS == "TR221" & location == "Balıkesir", location := "Balikesir"]
eurostat_high[NUTS == "TRC22" & location == "Diyarbakır", location := "Diyarbakir"]
eurostat_high[NUTS == "TR412" & location == "Eskişehir", location := "Eskisehir"]
eurostat_high[NUTS == "TR906" & location == "Gümüşhane", location := "Gümüshane"]
eurostat_high[NUTS == "TR632" & location == "Kahramanmaraş", location := "K. Maras"]
eurostat_high[NUTS == "TR213" & location == "Kırklareli", location := "Kirklareli"]
eurostat_high[NUTS == "TR711" & location == "Kırıkkale", location := "Kinkkale"]
eurostat_high[NUTS == "TR715" & location == "Kırşehir", location := "Kirsehir"]
eurostat_high[NUTS == "TR323" & location == "Muğla", location := "Mugla"]
eurostat_high[NUTS == "TRB22" & location == "Muş", location := "Mus"]
eurostat_high[NUTS == "TR714" & location == "Nevşehir", location := "Nevsehir"]
eurostat_high[NUTS == "TR713" & location == "Niğde", location := "Nigde"]
eurostat_high[NUTS == "TR211" & location == "Tekirdağ", location := "Tekirdag"]
eurostat_high[NUTS == "TR334" & location == "Uşak", location := "Usak"]
eurostat_high[NUTS == "TR811" & location == "Zonguldak", location := "Zinguldak"]
eurostat_high[NUTS == "TR822" & location == "Çankırı", location := "Çankiri"]
eurostat_high[NUTS == "TR100" & location == "İstanbul", location := "Istanbul"]
eurostat_high[NUTS == "TR310" & location == "İzmir", location := "Izmir"]
eurostat_high[NUTS == "TRC21" & location == "Şanlıurfa", location := "Sanliurfa"]
eurostat_high[NUTS == "TRC33" & location == "Şırnak", location := "Sirnak"]

#Slovenia
eurostat_high[NUTS == "SI036" & location == "Posavska", location := "Spodnjeposavska"]
eurostat_high[NUTS == "SI038" & location == "Primorsko-notranjska", location := "Notranjsko-kraška"]

#Slovakia
eurostat_high[NUTS == "SK032" & location == "Banskobystrický kraj", location := "Banskobystrický"]
eurostat_high[NUTS == "SK010" & location == "Bratislavský kraj", location := "Bratislavský"]
eurostat_high[NUTS == "SK042" & location == "Košický kraj", location := "Košický"]
eurostat_high[NUTS == "SK023" & location == "Nitriansky kraj", location := "Nitriansky"]
eurostat_high[NUTS == "SK041" & location == "Prešovský kraj", location := "Prešovský"]
eurostat_high[NUTS == "SK022" & location == "Trenčiansky kraj", location := "Trenčiansky"]
eurostat_high[NUTS == "SK021" & location == "Trnavský kraj", location := "Trnavský"]
eurostat_high[NUTS == "SK031" & location == "Žilinský kraj", location := "Žilinský"]

#Romania
eurostat_high[NUTS == "RO311" & location == "Argeş", location := "Argeș"]
eurostat_high[NUTS == "RO112" & location == "Bistriţa-Năsăud", location := "Bistrița-Năsăud"]
eurostat_high[NUTS == "RO212" & location == "Botoşani", location := "Botoșani"]
eurostat_high[NUTS == "RO122" & location == "Braşov", location := "Brașov"]
eurostat_high[NUTS == "RO321" & location == "Bucureşti", location := "Bucharest"]
eurostat_high[NUTS == "RO312" & location == "Călăraşi", location := "Călărași"]
eurostat_high[NUTS == "RO422" & location == "Caraş-Severin", location := "Caraș-Severin"]
eurostat_high[NUTS == "RO223" & location == "Constanţa", location := "Constanța"]
eurostat_high[NUTS == "RO313" & location == "Dâmboviţa", location := "Dâmbovița"]
eurostat_high[NUTS == "RO224" & location == "Galaţi", location := "Galați"]
eurostat_high[NUTS == "RO315" & location == "Ialomiţa", location := "Ialomița"]
eurostat_high[NUTS == "RO213" & location == "Iaşi", location := "Iași"]
eurostat_high[NUTS == "RO114" & location == "Maramureş", location := "Maramureș"]
eurostat_high[NUTS == "RO413" & location == "Mehedinţi", location := "Mehedinți"]
eurostat_high[NUTS == "RO125" & location == "Mureş", location := "Mureș"]
eurostat_high[NUTS == "RO214" & location == "Neamţ", location := "Neamț"]
eurostat_high[NUTS == "RO424" & location == "Timiş", location := "Timiș"]

#Lithuania
eurostat_high$location = gsub(" apskritis", "", eurostat_high$location)
eurostat_high[NUTS == "LT023" & location == "Klaipėdos", location := "Klaipedos"]
eurostat_high[NUTS == "LT024" & location == "Marijampolės", location := "Marijampoles"]
eurostat_high[NUTS == "LT025" & location == "Panevėžio", location := "Panevezio"]
eurostat_high[NUTS == "LT027" & location == "Tauragės", location := "Taurages"]
eurostat_high[NUTS == "LT028" & location == "Telšių", location := "Telšiai"]
eurostat_high[NUTS == "LT026" & location == "Šiaulių", location := "Šiauliai"]

#Netherlands
eurostat_high$location = gsub(" (NL)", "", eurostat_high$location)
eurostat_high[NUTS == "NL12" & location == "Friesland (NL)", location := "Friesland"]
eurostat_high[NUTS == "NL42" & location == "Limburg (NL)", location := "Limburg"]

#Finland
eurostat_high[NUTS == "FI1C5" & location == "Etelä-Karjala", location := "South Karelia"]
eurostat_high[NUTS == "FI194" & location == "Etelä-Pohjanmaa", location := "Southern Ostrobothnia"]
eurostat_high[NUTS == "FI1D1" & location == "Etelä-Savo", location := "Southern Savonia"]
eurostat_high[NUTS == "FI1D5" & location == "Keski-Pohjanmaa", location := "Central Ostrobothnia"]
eurostat_high[NUTS == "FI193" & location == "Keski-Suomi", location := "Central Finland"]
eurostat_high[NUTS == "FI1D7" & location == "Lappi", location := "Lapland"]
eurostat_high[NUTS == "FI195" & location == "Pohjanmaa", location := "Ostrobothnia"]
eurostat_high[NUTS == "FI1D3" & location == "Pohjois-Karjala", location := "North Karelia"]
eurostat_high[NUTS == "FI1D9" & location == "Pohjois-Pohjanmaa", location := "Northern Ostrobothnia"]
eurostat_high[NUTS == "RO424" & location == "Timiş", location := "Timiș"]
eurostat_high[NUTS == "FI1D2" & location == "Pohjois-Savo", location := "Northern Savonia"] 
eurostat_high[NUTS == "FI1C1" & location == "Varsinais-Suomi", location := "Finland Proper"]
eurostat_high[(NUTS == "FI200" & location == "Åland"), notes := "Extra region in source"]
eurostat_high[NUTS == "FI1B1" & location == "Helsinki-Uusimaa", location := "Uusimaa"]
eurostat_high[NUTS == "FI1C3" & location == "Päijät-Häme", location := "Päijänne Tavastia"]

#Greece
eurostat_high[NUTS == "EL30" & location == "Attikí", location := "Attica"]
eurostat_high[NUTS == "EL63" & location == "Dytikí Elláda", location := "West Greece"]
eurostat_high[NUTS == "EL53" & location == "Dytikí Makedonía", location := "West Macedonia"]
eurostat_high[NUTS == "EL51" & location == "Anatolikí Makedonía, Thráki", location := "East Macedonia and Thrace"]
eurostat_high[NUTS == "EL43" & location == "Kríti", location := "Crete"]
eurostat_high[NUTS == "EL42" & location == "Nótio Aigaío", location := "South Aegean"]
eurostat_high[NUTS == "EL65" & location == "Pelopónnisos", location := "Peloponnese"]
eurostat_high[NUTS == "EL64" & location == "Stereá Elláda", location := "Central Greece"]
eurostat_high[NUTS == "EL61" & location == "Thessalía", location := "Thessaly"]
eurostat_high[NUTS == "EL41" & location == "Vóreio Aigaío", location := "North Aegean"]
#eurostat_high[NUTS == "EL52" & location == "Kentrikí Makedonía", location := "Central Macedonia"] # doesn't match nicely, EL52 excludes Athos
eurostat_high[NUTS == "EL62" & location == "Iónia Nisiá", location := "Ionian Islands"]
eurostat_high[NUTS == "EL54" & location == "Ípeiros", location := "Epirus"]

#Croatia
eurostat_high$location = gsub(" (županija)", "", eurostat_high$location)
eurostat_high[NUTS == "HR021" & location == "Bjelovarsko-bilogorska", location := "Bjelovarska-Bilogorska"]
eurostat_high[NUTS == "HR024" & location == "Brodsko-posavska", location := "Brodsko-Posavska"]
eurostat_high[NUTS == "HR037" & location == "Dubrovačko-neretvanska", location := "Dubrovacko-Neretvanska"]
eurostat_high[NUTS == "HR027" & location == "Karlovačka", location := "Karlovacka"]
eurostat_high[NUTS == "HR063" & location == "Koprivničko-križevačka", location := "Koprivničko-Križevačka"]
eurostat_high[NUTS == "HR064" & location == "Krapinsko-zagorska", location := "Krapinsko-Zagorska"]
eurostat_high[NUTS == "HR032" & location == "Ličko-senjska", location := "Licko-Senjska"]
eurostat_high[NUTS == "HR061" & location == "Međimurska", location := "Medimurska"]
eurostat_high[NUTS == "HR025" & location == "Osječko-baranjska", location := "Osjecko-Baranjska"]
eurostat_high[NUTS == "HR023" & location == "Požeško-slavonska", location := "Požeško-Slavonska"]
eurostat_high[NUTS == "HR031" & location == "Primorsko-goranska", location := "Primorsko-Goranska"]
eurostat_high[NUTS == "HR028" & location == "Sisačko-moslavačka", location := "Sisacko-Moslavacka"]
eurostat_high[NUTS == "HR035" & location == "Splitsko-dalmatinska", location := "Splitsko-Dalmatinska"]
eurostat_high[NUTS == "HR022" & location == "Virovitičko-podravska", location := "Viroviticko-Podravska"]
eurostat_high[NUTS == "HR026" & location == "Vukovarsko-srijemska", location := "Vukovarsko-Srijemska"]
eurostat_high[NUTS == "HR034" & location == "Šibensko-kninska", location := "Šibensko-Kninska"]

#Poland
eurostat_high[NUTS == "PL61" & location == "Kujawsko-pomorskie", location := "Kujawsko-Pomorskie"]
eurostat_high[NUTS == "PL62" & location == "Warmińsko-mazurskie", location := "Warmińsko-Mazurskie"]

#Italy
eurostat_high[NUTS == "ITH10" & location == "Bolzano-Bozen", location := "Bolzano"]
eurostat_high[NUTS == "ITI14" & location == "Firenze", location := "Florence"]
eurostat_high[NUTS == "ITH58" & location == "Forlì-Cesena", location := "Forli' - Cesena"]
eurostat_high[NUTS == "ITC4B" & location == "Mantova", location := "Mantua"]
eurostat_high[NUTS == "ITI11" & location == "Massa-Carrara", location := "Massa Carrara"]
eurostat_high[NUTS == "ITC4D" & location == "Monza e della Brianza", location := "Monza and Brianza"]
eurostat_high[NUTS == "ITH36" & location == "Padova", location := "Padua"]
eurostat_high[NUTS == "ITI31" & location == "Pesaro e Urbino", location := "Pesaro E Urbino"]
eurostat_high[NUTS == "ITF65" & location == "Reggio di Calabria", location := "Reggio Di Calabria"]
eurostat_high[NUTS == "ITH53" & location == "Reggio nell'Emilia", location := "Reggio Nell'Emilia"]
eurostat_high[NUTS == "ITG19" & location == "Siracusa", location := "Syracuse"]
eurostat_high[NUTS == "ITC20" & location == "Valle d'Aosta/Vallée d'Aoste", location := "Aosta"]

eurostat_high[NUTS == "ITH1" & location == "Provincia Autonoma di Bolzano/Bozen", location := "Provincia autonoma di Bolzano"]
eurostat_high[NUTS == "ITH2" & location == "Provincia Autonoma di Trento", location := "Provincia autonoma di Trento"]
eurostat_high[NUTS == "ITC2" & location == "Valle d'Aosta/Vallée d'Aoste", location := "Valle d'Aosta"]

#Germany
eurostat_high$location = gsub(", Landkreis", "", eurostat_high$location)
eurostat_high$location = gsub(", Kreis", "", eurostat_high$location)
eurostat_high$location = gsub(", Kreisfreie Stadt", "", eurostat_high$location)
eurostat_high$location = gsub(", Stadt", "", eurostat_high$location)
eurostat_high$location = gsub("freie Stadt", "", eurostat_high$location)
eurostat_high[NUTS == "DEA2D" & location == "Aachen, St√§dteregion", location := "St√§dteregion Aachen"]
eurostat_high[NUTS == "DE711" & location == "Darmstadt, Wissenschaftsstadt", location := "Darmstadt"]
eurostat_high[NUTS == "DE277" & location == "Dillingen a.d.Donau", location := "Dillingen an der Donau"]
eurostat_high[NUTS == "DE929" & location == "Hannover, Region", location := "Region Hannover"]
eurostat_high[NUTS == "DE731" & location == "Kassel, documenta-Stadt", location := "Kassel (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DEF02" & location == "Kiel, Landeshauptstadt", location := "Kiel"]
eurostat_high[NUTS == "DEF03" & location == "L√ºbeck, Hansestadt", location := "L√ºbeck"]
eurostat_high[NUTS == "DED51" & location == "Leipzig", location := "Leipzig (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE21G" & location == "M√ºhldorf a.Inn", location := "M√ºhldorf am Inn"]
eurostat_high[NUTS == "DE212" & location == "M√ºnchen, Landeshauptstadt", location := "MÔøΩnchen (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DEE03" & location == "Magdeburg, Landeshauptstadt", location := "Magdeburg"]
eurostat_high[NUTS == "DE126" & location == "Mannheim, Universit√§tsstadt", location := "Mannheim"]
eurostat_high[NUTS == "DE236" & location == "Neumarkt i.d.OPf.", location := "Neumarkt in der Oberpfalz"]
eurostat_high[NUTS == "DE25A" & location == "Neustadt a.d.Aisch-Bad Windsheim", location := "Neustadt an der Aisch-Bad Windsheim"]
eurostat_high[NUTS == "DE237" & location == "Neustadt a.d.Waldnaab", location := "Neustadt an der Waldnaab"]
eurostat_high[NUTS == "DE943" & location == "Oldenburg (Oldenburg)", location := "Oldenburg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE21J" & location == "Pfaffenhofen a.d.Ilm", location := "Pfaffenhofen an der Ilm"]
eurostat_high[NUTS == "44531" & location == "Saarbr√ºcken, Regionalverband", location := "Regionalverband Saarbr√ºcken"]
eurostat_high[NUTS == "DE938" & location == "Soltau-Fallingbostel", location := "Heidekreis"]
eurostat_high[NUTS == "DE111" & location == "Stuttgart, Landeshauptstadt", location := "Stuttgart"]
eurostat_high[NUTS == "DE144" & location == "Ulm, Universit√§tsstadt", location := "Ulm"]
eurostat_high[NUTS == "DE233" & location == "Weiden i.d.OPf.", location := "Weiden in der Oberpfalz"]
eurostat_high[NUTS == "DEG0G" & location == "Weimarer-Land", location := "Weimarer Land"]
eurostat_high[NUTS == "DE714" & location == "Wiesbaden, Landeshauptstadt", location := "Wiesbaden"]
eurostat_high[NUTS == "DE24D" & location == "Wunsiedel i.Fichtelgebirge", location := "Wunsiedel im Fichtelgebirge"]
eurostat_high[NUTS == "DE251" & location == "Ansbach", location := "Ansbach (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE261" & location == "Aschaffenburgfreie Stadt", location := "Aschaffenburg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE271" & location == "Augsburg", location := "Augsburg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE241" & location == "Bamberg", location := "Bamberg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE242" & location == "Bayreuth", location := "Bayreuth (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE243" & location == "Coburg", location := "Coburg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DEB32" & location == "Kaiserslautern", location := "Kaiserslautern (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE122" & location == "Karlsruhe", location := "Karlsruhe (Stadtkreis)"]
eurostat_high[NUTS == "DE221" & location == "Landshut", location := "Landshut (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DEF03" & location == "Lübeck, Hansestadt", location := "L√ºbeck"]
eurostat_high[NUTS == "DE216" & location == "Mühldorf a.Inn", location := "Mühldorf am Inn"]
eurostat_high[NUTS == "DE126" & location == "Mannheim, Universitätsstadt", location := "Mannheim"]
eurostat_high[NUTS == "DE212" & location == "München, Landeshauptstadt", location := "MÔøΩnchen (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE222" & location == "Passau", location := "Passau (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE232" & location == "Regensburg", location := "Regensburg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "44531" & location == "Saarbr√ºcken, Regionalverband", location := "Regionalverband Saarbr√ºcken"]
eurostat_high[NUTS == "DE213" & location == "Rosenheim", location := "Rosenheim (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE803" & location == "Rostock", location := "Rostock (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE262" & location == "Schweinfurt", location := "Schweinfurt (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DEA2D" & location == "Aachen, Städteregion", location := "Städteregion Aachen"]
eurostat_high[NUTS == "DE144" & location == "Ulm, Universitätsstadt", location := "Ulm"]
eurostat_high[NUTS == "DE263" & location == "Würzburg", location := "W�rzburg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE253" & location == "Fürth", location := "F�rth (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE264" & location == "Aschaffenburg", location := "Aschaffenburg (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE249" & location == "Hof", location := "Hof (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DEF03" & location == "L√ºbeck", location := "Lübeck"]
eurostat_high[NUTS == "DE249" & location == "Hof", location := "Hof (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE249" & location == "Hof", location := "Hof (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE249" & location == "Hof", location := "Hof (Kreisfreie Stadt)"]
eurostat_high[NUTS == "DE117" & location == "Heilbronn", location := "Heilbronn (Stadtkreis)"]
eurostat_high[NUTS == "DE212" & location == "MÔøΩnchen (Kreisfreie Stadt)", location := "M�nchen (Kreisfreie Stadt)"]
eurostat_high[NUTS == "44531" & location == "Saarbrücken, Regionalverband", location := "Regionalverband Saarbrücken"]
eurostat_high[NUTS == "DE21G" & location == "Mühldorf a.Inn", location := "Mühldorf am Inn"]
eurostat_high[NUTS == "DE944" & location == "Osnabrück", location := "Osnabr�ck (Kreisfreie Stadt)"]

#Hungary 
eurostat_high[NUTS == "HU221" & location == "Győr-Moson-Sopron", location := "Gyor-Moson-Sopron"]

#Cyprus
eurostat_high[NUTS == "CY00" & location == "Kýpros", location := "Cyprus"]

#Montenegro
eurostat_high[NUTS == "ME000" & location == "Crna Gora", location := "Montenegro"]

#Macedonia
eurostat_high[NUTS == "MK000" & location == "Severna Makedonija", location := "North Macedonia"]

#Serbia
eurostat_high[NUTS == "RS00" & location == "Srbija", location := "Serbia"]

#Malta
eurostat_high[NUTS == "MT002" & location == "Gozo and Comino", location := "Malta"]

#Belgium
eurostat_high[NUTS == "BE24" & location == "Vlaams-Brabant", location := "Vlaams Brabant"]
eurostat_high[NUTS == "BE10" & location == "Région de Bruxelles-Capitale / Brussels Hoofdstedelijk Gewest", location := "Bruxelles"]
eurostat_high[NUTS == "BE34" & location == "Luxembourg (BE)", location := "Luxembourg"]
eurostat_high[NUTS == "BE22" & location == "Limburg (BE)", location := "Limburg"]

#merge with GBD locations and write to csv (for further exploration)
#### potentially change all.y to F - but need to figure out how to keep all the notes from below 
eurostat_merged <- merge(eurostat_high, locs_eurostat, by.x = c("location", "iso3"), by.y = c("location_name", "country_ISO3"), all.x = T, all.y = F)

# Manually fix a few duplicate locations at wrong admin level
eurostat_merged<- eurostat_merged[!(NUTS == "ES63" & location== "Ceuta" & location_type == "admin2")]
eurostat_merged<- eurostat_merged[!(NUTS == "ES64" & location== "Melilla" & location_type == "admin2")]
eurostat_merged<- eurostat_merged[!(NUTS == "ES13" & location== "Cantabria" & location_type == "admin2")]
eurostat_merged<- eurostat_merged[!(NUTS == "ES23" & location== "La Rioja" & location_type == "admin2")]
eurostat_merged<- eurostat_merged[!(NUTS == "ES630" & location== "Ceuta" & location_type == "admin1")]
eurostat_merged<- eurostat_merged[!(NUTS == "ES640" & location== "Melilla" & location_type == "admin1")]
eurostat_merged<- eurostat_merged[!(NUTS == "ES130" & location== "Cantabria" & location_type == "admin1")]
eurostat_merged<- eurostat_merged[!(NUTS == "ES230" & location== "La Rioja" & location_type == "admin1")]
# Test that no duplicate locations are being added by this merge
validate_merge_nrows(eurostat_high, eurostat_merged)
nrow(eurostat_high)
nrow(eurostat_merged)
list(duplicated(eurostat_merged$NUTS))
eurostat_merged %>% group_by(NUTS) %>% filter(n() > 1)

#write.csv(eurostat_merged, "LSAE/Eurostat/eurostat_merged.csv")


#Clean up merge with notes where boundaries don't align/exist 
# TODO: revisit these commented out portions and confirm they are due to shapefile changes.
#Germany
# DEU G√∂ttingen DE91C = 67142 + 67160
#eurostat_merged[NUTS == "DE91C" & location == "G√∂ttingen", notes := "Multiple GBD regions sum to source region"]
#eurostat_merged[NUTS == "DE91C" & location == "G√∂ttingen", c("location_id", "location_type") := NA]
#eurostat_merged[location_id == "66936" & location == "Bodensee", notes := "No associated source region"]
#eurostat_merged[location_id == "67142" & location == "Göttingen", notes := "Multiple GBD regions sum to source region"]
#eurostat_merged[location_id == "67160" & location == "Osterode am Harz", notes := "Multiple GBD regions sum to source region"]

#Spain
#69491 not showing up in sheet but shows up in shapefile 
eurostat_merged[NUTS == "ES230" & location == "La Rioja", location_id := 69491]
eurostat_merged[NUTS == "ES230" & location == "La Rioja", location_type := "admin2"]
eurostat_merged[NUTS == "ES531" & location == "Eivissa y Formentera", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES703" & location == "El Hierro", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES704" & location == "Fuerteventura", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES705" & location == "Gran Canaria", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES706" & location == "La Gomera", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES707" & location == "La Palma", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES708" & location == "Lanzarote", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES533" & location == "Menorca", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES709" & location == "Tenerife", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES70" & location == "Canarias", notes := "Extra region in source"]
eurostat_merged[NUTS == "ES64" & location == "Ciudad de Melilla", notes := "Extra region in source"]
#Greece
# GRC EL52 Kentrikí Makedonía = 70178 + 70173 
eurostat_merged[NUTS == "EL52" & location == "Kentrikí Makedonía", notes := "Multiple GBD regions sum to source region"]
eurostat_merged[location_id == "70173" & location == "Athos", notes := "Multiple GBD regions sum to source region"]
eurostat_merged[location_id == "70178" & location == "Central Macedonia", notes := "Multiple GBD regions sum to source region"]

#Finland
# FIN Helsinki-Uusimaa FI1B1 = sum(Eastern Uusimaa 69806 + Uusimaa 69810)
eurostat_merged[NUTS == "FI1B1" & location == "Helsinki-Uusimaa", notes := "Multiple GBD regions sum to source region"]
eurostat_merged[location_id == "69806" & location == "Eastern Uusimaa", notes := "Multiple GBD regions sum to source region"]
eurostat_merged[location_id == "69810" & location == "Uusimaa", notes := "Multiple GBD regions sum to source region"]
# FIN Päijät-Häme FI1C3 = sum(Päijänne Tavastia 69801 + Päijänne Tavastia 69808 + Päijänne Tavastia 69815)
eurostat_merged[NUTS == "FI1C3" & location == "Päijät-Häme", notes := "Multiple GBD regions sum to source region"]
eurostat_merged[location_id == "69801" & location == "Päijänne Tavastia", notes := "Multiple GBD regions sum to source region"]
eurostat_merged[location_id == "69808" & location == "Päijänne Tavastia", notes := "Multiple GBD regions sum to source region"]
eurostat_merged[location_id == "69815" & location == "Päijänne Tavastia", notes := "Multiple GBD regions sum to source region"]

#Netherlands
# NDL IJsselmeer 62205 and Zeeuwse meren 62212 don't exist in GBD shapefile (empty)
eurostat_merged[location_id == "62212" & location == "Zeeuwse meren", notes := "Extra region in source"]
eurostat_merged[location_id == "62205" & location == "IJsselmeer", notes := "Extra region in source"]

#Poland
# POL Mazowieckie 53666 = sum(Mazowiecki regionalny PL92 + Warszawski stołeczny PL91)
eurostat_merged[(NUTS == "PL92" & location == "Mazowiecki regionalny") | (NUTS == "PL91" & location == "Warszawski stołeczny"), notes := "Multiple source regions sum to GBD region"]
eurostat_merged[location_id == "53666" & location == "Mazowieckie", notes := "Multiple source regions sum to GBD region"]

# LVA Riga 61934 = sum(Pierīga LV007 + Rīga LV006)
eurostat_merged[(NUTS == "LV007" & location == "Pierīga") | (NUTS == "LV006" & location == "Rīga"), notes := "Multiple source regions sum to GBD region"]
eurostat_merged[location_id == "61934" & location == "Riga", notes := "Multiple source regions sum to GBD region"]

#France
eurostat_merged[NUTS == "FRK26" & location == "Rhône", c("location_id", "location_type") := NA]
eurostat_merged[(NUTS == "FRK26" & location == "Rhône"), notes := "Multiple GBD regions sum to source region"]
eurostat_merged[(location_id == "60232" & location == "Rhône"), notes := "Multiple GBD regions sum to source region"]
eurostat_merged[(location_id == "60233" & location == "Métropole de Lyon"), notes := "Multiple GBD regions sum to source region"]

#low concordance notes 
eurostat_merged[(nuts_level == "3" & country_code == "DK"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "EE"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "IE"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "PT"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "AT"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "CY"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "NL"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "PL"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "3" & country_code == "RS"), notes := "Most granular location not matching GBD"]
eurostat_merged[(nuts_level == "2" & country_code == "FI"), notes := "Less granular location not matching GBD"]
eurostat_merged[(nuts_level == "2" & country_code == "FR"), notes := "Less granular location not matching GBD"]
eurostat_merged[(nuts_level == "2" & country_code == "DE"), notes := "Less granular location not matching GBD"]

#additions from income dataset at NUTS2
eurostat_merged[(nuts_level == "2" & country_code == "HU"), notes := "Less granular location not matching GBD"]
eurostat_merged[(nuts_level == "2" & country_code == "NO"), notes := "Less granular location not matching GBD"]
eurostat_merged[(nuts_level == "2" & country_code == "HU"), location_id := NA]
eurostat_merged[(nuts_level == "2" & country_code == "NO"), location_id := NA]

# rename location_id where 'most/less granular location not matching GBD' since it is causing duplicates and incorrectly pulling location_id
eurostat_merged[(nuts_level == "3" & country_code == "DK"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "EE"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "IE"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "PT"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "AT"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "CY"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "NL"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "PL"), location_id := NA]
eurostat_merged[(nuts_level == "3" & country_code == "RS"), location_id := NA]
eurostat_merged[(nuts_level == "2" & country_code == "FI"), location_id := NA]
eurostat_merged[(nuts_level == "2" & country_code == "FR"), location_id := NA]
eurostat_merged[(nuts_level == "2" & country_code == "DE"), location_id := NA]

#Manually update 6 Germany regions where NUTS is a series of numbers not numbers and letters 
eurostat_merged[NUTS == "44531", NUTS := "DEC01"]
eurostat_merged[NUTS == "44532", NUTS := "DEC02"]
eurostat_merged[NUTS == "44533", NUTS := "DEC03"]
eurostat_merged[NUTS == "44534", NUTS := "DEC04"]
eurostat_merged[NUTS == "44535", NUTS := "DEC05"]
eurostat_merged[NUTS == "44536", NUTS := "DEC06"]

# Run some checks 
nrow(eurostat_merged[is.na(location_id) & is.na(notes)]) # should be 0
nrow(eurostat_merged[is.na(NUTS) & is.na(notes)]) # should be 0
list(eurostat_merged[is.na(location_id) & is.na(notes)]) # should be an empty data.table
list(eurostat_merged[is.na(NUTS) & is.na(notes)]) # should be an empty data.table

#working with Eurostat GDP data

# TODO: Unsure where nuts_recoded file comes from
#load table with historical recoded nuts keeping those that switched in 2016 (removing duplicates that changed multiple times)
nuts_recoded_updated_2021 <- fread(paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/geomatched_nuts.csv"))


#load in Eurostat GDP data (both at the NUTS0, NUTS2 and NUTS3 level)
eurostat_path <- paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/EUROSTAT_GDP_LCU/nama_10r_3gdp.xlsx")
eurostat_GDP <- data.table(read_excel(eurostat_path, sheet = "Sheet 4", skip = 7)) 

colnames(eurostat_GDP)[1] = "GEO"
colnames(eurostat_GDP)[2] = "name"
eurostat_GDP <- eurostat_GDP[-1,]
eurostat_GDP <- eurostat_GDP[!is.na(GEO) & GEO != ":"]
eurostat_GDP <- eurostat_GDP[!GEO %in% c("Special value", "Available flags:", "b", "e", "p")]


#merge GDP data with recoded NUTS informaton - keeping those that never changed and replacing those that did with their updated code
eurostat_GDP<- merge(eurostat_GDP, nuts_recoded_updated_2021, by.x = c("GEO"), by.y = c("geo"), all.x = T, all.y = F)

#Manually updated NUTS codes where the recoded information did not match
eurostat_merged[NUTS == "MK000", NUTS := "MK"]
eurostat_merged[NUTS == "RS00", NUTS := "RS"]
eurostat_merged[NUTS == "ITG2G", NUTS := "ITG28"]
eurostat_merged[NUTS == "NO060", NUTS := "NO06"]
eurostat_merged[NUTS == "NO020", NUTS := "NO02"]
eurostat_merged[NUTS == "NO0A2", NUTS := "NO05"]
eurostat_merged[NUTS == "NO092", NUTS := "NO04"]
eurostat_merged[NUTS == "ITG28", NUTS := "ITG2G"]

#NO074 needs further exploration - seems like this region (with location_id 60137) is the sum of NO072 and NO073 
eurostat_GDP$geo_updated <- ifelse(is.na(eurostat_GDP$code_2021), eurostat_GDP$GEO, eurostat_GDP$code_2021)

#merge GDP data with eurostat location data that contains GDB location information removing those locations that were not matched or had notes (likely at the wrong NUTS level)
eurostat_GDP_GBD_locs<- merge(eurostat_GDP, eurostat_merged, by.x = c("geo_updated"), by.y = c("NUTS"), all.x = T)

###Start there - keep extra regions in dataset 
eurostat_GDP_GBD_locs$extra_region <- ifelse(eurostat_GDP_GBD_locs$name=="Extra-Regio NUTS 1", 1, "NA")
eurostat_GDP_GBD_locs[name == "Extra-Regio NUTS 1", location := "Extra-Regio NUTS 1"]
eurostat_GDP_GBD_locs[name == "Extra-Regio NUTS 1", notes := "Not regionalized"]
eurostat_GDP_GBD_locs[GEO == "BEZ", iso3 := "BEL"]
eurostat_GDP_GBD_locs[GEO == "DKZ", iso3 := "DNK"]
eurostat_GDP_GBD_locs[GEO == "ESZ", iso3 := "ESP"]
eurostat_GDP_GBD_locs[GEO == "FRZ", iso3 := "FRA"]
eurostat_GDP_GBD_locs[GEO == "ITZ", iso3 := "ITA"]
eurostat_GDP_GBD_locs[GEO == "LVZ", iso3 := "LVA"]
eurostat_GDP_GBD_locs[GEO == "HUZ", iso3 := "HUN"]
eurostat_GDP_GBD_locs[GEO == "MTZ", iso3 := "MLT"]
eurostat_GDP_GBD_locs[GEO == "NLZ", iso3 := "NLD"]
eurostat_GDP_GBD_locs[GEO == "ATZ", iso3 := "AUT"]
eurostat_GDP_GBD_locs[GEO == "PTZ", iso3 := "PRT"]
eurostat_GDP_GBD_locs[GEO == "ROZ", iso3 := "ROU"]
eurostat_GDP_GBD_locs[GEO == "FIZ", iso3 := "FIN"]
eurostat_GDP_GBD_locs[GEO == "SEZ", iso3 := "SWE"]

eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[!(location=="NA" & notes=="NA" & location_id=="NA" & extra_region=="NA")]


#check total country GDP against original file - most match, those that don't either have some location errors with GBD or there are Extra-Regio NUTS not included in these estimates 
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2000`:=as.numeric(`2000`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2001`:=as.numeric(`2001`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2002`:=as.numeric(`2002`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2003`:=as.numeric(`2003`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2004`:=as.numeric(`2004`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2005`:=as.numeric(`2005`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2006`:=as.numeric(`2006`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2007`:=as.numeric(`2007`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2008`:=as.numeric(`2008`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2009`:=as.numeric(`2009`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2010`:=as.numeric(`2010`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2011`:=as.numeric(`2011`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2012`:=as.numeric(`2012`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2013`:=as.numeric(`2013`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2014`:=as.numeric(`2014`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2015`:=as.numeric(`2015`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2016`:=as.numeric(`2016`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2017`:=as.numeric(`2017`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2018`:=as.numeric(`2018`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2019`:=as.numeric(`2019`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2020`:=as.numeric(`2020`)]
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, `2021`:=as.numeric(`2021`)]


# Format data to extraction template

# First clean up columns, transform to long on year (keeping flag/footnotes), and reordering 
eurostat_GDP_GBD_locs <- eurostat_GDP_GBD_locs[, !c("typology", "typology_change")]

eurostat_GDP_GBD_locs_long <- data.table::melt(setDT(eurostat_GDP_GBD_locs), 
                                               id.vars = c("geo_updated", "GEO", "name", "nuts_year", "change_year", "iso2c", "code_2021", "location", "iso3", 
                                                           "country_code", "OBJECTID *", "nuts_level", "country", "Notes", "notes", "location_id", "location_type", "extra_region" ), 
                                               measure.vars = list(value = seq(4, 46, 2),
                                                                   flags = seq(5, 47, 2)),
                                               variable.name = "year")
eurostat_GDP_GBD_locs_long$year <- eurostat_GDP_GBD_locs_long[, recode(year, "1" = "2000", "2" = "2001", "3" = "2002", "4" = "2003", "5" = "2004", "6" = "2005",
                                                                       "7" = "2006", "8" = "2007", "9" = "2008", "10" = "2009", "11" = "2010",
                                                                       "12" = "2011", "13" = "2012", "14" = "2013", "15" = "2014", "16" = "2015",
                                                                       "17" = "2016", "18" = "2017", "19" = "2018", "20" = "2019", "21" = "2020", "22" = "2021")]

setorder(eurostat_GDP_GBD_locs_long, "iso3", "location_id", "year")

eurostat <- copy(eurostat_GDP_GBD_locs_long)

# Merge on geolocation info from shapefiles
eurostat_before <- copy(eurostat)
eurostat <- merge_shapefile_locations(eurostat)
validate_merge_nrows(eurostat_before, eurostat)

# Extract data based on extraction template and update a few variables with more details 
eurostat$flags <- eurostat[, recode(flags, "b"	= "break in time series",	"c"	= "confidential",	"d"	= "definition differs, see metadata",
                                    "e"	= "estimated",	"f"	= "forecast",	"n"	= "not significant",
                                    "p"	= "provisional",	"r"	= "revised",	"s"	= "Eurostat estimate",
                                    "u"	= "low reliability",	"z"	= "not applicable")]
table(eurostat$flags)
eurostat$value_type <- ifelse(is.na(eurostat$flags), "observed", "modeled")

eurostat <- eurostat[, .(nid = 498412, source = "EUROSTAT", data_type = "admin_tabulated", file_path = eurostat_path, year = year, iso3 = iso3, 
                         location_id, location_type, location_name = location, source_location_type= nuts_level, source_location_id= geo_updated, 
                         lat = NA, long = NA, location_code, shapefile, measure = "GDP", 
                         denominator = "total", multiplier = 1000000, value = value,
                         value_type, currency = "LCU", base_year = year, currency_detail = NA, notes = flags, geomatching_notes = notes, initials = "PN")]

eurostat$source_location_type <- eurostat[, recode(source_location_type, "2" = "NUTS2", "3" = "NUTS3")]

eurostat[value== ":", value := "NA"]

eurostat <- eurostat[!is.na(value)]
eurostat <- eurostat[value != 0]

fwrite(eurostat, paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/eurostat_GDP_extraction.csv"))


# Run some checks to validate data processing
validate_extractions(eurostat)
# For each country, check number of locations against what is expected from the LSAE location set
validate_geomatched_locations(eurostat, 2010)

# source summary table 
eurostat$year <- as.integer(eurostat$year)
dt <- create_source_summary_table(eurostat)

# #load in Eurostat income data (both at the NUTS0, NUTS2 and NUTS3 level)
nuts_recoded_updated_2021 <- data.table(fread(paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/geomatched_nuts.csv")))
eurostat_path <- paste0(j, "DATA/Incoming Data/LSAE_income/Eurostat_income_NUTS2/nama_10r_2hhinc.xlsx")
eurostat_income <- data.table(read_excel(eurostat_path, sheet = "Sheet 1", skip = 8)) 


colnames(eurostat_income)[1] = "GEO"
colnames(eurostat_income)[2] = "name"
eurostat_income <- eurostat_income[-1,]
eurostat_income <- eurostat_income[!is.na(GEO) & GEO != ":"]
eurostat_income <- eurostat_income[!GEO %in% c("Special value", "Available flags:", "b", "e", "p")]

#merge GDP data with recoded NUTS informaton - keeping those that never changed and replacing those that did with their updated code
eurostat_income<- merge(eurostat_income, nuts_recoded_updated_2021, by.x = c("GEO"), by.y = c("geo"), all.x = T, all.y = F)

#Manually updated NUTS codes where the recoded information did not match
eurostat_merged[NUTS == "MK000", NUTS := "MK"]
eurostat_merged[NUTS == "RS00", NUTS := "RS"]
eurostat_merged[NUTS == "ITG2G", NUTS := "ITG28"]
eurostat_merged[NUTS == "NO060", NUTS := "NO06"]
eurostat_merged[NUTS == "NO020", NUTS := "NO02"]
eurostat_merged[NUTS == "NO0A2", NUTS := "NO05"]
eurostat_merged[NUTS == "NO092", NUTS := "NO04"]
eurostat_merged[NUTS == "ITG28", NUTS := "ITG2G"]

#NO074 needs further exploration - seems like this region (with location_id 60137) is the sum of NO072 and NO073 
eurostat_income$geo_updated <- ifelse(is.na(eurostat_income$code_2021), eurostat_income$GEO, eurostat_income$code_2021)



#merge GDP data with eurostat location data that contains GBD location information removing those locations that were not matched or had notes (likely at the wrong NUTS level)
eurostat_income_GBD_locs<- merge(eurostat_income, eurostat_merged, by.x = c("geo_updated"), by.y = c("NUTS"), all.x = T)

###Start there - keep extra regions in dataset 
eurostat_income_GBD_locs$extra_region <- ifelse(eurostat_income_GBD_locs$name=="Extra-Regio NUTS 1", 1, "NA")
eurostat_income_GBD_locs[name == "Extra-Regio NUTS 1", location := "Extra-Regio NUTS 1"]
eurostat_income_GBD_locs[name == "Extra-Regio NUTS 1", notes := "Not regionalized"]
eurostat_income_GBD_locs[GEO == "BEZ", iso3 := "BEL"]
eurostat_income_GBD_locs[GEO == "DKZ", iso3 := "DNK"]
eurostat_income_GBD_locs[GEO == "ESZ", iso3 := "ESP"]
eurostat_income_GBD_locs[GEO == "FRZ", iso3 := "FRA"]
eurostat_income_GBD_locs[GEO == "ITZ", iso3 := "ITA"]
eurostat_income_GBD_locs[GEO == "LVZ", iso3 := "LVA"]
eurostat_income_GBD_locs[GEO == "HUZ", iso3 := "HUN"]
eurostat_income_GBD_locs[GEO == "MTZ", iso3 := "MLT"]
eurostat_income_GBD_locs[GEO == "NLZ", iso3 := "NLD"]
eurostat_income_GBD_locs[GEO == "ATZ", iso3 := "AUT"]
eurostat_income_GBD_locs[GEO == "PTZ", iso3 := "PRT"]
eurostat_income_GBD_locs[GEO == "ROZ", iso3 := "ROU"]
eurostat_income_GBD_locs[GEO == "FIZ", iso3 := "FIN"]
eurostat_income_GBD_locs[GEO == "SEZ", iso3 := "SWE"]

eurostat_income_GBD_locs <- eurostat_income_GBD_locs[!(location=="NA" & notes=="NA" & location_id=="NA" & extra_region=="NA")]


#check total country GDP against original file - most match, those that don't either have some location errors with GBD or there are Extra-Regio NUTS not included in these estimates 
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `1995`:=as.numeric(`1995`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `1996`:=as.numeric(`1996`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `1997`:=as.numeric(`1997`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `1998`:=as.numeric(`1998`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `1999`:=as.numeric(`1999`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2000`:=as.numeric(`2000`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2001`:=as.numeric(`2001`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2002`:=as.numeric(`2002`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2003`:=as.numeric(`2003`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2004`:=as.numeric(`2004`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2005`:=as.numeric(`2005`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2006`:=as.numeric(`2006`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2007`:=as.numeric(`2007`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2008`:=as.numeric(`2008`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2009`:=as.numeric(`2009`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2010`:=as.numeric(`2010`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2011`:=as.numeric(`2011`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2012`:=as.numeric(`2012`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2013`:=as.numeric(`2013`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2014`:=as.numeric(`2014`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2015`:=as.numeric(`2015`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2016`:=as.numeric(`2016`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2017`:=as.numeric(`2017`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2018`:=as.numeric(`2018`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2019`:=as.numeric(`2019`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2020`:=as.numeric(`2020`)]
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, `2021`:=as.numeric(`2021`)]


# Format data to extraction template

# First clean up columns, transform to long on year (keeping flag/footnotes), and reordering 
eurostat_income_GBD_locs <- eurostat_income_GBD_locs[, !c("typology", "typology_change")]

eurostat_income_GBD_locs_long <- data.table::melt(setDT(eurostat_income_GBD_locs), 
                                                  id.vars = c("geo_updated", "GEO", "name", "nuts_year", "change_year", "iso2c", "code_2021", "location", "iso3", 
                                                              "country_code", "OBJECTID *", "nuts_level", "country", "Notes", "notes", "location_id", "location_type", "extra_region" ), 
                                                  measure.vars = list(value = seq(4, 56, 2),
                                                                      flags = seq(5, 57, 2)),
                                                  variable.name = "year")
eurostat_income_GBD_locs_long$year <- eurostat_income_GBD_locs_long[, recode(year, "1" = "1995", "2" = "1996", "3" = "1997", "4" = "1998", "5" = "1999",
                                                                             "6" = "2000", "7" = "2001", "8" = "2002", "9" = "2003", "10" = "2004", "11" = "2005",
                                                                             "12" = "2006", "13" = "2007", "14" = "2008", "15" = "2009", "16" = "2010",
                                                                             "17" = "2011", "18" = "2012", "19" = "2013", "20" = "2014", "21" = "2015",
                                                                             "22" = "2016", "23" = "2017", "24" = "2018", "25" = "2019", "26" = "2020",
                                                                             "27" = "2021")]

setorder(eurostat_income_GBD_locs_long, "iso3", "location_id", "year")

eurostat <- copy(eurostat_income_GBD_locs_long)

# Merge on geolocation info from shapefiles
eurostat_before <- copy(eurostat)
eurostat <- merge_shapefile_locations(eurostat)
validate_merge_nrows(eurostat_before, eurostat)

# Extract data based on extraction template and update a few variables with more details 
eurostat$flags <- eurostat[, recode(flags, "b"	= "break in time series",	"c"	= "confidential",	"d"	= "definition differs, see metadata",
                                         "e"	= "estimated",	"f"	= "forecast",	"n"	= "not significant",
                                         "p"	= "provisional",	"r"	= "revised",	"s"	= "Eurostat estimate",
                                         "u"	= "low reliability",	"z"	= "not applicable")]
table(eurostat$flags)
eurostat$value_type <- ifelse(is.na(eurostat$flags), "observed", "modeled")

eurostat <- eurostat[, .(nid = 498413, source = "EUROSTAT", data_type = "admin_tabulated", file_path = eurostat_path, year = year, iso3 = iso3, 
                         location_id, location_type, location_name = location, source_location_type= nuts_level, source_location_id= geo_updated, 
                         lat = NA, long = NA, location_code, shapefile, measure = "income",
                         denominator = "total", multiplier = 1000000, value = value,
                         value_type, currency = "LCU", base_year = year, currency_detail = NA, notes = flags, geomatching_notes = notes, initials = "PN")]

eurostat$source_location_type <- eurostat[, recode(source_location_type, "2" = "NUTS2", "3" = "NUTS3")]

eurostat[value== ":", value := "NA"]

table(eurostat$value_type)
eurostat <- eurostat[!is.na(value)]
eurostat <- eurostat[value != 0]

fwrite(eurostat, paste0(j, "DATA/Incoming Data/LSAE_income/Eurostat_income_NUTS2/eurostat_income_extraction.csv"))

# 6 countries matched to a subnational level, 7 did not match, and 1 matched to country 
# countries that did not match: BEL; DEU; FIN; FRA; HUN; MLT; NOR 
# BEL - Belgium - NUTS2 has high concordance with admin2 - add in BE (7/11 matched) and then geomatch to check naming **
# DEU - Germany - there are more NUTS2 than admin1
# FIN - Finland - there are more admin1 than NUTS2 
# FRA - France - there are more NUTS2 than admin1 
# HUN - Hungary - there are more admin1 than NUTS2
# MLT - Malta - admin 0 - double check on why this wasnt merging - no data for Malta that is why  
# NOR - Norway - there are more admin1 than NUTS2

# Run some checks to validate data processing
validate_extractions(eurostat)
# For each country, check number of locations against what is expected from the LSAE location set
validate_geomatched_locations(eurostat, 2010)

# source summary table 
eurostat$year <- as.integer(eurostat$year)
dt <- create_source_summary_table(eurostat)



# Save compiled EUROSTAT extraction with GDP and income

rm(list = ls())

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

eurostat_gdp <- data.table(fread(paste0(j, "DATA/Incoming Data/LSAE_income/EUROSTAT_GDP_NUTS3/eurostat_GDP_extraction.csv")))
eurostat_income <- data.table(fread(paste0(j, "DATA/Incoming Data/LSAE_income/Eurostat_income_NUTS2/eurostat_income_extraction.csv")))

eurostat <- rbind(eurostat_gdp, eurostat_income)
eurostat <- eurostat[shapefile == "", shapefile := NA]
eurostat <- eurostat[source_location_type == "", source_location_type := NA]
eurostat <- eurostat[location_type == "", location_type := NA]
eurostat <- eurostat[geomatching_notes == "", geomatching_notes := NA]

validate_extractions(eurostat)
save_extraction(eurostat, compiled = F)
