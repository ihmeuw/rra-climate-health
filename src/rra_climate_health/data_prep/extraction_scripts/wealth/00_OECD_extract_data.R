#' @Title: [00_process_OECD_data.R]  
#' @Authors: Bianca Zlavog
#' @contact: zlavogb@uw.edu
#' @Date_code_last_updated:  10/24/2022
#' @Date_data_last_updated: 10/24/2022
#' @Purpose: Process OECD wealth data into extraction template format
#' 
#' @File_Input(s)
#'  J:/DATA/Incoming Data/LSAE_income/OECD/OECD_regional_GDP_income - May 2022.csv
#'  This file contains annual GDPpc and Disposable household income pc in nominal LCU from 1995-2020 by regions
#' @File_Output(s)
#' /ihme/resource_tracking/LSAE_income/1_data_extractions/extracted_OECD.csv, and a datestamped version
#' This file contains processed and geomatched data in extraction template format

##### Setup

rm(list = ls())
require(pacman)
p_load(data.table, dplyr, readxl, haven, tidyverse, foreign)

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

##### Extract OECD data
# Read incoming data and map regional data to parent country ISO3
  oecd_path <- paste0(j, "DATA/Incoming Data/LSAE_income/OECD/OECD_regional_GDP_income - May 2022.csv")
  oecd_iso3_mapping <- data.table(read_excel(paste0(j, "DATA/Incoming Data/LSAE_income/OECD/OECD Territorial grid and Regional typologies - October 2022.xlsx"), sheet = "List of regions - 2022", skip = 2))
  oecd <- data.table(read_csv(oecd_path))
  oecd <- merge(oecd, oecd_iso3_mapping[, .(country_ISO3 = ISO3, REG_ID)], by = "REG_ID", all.x = T)
  oecd <- oecd[`Territory Level and Typology` %in% c("Country", "Large regions (TL2)", "Small regions (TL3)")]

# Fix TL region naming for merge with GBD admin regions
  oecd[REG_ID == "AU8" & Region == "Canberra region (ACT)", Region := "Australian Capital Territory"]
  oecd[REG_ID == "AT21" & Region == "Carinthia", Region := "Kärnten"]
  oecd[REG_ID == "AT12" & Region == "Lower Austria", Region := "Niederösterreich"]
  oecd[REG_ID == "AT22" & Region == "Styria", Region := "Steiermark"]
  oecd[REG_ID == "AT33" & Region == "Tyrol", Region := "Tirol"]
  oecd[REG_ID == "AT31" & Region == "Upper Austria", Region := "Oberösterreich"]
  oecd[REG_ID == "AT13" & Region == "Vienna", Region := "Wien"]
  oecd[REG_ID == "BE1" & Region == "Brussels Capital Region", Region := "Bruxelles"]
  oecd[REG_ID == "BE2" & Region == "Flemish Region", Region := "Vlaanderen"]
  oecd[REG_ID == "BE3" & Region == "Walloon Region", Region := "Wallonie"]
  oecd[REG_ID == "BG312" & Region == "Montana (BG)", Region := "Montana"]
  oecd[REG_ID == "BG411" & Region == "Sofia (Stolitsa)", Region := "Grad Sofiya"]
  oecd[REG_ID == "BG344" & Region == "Starazagora", Region := "Stara Zagora"]
  oecd[REG_ID == "BR24" & Region == "Distrito Federal (BR)", Region := "Distrito Federal"]
  oecd[REG_ID == "BR27" & Region == "Mato Grosso Do Sul", Region := "Mato Grosso do Sul"]
  oecd[REG_ID == "BR14" & Region == "Piauí", Region := "Piaui"]
  oecd[REG_ID == "BR19" & Region == "Rio De Janeiro", Region := "Rio de Janeiro"]
  oecd[REG_ID == "BR15" & Region == "Rio Grande Do Norte", Region := "Rio Grande do Norte"]
  oecd[REG_ID == "BR22" & Region == "Rio Grande Do Sul", Region := "Rio Grande do Sul"]
  oecd[REG_ID == "CH022" & Region == "Fribourg (CH)", Region := "Fribourg"]
  oecd[REG_ID == "CH025" & Region == "Jura (CH)", Region := "Jura"]
  oecd[REG_ID == "CH055" & Region == "St. Gallen", Region := "Sankt Gallen"]
  oecd[REG_ID == "CHN" & Region == "China (People's Republic of)", Region := "China"]
  oecd[REG_ID == "CL02" & Region == "Antofagasta", Region := "De Antofagasta"]
  oecd[REG_ID == "CL09" & Region == "Araucanía", Region := "De La Araucanía"]
  oecd[REG_ID == "CL15" & Region == "Arica y Parinacota", Region := "De Arica y Parinacota"]
  oecd[REG_ID == "CL03" & Region == "Atacama", Region := "De Atacama"]
  oecd[REG_ID == "CL11" & Region == "Aysén", Region := "De Aisén del Gral. C. Ibáñez del Campo"]
  oecd[REG_ID == "CL04" & Region == "Coquimbo", Region := "De Coquimbo"]
  oecd[REG_ID == "CL10" & Region == "Los Lagos", Region := "De Los Lagos"]
  oecd[REG_ID == "CL14" & Region == "Los Ríos", Region := "De Los Ríos"]
  oecd[REG_ID == "CL12" & Region == "Magallanes and Chilean Antarctica", Region := "De Magallanes y de La Antártica Chilena"]
  oecd[REG_ID == "CL07" & Region == "Maule", Region := "Del Maule"]
  oecd[REG_ID == "CL06" & Region == "O'Higgins", Region := "Del Libertador B. O'Higgins"]
  oecd[REG_ID == "CL13" & Region == "Santiago Metropolitan Region", Region := "Metropolitana de Santiago"]
  oecd[REG_ID == "CL01" & Region == "Tarapacá", Region := "De Tarapacá"]
  oecd[REG_ID == "CL05" & Region == "Valparaíso", Region := "De Valparaíso"]
  oecd[REG_ID == "CO11" & Region == "Bogotá Capital District", Region := "Bogotá, D.C."]
  oecd[REG_ID == "CO23" & Region == "Córdoba (CO)", Region := "Córdoba"]
  oecd[REG_ID == "CO54" & Region == "Norte de Santander", Region := "Norte De Santander"]
  oecd[REG_ID == "CO63" & Region == "Quindío", Region := "Quindio"]
  oecd[REG_ID == "CO76" & Region == "Valle del Cauca", Region := "Valle Del Cauca"]
  oecd[REG_ID == "CO88" & Region == "San Andrés", Region := "Archipiélago De San Andrés Y Providencia"]
  oecd[REG_ID == "CZE" & Region == "Czech Republic", Region := "Czechia"]
  oecd[REG_ID == "CZ020" & Region == "Central Bohemia", Region := "Středočeský"]
  oecd[REG_ID == "CZ052" & Region == "Hradec Králové", Region := "Královéhradecký"]
  oecd[REG_ID == "CZ041" & Region == "Karlovy Vary", Region := "Karlovarský"]
  oecd[REG_ID == "CZ051" & Region == "Liberec", Region := "Liberecký"]
  oecd[REG_ID == "CZ080" & Region == "Moravia-Silesia", Region := "Moravskoslezský"]
  oecd[REG_ID == "CZ071" & Region == "Olomouc", Region := "Olomoucký"]
  oecd[REG_ID == "CZ053" & Region == "Pardubice", Region := "Pardubický"]
  oecd[REG_ID == "CZ032" & Region == "Plzen", Region := "Plzeňský"]
  oecd[REG_ID == "CZ031" & Region == "South Bohemia", Region := "Jihočeský"]
  oecd[REG_ID == "CZ064" & Region == "South Moravia", Region := "Jihomoravský"]
  oecd[REG_ID == "CZ063" & Region == "Vysocina", Region := "Kraj Vysočina"]
  oecd[REG_ID == "CZ072" & Region == "Zlín", Region := "Zlínský"]
  oecd[REG_ID == "CZ042" & Region == "Ústí nad Labem", Region := "Ústecký"]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, ", Kreisfreie Stadt", "")]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, " Kreisfreie Stadt", "")]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, ",Kreisfreie Stadt", "")]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, ", Landkreis", "")]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, " Landkreis", "")]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, ", Stadtkreis", "")]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, " Stadtkreis", "")]
  oecd[country_ISO3 == "DEU", Region := str_replace(Region, " (DE)", "")]
  oecd[REG_ID == "DEE08" & Region == "Burgenland (DE)", Region := "Burgenlandkreis"]
  oecd[REG_ID == "DE277" & Region == "Dillingen a.d. Donau", Region := "Dillingen an der Donau"]
  oecd[REG_ID == "DE94A" & Region == "Friesland (DE)", Region := "Friesland"]
  oecd[REG_ID == "DE80K" & Region == "Landkreis Rostock", Region := "Rostock (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE21G" & Region == "Mühldorf a. Inn", Region := "Mühldorf am Inn"]
  oecd[REG_ID == "DE236" & Region == "Neumarkt i. d. OPf.", Region := "Neumarkt in der Oberpfalz"]
  oecd[REG_ID == "DE25A" & Region == "Neustadt a. d. Aisch-Bad Windsheim", Region := "Neustadt an der Aisch-Bad Windsheim"]
  oecd[REG_ID == "DE237" & Region == "Neustadt a. d. Waldnaab", Region := "Neustadt an der Waldnaab"]
  oecd[REG_ID == "DE943" & Region == "Oldenburg (Oldenburg)", Region := "Oldenburg (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE21J" & Region == "Pfaffenhofen a. d. Ilm", Region := "Pfaffenhofen an der Ilm"]
  oecd[REG_ID == "DE233" & Region == "Weiden i. d. Opf", Region := "Weiden in der Oberpfalz"]
  oecd[REG_ID == "DE24D" & Region == "Wunsiedel i. Fichtelgebirge", Region := "Wunsiedel im Fichtelgebirge"]
  oecd[REG_ID == "DE2" & Region == "Bavaria", Region := "Bayern"]
  oecd[REG_ID == "DE7" & Region == "Hesse", Region := "Hessen"]
  oecd[REG_ID == "DE9" & Region == "Lower Saxony", Region := "Niedersachsen"]
  oecd[REG_ID == "DEA" & Region == "North Rhine-Westphalia", Region := "Nordrhein-Westfalen"]
  oecd[REG_ID == "DEB" & Region == "Rhineland-Palatinate", Region := "Rheinland-Pfalz"]
  oecd[REG_ID == "DED" & Region == "Saxony", Region := "Sachsen"]
  oecd[REG_ID == "DEE" & Region == "Saxony-Anhalt", Region := "Sachsen-Anhalt"]
  oecd[REG_ID == "DEG" & Region == "Thuringia", Region := "Thüringen"]
  # Fix a few DEU locations where multiple OECD regions have identical names by comparing to shapefiles
  oecd[REG_ID == "DE117" & Region == "Heilbronn", Region := "Heilbronn (Stadtkreis)"]
  oecd[REG_ID == "DE122" & Region == "Karlsruhe", Region := "Karlsruhe (Stadtkreis)"]
  oecd[REG_ID == "DE251" & Region == "Ansbach", Region := "Ansbach (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE261" & Region == "Aschaffenburg", Region := "Aschaffenburg (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE271" & Region == "Augsburg", Region := "Augsburg (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE241" & Region == "Bamberg", Region := "Bamberg (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE242" & Region == "Bayreuth", Region := "Bayreuth (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE243" & Region == "Coburg", Region := "Coburg (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE253" & Region == "Fürth", Region := "F�rth (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE244" & Region == "Hof", Region := "Hof (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE221" & Region == "Landshut", Region := "Landshut (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE212" & Region == "München", Region := "M�nchen (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE222" & Region == "Passau", Region := "Passau (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE232" & Region == "Regensburg", Region := "Regensburg (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE213" & Region == "Rosenheim", Region := "Rosenheim (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE262" & Region == "Schweinfurt", Region := "Schweinfurt (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE263" & Region == "Würzburg", Region := "W�rzburg (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE731" & Region == "Kassel", Region := "Kassel (Kreisfreie Stadt)"]
  oecd[REG_ID == "DE944" & Region == "Osnabrück", Region := "Osnabr�ck (Kreisfreie Stadt)"]
  oecd[REG_ID == "DEB32" & Region == "Kaiserslautern", Region := "Kaiserslautern (Kreisfreie Stadt)"]
  oecd[REG_ID == "DED51" & Region == "Leipzig", Region := "Leipzig (Kreisfreie Stadt)"]
  oecd[REG_ID == "DK04" & Region == "Central Jutland", Region := "Midtjylland"]
  oecd[REG_ID == "DK01" & Region == "Copenhagen region", Region := "Hovedstaden"]
  oecd[REG_ID == "DK05" & Region == "Northern Jutland", Region := "Nordjylland"]
  oecd[REG_ID == "DK03" & Region == "Southern Denmark", Region := "Syddanmark"]
  oecd[REG_ID == "DK02" & Region == "Zealand", Region := "Sjælland"]
  oecd[REG_ID == "ES213" & Region == "Biscay", Region := "Vizcaya"]
  oecd[REG_ID == "ES630" & Region == "Ceuta (ES)", Region := "Ceuta"]
  oecd[REG_ID == "ES613" & Region == "Córdoba (ES)", Region := "Córdoba"]
  oecd[REG_ID == "ES512" & Region == "Gerona", Region := "Girona"]
  oecd[REG_ID == "ES212" & Region == "Gipuzkoa", Region := "Guipúzcoa"]
  oecd[REG_ID == "ES111" & Region == "La Corunna", Region := "A Coruña"]
  oecd[REG_ID == "ES230" & Region == "La Rioja (ES)", Region := "La Rioja"]
  oecd[REG_ID == "ES513" & Region == "Lérida", Region := "Lleida"]
  oecd[REG_ID == "ES640" & Region == "Melilla (ES)", Region := "Melilla"]
  oecd[REG_ID == "ES220" & Region == "Navarre", Region := "Navarra"]
  oecd[REG_ID == "ES113" & Region == "Orense", Region := "Ourense"]
  oecd[REG_ID == "ES243" & Region == "Saragossa", Region := "Zaragoza"]
  oecd[REG_ID == "ES618" & Region == "Seville", Region := "Sevilla"]
  oecd[REG_ID == "ES211" & Region == "Araba/Álava", Region := "Álava"]
  oecd[REG_ID == "ES61" & Region == "Andalusia", Region := "Andalucía"]
  oecd[REG_ID == "ES24" & Region == "Aragon", Region := "Aragón"]
  oecd[REG_ID == "ES12" & Region == "Asturias", Region := "Principado de Asturias"]
  oecd[REG_ID == "ES53" & Region == "Balearic Islands", Region := "Islas Baleares"]
  oecd[REG_ID == "ES21" & Region == "Basque Country", Region := "País Vasco"]
  oecd[REG_ID == "ES62" & Region == "Murcia", Region := "Región de Murcia"]
  oecd[REG_ID == "ES30" & Region == "Madrid", Region := "Comunidad de Madrid"]
  oecd[REG_ID == "ES22" & Region == "Navarra", Region := "Comunidad Foral de Navarra"]
  oecd[REG_ID == "ES52" & Region == "Valencia", Region := "Comunidad Valenciana"]
  oecd[REG_ID == "ES41" & Region == "Castile and León", Region := "Castilla y León"]
  oecd[REG_ID == "ES42" & Region == "Castile-La Mancha", Region := "Castilla-La Mancha"]
  oecd[REG_ID == "ES51" & Region == "Catalonia", Region := "Cataluña"]
  oecd[REG_ID == "FI1B1" & Region == "Helsinki-Uusimaa", Region := "Uusimaa"]
  oecd[REG_ID == "FI1C3" & Region == "Päijät-Häme", Region := "Päijänne Tavastia"]
  oecd[REG_ID == "FI1D9" & Region == "North Ostrobothnia", Region := "Northern Ostrobothnia"]
  oecd[REG_ID == "FI194" & Region == "South Ostrobothnia", Region := "Southern Ostrobothnia"]
  oecd[REG_ID == "FI1D2" & Region == "North Savonia", Region := "Northern Savonia"]
  oecd[REG_ID == "FI1D1" & Region == "South Savonia", Region := "Southern Savonia"]
  oecd[REG_ID == "FI1C2" & Region == "Tavastia Proper", Region := "Kanta-Häme"]
  oecd[REG_ID == "FRH" & Region == "Brittany", Region := "Bretagne"]
  oecd[REG_ID == "FRB" & Region == "Centre - Val de Loire", Region := "Centre-Val de Loire"]
  oecd[REG_ID == "FRM" & Region == "Corsica", Region := "Corse"]
  oecd[REG_ID == "FRD" & Region == "Normandy", Region := "Normandie"]
  oecd[REG_ID == "FRL" & Region == "Provence-Alpes-Côte d’Azur", Region := "Provence-Alpes-Côte d'Azur"]
  oecd[REG_ID == "FRE11" & Region == "Nord (FR)", Region := "Nord"]
  oecd[REG_ID == "UKE" & Region == "Yorkshire and The Humber", Region := "Yorkshire and the Humber"]
  oecd[REG_ID == "EL51" & Region == "Eastern Macedonia, Thrace", Region := "East Macedonia and Thrace"]
  oecd[REG_ID == "EL63" & Region == "Western Greece", Region := "West Greece"]
  oecd[REG_ID == "EL53" & Region == "Western Macedonia", Region := "West Macedonia"]
  oecd[country_ISO3 == "HRV", Region := str_replace(Region, " County", "")]
  oecd[REG_ID == "HR021" & Region == "Bjelovar-Bilogora", Region := "Bjelovarska-Bilogorska"]
  oecd[REG_ID == "HR024" & Region == "Brod-Posavina", Region := "Brodsko-Posavska"]
  oecd[REG_ID == "HR037" & Region == "Dubrovnik-Neretva", Region := "Dubrovacko-Neretvanska"]
  oecd[REG_ID == "HR036" & Region == "Istria", Region := "Istarska"]
  oecd[REG_ID == "HR027" & Region == "Karlovac", Region := "Karlovacka"]
  oecd[REG_ID == "HR063" & Region == "Koprivnica-Krizevci", Region := "Koprivničko-Križevačka"]
  oecd[REG_ID == "HR064" & Region == "Krapina-Zagorje", Region := "Krapinsko-Zagorska"]
  oecd[REG_ID == "HR032" & Region == "Lika-Senj", Region := "Licko-Senjska"]
  oecd[REG_ID == "HR061" & Region == "Medimurje", Region := "Medimurska"]
  oecd[REG_ID == "HR025" & Region == "Osijek-Baranja", Region := "Osjecko-Baranjska"]
  oecd[REG_ID == "HR023" & Region == "Pozega-Slavonia", Region := "Požeško-Slavonska"]
  oecd[REG_ID == "HR031" & Region == "Primorje-Gorski Kotar", Region := "Primorsko-Goranska"]
  oecd[REG_ID == "HR034" & Region == "Sibenik-Knin", Region := "Šibensko-Kninska"]
  oecd[REG_ID == "HR028" & Region == "Sisak-Moslavina", Region := "Sisacko-Moslavacka"]
  oecd[REG_ID == "HR035" & Region == "Split-Dalmatia", Region := "Splitsko-Dalmatinska"]
  oecd[REG_ID == "HR062" & Region == "Varazdin", Region := "Varaždinska"]
  oecd[REG_ID == "HR022" & Region == "Virovitica-Podravina", Region := "Viroviticko-Podravska"]
  oecd[REG_ID == "HR026" & Region == "Vukovar-Srijem", Region := "Vukovarsko-Srijemska"]
  oecd[REG_ID == "HR033" & Region == "Zadar", Region := "Zadarska"]
  oecd[REG_ID == "HR065" & Region == "Zagreb", Region := "Zagrebačka"]
  oecd[REG_ID == "HR050" & Region == "City of Zagreb", Region := "Grad Zagreb"]
  oecd[REG_ID == "HU333" & Region == "Csongrád-Csanád", Region := "Csongrád"]
  oecd[country_ISO3 == "IDN", Region := str_replace(Region, " Province", "")]
  oecd[REG_ID == "ID15" & Region == "Bangka Belitung", Region := "Bangka-Belitung Islands"]
  oecd[REG_ID == "ID03" & Region == "D.I. Yogyakarta", Region := "Yogyakarta"]
  oecd[REG_ID == "ID01" & Region == "DKI Jakarta", Region := "Jakarta"]
  oecd[REG_ID == "ID23" & Region == "Middle Kalimantan", Region := "Central Kalimantan"]
  oecd[REG_ID == "ID30" & Region == "Middle Sulawesi", Region := "Central Sulawesi"]
  oecd[REG_ID == "ID27" & Region == "South East Sulawesi", Region := "Southeast Sulawesi"]
  oecd[REG_ID == "ID11" & Region == "Riau", Region := "Riau Islands"]
  oecd[REG_ID == "ID12" & Region == "Riau Mainland", Region := "Riau"]
  oecd[REG_ID == "ID09" & Region == "North Sumatera", Region := "North Sumatra"]
  oecd[REG_ID == "ID14" & Region == "South Sumatera", Region := "South Sumatra"]
  oecd[REG_ID == "ID13" & Region == "West Sumatera", Region := "West Sumatra"]
  oecd[REG_ID == "ID33" & Region == "Eastern Lesser Sundas", Region := "East Nusa Tenggara"]
  oecd[REG_ID == "ID32" & Region == "Western Lesser Sundas", Region := "West Nusa Tenggara"]
  oecd[REG_ID == "IL040" & Region == "Central District", Region := "HaMerkaz"]
  oecd[REG_ID == "IL030" & Region == "Haifa District", Region := "Haifa"]
  oecd[REG_ID == "IL010" & Region == "Jerusalem District", Region := "Jerusalem"]
  oecd[REG_ID == "IL020" & Region == "Northern District", Region := "HaZafon"]
  oecd[REG_ID == "IL060" & Region == "Southern District", Region := "HaDarom"]
  oecd[REG_ID == "IL050" & Region == "Tel Aviv District", Region := "Tel Aviv"]
  oecd[REG_ID == "IN01" & Region == "Jammu and Kashmir", Region := "Jammu & Kashmir and Ladakh"]
  oecd[REG_ID == "IN03" & Region == "National Capital Territory of Delhi", Region := "Delhi"]
  oecd[REG_ID == "IN27" & Region == "Orissa", Region := "Odisha"]
  oecd[REG_ID == "IN21" & Region == "Uttaranchal", Region := "Uttarakhand"]
  oecd[REG_ID == "ITC20" & Region == "Aosta Valley", Region := "Aosta"]
  oecd[REG_ID == "ITH10" & Region == "Bolzano-Bozen", Region := "Bolzano"]
  oecd[REG_ID == "ITH58" & Region == "Forlì-Cesena", Region := "Forli' - Cesena"]
  oecd[REG_ID == "ITC33" & Region == "Genoa", Region := "Genova"]
  oecd[REG_ID == "ITI11" & Region == "Massa-Carrara", Region := "Massa Carrara"]
  oecd[REG_ID == "ITC4C" & Region == "Milan", Region := "Milano"]
  oecd[REG_ID == "ITC4D" & Region == "Monza e della Brianza", Region := "Monza and Brianza"]
  oecd[REG_ID == "ITF33" & Region == "Naples", Region := "Napoli"]
  oecd[REG_ID == "ITI31" & Region == "Pesaro e Urbino", Region := "Pesaro E Urbino"]
  oecd[REG_ID == "ITF65" & Region == "Reggio di Calabria", Region := "Reggio Di Calabria"]
  oecd[REG_ID == "ITH53" & Region == "Reggio nell'Emilia", Region := "Reggio Nell'Emilia"]
  oecd[REG_ID == "ITI43" & Region == "Rome", Region := "Roma"]
  oecd[REG_ID == "ITG19" & Region == "Siracusa", Region := "Syracuse"]
  oecd[REG_ID == "ITC11" & Region == "Turin", Region := "Torino"]
  oecd[REG_ID == "ITH35" & Region == "Venice", Region := "Venezia"]
  oecd[REG_ID == "ITC2" & Region == "Aosta Valley", Region := "Valle d'Aosta"]
  oecd[REG_ID == "ITF4" & Region == "Apulia", Region := "Puglia"]
  oecd[REG_ID == "ITC4" & Region == "Lombardy", Region := "Lombardia"]
  oecd[REG_ID == "ITC1" & Region == "Piedmont", Region := "Piemonte"]
  oecd[REG_ID == "ITH1" & Region == "Province of Bolzano-Bozen", Region := "Provincia autonoma di Bolzano"]
  oecd[REG_ID == "ITH2" & Region == "Province of Trento", Region := "Provincia autonoma di Trento"]
  oecd[REG_ID == "ITG2" & Region == "Sardinia", Region := "Sardegna"]
  oecd[REG_ID == "ITG1" & Region == "Sicily", Region := "Sicilia"]
  oecd[REG_ID == "ITI1" & Region == "Tuscany", Region := "Toscana"]
  oecd[REG_ID == "IS011" & Region == "Capital Region", Region := "Höfuðborgarsvæði"]
  oecd[REG_ID == "IS026" & Region == "Northeastern Region (IS)", Region := "Hálshreppur"]
  oecd[REG_ID == "IS024" & Region == "Northwestern Region (IS)", Region := "Norðurland vestra"]
  oecd[REG_ID == "IS021" & Region == "Southern Peninsula", Region := "Suðurnes"]
  oecd[REG_ID == "IS025" & Region == "Southern Region (IS)", Region := "Suðurland"]
  oecd[REG_ID == "IS022" & Region == "Western Region (IS)", Region := "Vesturland"]
  oecd[REG_ID == "IS023" & Region == "Westfjords", Region := "Vestfirðir"]
  oecd[REG_ID == "JPC10" & Region == "Gumma", Region := "Gunma"]
  oecd[REG_ID == "JPA01" & Region == "Hokkaido", Region := "Hokkaidō"]
  oecd[REG_ID == "JPG28" & Region == "Hyogo", Region := "Hyōgo"]
  oecd[REG_ID == "JPI39" & Region == "Kochi", Region := "Kōchi"]
  oecd[REG_ID == "JPG26" & Region == "Kyoto", Region := "Kyōto"]
  oecd[REG_ID == "JPJ44" & Region == "Oita", Region := "Ōita"]
  oecd[REG_ID == "JPG27" & Region == "Osaka", Region := "Ōsaka"]
  oecd[REG_ID == "JPD13" & Region == "Tokyo", Region := "Tōkyō"]
  oecd[REG_ID == "KOR" & Region == "Korea", Region := "Republic of Korea"]
  oecd[REG_ID == "KR071" & Region == "Jeju-do", Region := "Jeju"]
  oecd[country_ISO3 == "LTU", Region := str_replace(Region, " county", "")]
  oecd[REG_ID == "LT021" & Region == "Alytus", Region := "Alytaus"]
  oecd[REG_ID == "LT022" & Region == "Kaunas", Region := "Kauno"]
  oecd[REG_ID == "LT023" & Region == "Klaipeda", Region := "Klaipedos"]
  oecd[REG_ID == "LT024" & Region == "Marijampole", Region := "Marijampoles"]
  oecd[REG_ID == "LT027" & Region == "Taurage", Region := "Taurages"]
  oecd[REG_ID == "LT029" & Region == "Utena", Region := "Utenos"]
  oecd[REG_ID == "LT011" & Region == "Vilnius", Region := "Vilniaus"]
  oecd[REG_ID == "LT025" & Region == "Panevežys", Region := "Panevezio"]
  oecd[REG_ID == "ME15" & Region == "Mexico", Region := "México"]
  oecd[REG_ID == "ME16" & Region == "Michoacan", Region := "Michoacán de Ocampo"]
  oecd[REG_ID == "ME19" & Region == "Nuevo Leon", Region := "Nuevo León"]
  oecd[REG_ID == "ME22" & Region == "Queretaro", Region := "Querétaro"]
  oecd[REG_ID == "ME24" & Region == "San Luis Potosi", Region := "San Luis Potosí"]
  oecd[REG_ID == "ME30" & Region == "Veracruz", Region := "Veracruz de Ignacio de la Llave"]
  oecd[REG_ID == "ME31" & Region == "Yucatan", Region := "Yucatán"]
  oecd[REG_ID == "NL41" & Region == "North Brabant", Region := "Noord-Brabant"]
  oecd[REG_ID == "NL32" & Region == "North Holland", Region := "Noord-Holland"]
  oecd[REG_ID == "NL33" & Region == "South Holland", Region := "Zuid-Holland"]
  oecd[REG_ID == "NO074" & Region == "Troms and Finnmark", Region := "Troms og Finnmark"]
  oecd[REG_ID == "NO091" & Region == "Vestfold and Telemark", Region := "Vestfold og Telemark"]
  oecd[country_ISO3 == "NZL", Region := str_replace(Region, " Region", "")]
  oecd[REG_ID == "PE01" & Region == "Amazonas (PE)", Region := "Amazonas"]
  oecd[REG_ID == "PE03" & Region == "Apurímac", Region := "Apurimac"]
  oecd[REG_ID == "PE10" & Region == "Huánuco", Region := "Huanuco"]
  oecd[REG_ID == "PE12" & Region == "Junín", Region := "Junin"]
  oecd[REG_ID == "PE17" & Region == "Madre de dios", Region := "Madre De Dios"]
  oecd[REG_ID == "PE07" & Region == "Prov. const. del Callao", Region := "Callao"]
  oecd[REG_ID == "PE02" & Region == "Áncash", Region := "Ancash"]
  oecd[REG_ID == "PL41" & Region == "Greater Poland", Region := "Wielkopolskie"]
  oecd[REG_ID == "PL61" & Region == "Kuyavian-Pomerania", Region := "Kujawsko-Pomorskie"]
  oecd[REG_ID == "PL21" & Region == "Lesser Poland", Region := "Małopolskie"]
  oecd[REG_ID == "PL71" & Region == "Lodzkie", Region := "Łódzkie"]
  oecd[REG_ID == "PL51" & Region == "Lower Silesia", Region := "Dolnośląskie"]
  oecd[REG_ID == "PL81" & Region == "Lublin Province", Region := "Lubelskie"]
  oecd[REG_ID == "PL43" & Region == "Lubusz", Region := "Lubuskie"]
  oecd[REG_ID == "PL52" & Region == "Opole region", Region := "Opolskie"]
  oecd[REG_ID == "PL82" & Region == "Podkarpacia", Region := "Podkarpackie"]
  oecd[REG_ID == "PL63" & Region == "Pomerania", Region := "Pomorskie"]
  oecd[REG_ID == "PL22" & Region == "Silesia", Region := "Śląskie"]
  oecd[REG_ID == "PL72" & Region == "Swietokrzyskie", Region := "Świętokrzyskie"]
  oecd[REG_ID == "PL62" & Region == "Warmian-Masuria", Region := "Warmińsko-Mazurskie"]
  oecd[REG_ID == "PL42" & Region == "West Pomerania", Region := "Zachodniopomorskie"]
  oecd[REG_ID == "RO311" & Region == "Arges", Region := "Argeș"]
  oecd[REG_ID == "RO211" & Region == "Bacau", Region := "Bacău"]
  oecd[REG_ID == "RO112" & Region == "Bistrita-Nasaud", Region := "Bistrița-Năsăud"]
  oecd[REG_ID == "RO212" & Region == "Botosani", Region := "Botoșani"]
  oecd[REG_ID == "RO221" & Region == "Braila", Region := "Brăila"]
  oecd[REG_ID == "RO122" & Region == "Brasov", Region := "Brașov"]
  oecd[REG_ID == "RO321" & Region == "Bucuresti", Region := "Bucharest"]
  oecd[REG_ID == "RO222" & Region == "Buzau", Region := "Buzău"]
  oecd[REG_ID == "RO312" & Region == "Calarasi", Region := "Călărași"]
  oecd[REG_ID == "RO422" & Region == "Caras-Severin", Region := "Caraș-Severin"]
  oecd[REG_ID == "RO223" & Region == "Constanta", Region := "Constanța"]
  oecd[REG_ID == "RO313" & Region == "Dambovita", Region := "Dâmbovița"]
  oecd[REG_ID == "RO224" & Region == "Galati", Region := "Galați"]
  oecd[REG_ID == "RO315" & Region == "Ialomita", Region := "Ialomița"]
  oecd[REG_ID == "RO213" & Region == "Iasi", Region := "Iași"]
  oecd[REG_ID == "RO114" & Region == "Maramures", Region := "Maramureș"]
  oecd[REG_ID == "RO413" & Region == "Mehedinti", Region := "Mehedinți"]
  oecd[REG_ID == "RO125" & Region == "Mures", Region := "Mureș"]
  oecd[REG_ID == "RO214" & Region == "Neamt", Region := "Neamț"]
  oecd[REG_ID == "RO116" & Region == "Salaj", Region := "Sălaj"]
  oecd[REG_ID == "RO115" & Region == "Satumare", Region := "Satu Mare"]
  oecd[REG_ID == "RO424" & Region == "Timis", Region := "Timiș"]
  oecd[REG_ID == "RO415" & Region == "Valcea", Region := "Vâlcea"]
  oecd[country_ISO3 == "RUS", Region := str_replace(Region, "Oblast", "oblast")]
  oecd[country_ISO3 == "RUS", Region := str_replace(Region, "Krai", "kray")]
  oecd[REG_ID == "RUS" & Region == "Russia", Region := "Russian Federation"]
  oecd[REG_ID == "RU21" & Region == "Arkhangelsk oblast", Region := "Arkhangelsk oblast without Nenets autonomous district"]
  oecd[REG_ID == "RU64" & Region == "Buryat Republic", Region := "Republic of Buryatia"]
  oecd[REG_ID == "RU83" & Region == "Chukotka Autonomous Okrug", Region := "Chukotka Autonomous Area"]
  oecd[REG_ID == "RU18" & Region == "City of Moscow", Region := "Moscow City"]
  oecd[REG_ID == "RU29" & Region == "Federal City of Saint Petersburg", Region := "Saint Petersburg"]
  oecd[REG_ID == "RU82" & Region == "Jewish Autonomous oblast", Region := "Jewish autonomous oblast"]
  oecd[REG_ID == "RU39" & Region == "Karachay-Cherkhess Republic", Region := "Karachay-Cherkess Republic"]
  oecd[REG_ID == "RU60" & Region == "Khanty-Mansi Autonomous Okrug - Yugra", Region := "Khanty-Mansi autonomous area"]
  oecd[REG_ID == "RU20" & Region == "Komi republic", Region := "Komi Republic"]
  oecd[REG_ID == "RU09" & Region == "Lipetsk oblast", Region := "Lipetzk oblast"]
  oecd[REG_ID == "RU44" & Region == "Mari El Republic", Region := "Republic of Mari El"]
  oecd[REG_ID == "RU22" & Region == "Nenets Autonomous Okrug", Region := "Nenets autonomous district"]
  oecd[REG_ID == "RU30" & Region == "Republic of Adygea", Region := "Republic of Adygeya"]
  oecd[REG_ID == "RU43" & Region == "Republic of Bashkorstostan", Region := "Republic of Bashkortostan"]
  oecd[REG_ID == "RU75" & Region == "Sakha Republic (Yakutia)", Region := "Republic of Sakha (Yakutia)"]
  oecd[REG_ID == "RU59" & Region == "Tyumen oblast", Region := "Tyumen oblast without autonomous areas"]
  oecd[REG_ID == "RU65" & Region == "Tyva Republic", Region := "Republic of Tuva"]
  oecd[REG_ID == "RU56" & Region == "Ulianov oblast", Region := "Ulyanovsk oblast"]
  oecd[REG_ID == "RU61" & Region == "Yamalo-Nenets Autonomous Okrug", Region := "Yamalo-Nenets autonomous area"]
  oecd[REG_ID == "RU68" & Region == "Zabaykalsky kray", Region := "Zabaikalsk kray"]
  oecd[REG_ID == "SK032" & Region == "Banská Bystrica Region", Region := "Banskobystrický"]
  oecd[REG_ID == "SK010" & Region == "Bratislava Region", Region := "Bratislavský"]
  oecd[REG_ID == "SK042" & Region == "Košice Region", Region := "Košický"]
  oecd[REG_ID == "SK023" & Region == "Nitra Region", Region := "Nitriansky"]
  oecd[REG_ID == "SK041" & Region == "Prešov Region", Region := "Prešovský"]
  oecd[REG_ID == "SK022" & Region == "Trencín Region", Region := "Trenčiansky"]
  oecd[REG_ID == "SK021" & Region == "Trnava Region", Region := "Trnavský"]
  oecd[REG_ID == "SK031" & Region == "Žilina Region", Region := "Žilinský"]
  oecd[REG_ID == "SI031" & Region == "Mura", Region := "Pomurska"]
  oecd[REG_ID == "SI032" & Region == "Drava", Region := "Podravska"]
  oecd[REG_ID == "SI033" & Region == "Carinthia", Region := "Koroška"]
  oecd[REG_ID == "SI034" & Region == "Savinja", Region := "Savinjska"]
  oecd[REG_ID == "SI035" & Region == "Central Sava", Region := "Zasavska"]
  oecd[REG_ID == "SI036" & Region == "Lower Sava", Region := "Spodnjeposavska"]
  oecd[REG_ID == "SI037" & Region == "Southeast Slovenia", Region := "Jugovzhodna Slovenija"]
  oecd[REG_ID == "SI038" & Region == "Littoral–Inner Carniola", Region := "Notranjsko-kraška"]
  oecd[REG_ID == "SI041" & Region == "Central Slovenia", Region := "Osrednjeslovenska"]
  oecd[REG_ID == "SI042" & Region == "Upper Carniola", Region := "Gorenjska"]
  oecd[REG_ID == "SI043" & Region == "Goriska", Region := "Goriška"]
  oecd[REG_ID == "SI044" & Region == "Coastal-Karst", Region := "Obalno-kraška"]
  oecd[country_ISO3 == "SWE", Region := str_replace(Region, " County", "")]
  oecd[REG_ID == "SE332" & Region == "Norrbottens", Region := "Norrbotten"]
  oecd[REG_ID == "SE124" & Region == "Örebro", Region := "Orebro"]
  oecd[REG_ID == "SE311" & Region == "Värmlands", Region := "Värmland"]
  oecd[REG_ID == "SE331" & Region == "Västerbottens", Region := "Västerbotten"]
  oecd[REG_ID == "SVK" & Region == "Slovak Republic", Region := "Slovakia"]
  oecd[REG_ID == "TUR" & Region == "Türkiye", Region := "Turkey"]
  oecd[REG_ID == "TR813" & Region == "Bartin", Region := "Bartın"]
  oecd[REG_ID == "TRB12" & Region == "Elazig", Region := "Elazığ"]
  oecd[REG_ID == "TRA23" & Region == "Igdir", Region := "Iğdır"]
  oecd[REG_ID == "TR632" & Region == "Kahramanmaras", Region := "K. Maras"]
  oecd[REG_ID == "TR711" & Region == "Kirikkale", Region := "Kinkkale"]
  oecd[REG_ID == "TR811" & Region == "Zonguldak", Region := "Zinguldak"]
  oecd[REG_ID == "USA" & Region == "United States", Region := "United States of America"]
  oecd[REG_ID == "ZA08" & Region == "North West", Region := "North-West"]

  # For some nonsovereign regions, match to GBD admin0 and keep most granular level of data at TL3 level since identical to TL2 level
  # FIN Åland in OECD data but is a GBD admin0 location
  oecd <- oecd[REG_ID == "FI200" & Region == "Åland", Region := "Aland Islands"]
  oecd <- oecd[REG_ID == "FI200" & Region == "Aland Islands", country_ISO3 := "ALA"]
  oecd <- oecd[!(REG_ID == "FI20")]
  # ESP Canary Islands in OECD which is a GBD admin0 location
  oecd <- oecd[REG_ID == "ES70" & Region == "Canary Islands", country_ISO3 := "XCA"]
  # FRA contains islands at the GBD admin0 level
  oecd <- oecd[REG_ID == "FRY30" & Region == "French Guiana", country_ISO3 := "GUF"]
  oecd <- oecd[REG_ID == "FRY10" & Region == "Guadeloupe", country_ISO3 := "GLP"]
  oecd <- oecd[REG_ID == "FRY40" & Region == "La Réunion", Region := "Reunion"]
  oecd <- oecd[REG_ID == "FRY40" & Region == "Reunion", country_ISO3 := "REU"]
  oecd <- oecd[REG_ID == "FRY20" & Region == "Martinique", country_ISO3 := "MTQ"]
  oecd <- oecd[REG_ID == "FRY50" & Region == "Mayotte", country_ISO3 := "MYT"]
  oecd <- oecd[!(REG_ID %in% c("FRY1", "FRY2", "FRY3", "FRY4", "FRY5"))]

  locs_oecd <- copy(locs)[, c("location_name", "location_id", "location_type", "country_ISO3")]
  locs_oecd <- locs_oecd[country_ISO3 %in% unique(oecd$country_ISO3)]

# Drop unneeded location info where keeping it yields duplicate rows
  locs_oecd <- locs_oecd[!(country_ISO3 %in% c("TUR", "SWE", "SVN", "SVK", "ROU", "POL", "PER", "NZL", "NOR", "NLD", "MEX", "LVA", "MLT", "USA", "LTU", "KOR", "IRL", "IDN", "HUN", "HRV", "COL", "CHN", "CHE", "CAN", "BRA", "BGR", "BEL", "JPN", "LUX", "ISL", "RUS", "AUS", "PRT", "EST", "FIN") & location_type == "admin2")]
  locs_oecd <- locs_oecd[!(country_ISO3 %in% c("GRC", "LUX", "ISR") & location_type == "admin1")]
  
  oecd <- oecd[!(REG_ID == "NO011" & Region == "Oslo")]
  oecd <- oecd[!(REG_ID == "NO043" & Region == "Rogaland")]
  oecd <- oecd[!(REG_ID == "NO053" & Region == "Møre og Romsdal")]

# Merge on location names and parent country from OECD and GBD
  oecd_before_merge <- copy(oecd)
  oecd <- merge(oecd_before_merge, locs_oecd, by.x = c("Region", "country_ISO3"), by.y = c("location_name", "country_ISO3"), all.x = T)

# Manually fix a few duplicate locations at wrong admin level
  oecd <- oecd[!(REG_ID == "AT32" & Region == "Salzburg" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "AT13" & Region == "Wien" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "DE3" & Region == "Berlin" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "DE300" & Region == "Berlin" & location_type == "admin1")]
  oecd <- oecd[!(REG_ID == "DE5" & Region == "Bremen" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "DE501" & Region == "Bremen" & location_type == "admin1")]
  oecd <- oecd[!(REG_ID == "DE6" & Region == "Hamburg" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "DE600" & Region == "Hamburg" & location_type == "admin1")]
  oecd <- oecd[!(REG_ID == "ES130" & Region == "Cantabria" & location_type == "admin1")]
  oecd <- oecd[!(REG_ID == "ES13" & Region == "Cantabria" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "ES23" & Region == "La Rioja" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "ES230" & Region == "La Rioja" & location_type == "admin1")]
  oecd <- oecd[!(REG_ID == "ES63" & Region == "Ceuta" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "ES630" & Region == "Ceuta" & location_type == "admin1")]
  oecd <- oecd[!(REG_ID == "ES64" & Region == "Melilla" & location_type == "admin2")]
  oecd <- oecd[!(REG_ID == "ES640" & Region == "Melilla" & location_type == "admin1")]
  
# Test that no duplicate locations are being added by this merge
  validate_merge_nrows(oecd_before_merge, oecd)

# Create source_location_type column
  oecd[, source_location_type := paste0("TL", TL)]

# Add notes here and to data, and fixes for imperfect location matching
  # "Not regionalised" data exists that summed with the regional data adds up to the country total GDP. Per decision from team, will exclude for now, since raking will re-scale values proportionally.
  oecd <- oecd[grepl(" - not regionalised", Region), notes := "Not regionalized"]
  
  # The sum of multiple GBD regions comprises one OECD region
  # CHL Biobío CL08 = sum(Del Bíobío 59700 + Ñuble 61221)
  oecd[REG_ID == "CL08" & Region == "Biobío (Región)", notes := "Multiple GBD regions sum to source region"]
  # FRA Rhône FRK26 = sum (Rhône 60232 + Métropole de Lyon 60233)
  oecd[REG_ID == "FRK26" & Region == "Rhône", c("location_id", "location_type") := NA]
  oecd[REG_ID == "FRK26" & Region == "Rhône", notes := "Multiple GBD regions sum to source region"]
  # IND Andaman and Nicobar Islands IN35 = sum(admin2 Nicobars 90741 + North & Middle Andaman 93917 + South Andaman 90743)
  oecd[REG_ID == "IN35" & Region == "Andaman and Nicobar Islands", notes := "Multiple GBD regions sum to source region"]
  # NZL Tasman-Nelson-Marlborough NZ021 = sum(Tasman 62237 + Nelson 62232 + Marlborough 62231)
  oecd[REG_ID == "NZ021" & Region == "Tasman-Nelson-Marlborough", notes := "Multiple GBD regions sum to source region"] 
  # PER Lima PE15 = sum(admin1 Lima 54138 + Lima province 54139). 
  oecd[REG_ID == "PE15" & Region == "Lima", c("location_id", "location_type") := NA]
  oecd[REG_ID == "PE15" & Region == "Lima", notes := "Multiple GBD regions sum to source region"]
  
  # The sum of multiple OECD regions comprises one GBD region. Revisit if we want to sum across these or keep more granular, depending on modeling method.
  # ESP Baleares 69488 = sum(Mallorca ES532 + Menorca ES533 + Eivissa y Formentera ES531)
  oecd[(REG_ID == "ES532" & Region == "Mallorca") | (REG_ID == "ES533" & Region == "Menorca") | (REG_ID == "ES531" & Region == "Eivissa y Formentera"), notes := "Multiple source regions sum to GBD region"]
  # ESP GBD regions are Santa Cruz de Tenerife, Las Palmas, Islas Canarias. We have OECD data on individual islands: El Hierro, Fuerteventura, Gran Canaria, La Gomera, La Palma, Lanzarote, Tenerife. Figure our how to map these to use XCA country_ISO3
  oecd[REG_ID %in% c("ES703", "ES704", "ES705", "ES706", "ES707", "ES708", "ES709"), notes := "Multiple source regions sum to GBD region"]
  # ITA	Sud Sardegna 20670 = sum(Carbonia-Iglesias ITG2C + Medio Campidano ITG2B + part of Cagliari ITG27). In future, this sum region may be recoded to ITGH. Note that Cagliari ITG2F is the correct region to be geomatched to Cagliari 94050.
  oecd[REG_ID %in% c("ITG2C", "ITG2B", "ITG27"), c("location_id", "location_type") := NA]
  oecd[REG_ID %in% c("ITG2C", "ITG2B", "ITG27"), notes := "Multiple source regions sum to GBD region"]
  # ITA	Nuoro 94051 = sum(Ogliastra ITG2A + Nuoro ITG26). This correct sum region to use has been recoded to REG_ID = ITG2E
  oecd[REG_ID %in% c("ITG26", "ITG2A"), c("location_id", "location_type") := NA]
  oecd[REG_ID %in% c("ITG26", "ITG2A"), notes := "Multiple source regions sum to GBD region"]
  # ITA	Oristano 94052 is not the same as Oristano ITG28. This correct region to use has been recoded to REG_ID = ITG2G
  oecd <- oecd[!REG_ID %in% c("ITG28")]
  # ITA	Sassari 20669 = sum(Olbia-Tempio ITG29 + Sassari ITG25). In future, this sum region may be recoded to REG_ID = ITG2D
  oecd[REG_ID %in% c("ITG29", "ITG25"), c("location_id", "location_type") := NA]
  oecd[REG_ID %in% c("ITG29", "ITG25"), notes := "Multiple source regions sum to GBD region"]
  # LVA Riga 61934 = sum(Riga LV006 + Pieriga LV007) but keep more granular data unmapped to GBD for now.
  oecd[REG_ID == "LV006" & Region == "Riga", c("location_id", "location_type") := NA]
  oecd[(REG_ID == "LV007" & Region == "Pieriga") | (REG_ID == "LV006" & Region == "Riga"), notes := "Multiple source regions sum to GBD region"]
  # NOR admin1 Viken 60136 = sum(Akershus NO012 + Buskerud NO032 + Østfold NO031)
  # NOR Agder 60133 = sum(Aust-Agder NO041 + Vest-Agder NO042) 
  # NOR Innlandet 60135 = sum(Oppland NO022 + Hedmark NO021)
  # NOR Troms og Finnmark 60137 = sum(Finnmark NO073 + Troms NO072)
  # NOR Vestfold og Telemark 60134 = sum(Telemark NO034 + Vestfold NO033)
  # NOR Vestland 60132 = sum(Hordaland NO051 + Sogn og Fjordane NO052)
  oecd[(REG_ID %in% c("NO012", "NO032", "NO031", "NO041", "NO042", "NO022", "NO021", "NO073", "NO072", "NO034", "NO033", "NO051", "NO052")), notes := "Multiple source regions sum to GBD region"]
  # POL POL Mazowieckie 53666 = sum(Mazowiecki region PL92 + Warsaw PL91)
  oecd[(REG_ID == "PL92" & Region == "Mazowiecki region") | (REG_ID == "PL91" & Region == "Warsaw"), notes := "Multiple source regions sum to GBD region"]
  
  # The most granular level of OECD regions does not map nicely to GBD regions. Revisit if using an MBG approach.
  # AUT TL3 less granular than admin2
  # BEL TL3 more granular than admin2
  # DNK TL3 more granular than admin1 but less granular than admin2
  # EST TL3 less granular than admin1
  # GBR TL3 more granular than admin2
  # GRC TL3 more granular than admin2
  # IRL TL3 less granular than admin1
  # MLT TL3 has only 2 regions, less granular than admin1. Also both regions have same name which needs to be fixed for any geomatching. In shapefile "Gozo And Comino/Ghawdex U Kemmuna" is the name of MT002, which aligns with this report: https://nso.gov.mt/en/nso/Media/Salient-Points-of-Publications/Documents/2021/Regional%202021/Regional%20Statistics%202021_full%20publication.pdf
  # NLD TL3 more granular than admin1 but less granular than admin2
  # POL TL3 more granular than admin1 but less granular than admin2
  # PRT TL3 doesn't align well with GBD
  # USA TL3 contains cities, with locations in shapefile, not sure if we can use
  oecd[country_ISO3 %in% c("AUT", "BEL", "DNK", "EST", "GBR", "GRC", "IRL", "MLT", "NLD", "POL", "PRT", "USA") & `Territory Level and Typology` == "Small regions (TL3)", notes := "Most granular location not matching GBD"]
  oecd[country_ISO3 %in% c("AUT", "BEL", "DNK", "EST", "GBR", "GRC", "IRL", "MLT", "NLD", "POL", "PRT", "USA") & `Territory Level and Typology` == "Small regions (TL3)", c("location_id", "location_type") := NA]
  # Data not at the most granular level is available and does not map nicely to GBD regions. Often more granular data is available and has higher concordance. Check with Joe if we still want to keep.
  oecd[country_ISO3 %in% c("BGR", "CZE", "CHE", "EST", "FIN", "HRV", "HUN", "IRL", "ISL", "JPN", "KOR", "LTU", "LVA", "MLT", "NOR", "NZL", "PRT", "TUR", "SWE", "SVN", "SVK", "ROU") & `Territory Level and Typology` == "Large regions (TL2)", notes := "Less granular location not matching GBD"]
  oecd[country_ISO3 %in% c("BGR", "CZE", "CHE", "EST", "FIN", "HRV", "HUN", "IRL", "ISL", "JPN", "KOR", "LTU", "LVA", "MLT", "NOR", "NZL", "PRT", "TUR", "SWE", "SVN", "SVK", "ROU") & `Territory Level and Typology` == "Large regions (TL2)", c("location_id", "location_type") := NA]
  
  # Noting some GBD regions missing from OECD regions, don't need to do anything with these.
  # AUS 61019, 61021, 61022 aren't included in OECD TL2 data
  # CHN HKG and MAC missing
  # GRC Athos missing
  # IDN missing North Kalimantan
  # ISR missing Golan region
  # KOR missing Sejong
  # NZL Chatham Islands, Northern Islands, Southern Islands, all in admin1 but missing from OECD. Hawke's Bay, West Coast are present in shapefiles but not in OECD data.
  # Noting locations with data at multiple admin levels
  # GBR is at admin1 for Northern Ireland, Scotland, and Wales, and admin2 for England subregions
  # IND Puducherry and Chandigarh are at admin2 level, rest of IND data is admin1
  # Data for DEU, ESP, FRA, ITA is matched at both admin1 and admin2 levels
  
  # Noting some miscellaneous issues
  # Drop aggregate EU region
  oecd <- oecd[!(REG_ID == "EU27_2020" & Region == "EU-27")]
  oecd <- oecd[!(REG_ID == "EU28" & Region == "EU-28")]
  # Drop ISR TL2 which is identical to TL3
  oecd <- oecd[!(country_ISO3 == "ISR" & `Territory Level and Typology` == "Large regions (TL2)")]
  # LUX TL1, TL2, TL3 are all identical at admin0 level
  oecd <- oecd[!(country_ISO3 == "LUX" & `Territory Level and Typology` != "Country")]
  # Drop LVA TL2 which is identical to TL1
  oecd <- oecd[!(country_ISO3 == "LVA" & `Territory Level and Typology` == "Large regions (TL2)")]
  # Drop EST TL2 which is identical to TL1
  oecd <- oecd[!(country_ISO3 == "EST" & `Territory Level and Typology` == "Large regions (TL2)")]
  
# Merge on geolocation info from shapefiles
  oecd_before_merge <- copy(oecd)
  oecd <- merge_shapefile_locations(oecd)
  validate_merge_nrows(oecd_before_merge, oecd)
  
# Drop zero values (this gets rid of KOR Sejong and some CAN provinces in years when these regions did not exist)
  oecd <- oecd[Value != 0]
  
# Format data to extraction template
  oecd <- oecd[, .(nid = 498411, source = "OECD", data_type = "admin_tabulated", file_path = oecd_path, year = Year, iso3 = country_ISO3, location_id, location_type,
                   location_name = Region, source_location_id = REG_ID, source_location_type, lat = NA, long = NA, location_code, shapefile, measure = VAR, denominator = "per capita", 
                   multiplier = 1, value = Value, value_type = "", currency = "LCU", base_year = Year, currency_detail = NA, notes = Flags, geomatching_notes = notes, initials = "BZ")]
  oecd[measure == "INCOME_DISP", measure := "income"]
  oecd[notes != "", value_type := "modeled"]
  oecd[value_type == "", value_type := "observed"]
  oecd <- oecd[geomatching_notes == "", geomatching_notes := NA]
  setorder(oecd, "measure", "iso3", "location_id", "year")

# Run some checks to validate data processing
  validate_extractions(oecd)
  # For each country, check number of locations against what is expected from the LSAE location set
  validate_geomatched_locations(oecd, 2010)
  
# Save out data
  save_extraction(oecd)
  
# Create source summary table
  source_summary_table <- create_source_summary_table(oecd)
  