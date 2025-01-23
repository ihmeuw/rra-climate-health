#' @Title: [00_preprocess_LSMS.R]  
#' @Authors: Bianca Zlavog, Audrey Serfes
#' @contact: zlavogb@uw.edu, aserfes@uw.edu
#' @Purpose: Preprocess LSMS surveys for UbCov extraction. A DSS note about outputs:
#' A DSS note about outputs: If it's an 'L:/IDENT" file it can be saved with J:/ drive files. If it's "limited_use" use the seperate L:/ folder. 

rm(list = ls())
require(pacman)
p_load(data.table, haven, dplyr,readstata13,janitor)

if (Sys.info()["sysname"] == "Linux") {
  j <- "/home/j/" 
  h <- paste0("/homes/", Sys.getenv("USER"), "/")
  k <- "/ihme/cc_resources/"
  l <- "/ihme/limited_use/"
} else { 
  j <- "J:/"
  h <- "H:/"
  k <- "K:/"
  l <- "L:/"
}

out_dir <- "/ihme/resource_tracking/LSAE_income/0_preprocessing/WB_LSMS/"

# Preprocess surveys by merging on variables saved across different files

  # BGR 1698 1995
  #Adding hhsize, income file, hhweight
  bgr_1995_hhsize <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/1995/BGR_LSMS_1995_CONSTRUCTED_HH_SIZE.DTA")))[, .(hhnumber, hhsizea)]
  bgr_1995_hh <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/1995/BGR_LSMS_1995_HH_INFO.DTA")))[, .(hhnumber, distr, clust)]
  bgr_1995_exp <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/1995/BGR_LSMS_1995_CONSTRUCTED_TOTAL_HH_EXPENDITURE_BY_ITEM.DTA")))[, .(hhnumber, exptot)]
  bgr_1995_inc <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/1995/BGR_LSMS_1995_CONSTRUCTED_HH_LEVEL_INCOME_FILE.DTA")))[, .(hhnumber, distr, loc_type, hhtminc)]
  bgr_1995_all <- merge(bgr_1995_exp, bgr_1995_inc, by = "hhnumber", all = T)
  bgr_1995_all <- merge(bgr_1995_all, bgr_1995_hhsize, by = "hhnumber", all = T)
  bgr_1995_all <- merge(bgr_1995_all, bgr_1995_hh, by = c("hhnumber", "distr"), all = T)
  bgr_1995_all$hhweight <- "1"
  fwrite(bgr_1995_all, paste0(out_dir, "BGR_LSMS_1995_preprocess_.csv"))
  
  # BGR 1775 1997
  #Added hhsize and monthly income to monthly expenditure file
  bgr_97_basic <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/BGR_LSMS_1997.dta'))
  bgr_97_basic <- bgr_97_basic[, c('hhnumber', 'clust', 'region', 'distr')]
  bgr_97_cons <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/BGR/1997/BGR_LSMS_1997_CONSTRUCTED_PER_CAPITA_EXPENDITURES.DTA')))
  bgr_97_cons <- merge(bgr_97_basic, bgr_97_cons, by = "hhnumber", all = T)
  bgr_97_income <- data.table(read_dta(paste0(j, '/DATA/WB_LSMS/BGR/1997/BGR_LSMS_1997_CONSTRUCTED_HH_INCOME.DTA')))
  bgr_97_income <- bgr_97_income[, c('hhnumber', 'hhtminc')]
  bgr_97_all <- merge(bgr_97_cons, bgr_97_income, by = "hhnumber", all = T)
  bgr_97_all$hhweight <- "1"
  fwrite(bgr_97_all, paste0(out_dir, "BGR_LSMS_1997_preprocess.csv"))
  
  
  # BGR 46123 2001
  #Adding hhsize, income file, asset file, hhweight
  bgr_2001_hh <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/2001/BGR_LSMS_2001_HH_ROSTER.DTA")))[, .(hhnumber, region, clust, idcode)][idcode == 1]
  bgr_2001_hhsize <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/2001/BGR_LSMS_2001_CONSTRUCTED_CONSUMPTION_AGGREGATES_ADJUSTED_USING_2000_HBS_HH_SIZE_HH_ROSTER.DTA")))[, .(hhnumber, hhsize_resident)]
  bgr_2001_exp <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/2001/BGR_LSMS_2001_CONSTRUCTED_CONSUMPTION_AGGREGATES_ADJUSTED_USING_2000_HBS_TOTAL_AGGREGATE_HH_MONTHLY_EXPENDITURE_NOMINAL.DTA")))[, .(hhnumber, exptot_n)] 
  bgr_2001_inc <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/2001/BGR_LSMS_2001_CONSTRUCTED_INCOME_AGGREGATES_ADJUSTED_USING_2000_HBS_HH.DTA")))[, .(hhnumber, distr, loc_type, hhtminc)]
  bgr_2001_asset <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/2001/BGR_LSMS_2001_CONSTRUCTED_ASSET_SCORE.DTA")))[, .(hhnumber, assetscore)]
  bgr_2001_all <- merge(bgr_2001_exp, bgr_2001_inc, by = "hhnumber", all = T)
  bgr_2001_all <- merge(bgr_2001_all, bgr_2001_hh, by = "hhnumber", all = T)
  bgr_2001_all <- merge(bgr_2001_all, bgr_2001_hhsize, by = "hhnumber", all = T)
  bgr_2001_all <- merge(bgr_2001_all, bgr_2001_asset, by = "hhnumber", all = T)
  bgr_2001_all$hhweight <- "1"
  fwrite(bgr_2001_all, paste0(out_dir, "BGR_LSMS_2001_preprocess_.csv"))
  
  # BGR 1813 2003
  #Adding hhsize, income file, hhweight, added a "z" to preserve zeros in geospatial_id, removed "z" before post_processing
  bgr_2003_hhsize <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/2003/BGR_LSMS_2003_HH_MEMBERS.DTA")))[, .(loc_type,hhcode = paste0(id1, id2, id3, id4), id1, id2, id3, id4, id5)]
  bgr_2003_hhsize <- bgr_2003_hhsize[, .(hhsize = .N), by = c("hhcode", "id1", "id2", "id3", "id4", "loc_type")]
  bgr_2003_exp <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/BGR/2003/BGR_LSMS_2003_EXPENDITURE_VARIABLES_CALCULATED_BY_NSI.DTA")))[, .(hhcode, texp, tconsexp)]
  bgr_2003_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/BGR/2003/BGR_LSMS_2003_INCOME_VARIABLES_CALCULATED_BY_NSI.DTA")))[, .(hhcode = v1, v2)]
  bgr_2003_all <- merge(bgr_2003_exp, bgr_2003_inc, by = "hhcode", all = T)
  bgr_2003_all <- merge(bgr_2003_all, bgr_2003_hhsize, by = "hhcode", all = T)
  bgr_2003_all$id3<- paste0("Z",  bgr_2003_all$id3)
  bgr_2003_all$hhweight <- "1"
  fwrite(bgr_2003_all, paste0(out_dir, "BGR_LSMS_2003_preprocess_.csv"))
  
  # CIV 21364 1985
  civ_1985_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1985/CIV_LSMS_1985_INCOME_AGG.DTA")))
  civ_1985_exp <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1985/CIV_LSMS_1985_EXPENDITURE_AGG.DTA")))
  civ_1985_wgt <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1985/CIV_LSMS_1985_WEIGHTS.DTA")))
  civ_1985_all <- merge(civ_1985_inc, civ_1985_exp, by = c("clust", "nh", "reg", "hhsize5"), all = T)
  civ_1985_all <- merge(civ_1985_all, civ_1985_wgt, by = c("clust", "nh", "reg"))
  fwrite(civ_1985_all, paste0(out_dir, "CIV_LSMS_1985_preprocess.csv"))
  
  # CIV 45376 1986
  civ_1986_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1986/CIV_LSMS_1986_HHINC_AGG.DTA")))
  civ_1986_exp <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1986/CIV_LSMS_1986_HHEXP_AGG.DTA")))
  civ_1986_wgt <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1986/CIV_LSMS_1986_WEIGHTS.DTA")))
  civ_1986_all <- merge(civ_1986_inc, civ_1986_exp, by = c("clust", "nh", "reg", "hhsize6"), all = T)
  civ_1986_all <- merge(civ_1986_all, civ_1986_wgt, by = c("clust", "nh", "reg"))
  fwrite(civ_1986_all, paste0(out_dir, "CIV_LSMS_1986_preprocess.csv"))
  
  # CIV 45464 1987
  civ_1987_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1987/CIV_LSMS_1987_HHINC.DTA")))
  civ_1987_exp <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1987/CIV_LSMS_1987_HHEXP.DTA")))
  civ_1987_wgt <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1987/CIV_LSMS_1987_WEIGHTS.DTA")))
  civ_1987_all <- merge(civ_1987_inc, civ_1987_exp, by = c("clust", "nh", "reg", "hhsize7"), all = T)
  civ_1987_all <- merge(civ_1987_all, civ_1987_wgt, by = c("clust", "nh", "reg"))
  fwrite(civ_1987_all, paste0(out_dir, "CIV_LSMS_1987_preprocess.csv"))
  
  # CIV 45464 1988
  civ_1988_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1988/CIV_LSMS_1988_HHINC.DTA")))
  civ_1988_exp <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1988/CIV_LSMS_1988_HHEXP.DTA")))
  civ_1988_wgt <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/CIV/1988/CIV_LSMS_1988_WEIGHTS.DTA")))
  civ_1988_all <- merge(civ_1988_inc, civ_1988_exp, by = c("clust", "nh", "reg", "hhsize8"), all = T)
  civ_1988_all <- merge(civ_1988_all, civ_1988_wgt, by = c("clust", "nh", "reg"))
  fwrite(civ_1988_all, paste0(out_dir, "CIV_LSMS_1988_preprocess.csv"))
  
  # TJK 12489 2003
  #Adding income file
  tjk_2003_hh <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/TJK/2003/TJK_LSMS_2003_HH_ROSTER.DTA")))[, .(hhid, oblast, type_reg, tlss_psu)]
  tjk_2003_hh <- unique(tjk_2003_hh, by=c("hhid", "oblast", "type_reg"))
  tjk_2003_exp <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/TJK/2003/TJK_LSMS_2003_EXPENDITURES.DTA")))[, .(hhid, oblast, type_reg, tot_ind, tot_exp, wgt_nat)]
  tjk_2003_inc <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/TJK/2003/TJK_LSMS_2003_INCOME_3.DTA")))[, .(hhid, oblast, type_reg, tot_ind, tot_inc)]
  tjk_2003_all <- merge(tjk_2003_exp, tjk_2003_inc, by = c("hhid", "oblast", "type_reg", "tot_ind"), all = T)
  tjk_2003_all <- merge(tjk_2003_all, tjk_2003_hh, by = c("hhid", "oblast", "type_reg"), all = T)
  tjk_2003_all$hhweight <- "1"
  fwrite(tjk_2003_all, paste0(out_dir, "TJK_LSMS_2003_preprocess_.csv"))

  # NGA 151802 2010-2011 HH post harvest
  #Adding basic information
  nga_2010_hh <- data.table(read_dta(paste0(j, "DATA/NGA/GENERAL_HOUSEHOLD_SURVEY/2010_2011/NGA_GHS_2010_2011_POST_HARVEST_HH_SECTA.DTA")))
  nga_2010_ph <- data.table(read_dta(paste0(j, "DATA/NGA/GENERAL_HOUSEHOLD_SURVEY/2010_2011/NGA_GHS_2010_2011_POST_HARVEST_HH_ANNUAL.DTA")))
  nga_2010_all <- merge(nga_2010_hh, nga_2010_ph, by = 'hhid')
  nga_2010_all <- nga_2010_all[, c("hhid", "sector", "state", "lga", "ea", "zone", "wght", "saq13d","saq13m","saq13y", "hhsize", "cons", "totexp", "percap")]
  fwrite(nga_2010_all, paste0(out_dir, "NGA_LSMS_2010_2011_preprocess.csv"))
  
  # PNG 46563 1996
  #Adding weights, hhsize, and geo-information
  png_1996_hh <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/PNG/1996/PNG_LSMS_1996_HH_WEIGHTS.DTA")))
  png_1996_cons <- fread(paste0(j, "/DATA/WB_LSMS/PNG/1996/PNG_LSMS_1996_ANNUAL_HH_CONSUMPTION_VALUE.CSV"))
  names(png_1996_cons) <- tolower(names(png_1996_cons))
  png_1996_cons <- png_1996_cons[, c('prov', 'cd', 'cu','dwg', 'total')]
  png_1996_lookup <- data.table(read_dta(paste0(j, 'DATA/WB_LSMS/PNG/1996/PNG_LSMS_1996_HH_ROSTER_EDUCATION.DTA')))
  png_1996_lookup <- data.table(aggregate(png_1996_lookup$p00, by=list(png_1996_lookup$province, png_1996_lookup$c, png_1996_lookup$census, png_1996_lookup$dwelling), FUN=tail, n = 1))
  setnames(png_1996_lookup,"Group.1", "prov")
  setnames(png_1996_lookup, "Group.2", "cd")
  setnames(png_1996_lookup, "Group.3", "cu")
  setnames(png_1996_lookup, "Group.4", "dwg")
  setnames(png_1996_lookup, "x", "hhsize")
  png_1996_cons <- merge(png_1996_lookup, png_1996_cons, by = c('prov', 'cd', 'cu', 'dwg'), all = T)
  png_1996_all <- merge(png_1996_hh, png_1996_cons, by = c('prov', 'cd', 'cu', 'dwg'), all = T)
  png_1996_all[, year := 1996]
  # Since Consumption has 71 more columns, I'm replacing NAs in hhweight with 1.
  png_1996_all[is.na(png_1996_all$hhweight), hhweight := 1]
  #Add column with location names matching PNG survey to lbd_standard_admin_1
  png_1996_all <- png_1996_all %>% mutate(location_name =
                                  case_when(prov == 1 ~ "Western",
                                            prov == 2  ~ "Gulf",
                                            prov == 3 ~ "Central",
                                            prov == 4 ~ "National Capital District",
                                            prov == 5 ~ "Miline Bay",
                                            prov == 6 ~ "Oro",
                                            prov == 7 ~ "Southern Highlands",
                                            prov ==  8 ~ "Enga",
                                            prov == 9 ~ "Western Highlands",
                                            prov == 10 ~ "Chimbu",
                                            prov == 11 ~ "Eastern Highlands",
                                            prov == 12 ~ "Morobe",
                                            prov == 13 ~ "Madang",
                                            prov == 14 ~ "East Sepik",
                                            prov == 15 ~ "West Sepik",
                                            prov == 16 ~ "Manus",
                                            prov == 17 ~ "New Ireland",
                                            prov == 18 ~ "East New Britain",
                                            prov == 19 ~ "West New Britain",
                                            prov == 20 ~ "North Solomons"))
  write.csv(png_1996_all, paste0(out_dir, "PNG_LSMS_1996_preprocess.csv"))
  
#GHA 438439 2016-2017
#Added income file
gha_exp <- data.table(read_dta(paste0(l,'/IDENT/PROJECT_FOLDERS/WB_LSMS/GHA/2016_2017/GHA_LSMS_2016_2017_13_GHA_2017_E_Y2020M03D25.DTA')))
gha_inc <- data.table(read_dta(paste0(l,'/IDENT/PROJECT_FOLDERS/WB_LSMS/GHA/2016_2017/GHA_LSMS_2016_2017_17_GHA_2017_INCOME_Y2020M03D25.DTA')))
gha_inc <- gha_inc[, c("hid", "TOT_HH_INC_GRS" )]
gha_16_all <- merge(gha_exp, gha_inc, by = 'hid')
fwrite(gha_16_all, paste0(out_dir, "GHA_LSMS_2016_2017_preprocess.csv"))

#NPL 9212 1995-1996
#Added income file
npl_cons <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/NPL_LSMS_1995_1996.dta'))
npl_cons$WWWHH <- as.numeric(npl_cons$WWWHH)
npl_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/NPL/1995_1996/NPL_LSMS_1995_1996_INCOME.DTA"))) 
npl_inc <- npl_inc[, c("WWWHH", "income" )]
npl_95_all <- merge(npl_cons, npl_inc, by = 'WWWHH')
fwrite(npl_95_all, paste0(out_dir, "NPL_LSMS_1995_1996_preprocess.csv"))
                        
#KGZ 45942 1997
#Added income file
kgz_cons <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/KGZ_LSMS_1997.dta'))
kgz_inc <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/KGZ/1997/KGZ_LSMS_1997_INCOME_AGGREGATE.DTA")))
kgz_inc <- kgz_inc[, c("fprimary", "hhinc_re" )]
kgz_97_all <- merge(kgz_cons, kgz_inc, by = 'fprimary')
fwrite(kgz_97_all, paste0(out_dir, "KGZ_LSMS_1997_preprocess.csv"))


#ZAF 12101 1993
#Added income file
zaf_cons <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/ZAF_LSMS_1993.dta'))
zaf_inc <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/ZAF/1993/ZAF_LSMS_1993_CONSTRUCTED_TOTAL_MONTHLY_INCOME.DTA")))
zaf_inc <- zaf_inc[, c("hhid", "totminc" )]
zaf_93_all <- merge(zaf_cons, zaf_inc, by = 'hhid')
fwrite(zaf_93_all, paste0(out_dir, "ZAF_LSMS_1993_preprocess.csv"))


#TZA 14340 1991-1994
#added hhweight, added hhsize file, added income file, using interview year as base_year
tza_hh <- data.table(read_dta(paste0(j,"/DATA/WB_LSMS/TZA/1991_1994/TZA_KAGERA_LSMS_1991_1994_CONSTRUCTED_HH_KEY.DTA")))
tza_hh <- data.table(tza_hh[, c("hh", "cluster", "hhwt")])
tza_cons <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/TZA_LSMS_1991_1994.dta'))
tza_cons <- merge(tza_hh, tza_cons, by = c("hh", "cluster"))
tza_hhsize <- data.table(read_dta(paste0(j,'DATA/WB_LSMS/TZA/1991_1994/TZA_KAGERA_LSMS_1991_1994_HH_ENUMERATION.DTA')))
tza_hhsize <- tza_hhsize[, c("hh", "cluster", "hhsize" )]
tza_cons <- merge(tza_cons,tza_hhsize, by = c("hh", "cluster"))
tza_inc <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/TZA/1991_1994/TZA_KAGERA_LSMS_1991_1994_CONSTRUCTED_HH_INCOME.DTA")))
tza_inc <- tza_inc[, c("hh", "inchh1", "wave", "cluster" )]
tza_91_all <- merge(tza_cons, tza_inc, by = c("hh", "wave", "cluster"))
tza_91_all[is.na(tza_91_all$hhwt)] <- 1
#where hhweight is blank, entering a weight of 1
tza_91_all <- data.table(tza_91_all)
tza_91_all$base_year <- 1993 # interview year
fwrite(tza_91_all, paste0(out_dir, "TZA_LSMS_1991_1994_preprocess.csv"))

# TZA 476073 2019-2020
tza_2019_cons <- data.table(read_dta(paste0(l,"/IDENT/PROJECT_FOLDERS/WB_LSMS_ISA/TZA/2019_2020_EXTENDED_PANEL/TZA_LSMS_ISA_NPS_2019_2020_CONSUMPTIONSDD_V05_Y2022M08D29.DTA")))
tza_2019_hh <- data.table(read_dta(paste0(l,"/IDENT/PROJECT_FOLDERS/WB_LSMS_ISA/TZA/2019_2020_EXTENDED_PANEL/TZA_WB_LSMS_NPS_2019_2020_EXTENDED_PANEL_HH_SEC_A_V04_Y2022M07D01.DTA")))[, c("sdd_hhid", "sdd_weights")]
tza_2019_all <- merge(tza_2019_cons, tza_2019_hh, by = "sdd_hhid")
fwrite(tza_2019_all, paste0(out_dir, "TZA_LSMS_2019_2020_preprocess.csv"))

#PAK 9919 1991
#Adding geospatial information and hhweight
pak_cons <- data.table(read_dta(paste0(j,'DATA/WB_LSMS/PAK/1991/PAK_LSMS_1991_CONSTRUCTED.DTA')))
pak_basic_info <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/PAK/1991/PAK_LSMS_1991_F07BA.DTA')))
pak_basic_info <- pak_basic_info[, c("hid", "clust", "nh" )]
pak_91_all <- merge(pak_cons, pak_basic_info, by = c('hid'))
pak_91_all$hhweight <- "1"
fwrite(pak_91_all, paste0(out_dir, "PAK_LSMS_1991_preprocess.csv"))

#PER 10877 1994
#Added income and hhsize files to expenditure details
per_cons <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/PER_LSMS_1994.dta'))
per_hsize <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/PER/1994/PER_LSMS_1994_REG02.DTA')))
per_hsize_lookup <- data.table(aggregate(per_hsize$b00, by=list(per_hsize$segmento, per_hsize$vivienda), FUN=tail, n = 1))
setnames(per_hsize_lookup, "Group.1", "segmento")
setnames(per_hsize_lookup, "Group.2", "vivienda")
setnames(per_hsize_lookup, "x", "hhsize")
per_cons <- merge(per_cons, per_hsize_lookup , by = c("segmento", "vivienda"))
per_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/PER/1994/PER_LSMS_1994_INGHOG94.DTA")))
per_inc <- per_inc[, c("segmento", "vivienda", "totinhog")]
per_94_all <- merge(per_cons, per_inc, by = c("segmento", "vivienda"))
fwrite(per_94_all, paste0(out_dir, "PER_LSMS_1994_preprocess.csv"))

#JAM 7245 1992
#Adding income file
jam92_cons <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/JAM_LSMS_1992.dta'))
jam92_inc <- data.table(read_dta(paste0(j, "DATA/WB_LSMS/JAM/1992/JAM_LSMS_1992_ANNUAL.DTA")))
jam92_inc <- jam92_inc[, c("serial", "TOT_EXP")]
jam92_all <- merge(jam92_cons, jam92_inc, by = "serial")
fwrite(jam92_all, paste0(out_dir, "JAM_LSMS_1992_preprocess.csv"))

# TJK 12455 1999
#Adding hhweight since it's missing in raw data.
tjk99_cons <- data.table(read_dta(paste0(j,'DATA/WB_LSMS/TJK/1999/TJK_LSMS_1999_INCOME_EXPENDITURE.DTA')))
tjk99_cons$hhweight <- "1"
fwrite(tjk99_cons,paste0(out_dir, "TJK_LSMS_1999_preprocess.csv"))

#TZA 81005 2008-2011
#We lose 19 houses that don't have total exp. in 2010, but have it in 2008.
tza_main <- data.table(read_dta(paste0(j,'DATA/WB_LSMS_ISA/TZA/2010_2011/TZA_LSMS_ISA_2010_2011_HH_SEC_A_Y2012M12D24.DTA')))
tza_main <- tza_main[, c("y2_hhid", "clusterid")]
tza2010_cons <- data.table(read_dta(paste0(j,'DATA/WB_LSMS_ISA/TZA/2010_2011/TZA_LSMS_ISA_2010_2011_TZY2_HH_CONSUMPTION_Y2012M12D24.DTA')))
tza_2010_main <- merge(tza_main, tza2010_cons, by = 'y2_hhid' )


tza2008_cons <- data.table(read_dta(paste0(j,'DATA/WB_LSMS_ISA/TZA/2010_2011/TZA_LSMS_ISA_2010_2011_TZY1_HH_CONSUMPTION_Y2012M12D24.DTA')))
tzahh_all <- data.table(plyr::rbind.fill(tza2008_cons,tza_2010_main))
tza_hhclust2008 <- tzahh_all[, c("hhid_2008", "clusterid")]
tza_hhclust2008 <- tza_hhclust2008[!is.na(tza_hhclust2008$clusterid)]
setnames(tza_hhclust2008, "hhid_2008", "hhid" )
tza2008_cons <- merge(tza2008_cons,tza_hhclust2008, by = "hhid")
tzahh_all <- data.table(plyr::rbind.fill(tza2008_cons,tza_2010_main))
fwrite(tzahh_all,paste0(out_dir, "TZA_LSMS_2008_2011_preprocess.csv"))

#KGZ 7575 1993
#Adding geo_id
kgz93_cons <- data.table(read_dta(paste0(j,'DATA/WB_LSMS/KGZ/1993/KGZ_LSMS_1993_INCOME_EXPENDITURE.DTA')))
kgz93_hh <- data.table(read_dta(paste0(j,'DATA/WB_LSMS/KGZ/1993/KGZ_LSMS_1993_HOUSEHOLD_CHARACTERISTICS.DTA')))
kgz93_hh <- kgz93_hh[, c("hid", "region", "hhsize")]
kgz93_merge <- merge(kgz93_cons,kgz93_hh, by = "hid")
fwrite(kgz93_merge,paste0(out_dir, "KGZ_LSMS_1993_preprocess.csv"))

#MWI 327852 2016_2017
#Adding geospatial information
mwi16_17_cons <- data.table(read_dta(paste0(l,'IDENT/PROJECT_FOLDERS/WB_LSMS_ISA/MWI/2016_2017/MWI_LSMS_ISA_IHS4_2016_2017_IHS4 CONSUMPTION AGGREGATE_V04_Y2021M12D14.DTA')))
mwi16_17_geo_a <- data.table(read_dta(paste0(l, 'IDENT/PROJECT_FOLDERS/WB_LSMS_ISA/MWI/2016_2017/MWI_LSMS_ISA_IHS4_2016_2017_HH_METADATA_Y2017M12D18.DTA')))
mwi16_17_merge_a <- merge(mwi16_17_cons,mwi16_17_geo_a, by = c("case_id"))
mwi16_17_merge_a <- mwi16_17_merge_a[, c("HHID", "case_id")]
mwi16_17_geo_b <- data.table(read_dta(paste0('/ihme/scratch/projects/hssa/frp/processed/MWI_LSMS_ISA_2016_2017.dta')))
mwi16_17_hh_id <- mwi16_17_geo_b[, c("HHID", "reside")]
mwi16_17_hh_subset <- merge(mwi16_17_merge_a,mwi16_17_hh_id, by = c("HHID"))
mwi16_17_cons_all <- merge(mwi16_17_cons,mwi16_17_hh_subset, by = c("case_id"))
fwrite(mwi16_17_cons_all,paste0(out_dir, "MWI_LSMS_16_17_preprocess.csv"))

#SRB KOSOVO 168020 2000
#Adding geospatial information
kos_cons <- data.table(read_dta(paste0(l,'IDENT/PROJECT_FOLDERS/WB_LSMS/KOSOVO/2000/KOSOVO_LSMS_2000_CONSVARS_Y2014M11D18.DTA')))
kos_hh <- data.table(read_dta(paste0(l, 'IDENT/PROJECT_FOLDERS/WB_LSMS/KOSOVO/2000/KOSOVO_LSMS_2000_ID_Y2014M11D18.DTA')))
kos_hh <- kos_hh[, c("hhid", "psu", "weight", "s0i_q07", "s0i_q08", "s0i_q09", )] # 07- municipality, 08- village, 09- urban/rural
kos_merge <- merge(kos_cons,kos_hh, by = "hhid")
fwrite(kos_merge,paste0(out_dir, "KOSOVO_LSMS_2000_preprocess.csv"))

#NGA 274160 2015 - 2016
#Adding geospatial_id
nga2016_cons <- data.table(read_dta(paste0(l,'/IDENT/PROJECT_FOLDERS/NGA/GENERAL_HOUSEHOLD_SURVEY/2015_2016/NGA_GHS_2015_2016_CONS_AGG_WAVE3_VISIT1_V02_Y2021M12D22.DTA')))
nga2016_sector <- data.table(read_dta(paste0(j,'/DATA/NGA/GENERAL_HOUSEHOLD_SURVEY/2015_2016/NGA_GHS_2015_2016_POST_HARVEST_SECT3_Y2017M12D28.DTA')))
nga2016_sector <- nga2016_sector[, c("hhid", "sector",  "zone", "state", "lga", "ea")]
nga2016_merge <- merge(nga2016_cons,nga2016_sector, by = c("hhid", "zone", "state", "lga", "ea"))
fwrite(nga2016_merge,paste0(out_dir, "NGA_LSMS_2015_2016_preprocess.csv"))
                                    
#ECU 46804 1994
#Updating version, grouping for hhsize
ecu1994_cons <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/ECU/1994/ECU_LSMS_1994_HH.DTA')))
ecu1994_cons <- ecu1994_cons %>% group_by(hhid, hhid1, hhid2) %>%
  filter(id == max(id)) %>%
  arrange(hhid, hhid1,hhid2, id)
ecu1994_cons <- data.table(ecu1994_cons)
fwrite(ecu1994_cons,paste0(out_dir, "ECU_LSMS_1994_preprocess.csv"))

#ECU 46837 1995
#Updating version
ecu1995_cons <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/ECU/1995/ECU_LSMS_1995_HH_EN.DTA')))
fwrite(ecu1995_cons,paste0(out_dir, "ECU_LSMS_1995_preprocess.csv"))

#JAM 45779 1993
#Added weight
jam1993_cons <- data.table(read.dta13(paste0(j,'DATA/WB_LSMS/JAM/1993/JAM_LSMS_1993_ANNUAL.DTA')))
jam1993wgt <- data.table(read.dta13(paste0(j,'DATA/WB_LSMS/JAM/1993/JAM_LSMS_1993_REC001.DTA')))
jam1993wgt <- jam1993wgt[, c("serial", "edwght")]
jam93_merge <- merge(jam1993_cons,jam1993wgt, by = "serial")
fwrite(jam93_merge,paste0(out_dir, "JAM_LSMS_1993_preprocess.csv"))

#ETH 286657 2015-2016
#Added a "Z" to geospatial_id, winnower was removing leading zeros in geo_id. Removed "Z" before post-processing.
eth_15_16_con <- data.table(read.dta13(paste0(j,'/DATA/WB_LSMS_ISA/ETH/2015_2016/ETH_LSMS_ISA_2015_2016_CONSUMPTION_AGGREGATE_Y2017M03D17.DTA')))
eth_15_16_con$household_id2 <- paste0("Z", eth_15_16_con$household_id2)
write_csv(eth_15_16_con,paste0(out_dir, "ETH_LSMS_2015_2016_preprocess.csv"))


#IRQ 34524 2006-2007
#Merged for geospatial information
irq_06_07_hh <- data.table(read_dta(paste0(j, '/DATA/IRQ/HH_SOCIOECONOMIC_SURVEY/2006_2007/IRQ_HH_SOCIOECONOMIC_SURVEY_2006_2007_01_HOUSEHOLD_Y2011M01D12.DTA')))
irq_06_07_hh <- irq_06_07_hh[, c("xhhkey", "weight", "xcluster"), ]
irq_06_07_con <- data.table(read_dta(paste0(j, '/DATA/IRQ/HH_SOCIOECONOMIC_SURVEY/2006_2007/IRQ_HH_SOCIOECONOMIC_SURVEY_2006_2007_CONSTR_8_EXPENDITURE_Y2011M01D12.DTA')))
irq_06_07_merge <- merge(irq_06_07_hh,irq_06_07_con, by = "xhhkey")
write.csv(irq_06_07_merge,paste0(out_dir, "IRQ_LSMS_2006_2007_preprocess.csv"))


#	SRB 11606	2003
#Merged for geospatial information
srb03_con <- data.table(read_dta(paste0(j, '/DATA/WB_LSMS/SRB/2003/SRB_LSMS_2003_INCOMECONSUMPTION_Y2010M05D18.DTA')))
srb03_hh <- data.table(read_dta(paste0(j, '/DATA/WB_LSMS/SRB/2003/SRB_LSMS_2003_ROMA_1_DEMOGRAPHY_INDIV_Y2010M05D18.DTA')))
srb03_hh <- srb03_hh[, c( "mesto", "rbd", "naselje"), ]
srb03_all <- merge(srb03_con, srb03_hh, by = c("mesto", "rbd"))
write.csv(srb03_all,paste0(out_dir, "SRB_LSMS_2003_preprocess.csv"))


#TZA	311265 2014-2016
#This file will include total aggregate and sums of expenditure from previous FRP file
tza_14_16 <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/TZA_LSMS_ISA_2014_2016.dta'))
tza_14_16_tot <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS_ISA/TZA/2014_2016_WAVE_4/TZA_LSMS_ISA_2014_2016_CONSUMPTIONNPS4_Y2017M08D14.DTA')))
tza_14_16_merge <- merge(tza_14_16,tza_14_16_tot, by = "y4_hhid")
write.csv(tza_14_16_merge,paste0(out_dir, "TZA_LSMS_2014_2016_preprocess.csv"))


#TZA	224096	2012-2013
#Merged for geospatial information
tza_12_13 <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/TZA_LSMS_ISA_2012_2013.dta'))
tza_12_13_tot <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS_ISA/TZA/2012_2013/TZA_LSMS_ISA_2012_2013_CONSUMPTION_Y2015M09D15.DTA')))
tza_12_13_merge <- merge(tza_12_13,tza_12_13_tot, by = "y3_hhid")
write.csv(tza_12_13_merge,paste0(out_dir, "TZA_LSMS_2012_2013_preprocess.csv"))

#BIH 44844	2004-2005
#Merged for geospatial information
bih_04_05 <- data.table(read_dta(paste0(l,'/IDENT/PROJECT_FOLDERS/WB_LSMS/BIH/2004_2005_WAVE_4/BIH_LSMS_2004_WAVE4_FARM.DTA')))
bih_04_05_tot <- data.table(read_dta(paste0(l, '/IDENT/PROJECT_FOLDERS/WB_LSMS/BIH/2004_2005_WAVE_4/BIH_LSMS_2004_WAVE4_POVERTY.DTA')))
bih_04_05_merge <- merge(bih_04_05,bih_04_05_tot, by = "hid" )
write.csv(bih_04_05_merge,paste0(out_dir, "BIH_LSMS_2004_2005_preprocess.csv"))

#GHA 4043 1987-1988
gha_87_88_exp <- data.table(read_dta('/home/j/DATA/WB_LSMS/GHA/1987_1988/GHA_LSMS_1987_1988_EXPEND_Y2017M03D21.DTA'))
gha_87_88_hh <- data.table(read_dta('/home/j/DATA/WB_LSMS/GHA/1987_1988/GHA_LSMS_1987_1988_HEAD_Y2017M03D21.DTA'))[, clust := clust + 1000]
gha_87_88_merge <- merge(gha_87_88_exp, gha_87_88_hh, by = c("hid"))
write.csv(gha_87_88_merge, paste0(out_dir, "GHA_LSMS_1987_1988_preprocess.csv"))

#GHA 4075 1988-1989
gha_88_89_exp <- data.table(read_dta('/home/j/DATA/WB_LSMS/GHA/1988_1989/GHA_LSMS_1988_1989_EXPEND_Y2017M03D21.DTA'))
gha_88_89_hh <- data.table(read_dta('/home/j/DATA/WB_LSMS/GHA/1988_1989/GHA_LSMS_1988_1989_HEAD_Y2017M03D21.DTA'))[, clust := NULL]
gha_88_89_merge <- merge(gha_88_89_exp, gha_88_89_hh, by = c("hid", "size"))
write.csv(gha_88_89_merge, paste0(out_dir, "GHA_LSMS_1988_1989_preprocess.csv"))

#GHA 165101 2012-2013
#Merged for income file information
gha_12_13 <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/GHA_LSMS_ISA_2012_2013.dta'))
gha_12_13_inc <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/GHA/2012_2013/GHA_LSMS_2012_2013_AGGREGATE_INCOME_Y2016M03D17.DTA')))
gha_12_13_inc <- gha_12_13_inc [, c( "HID", "TOT_HH_INC_GRS"), ]
gha_12_13_merge <- merge(gha_12_13,gha_12_13_inc, by = "HID")
write.csv(gha_12_13_merge,paste0(out_dir, "GHA_LSMS_2012_2013_preprocess.csv"))

#PAN 10277 1997
#Merged for geospatial information
pan_97 <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/PAN_LSMS_1997.dta'))
pan_97_con <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/PAN/1997/PAN_LSMS_1997_CONSUMPTION_AND_POVERTY.DTA')))
pan_97_con <- pan_97_con[, c("form", "hogar", "provinci", "upm", "cons2pc"), ]
pan_97_all <- merge(pan_97, pan_97_con, by = c("form", "hogar", "provinci", "upm"))
write.csv(pan_97_all,paste0(out_dir, "PAN_LSMS_1997_preprocess.csv"))

#ROU 11117 1994 
#winnower would only let me save as this dta file, not csv
#added hhsize, and hhweight
rou_94_hh <- data.table(read_dta(paste0(j, '/DATA/WB_LSMS/ROU/1994/ROU_LSMS_1994_HH_HH_ROSTER.DTA')))
rou_94_con <- data.table(read_dta(paste0(j, '/DATA/WB_LSMS/ROU/1994/ROU_LSMS_1994_HH_CONSTRUCTED.DTA')))
rou_94_all <- merge(rou_94_hh, rou_94_con, by= c('hshld'))
rou_94_all <- setDT(rou_94_all)[, hhsize := .N, by = hshld]
rou_94_all$hhweight <- "1"
write_dta(rou_94_all,paste0(out_dir, "rou_LSMS_1994_1995.dta"))

#NIC 9310 1993 
#winnower would only let me save as this dta file, not csv
nic_93 <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/NIC_LSMS_1993.dta'))
nic_93_cons <- data.table(read_dta(paste0(j, '/DATA/WB_LSMS/NIC/1993/NIC_LSMS_1993_TOTAL_EXPENDITURES.DTA')))
nic_93_cons <- nic_93_cons[, c('_casenum', "region", "loc", "pc30"), ]
nic_93_all <- merge(nic_93,nic_93_cons, by= c('_casenum', "region", "loc"))
janitor::clean_names(nic_93_all)
haven::write_dta(nic_93_all,paste0(out_dir, "nic_lsms_1993.dta"))
                       
#NIC 9422 2001
#outdated version in winnower
nic_01_cons <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/NIC/2001/NIC_LSMS_2001_POVERTY.DTA')))
write.csv(nic_01_cons,paste0(out_dir, "NIC_LSMS_2001_preprocess.csv"))

#NIC 2014 438798 LSMS
dir <- '/ihme/limited_use/IDENT/PROJECT_FOLDERS/WB_LSMS/NIC/2014'
nic_2014_hh_info <- data.table(read.dta13(file.path(dir, "NIC_LSMS_2014_02_DATOS_DE_LA_VIVIENDA_Y_EL_HOGAR_Y2020M03D30.DTA"), convert.factors=FALSE), stringsAsFactors = FALSE)
nic_2014_pov <- data.table(read.dta13(file.path(dir, "NIC_LSMS_2014_POBREZA_Y2020M03D30.DTA"), convert.factors=FALSE), stringsAsFactors = FALSE)[, c("Peso2", "Peso3") := NULL]
nic_2014_geo <- data.table(read.dta13(file.path(dir, "NIC_LSMS_2014_07_VARIABLES_SIMPLES_DE_LA_SECCION_7_Y2020M03D30.DTA"), convert.factors=FALSE), stringsAsFactors = FALSE)[, c("I00", "DOMINIO4", "I06", "S7P15D")]
nic_2014_total <- merge(nic_2014_hh_info, nic_2014_pov, by = c("I00", "DOMINIO4", "I06"))
nic_2014_total <- merge(nic_2014_total, nic_2014_geo, by = c("I00", "DOMINIO4", "I06"))
save.dta13(nic_2014_total, paste0(out_dir, "/NIC_LSMS_2014_2014.dta"))

#MAR 46255 1991 
#would only let me save as this dta file, not csv, added hhweight
mar_91_hhsize <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/MAR/1991/MAR_LSMS_1991_HH_SIZE.DTA')))
mar_91_con <- data.table(read_dta(paste0(j,'DATA/WB_LSMS/MAR/1991/MAR_LSMS_1991_PER_CAPITA_EXPENDITURE.DTA')))
mar_91_all <- merge(mar_91_hhsize,mar_91_con, by = c("ident"))
mar_91_all$hhweight <- "1"
write_dta(mar_91_all,paste0(out_dir, "mar_lsms_1991.dta")) 

#	ALB 137374 2012	
#Needed constructed hhsize
alb_2012 <- data.table(read_dta('/ihme/scratch/projects/hssa/frp/processed/ALB_LSMS_2012.dta'))
alb_2012 <- alb_2012[ , count := .N, by = .(psu,hh)]
setnames(alb_2012, "count", "hhsize" )
write.csv(alb_2012,paste0(out_dir, "ALB_LSMS_2012_preprocess.csv"))

# ALB 44126 2002
#outdated version in winnower
alb_2002_cons <- data.table(read_dta(paste0(j,'/DATA/WB_LSMS/ALB/2002/ALB_LSMS_2002_POVERTY.DTA')))
write.csv(alb_2002_cons,paste0(out_dir, "ALB_LSMS_2002_preprocess.csv"))

# ALB 44253 2005
# merge income and consumption files together
alb_2005_inc <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/ALB/2005/ALB_LSMS_2005_SUBJECTIVE_POVERTY.DTA")))
alb_2005_cons <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/ALB/2005/ALB_LSMS_2005_POVERTY.DTA")))
alb_2005_all <- merge(alb_2005_inc, alb_2005_cons, by = c("hhid"))
write.csv(alb_2005_all,paste0(out_dir, "ALB_LSMS_2005_preprocess.csv")) 


# ALB 135713 2008
# add consumption onto file processed by FRP team
alb_2008_all <- data.table(read_dta("/ihme/scratch/projects/hssa/frp/processed/ALB_LSMS_2008.dta"))
alb_2008_cons <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS/ALB/2008/ALB_LSMS_2008_POVERTY_Y2022M05F26.DTA")))
alb_2008_all <- merge(alb_2008_all, alb_2008_cons, by = c("hhid", "hh", "psu", "stratum"))
write.csv(alb_2008_all,paste0(out_dir, "ALB_LSMS_2008_preprocess.csv")) 

# UGA 264959 2013
# add region codes needed to create geospatial_id
uga_2013_all <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS_ISA/UGA/2013_2014/UGA_LSMS_ISA_2013_2014_CONSUMPTION_AGGREGATE_Y2016M03D09.DTA")))
uga_2013_reg <- data.table(read_dta(paste0(j, "/DATA/WB_LSMS_ISA/UGA/2013_2014/UGA_LSMS_ISA_2013_2014_CSEC1A_Y2016M03D09.DTA")))[, .(HHID, h1aq4a)]
uga_2013_all <- merge(uga_2013_all, uga_2013_reg, by = "HHID")
write.csv(uga_2013_all,paste0(out_dir, "UGA_LSMS_2013_preprocess.csv")) 

# MLI 508576 2018
# need to merge on 'sousregion' variable which is used in creating geospatial_id
mli_2018_all <- data.table(read_dta(paste0(l, "/IDENT/PROJECT_FOLDERS/WB_LSMS/MLI/HOUSEHOLD_LIVING_STANDARDS_HARMONIZED_SURVEY_EHCVM/2018_2019/MLI_LSMS_EHCVM_W1_W2_2018_2019_WELFARE_V01_Y2022M02D16.DTA")))
mli_2018_reg <- data.table(read_dta(paste0(l, "/IDENT/PROJECT_FOLDERS/WB_LSMS/MLI/HOUSEHOLD_LIVING_STANDARDS_HARMONIZED_SURVEY_EHCVM/2018_2019/MLI_LSMS_EHCVM_W1_W2_2018_2019_INDIVIDU_V01_Y2022M02D16.DTA")))[numind==1][, .(hhid, sousregion)]
mli_2018_all <- merge(mli_2018_all, mli_2018_reg, by = "hhid", all.x = T)
write.csv(mli_2018_all,paste0(out_dir, "MLI_LSMS_2018_preprocess.csv")) 
