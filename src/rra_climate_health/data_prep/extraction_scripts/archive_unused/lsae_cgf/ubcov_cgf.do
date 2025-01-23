/***********************************************************************************************************
 Author: Patrick Liu (pyliu@uw.edu) edited by Manny Garcia                 
 Date: 7/13/2015
 Project: ubCov
 Purpose: Run Script
                     
***********************************************************************************************************/
//do "H:/repos/cgf/data_prep/ubcov_cgf.do"

//////////////////////////////////
// Setup
//////////////////////////////////

if c(os) == "Unix" {
 local l "/ihme/limited_use/"
 local j "/home/j/"
 set odbcmgr unixodbc
}
else if c(os) == "Windows" {
 local l "L:"
 local j "J:"
}

clear all
set more off
set obs 1

// Settings
local central_root "`j'/WORK/01_covariates/common/ubcov_central"
local topics anthropometrics //child_growth_failure //demographics_child bmi

//topics for birth weight imputation extracts
//local topics child_growth_failure bmi vaccination maternal diarrhea lri tobacco wash contraception stroke anemia metabolics maternal_mortality injuries msk_pain_lowback demographics_child diet_nutrients digestives alcohol diabetes sbp_chl diet_other occupational sexual_violence ckd sth_covs resp_asthma lead_exposure inj_road_traffic inj_falls cbh sbh resp_copd birthhistories hfs tb hiv abuse angina ihd msk_pain_neck pad msk_osteoarthritis msk_rheumarthritis womans_ed abuse_csa neuro_tensache neuro_migraine msk_gout test hiv_biomarker bra_cov iq imp_vision rsp tb_treatment male_circumcision hiv_anc breastfeeding sti_symptoms education pooled_cohorts hiv_knowledge fin_risk_protection


// Load functions
cd "`central_root'"
do "`central_root'/modules/extract/core/load.do"


// Initialize the system
/* 
 Brings in the databases, after which you can run
 extraction or sourcing functions like: new_topic_rows

 You can view each of the loaded databases by running: get, *db* (eg. get, codebook)
*/
ubcov_path
init, topics(`topics')
// Run extraction
/* Launches extract

 Arguments:
  - ubcov_id: The id of the codebook row
 Optional:
  - keep: Keeps 
  - bypass: Skips the extraction check
  - run_all: Loops through all ubcov_ids in the codebook.
*/
local outpath "`l'/LIMITED_USE/LU_GEOSPATIAL/ubCov_extractions/cgf/all/cgf/COUNTRY_SPECIFIC"
//local outpath "`l'/LIMITED_USE/LU_GEOSPATIAL/ubCov_extractions/birth_weight"
local array 15276

foreach number in `array'{
    local i `number'
    run_extract `i' //keep //bypass //run_all
    tostring year_start, gen(year)
    tostring year_end, gen(end_year)
 tostring nid, gen(nid_1)
 if survey_name == "YEM/NUTRITION_MORTALITY_SURVEY" {
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/HAJJAH_2012/YEM_HAJJAH_NUTRITION_MORTALITY_SURVEY_2012_HIGHLAND_CHILD_Y2016M03D10.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_244471_HIGHLAND_CH_YEM_2012_2012"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/HAJJAH_2012/YEM_HAJJAH_NUTRITION_MORTALITY_SURVEY_2012_LOWLAND_CHILD_Y2016M03D10.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_244471_LOWLAND_CH_YEM_2012_2012"
		}		
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/LAHJ_2012/YEM_LAHJ_NUTRITION_MORTALITY_SURVEY_2012_LOWLAND_CHILD_Y2016M03D16.XLSX" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246254_LOWLAND_CH_YEM_2012_2012"
		}			
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/LAHJ_2012/YEM_LAHJ_NUTRITION_MORTALITY_SURVEY_2012_MOUNTAIN_CHILD_Y2016M03D16.XLSX" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246254_MOUNTAIN_CH_YEM_2012_2012"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/TAIZ_2012/YEM_TAIZ_NUTRITION_MORTALITY_SURVEY_2012_PLAIN_COASTAL_CHILD_ONLY_Y2016M03D10.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_244473_PLAIN_COASTAL_CH_YEM_2012_2012"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/TAIZ_2012/YEM_TAIZ_NUTRITION_MORTALITY_SURVEY_2012_MOUNTAINS_CHILD_ONLY_Y2016M03D10.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_244473_LOWLAND_CH_YEM_2012_2012"
		}	
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/DHAMAR_2013/YEM_DHAMAR_NUTRITION_MORTALITY_SURVEY_2013_EASTERN_DISTRICT_CHILD_Y2016M03D16.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246209_EASTERN_DISTRICT_CH_YEM_2013_2013"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/DHAMAR_2013/YEM_DHAMAR_NUTRITION_MORTALITY_SURVEY_2013_WESTERN_DISTRICT_CHILD_Y2016M03D16.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246209_WESTERN_DISTRICT_CH_YEM_2013_2013"
		}	
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/MAHWEET_2013/YEM_MAHWEET_NUTRITION_MORTALITY_SURVEY_2013_HIGHLAND_CHILD_Y2016M03D16.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246250_HIGHLAND_CH_YEM_2013_2013"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/MAHWEET_2013/YEM_MAHWEET_NUTRITION_MORTALITY_SURVEY_2013_LOWLAND_CHILD_Y2016M03D16.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246250_LOWLAND_CH_YEM_2013_2013"
		}	
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/HAJJAH_2014/YEM_HAJJAH_NUTRITION_MORTALITY_SURVEY_2014_CH_ONLY_HIGHLAND_Y2016M03D16.XLSX" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246246_HIGHLAND_CH_YEM_2014_2014"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/HAJJAH_2014/YEM_HAJJAH_NUTRITION_MORTALITY_SURVEY_2014_CH_LOWLAND_ENTRY_Y2016M03D16.XLSX" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246246_LOWLAND_CH_YEM_2014_2014"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/HODEIDAH_2014/YEM_HODEIDAH_NUTRITION_MORTALITY_SURVEY_2014_CH_HIGHLAND_Y2016M03D16.xlsx" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246248_HIGHLAND_CH_YEM_2014_2014"
		}
    if "$file_path" == "J:/DATA/YEM/NUTRITION_MORTALITY_SURVEY/HODEIDAH_2014/YEM_HODEIDAH_NUTRITION_MORTALITY_SURVEY_2014_CH_LOWLAND_Y2016M03D16.XLS" {
		cap local filename = "YEM_NUTRITION_MORTALITY_SURVEY_246248_LOWLAND_CH_YEM_2014_2014"
		}				
 }
 else {
    cap local filename = survey_name + "_" + nid_1 + "_" + survey_module + "_" + ihme_loc_id + "_" + year + "_" + end_year
 }

if "$file_path" == "J:/DATA/WB_LSMS/JAM/1998/JAM_LSMS_1998_ANTHRO.DTA" {
	duplicates drop metab_weight metab_height age_month
}


if "$file_path" == "J:/DATA/WB_LSMS/JAM/2000/JAM_LSMS_2000_REC009.DTA" {
	drop if geospatial_id == .
}

if "$file_path" == "J:/DATA/WB_LSMS_ISA/ETH/2011_2012/ETH_LSMS_ISA_2011_2012_HH_SEC1_Y2013M05D31.DTA" {		
	replace int_year = 2012
	replace int_month = int_month-4
	replace int_month = 0 if int_month = 12	
}

if "$file_path" == "J:/DATA/WB_LSMS_ISA/ETH/2013_2014/ETH_LSMS_ISA_2013_2014_HH_SECT3_Y2015M09D15.DTA" {
	replace int_year = 2014 	
	replace int_month = 2013 if int_month <6	
	replace int_month = int_month+8	
	replace int_month = mod(int_month,12)
	replace int_month = 0 if int_month = 12
}

if "$file_path" == "J:/DATA/WB_LSMS_ISA/ETH/2015_2016/ETH_LSMS_ISA_2015_2016_HH_SECT1_Y2017M03D17.DTA" {
	replace int_year = 2016 	
	replace int_month = int_month-4
	replace int_month = 0 if int_month = 12	
}

//These are the only two cases I've foud where the missingness logic can't take remove a number.
if "$file_path" == "J:/DATA/MEX/NATIONAL_NUTRITION_SURVEY_ENN/1999/MEX_ENN_1999_ENSE299V.CSV" {
	replace metab_weight = . if metab_weight >222.22 & metab_weight < 222.23
}

if "$file_path" == "J:/DATA/MEX/SURVEY_HEALTH_AND_NUTRITION_ENSANUT/2005_2006/MEX_ENSANUT_2005_2006_CH_INTERVIEW_EXAM_NEW_CHILD_WEIGHT_NAME_Y2006M07D24.DTA" {
	replace metab_weight = . if metab_weight > 150 
}

 drop nid_1
 local filename = subinstr("`filename'", "/", "_",.)
    cd  `outpath'
    saveold `filename', replace
}

