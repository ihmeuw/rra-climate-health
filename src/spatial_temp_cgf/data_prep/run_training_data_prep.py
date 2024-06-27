from pathlib import Path

import click


SURVEY_DATA_ROOT = Path(
    '/mnt/team/integrated_analytics/pub/goalkeepers/goalkeepers_2024/data'
)

SURVEY_DATA_PATHS = {
    "bmi": SURVEY_DATA_ROOT / "bmi" / "bmi_data_outliered_wealth_rex.csv",
    "wasting": SURVEY_DATA_ROOT / "wasting_stunting" / "wasting_stunting_outliered_wealth_rex.csv",
    "stunting": SURVEY_DATA_ROOT / "wasting_stunting" / "wasting_stunting_outliered_wealth_rex.csv",
}


############################
# Wasting/Stunting columns #
############################

ihme_meta_columns = [
    'nid',
    'file_path',    
]
survey_meta_columns = [
    'survey_name',
    'survey_module',
    'year_start',
    'year_end',
]
sample_meta_columns = [
    'psu',    
    'psu_id',
    'pweight',
    'pweight_admin_1'
    'urban',
    'strata'    
    'strata_id',
    'hh_id',
    'hhweight',
    'line_id',
    'int_year', 
    'int_month',
    'int_day',
]
location_meta_columns = [
    'ihme_loc_id', 
    'location_name',
    'super_region_name',
    'geospatial_id',
    'admin_1',
    'admin_1_id',
    'admin_1_mapped',
    'admin_1_urban_id',
    'admin_1_urban_mapped',        
    'admin_2',
    'admin_2_id',
    'admin_2_mapped',
    'admin_3',
    'admin_4',
    'latnum',
    'longnum',
]
individual_meta_columns = [
    'individual_id',
    'sex_id',
    'age_year',
    'age_month', 
    'age_day',
    'age_categorical'
]
value_columns = [
    'metab_height',
    'metab_height_unit',
    'metab_weight',
    'metab_weight_unit',
    'bmi',
    'overweight',
    'obese',
    'pregnant',
    'birth_weight',
    'birth_weight_unit',
    'birth_order',
    'mother_weight',
    'mother_height',
    'mother_age_month',
    'maternal_ed_yrs',
    'paternal_ed_yrs',
    'wealth_factor',
    'wealth_index_dhs',
    'suspicious.heights',
    'suspicious.weights',
    'HAZ', 'HAZ_b1', 'HAZ_b2', 'HAZ_b3', 'HAZ_b4',
    'WAZ', 'WAZ_b1', 'WAZ_b2', 'WAZ_b3', 'WAZ_b4',
    'WHZ', 'WHZ_b1', 'WHZ_b2', 'WHZ_b3', 'WHZ_b4'
]


def examine_survey_schema(df, columns):
    print('Records:', len(df))
    print()
    
    template = "{:<20} {:>10} {:>10} {:>10}"
    header = template.format('COLUMN', 'N_UNIQUE', 'N_NULL', "DTYPE")
    
    print(header)
    print('='*len(header))
    for col in columns:
        unique = df[col].nunique()
        nulls = df[col].isnull().sum()
        dtype = str(df[col].dtype)
        print(template.format(col, unique, nulls, dtype))


def run_training_data_prep_main(
    measure: str,
):
    survey_data_path = SURVEY_DATA_PATHS[measure]
    print(f"Running training data prep for {measure}...")
    print(f"Survey data path: {survey_data_path}")

    df = pd.read_csv(survey_data_path)

    # Filter bad rows, subset to columns of interest

    # Column transformations

    # Crosswalk asset score to ldi

    # Merge with climate data

    # Write to output

    print("Done!")


@click.command()
def run_training_data_prep():
    """Run training data prep."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    run_training_data_prep_main()
