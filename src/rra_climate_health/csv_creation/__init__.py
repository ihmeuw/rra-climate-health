from rra_climate_health.csv_creation.concat_final_csvs import concat_final_csvs
from rra_climate_health.csv_creation.run_csv_creation import run_csv_creation

RUNNERS = {
    "csv_creation": run_csv_creation,
    "concat_final_csvs": concat_final_csvs,
}
