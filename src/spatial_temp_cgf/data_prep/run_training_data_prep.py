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


def run_training_data_prep_main(
    measure: str,
):
    survey_data = pd.read_csv(SURVEY_DATA_PATHS[measure])
    print(f"Running training data prep for {measure}...")
    print(f"Survey data path: {SURVEY_DATA_PATHS[measure]}")
    print("Done!")


@click.command()
def run_training_data_prep():
    """Run training data prep."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    run_training_data_prep_main()
