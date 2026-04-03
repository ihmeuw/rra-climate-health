import pandas as pd
import os
from pathlib import Path
import yaml
import re
from rra_climate_health.data import DEFAULT_ROOT

output_root = Path(DEFAULT_ROOT)
RESULTS_ROOT = output_root / "neonatal_mortality" / "models"  # "spec_v2" /
validation_results = pd.read_csv(Path(RESULTS_ROOT) / "validation_results.csv")

var_to_predictor = {
    "consumption_pd": "Daily consumption per capita",
    "sex_id": "Sex",
    "birth_year": "Birth year",
    "ihme_loc_id": "Country",
}


def climate_var_to_predictor(var_name: str) -> str:
    # Regex patterns for different formats
    percentile_pattern = r"q(?P<percentile>\d+)_prev_(?P<time>\d+)_mo(?P<avg>_avg)?"
    temperature_pattern = (
        r"days_over_(?P<temp>\d+)C_prev_(?P<time>\d+)_mo(?P<avg>_avg)?"
    )
    precipitation_pattern = r"total_precipitation_prev_(?P<time>\d+)_mo(?P<avg>_avg)?"

    match = re.match(percentile_pattern, var_name)
    if match:
        percentile = match.group("percentile")
        time = match.group("time")
        avg = match.group("avg")
        if avg:
            return f"Average days above the {percentile}th percentile during {time} months before birth month"
        else:
            return (
                f"Days above the {percentile}th percentile during birth month"
                if time == "0"
                else f"Days above the {percentile}th percentile during {time} months before birth month"
            )

    match = re.match(temperature_pattern, var_name)
    if match:
        temp = match.group("temp")
        time = match.group("time")
        avg = match.group("avg")
        if avg:
            return f"Average days over {temp}C during {time} months before birth month"
        else:
            return (
                f"Days above {temp}C during birth month"
                if time == "0"
                else f"Days above {temp}C during {time} months before birth month"
            )

    match = re.match(precipitation_pattern, var_name)
    if match:
        time = match.group("time")
        avg = match.group("avg")
        if avg:
            return (
                f"Average total precipitation during {time} months before birth month"
            )
        else:
            return (
                f"Total precipitation during birth month"
                if time == "0"
                else f"Total precipitation during {time} months before birth month"
            )

    # Default fallback if no pattern matches
    return var_name


def build_formula_from_spec(spec: dict) -> str:
    consumption_terms = []  # only 1
    climate_terms = []  # 1-2
    plain_terms = []  # 3

    for pred in spec.get("predictors", []):
        name = pred["name"]

        # ignore intercept
        if name != "intercept":
            if name in var_to_predictor:
                predictor = var_to_predictor[name]
                if name == "consumption_pd":
                    consumption_terms.append(predictor)
                else:
                    plain_terms.append(predictor)
            else:
                # use regex to parse out climate variables
                predictor = climate_var_to_predictor(name)
                climate_terms.append(predictor)

    ordered = sorted(climate_terms) + consumption_terms + sorted(plain_terms)
    ordered = ", ".join(ordered)
    return ordered


specs = []
ldi_knots = []
climate_knots = []
for mv in validation_results["model_version"]:
    spec_path = RESULTS_ROOT / mv / "specification.yaml"
    if spec_path.exists():
        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        specs.append(build_formula_from_spec(spec))

        for pred in spec.get("predictors", []):
            if pred["name"] == "consumption_pd":
                ldi_formatted = ", ".join(
                    [str(int(k)) for k in pred["spline"]["knots"]]
                )
                ldi_knots.append(ldi_formatted)
            elif pred["name"].startswith("days_over") or pred["name"].startswith("q"):
                climate_formatted = ", ".join(
                    [str(int(k)) for k in pred["spline"]["knots"]]
                )
                climate_knots.append(climate_formatted)
    else:
        specs.append(None)

validation_results["Covariates included"] = specs
validation_results["Knot placement for LDI"] = ldi_knots
validation_results["Knot placement for Threshold"] = climate_knots

# Other formatting
validation_results["Threshold"] = validation_results["Covariates included"].apply(
    lambda x: x.split(",")[0]
)
validation_results.sort_values("prev_rmse", inplace=True)
validation_results.rename(columns={"prev_rmse": "RMSE"}, inplace=True)
validation_results["Rank"] = validation_results.reset_index().index + 1

validation_results = validation_results[
    [
        "model_version",
        "submodel",
        "timestamp",
        "prev_mae",
        "auc_pr",
        "ece",
        "log_loss",
        "brier_score",
        "gbd_rmse",
        "Rank",
        "Threshold",
        "Knot placement for LDI",
        "Knot placement for Threshold",
        "RMSE",
        "Covariates included",
    ]
]

# save out
validation_results.to_csv(
    Path(RESULTS_ROOT) / "validation_results_with_specs.csv", index=False
)
