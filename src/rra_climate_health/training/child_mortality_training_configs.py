"""
Overview of iterations tested for child_mortality models:
Climate variables:
- Absolute thresholds: days_over_{x}C_monthly_cumul for x in 24-32
- Relative thresholds: q{x}_monthly_cumul for x in 75,80,85,90,95

Other iterations:
- With and without total_precipitation_monthly_cumul


Total number of models run = 28

"""

import itertools
from pathlib import Path
from typing import Any
import click
import numpy as np

import pandas as pd
import rasterra as rt
from pymer4.models.Lmer import Lmer
from rra_tools import jobmon

from rra_climate_health import cli_options as clio
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from rra_climate_health.model_specification import (
    ModelSpecification,
)
from rra_climate_health.transforms import transform_column
from rra_climate_health import utils
import copy

specification_filepath = Path(
    "/ihme/homes/elyeb/repos/rra-climate-health/specifications"
)
base_spec = "child_mortality.yaml"

measure = "child_mortality"
output_root = Path(DEFAULT_ROOT)
queue = "all.q"
model_versions = []
node_args = dict()
node_args["measure"] = [measure]
node_args["model-version"] = []

model_specification_path = specification_filepath / base_spec
print("Using model specification:", model_specification_path)
base_model_spec = ModelSpecification.from_yaml(model_specification_path)

measure_root = Path(output_root) / measure
cm_data = ClimateMalnutritionData(measure_root)


# Make yaml-building function
def build_model_spec(
    # k: int,
    # s: str,
    climate_var: str,
    precip_var: str,
    model_specification_path: Path,
    base_model_spec: ModelSpecification,
) -> ModelSpecification:

    model_spec = ModelSpecification.from_yaml(model_specification_path)

    other_predictors = [
        p
        for p in base_model_spec.predictors
        if p.name
        not in [
            # "consumption_pd_cumul",
            "days_over_30C_monthly_cumul",
            "total_precipitation_monthly_cumul",
        ]
    ]
    model_spec.predictors = other_predictors

    # Add climate var
    new_pred = copy.deepcopy(
        next(
            p
            for p in base_model_spec.predictors
            if p.name == "days_over_30C_monthly_cumul"
        )
    )
    new_pred.name = climate_var
    # new_pred.spline.k = k
    # new_pred.spline.knot_strategy = s_climate
    model_spec.predictors.append(new_pred)

    # Add precip if applicable
    if precip_var:
        new_pred = copy.deepcopy(
            next(
                p
                for p in base_model_spec.predictors
                if p.name == "total_precipitation_monthly_cumul"
            )
        )
        new_pred.name = precip_var
        model_spec.predictors.append(new_pred)

    return model_spec


for absolute in [False]:  # True,
    for precip in [True, False]:

        # determine primary climate variable
        climate_var = ""
        precip_var = None
        if absolute:
            for threshold in range(24, 33):

                climate_var = f"days_over_{threshold}C_monthly_cumul"
                if precip:
                    precip_var = "total_precipitation_monthly_cumul"

                # Create yaml at this level of loop
                model_spec = build_model_spec(
                    climate_var=climate_var,
                    precip_var=precip_var,
                    model_specification_path=model_specification_path,
                    base_model_spec=base_model_spec,
                )
                model_spec.measure = measure
                model_version = cm_data.new_model_version()
                version_root = cm_data.models / model_version

                model_spec.version.model = model_version
                cm_data.save_model_specification(model_spec, model_version)
                model_versions.append(model_version)
                print(
                    "Running model training for model version",
                    model_version,
                )

        else:
            for threshold in [75, 80, 85, 90, 95]:
                climate_var = f"q{threshold}_monthly_cumul"
                if precip:
                    precip_var = "total_precipitation_monthly_cumul"

                # Create yaml at this level of loop
                model_spec = build_model_spec(
                    climate_var=climate_var,
                    precip_var=precip_var,
                    model_specification_path=model_specification_path,
                    base_model_spec=base_model_spec,
                )
                model_spec.measure = measure
                model_version = cm_data.new_model_version()
                version_root = cm_data.models / model_version
                model_spec.version.model = model_version
                cm_data.save_model_specification(model_spec, model_version)
                model_versions.append(model_version)
                print(
                    "Running model training for model version",
                    model_version,
                )

model_versions = list(set(model_versions))
node_args["model-version"] = model_versions

jobmon.run_parallel(
    runner="sttask",
    task_name="training",
    node_args=node_args,
    task_args={
        "output-root": output_root,
    },
    task_resources={
        "queue": queue,
        "cores": 1,
        "memory": "600Gb",
        "runtime": "180h",
        "project": "proj_rapidresponse",
    },
    max_attempts=1,
    log_root=str(version_root),
)

print("Model training complete. Results can be found at", version_root)
