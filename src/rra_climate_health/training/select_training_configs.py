"""
Overview of iterations tested for neonatal mortality models:
Climate variables:
- Absolute thresholds: days_over_xC_prev_0_mo and days_over_xC_prev_3_mo_avg for x in 28, 30, 32
- Relative thresholds: q{x}_prev_0_mo and q{x}_prev_3_mo_avg for x in 8, 85, 9, 95, 99
Time horizons:
- t=0 (birth month) and t=3 (3-month average)
Other iterations:
- With and without total_precipitation_prev_0_mo or total_precipitation_prev_3_mo_avg
- Knots k in 6, 9, 12
- Knot strategies: "quantiles" and "equal"

Total number of models run = 192

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
base_spec = "neonatal_sensitivity.yaml"

measure = "neonatal_mortality"
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
    k: int,
    s: str,
    climate_var: str,
    precip_var: str,
    model_specification_path: Path,
    base_model_spec: ModelSpecification,
) -> ModelSpecification:

    # custom knots only applies to consumption:
    if s == "custom_knots":
        s_climate = "quantiles"
    else:
        s_climate = s
    model_spec = ModelSpecification.from_yaml(model_specification_path)

    other_predictors = [
        p
        for p in base_model_spec.predictors
        if p.name
        not in [
            "consumption_pd",
            "days_over_30C_prev_0_mo",
            "total_precipitation_prev_0_mo",
        ]
    ]
    model_spec.predictors = other_predictors

    # Add climate var
    new_pred = copy.deepcopy(
        next(
            p for p in base_model_spec.predictors if p.name == "days_over_30C_prev_0_mo"
        )
    )
    new_pred.name = climate_var
    new_pred.spline.k = k
    new_pred.spline.knot_strategy = s_climate
    model_spec.predictors.append(new_pred)

    # Add precip if applicable
    if precip_var:
        new_pred = copy.deepcopy(
            next(
                p
                for p in base_model_spec.predictors
                if p.name == "total_precipitation_prev_0_mo"
            )
        )
        new_pred.name = precip_var
        model_spec.predictors.append(new_pred)

    # Add consumption_pd
    new_pred = copy.deepcopy(
        next(p for p in base_model_spec.predictors if p.name == "consumption_pd")
    )
    new_pred.spline.k = k
    new_pred.spline.knot_strategy = s
    if s == "custom_knots":
        new_pred.spline.knots = [2, 5, 10, 20, 40]
        new_pred.spline.k = 9
    model_spec.predictors.append(new_pred)

    return model_spec


for k in [9]:  # , 6, 12
    # number of knots
    for strategy in ["quantiles"]:  # , "custom_knots"
        # knot_stragey
        for t in [0]:  # , 3
            precip = True
            absolute = True

            # birth month or prev month avgs time horizon

            # determine primary climate variable
            climate_var = ""
            precip_var = None
            if absolute:
                for threshold in [30]:  # 28,
                    if t == 0:
                        climate_var = f"days_over_{threshold}C_prev_{t}_mo"
                        if precip:
                            precip_var = f"total_precipitation_prev_{t}_mo"

                    else:
                        climate_var = f"days_over_{threshold}C_prev_{t}_mo_avg"
                        if precip:
                            precip_var = f"total_precipitation_prev_{t}_mo_avg"

                    # Create yaml at this level of loop
                    model_spec = build_model_spec(
                        k=k,
                        s=strategy,
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
                for threshold in [95, 9, 99]:  # 75,
                    if t == 0:
                        climate_var = f"q{threshold}_prev_{t}_mo"
                        if precip:
                            precip_var = f"total_precipitation_prev_{t}_mo"

                    else:
                        climate_var = f"q{threshold}_prev_{t}_mo_avg"
                        if precip:
                            precip_var = f"total_precipitation_prev_{t}_mo_avg"

                    # Create yaml at this level of loop
                    model_spec = build_model_spec(
                        k=k,
                        s=strategy,
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
        "memory": "200Gb",
        "runtime": "48h",
        "project": "proj_rapidresponse",
    },
    max_attempts=1,
    log_root=str(version_root),
)

print("Model training complete. Results can be found at", version_root)
