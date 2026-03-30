""" """

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

spec_versions = [
    "neonatal_custom_factor_small.yaml",
    "neonatal_custom_linear_small.yaml",
    "neonatal_k6_factor_small.yaml",
    "neonatal_k6_linear_small.yaml",
]


measure = "neonatal_mortality"
output_root = Path(DEFAULT_ROOT)
queue = "all.q"
model_versions = []
node_args = dict()
node_args["measure"] = [measure]
node_args["model-version"] = []

measure_root = Path(output_root) / measure
cm_data = ClimateMalnutritionData(measure_root)


for v in spec_versions:

    model_specification_path = specification_filepath / v
    print("Using model specification:", model_specification_path)
    model_spec = ModelSpecification.from_yaml(model_specification_path)

    model_version = cm_data.new_model_version()
    model_spec.version.model = model_version
    version_root = cm_data.models / model_version

    model_versions.append(model_version)
    cm_data.save_model_specification(model_spec, model_version)

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
        "runtime": "72h",
        "project": "proj_rapidresponse",
    },
    max_attempts=1,
    log_root=str(version_root),
)
