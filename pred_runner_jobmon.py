import getpass
import os
import sys
import uuid
import mf
import pandas as pd

from jobmon.client.tool import Tool

user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="spatial_malnutrition_tool")

# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"spatial_malnutrition_forecaster_{wf_uuid}",
)

general_default_compute_resources_dict = {
    "queue": "all.q",
    "project": "proj_rapidresponse",
    "stdout": f"/share/temp/slurmoutput/victorvt/errors",
    "stderr": f"/share/temp/slurmoutput/victorvt/errors",
}

create_model_tt = tool.get_task_template(
    template_name="create_model",
    command_template="{python} {script} {measure} {model_identifier} {sex_id} {age_group_id} {model_spec} {grid_vars}",
    node_args=["measure", "model_identifier", 'sex_id', 'age_group_id', 'model_spec', 'grid_vars'],
    op_args=["python", "script"],
    default_cluster_name="slurm",
    default_compute_resources=general_default_compute_resources_dict,
)

predict_tt = tool.get_task_template(
    template_name="predict",
    command_template="{python} {script} {measure} {model_identifier} {fhs_location_id} {scenario} {year}",
    node_args=["measure", "model_identifier", "fhs_location_id", "scenario", "year"],
    op_args=["python", "script"],
    default_cluster_name="slurm",
    default_compute_resources=general_default_compute_resources_dict,
)

model_identifier = 'tempgrid_o30_precip'
pred_task_name = "o_{model_identifier}_{measure}_{fhs_location_id}_{scenario}_{year}"
#model_spec = '"stunting ~ (1 | ihme_loc_id) + (1 | grid_cell)"'
grid_vars = '"' + ','.join(['ldi_pc_pd', 'over_30']) + '"'

loc_mapping = pd.read_parquet("/mnt/share/scratch/users/victorvt/cgfwealth_spatial/fhs_lsae_location_mapping.parquet")
fhs_locs = loc_mapping.fhs_location_id.unique()

for measure in ['stunting', 'wasting']:
    # for sex_id in [1, 2]:
    #     for age_group_id in [4, 5]:
    for year in range(2022, 2101):
        for scenario in ['ssp119', 'ssp245']:
            for fhs_location_id in fhs_locs:
                pred_task = predict_tt.create_task(
                    compute_resources={"cores": 2, "memory": "15Gb", "runtime": "60m", "constraints": "archive"},
                    max_attempts=5,
                    name = pred_task_name.format(model_identifier=model_identifier, measure=measure, #sex_id=sex_id, #age_group_id=age_group_id, 
                        fhs_location_id=fhs_location_id, scenario=scenario, year=year),
                    python="conda run -p /mnt/share/homes/victorvt/envs/cgf_temperature python",
                    #python="/mnt/share/homes/victorvt/envs/cgf_temperature/bin/python3.12",
                    script='/mnt/share/homes/victorvt/repos/spatial_temp_cgf/predict_using_model.py',
                    measure = measure,
                    model_identifier = model_identifier,
                    fhs_location_id = fhs_location_id,
                    scenario = scenario,
                    year = year,
                    # sex_id = sex_id,
                    #age_group_id = age_group_id,
                    resource_scales = {"memory": 2, "runtime": 2},
                    fallback_queues=["long.q"]
                )
                workflow.add_tasks([pred_task])

# run workflow
workflow.run(fail_fast=True)

