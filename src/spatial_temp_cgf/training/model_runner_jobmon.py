import getpass
import uuid
import pandas as pd

from jobmon.client.tool import Tool

user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="spatial_malnutrition_tool")

# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"spatial_malnutrition_modelmaker_{wf_uuid}",
)

general_default_compute_resources_dict = {
    "queue": "long.q",
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
    command_template="{python} {script} {measure} {model_identifier} {lsae_location_id} {scenario} {year} {sex_id} {age_group_id}",
    node_args=["measure", "model_identifier", "lsae_location_id", "scenario", "year", "sex_id", "age_group_id"],
    op_args=["python", "script"],
    default_cluster_name="slurm",
    default_compute_resources=general_default_compute_resources_dict,
)

# Test task 1
model_identifier = 'me_grid_o30'
model_task_name = "{model_identifier}_{measure}_{sex_id}_{age_group_id}"
grid_vars = '"' + ','.join(['ldi_pc_pd', 'over_30']) + '"'

loc_mapping = pd.read_parquet("/mnt/share/scratch/users/victorvt/cgfwealth_spatial/fhs_lsae_location_mapping.parquet")
lsae_locs = loc_mapping.lsae_location_id.unique()

for measure in ['stunting', 'wasting']:
    model_spec = f'"{measure} ~ (1 | ihme_loc_id) + grid_cell"'
    for sex_id in [1, 2]:
        for age_group_id in [4, 5]:
            model_task = create_model_tt.create_task(
                compute_resources={"cores": 4, "memory": "20Gb", "runtime": "96h"},
                max_attempts=3,
                name = model_task_name.format(model_identifier=model_identifier, measure=measure, sex_id=sex_id, age_group_id=age_group_id),
                #python="/mnt/share/homes/victorvt/envs/cgf_temperature/bin/python3.12",
                python="conda run -p /mnt/share/homes/victorvt/envs/cgf_temperature python",
                script='/mnt/share/homes/victorvt/repos/spatial_temp_cgf/make_model.py',
                measure = measure,
                model_identifier = model_identifier,
                sex_id = sex_id,
                age_group_id = age_group_id,
                model_spec = model_spec,
                grid_vars = grid_vars
                )
            workflow.add_tasks([model_task])

# run workflow
workflow.run()

