import logging
import sys
from pathlib import Path
import itertools

import click
import numpy as np
import pandas as pd
import rasterra as rt
from rra_tools import jobmon
import xarray as xr
import tqdm
import geopandas as gpd

from spatial_temp_cgf import paths, utils
from spatial_temp_cgf.data_prep.location_mapping import FHS_SHAPE_PATH
from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData


def model_inference_main(
    output_dir: Path,    
    measure: str,
    model_version: str,
    cmip6_scenario: str,
    year: int,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir / measure)

    print('loading raster template')
    raster_template = cm_data.load_raster_template()
    print('loading models')
    models = cm_data.load_model_family(model_version)

    print('loading predictors')
    m = models[0]
    variables = {}
    for variable, info in m['model'].var_info.items():
        print(variable)
        if variable == 'ldi_pc_pd':
            v = cm_data.load_ldi(year, 0.1)
            v = info['transformer'](v)
        else:
            ds = cm_data.load_climate_raster(variable, cmip6_scenario, year)
            v = utils.xarray_to_raster(ds, nodata=np.nan).resample_to(raster_template)
        variables[variable] = v.to_numpy().reshape(-1, 1)

    print('predicting...')
    x = pd.DataFrame(variables)
    for model_dict in models:
        model = model_dict['model']
        raw_prediction = model.predict(x)
        prediction = rt.RasterArray(
            raw_prediction.reshape(raster_template.shape),
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan,
        )
        a_id = model_dict['age_group_id']
        s_id = model_dict['sex_id']
        cm_data.save_raster_results(prediction, model_version, cmip6_scenario, year, a_id, s_id)
        model_dict['prediction'] = prediction

    print("loading fhs shape data")
    fhs_shapes = gpd.read_parquet(FHS_SHAPE_PATH).set_index('loc_id').geometry.to_dict()
    print("loading population")
    fhs_pop_raster = rt.load_raster(paths.GLOBAL_POPULATION_FILEPATH).set_no_data_value(np.nan)

    print('Computing zonal statistics')
    out = []
    for model_dict in models:
        count = model_dict['prediction'] * fhs_pop_raster
        for shape_id, shape in fhs_shapes.items():
            numerator = np.nansum(count.clip(shape).mask(shape))
            denominator = np.nansum(fhs_pop_raster.clip(shape).mask(shape))
            out.append((
                shape_id,
                model_dict['age_group_id'],
                model_dict['sex_id'],
                numerator / denominator,
            ))

    print('saving results table')
    df = pd.DataFrame(out, columns=['location_id', 'age_group_id', 'sex_id', 'value'])
    cm_data.save_results_table(df, model_version, cmip6_scenario, year)


@click.command()
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_model_id()
@clio.with_measure()
@clio.with_location_id()
@clio.with_cmip6_scenario()
@clio.with_year()
def model_inference_task(
    output_dir: str,
    model_id: str,
    measure: str,
    location_id: str,
    cmip6_scenario: str,
    year: str,
):
    """Run model inference."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    model_inference_main(
        Path(output_dir),
        model_id,
        measure,
        int(location_id),
        cmip6_scenario,
        int(year),
    )


@click.command()
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_model_id()
@clio.with_measure(allow_all=True)
@clio.with_location_id(allow_all=True)
@clio.with_cmip6_scenario(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_queue()
@clio.with_overwrite()
def model_inference(
    output_dir: str,
    model_id: str,
    measure: list[str],
    location_id: list[str],
    cmip6_scenario: list[str],
    year: list[str],
    queue: str,
    overwrite: bool,
) -> None:
    """Run model inference."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    cm_data = ClimateMalnutritionData(output_dir)

    complete = []
    to_run = []
    if not overwrite:
        for m, l, s, y in tqdm.tqdm(list(itertools.product(measure, location_id, cmip6_scenario, year))):
            if cm_data.results_path(model_id, l, m, s, y).exists():
                complete.append((m, l, s, y))
            else:
                to_run.append((m, l, s, y))

    print(
        f"Run configuration has {len(complete)} jobs complete "
        f"and {len(to_run)} jobs to run."
    )

    jobmon.run_parallel(
        runner="sttask",
        task_name="inference",
        flat_node_args=(
            ("measure", "location-id", "cmip6-scenario", "year"),
            to_run,
        ),
        task_args={
            "output-dir": output_dir,
            "model-id": model_id,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "45Gb",
            "runtime": "60m",
            "project": "proj_rapidresponse",
            "constraints": "archive",

        },
        max_attempts=1,
    )
