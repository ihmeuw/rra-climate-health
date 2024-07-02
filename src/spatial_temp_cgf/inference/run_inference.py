import logging
import sys
from pathlib import Path
import itertools

import click
import numpy as np
import pandas as pd
import rasterra as rt
from rasterio.features import rasterize
from rra_tools import jobmon
import tqdm
import geopandas as gpd

from spatial_temp_cgf import utils
from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData
from spatial_temp_cgf.model_specification import PredictorSpecification


def get_intercept_raster(
    pred_spec: PredictorSpecification,
    coefs: pd.DataFrame,
    ranefs: pd.DataFrame,
    fhs_shapes: gpd.GeoDataFrame,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    icept = coefs.loc['(Intercept)']
    if pred_spec.random_effect == 'ihme_loc_id':
        shapes = list(
            ranefs['X.Intercept.'].reset_index()
            .merge(fhs_shapes, left_on='index', right_on='ihme_lc_id', how='left')
            .loc[:, ['geometry', 'X.Intercept.']]
            .itertuples(index=False, name=None)
        )
        icept_arr = rasterize(
            shapes,
            out=icept*np.ones_like(raster_template),
            transform=raster_template.transform,
        )
        icept_raster = rt.RasterArray(
            icept_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan
        )
    elif not pred_spec.random_effect:
        icept_raster = raster_template + icept
    else:
        msg = 'Only location random intercepts are supported'
        raise NotImplementedError(msg)
    return icept_raster


def model_inference_main(
    output_dir: Path,    
    measure: str,
    model_version: str,
    cmip6_scenario: str,
    year: int,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir / measure)
    spec = cm_data.load_model_specification(model_version)
    print('loading raster template and shapes')
    raster_template = cm_data.load_raster_template()
    fhs_shapes = cm_data.load_fhs_shapes()
    print('loading models')
    models = cm_data.load_model_family(model_version)
    # =============

    model = models[0]['model']

    coefs = model.coefs['Estimate']
    ranefs = model.ranef
    
    partial_estimates = {}
    
    for predictor in spec.predictors:
        if predictor.name == 'intercept':
            partial_estimates[predictor.name] = get_intercept_raster(
                predictor, coefs, ranefs, fhs_shapes, raster_template,
            )
        elif predictor.name == 'ldi_pc_pd':
            ldi_spec = predictor  # deal with later
        elif predictor.name == 'elevation':
            v = rt.load_raster('/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition/input/srtm_elevation.tif')
            beta = coefs.loc[predictor.name]
            partial_estimates[predictor.name] = rt.RasterArray(
                beta * np.array(model.var_info[predictor.name]['transformer'](v)),
                transform=v.transform,
                crs=v.crs,
                no_data_value=np.nan,
            )
        else:
            if predictor.random_effect:
                msg = 'Random slopes not implemented'
                raise NotImplementedError(msg)
                
            transform = predictor.transform
            variable = transform.from_column if hasattr(transform, 'from_column') else predictor.name    
            ds = cm_data.load_climate_raster(variable, cmip6_scenario, year)
            v = utils.xarray_to_raster(ds, nodata=np.nan).resample_to(raster_template)
            beta = coefs.loc[predictor.name]
            partial_estimates[predictor.name] = rt.RasterArray(
                beta * np.array(model.var_info[predictor.name]['transformer'](v)),
                transform=v.transform,
                crs=v.crs,
                no_data_value=np.nan,
            )
    
    assert spec.extra_terms == ['any_days_over_30C * ldi_pc_pd']
    
    beta_interaction = coefs.loc['ldi_pc_pd:any_days_over_30C'] / coefs.loc['any_days_over_30C']
    beta_ldi = beta_interaction * partial_estimates['any_days_over_30C'] + coefs.loc['ldi_pc_pd']
    z_partial = sum(partial_estimates.values())

    prevalence = 0
    for i in range(1, 11):
        print(i)
        ldi = cm_data.load_ldi(year, f"{i/10.:.1f}")
        prevalence += 0.1 * 1 / (1 + np.exp(-z_partial - beta_ldi * ldi))
# =====================

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
    fhs_shapes = cm_data.load_fhs_shapes().set_index('loc_id').geometry.to_dict()
    print("loading population")
    fhs_pop_raster = cm_data.load_population_raster().set_no_data_value(np.nan)

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
