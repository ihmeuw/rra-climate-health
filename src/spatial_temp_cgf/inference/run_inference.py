from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterra as rt
from rasterio.features import rasterize
from rra_tools import jobmon
import geopandas as gpd

from spatial_temp_cgf import utils
from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.data import ClimateMalnutritionData, DEFAULT_ROOT
from spatial_temp_cgf.model_specification import PredictorSpecification, ModelSpecification


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


def get_model_prevalence(
    model,
    spec: ModelSpecification,
    cmip6_scenario: str,
    year: int,
    cm_data: ClimateMalnutritionData,
    fhs_shapes: gpd.GeoDataFrame,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    coefs = model.coefs['Estimate']
    ranefs = model.ranef

    partial_estimates = {}
    for predictor in spec.predictors:
        if predictor.name == 'ldi_pc_pd':
            continue  # deal with after
        elif predictor.name == 'intercept':
            partial_estimates[predictor.name] = get_intercept_raster(
                predictor, coefs, ranefs, fhs_shapes, raster_template,
            )
        else:
            if predictor.random_effect:
                msg = 'Random slopes not implemented'
                raise NotImplementedError(msg)

            if predictor.name == 'elevation':
                v = cm_data.load_elevation()
            else:
                transform = predictor.transform
                variable = (
                    transform.from_column
                    if hasattr(transform, 'from_column') else predictor.name
                )
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

    beta_interaction = (
        coefs.loc['ldi_pc_pd:any_days_over_30C'] / coefs.loc['any_days_over_30C']
    )
    beta_ldi = (
        beta_interaction * partial_estimates['any_days_over_30C']
        + coefs.loc['ldi_pc_pd']
    )
    z_partial = sum(partial_estimates.values())

    prevalence = 0
    for i in range(1, 11):
        print(i)
        ldi = cm_data.load_ldi(year, f"{i / 10.:.1f}")
        ldi = rt.RasterArray(
            model.var_info['ldi_pc_pd']['transformer'](ldi),
            transform=ldi.transform,
            crs=ldi.crs,
            no_data_value=np.nan,
        )
        prevalence += 0.1 * 1 / (1 + np.exp(-z_partial - beta_ldi * ldi))

    return prevalence.astype(np.float32)


def model_inference_main(
    output_dir: Path,    
    measure: str,
    results_version: str,
    model_version: str,
    cmip6_scenario: str,
    year: int,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir / measure)
    spec = cm_data.load_model_specification(model_version)
    print('loading raster template and shapes')
    raster_template = cm_data.load_raster_template()
    fhs_shapes = cm_data.load_fhs_shapes()
    shape_map = fhs_shapes.set_index('loc_id').geometry.to_dict()
    print("loading population")
    fhs_pop_raster = cm_data.load_population_raster().set_no_data_value(np.nan)
    print('loading models')
    models = cm_data.load_model_family(model_version)

    for i, model_dict in enumerate(models):
        print(f'Computing prevalence for model {i} of {len(models)}')
        model_prevalence = get_model_prevalence(
            model_dict['model'],
            spec,
            cmip6_scenario,
            year,
            cm_data,
            fhs_shapes,
            raster_template,
        )
        cm_data.save_raster_results(
            model_prevalence,
            results_version,
            cmip6_scenario,
            year,
            model_dict['age_group_id'],
            model_dict['sex_id'],
        )

        model_dict['prediction'] = model_prevalence

    print('Computing zonal statistics')
    out = []
    for model_dict in models:
        count = model_dict['prediction'] * fhs_pop_raster
        for shape_id, shape in shape_map.items():
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
    cm_data.save_results_table(df, results_version, cmip6_scenario, year)


@click.command()
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@clio.with_model_version()
@clio.with_cmip6_scenario()
@clio.with_year()
def model_inference_task(
    output_root: str,
    measure: str,
    results_version: str,
    model_version: str,
    cmip6_scenario: str,
    year: str,
):
    """Run model inference."""
    model_inference_main(
        Path(output_root),
        measure,
        results_version,
        model_version,
        cmip6_scenario,
        int(year),
    )


@click.command()
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_model_version()
@clio.with_measure()
@clio.with_cmip6_scenario(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_queue()
def model_inference(
    output_root: str,
    model_version: str,
    measure: str,
    cmip6_scenario: list[str],
    year: list[str],
    queue: str,
) -> None:
    """Run model inference."""
    cm_data = ClimateMalnutritionData(Path(output_root) / measure)
    results_version = cm_data.new_results_version()

    jobmon.run_parallel(
        runner="sttask",
        task_name="inference",
        node_args={
            "measure": [measure],
            "cmip6-scenario": cmip6_scenario,
            "year": year,
        },
        task_args={
            "output-root": output_root,
            "model-version": model_version,
            "results-version": results_version,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "90Gb",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(cm_data.results / results_version),
    )
