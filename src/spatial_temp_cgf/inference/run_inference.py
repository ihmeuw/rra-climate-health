from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterra as rt
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize
from rra_tools import jobmon

from spatial_temp_cgf import utils
from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.data import ClimateMalnutritionData, DEFAULT_ROOT
from spatial_temp_cgf.model_specification import PredictorSpecification, ModelSpecification
from spatial_temp_cgf.inference.inference_diagnostics import create_inference_diagnostics_report

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
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        icept_raster = rt.RasterArray(
            icept + icept_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan
        )
    elif pred_spec.random_effect == 'lbd_admin2_id':
        shapes = list(
            ranefs['X.Intercept.'].reset_index()
            .merge(fhs_shapes, left_on='index', right_on='loc_id', how='left')
            .loc[:, ['geometry', 'X.Intercept.']]
            .itertuples(index=False, name=None)
        )
        icept_arr = rasterize(
            shapes,
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        icept_raster = rt.RasterArray(
            icept + icept_arr,
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

def get_nonspatial_predictor_raster(
    pred_spec: PredictorSpecification,
    pred_value: float,
    coefs: pd.DataFrame,
    ranefs: pd.DataFrame,
    fhs_shapes: gpd.GeoDataFrame,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    var_fixef = coefs.loc[pred_spec.name]
    if pred_spec.random_effect == 'ihme_loc_id':
        shapes = list(
            ranefs[pred_spec.name].reset_index()
            .merge(fhs_shapes, left_on='index', right_on='ihme_lc_id', how='left')
            .loc[:, ['geometry', pred_spec.name]]
            .itertuples(index=False, name=None)
        )
        var_arr = rasterize(
            shapes,
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        var_raster = rt.RasterArray(
            var_fixef + var_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan
        )
    elif not pred_spec.random_effect:
        var_raster = raster_template + var_fixef
    else:
        msg = 'Only location random intercepts are supported'
        raise NotImplementedError(msg)
    return var_raster * pred_value

def get_model_prevalence(
    model,
    spec: ModelSpecification,
    cmip6_scenario: str,
    year: int,
    cm_data: ClimateMalnutritionData,
    location_effect_shapes: gpd.GeoDataFrame,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    coefs = model.coefs['Estimate']
    ranefs = model.ranef
    
    partial_estimates = {}
    for predictor in spec.predictors:
        print(predictor.name)
        if predictor.name == 'ldi_pc_pd':
            continue  # deal with after
        elif predictor.name == 'intercept':
            partial_estimates[predictor.name] = get_intercept_raster(
                predictor, coefs, ranefs, location_effect_shapes, raster_template,
            )
        elif predictor.name == 'year_start': 
            #For now. Gotta think of a way for this not to become a long list of if-elses
            transformed_value = model.var_info[predictor.name]['transformer'](np.array([[year]]))[0][0]
            partial_estimates[predictor.name] = get_nonspatial_predictor_raster(
                predictor, transformed_value, 
                coefs, ranefs, location_effect_shapes, raster_template,
            )
        elif predictor.name == 'sdi':
            sdi = cm_data.load_rasterized_variable(predictor.name, year)
            partial_estimates[predictor.name] = rt.RasterArray(
                coefs.loc['sdi'] * np.array(model.var_info['sdi']['transformer'](sdi)),
                transform=sdi.transform,
                crs=sdi.crs,
                no_data_value=np.nan,
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
    ).to_numpy()
    z_partial = sum(partial_estimates.values())
    
    prevalence = 0
    for i in range(1, 11):
        print(i)
        ldi = cm_data.load_ldi(year, f"{i / 10.:.1f}")
        
        z_ldi = rt.RasterArray(
            beta_ldi * np.array(model.var_info['ldi_pc_pd']['transformer'](ldi)),
            transform=ldi.transform,
            crs=ldi.crs,
            no_data_value=np.nan,
        )
        prevalence += 0.1 * 1 / (1 + np.exp(-(z_partial + z_ldi)))
    
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
    result_shapes = cm_data.load_fhs_shapes(most_detailed_only=False)
    if 'lbd_admin2_id' in spec.random_effects:
        model_location_effect_shapes = cm_data.load_lbd_admin2_shapes()
    else:
        model_location_effect_shapes = result_shapes
    print("loading population")
    pop_raster = cm_data.load_population_raster().set_no_data_value(np.nan)
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
            model_location_effect_shapes,
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
    shape_map = result_shapes[result_shapes.most_detailed == 1].set_index('loc_id').geometry.to_dict()
    out = []
    for model_dict in models:
        count = model_dict['prediction'] * pop_raster
        for shape_id, shape in shape_map.items():
            numerator = np.nansum(count.clip(shape).mask(shape))
            denominator = np.nansum(pop_raster.clip(shape).mask(shape))
            out.append((
                shape_id,
                model_dict['age_group_id'],
                model_dict['sex_id'],
                numerator / denominator,
            ))

    print('saving results table')
    df = pd.DataFrame(out, columns=['location_id', 'age_group_id', 'sex_id', 'value'])
    cm_data.save_results_table(df, results_version, cmip6_scenario, year)


def load_population_timeseries(
    cm_data: ClimateMalnutritionData,
    locs_of_interest: list[int], 
    age_group_ids:list[int] = [4,5]):
    #TODO consider moving this to the data module, though I guess we don't own this location
    FORECASTED_POPULATIONS_FILEPATH = '/mnt/share/forecasting/data/7/future/population/20240529_500d_2100_lhc_ref_squeeze_hiv_shocks_covid_all_gbd_7_shifted/population.nc'
    forecast_pop = xr.open_dataset(FORECASTED_POPULATIONS_FILEPATH).mean(dim='draw').sel(age_group_id=age_group_ids, year_id = range(2022,2101), location_id=locs_of_interest).to_dataframe().droplevel('scenario')
    historical_pop = pd.read_parquet(cm_data._PROCESSED_DATA_ROOT / 'ihme' / 'global_population_by_age_group_and_sex.parquet')
    historical_pop = historical_pop.loc[historical_pop.year_id < forecast_pop.index.get_level_values('year_id').min()].set_index(forecast_pop.index.names)
    pop = pd.concat([historical_pop, forecast_pop])
    return pop

def forecast_scenarios(
    output_dir: Path, 
    measure: str, 
    results_version: str, 
    scenarios: list[str],):
    cm_data = ClimateMalnutritionData(output_dir / measure)
    REFERENCE_SCENARIO = 'ssp245'
    inference_scenarios = ['ssp119', 'ssp245', 'ssp585', 'constant_climate']

    locs_of_interest = cm_data.load_fhs_hierarchy().query("most_detailed == 1").location_id.unique()

    pop = load_population_timeseries(cm_data, locs_of_interest)
    loc_region_map = cm_data.load_fhs_hierarchy()[['location_id', 'location_name', 'region_name', 'super_region_name']]

    results_path = Path(DEFAULT_ROOT) / measure / 'results' / results_version

    scenarios_present = []
    for scenario in (inference_scenarios):
        dfs = []
        for fp in list(results_path.glob(f"*_{scenario}.parquet")):
            df = pd.read_parquet(fp)
            df['year_id'] = int(fp.name.split('_')[0])
            df['scenario'] = scenario
            dfs.append(df)
        if dfs:
            scenarios_present.append(scenario)
            pd.concat(dfs).to_parquet(results_path / f'{scenario}.parquet')

    idx_cols = ['location_id', 'age_group_id', 'sex_id', 'year_id', 'scenario']

    combined = pd.concat([pd.read_parquet(results_path / f'{scenario}.parquet').set_index(idx_cols).value.rename("prevalence") for scenario in scenarios_present]).sort_index()
    combined = combined.reset_index().assign(location_id = lambda x: x.location_id.astype(int))
    combined = combined.set_index(['location_id', 'age_group_id', 'sex_id', 'year_id']).sort_index()
    merged = combined.merge(pop, left_index=True, right_index=True, how='left')
    merged['affected'] = merged['prevalence'] * merged['population']

    if REFERENCE_SCENARIO in scenarios_present:
        reference = pd.read_parquet(results_path / f'{REFERENCE_SCENARIO}.parquet').set_index(['location_id', 'age_group_id', 'sex_id', 'year_id']).value.rename(f'ref_prev').drop(columns='scenario')
        merged = merged.join(reference, how='left')

        merged['delta'] = (merged['ref_prev'] * merged['population']) - merged['affected']
    else:
        merged['delta'] = np.nan

    merged = merged.merge(loc_region_map.set_index('location_id'), left_index=True, right_index=True)
    cm_data.save_forecast(merged, results_version)

@click.command()
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_measure()
@clio.with_results_version()
@clio.with_model_version()
@clio.with_cmip6_scenario(allow_all=True)
def forecast_scenarios_task(
    output_root: str,
    measure: str,
    results_version: str,
    model_version: str,
    cmip6_scenario: list[str],
):
    """Run forecasting applying the inference results, and output diagnostics."""
    forecast_scenarios(
        Path(output_root),
        measure,
        results_version,
        cmip6_scenario,
    )
    create_inference_diagnostics_report(
        Path(output_root),
        measure,
        results_version,
        model_version)



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
    results_version = cm_data.new_results_version(model_version)
    print(f"Running inference for {measure} using {model_version}. Results version: {results_version}")

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

    jobmon.run_parallel(
        runner="sttask",
        task_name="forecast",
        node_args={
            "measure": [measure],
            "cmip6-scenario": [" --cmip6-scenario ".join(cmip6_scenario)],
        },
        task_args={
            "output-root": output_root,
            "model-version": model_version,
            "results-version": results_version,
        },
        task_resources={
            "queue": queue,
            "cores": 2,
            "memory": "30Gb",
            "runtime": "30m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(cm_data.results / results_version),
    )
    print(f"Inference complete, results can be found at {cm_data.results / results_version}")
