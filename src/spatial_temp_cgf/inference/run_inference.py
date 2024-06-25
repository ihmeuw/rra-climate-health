import logging
import sys
from pathlib import Path
import dill as pickle

import click
import numpy as np
import pandas as pd
import rasterra as rt
from rra_tools import jobmon
from scipy.special import expit
import xarray as xr

from spatial_temp_cgf import income_funcs, paths
from spatial_temp_cgf.data_prep.location_mapping import load_fhs_lsae_mapping
from spatial_temp_cgf import cli_options as clio
from spatial_temp_cgf.data import DEFAULT_ROOT, ClimateMalnutritionData


def xarray_to_raster(ds: xr.DataArray, nodata: float | int) -> rt.RasterArray:
    from affine import Affine
    """Convert an xarray DataArray to a RasterArray."""
    lat, lon = ds["latitude"].data, ds["longitude"].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-dlat,
        f=lat[-1],
    )
    raster = rt.RasterArray(
        data=ds.data[::-1],
        transform=transform,
        crs="EPSG:4326",
        no_data_value=nodata,
    )
    return raster


def get_climate_variable_dataset(scenario, year, climate_var):
    import xarray as xr
    climate_variables = {'temp': 'mean_temperature',
                         'precip': 'total_precipitation'}
    for threshold in range(30, 31):
        climate_variables[f'over_{threshold}'] = f'days_over_{threshold}C'

    if climate_var not in climate_variables.keys():
        raise ValueError(f"Climate variable {climate_var} not recognized")

    subfolder = scenario if year >= 2024 else 'historical'

    filepath = paths.CLIMATE_PROJECTIONS_ROOT / subfolder / climate_variables[
        climate_var] / (str(year) + ".nc")

    projection_ds = xr.open_dataset(filepath)
    projection_ds = projection_ds.loc[dict(year=year)]
    return projection_ds


def get_climate_variable_raster(scenario, year, climate_var, shapefile,
                                reference_raster, nodata=np.nan, untreated=False):
    projection_da = get_climate_variable_dataset(scenario, year, climate_var).value
    result_raster = xarray_to_raster(projection_da, nodata)
    if untreated:
        return result_raster
    result_raster = result_raster.to_crs(reference_raster.crs) \
        .clip(shapefile) \
        .mask(shapefile) \
        .resample_to(reference_raster)
    return result_raster

def scale_like_input_data(to_scale, input_min, input_max):
    return (to_scale - input_min) / (input_max - input_min)


def reverse_scaling(X_scaled, original_min, original_max, scaled_min, scaled_max):
    X_std = (X_scaled - scaled_min) / (scaled_max - scaled_min)
    X_original = X_std * (original_max - original_min) + original_min
    return X_original


def model_inference_main(
    output_dir: Path,
    model_id: str,
    measure: str,
    fhs_location_id: int,
    cmip6_scenario: str,
    year: int,
) -> None:
    cm_data = ClimateMalnutritionData(output_dir)

    loc_mapping = load_fhs_lsae_mapping(fhs_location_id)
    fhs_shapefile = loc_mapping.iloc[0].fhs_shape
    income_df = income_funcs.load_binned_income_distribution_proportions(
        fhs_location_id=fhs_location_id, measure=measure, year_id=year
    )

    models = cm_data.load_model_family(model_id, measure)
    model = models[0]['model'] #getting the first model to base bins and variable info off of

    fhs_pop_raster = rt.load_raster(paths.GLOBAL_POPULATION_FILEPATH, fhs_shapefile.bounds).set_no_data_value(np.nan)
    fhs_pop_raster = fhs_pop_raster.clip(fhs_shapefile)

    possible_climate_variables = ['temp', 'precip', 'over_30']
    climate_vars_to_match_bins = [v for v in model.vars_to_bin if v in possible_climate_variables]
    continuous_climate_vars = [v for v in ['temp', 'over_30'] if (v in possible_climate_variables) and (v not in climate_vars_to_match_bins) and (v in model.model_vars)]
    climate_vars = [x for x in possible_climate_variables if x in set(model.model_vars + model.vars_to_bin)]

    climate_rasters = {}
    for var in climate_vars:
        climate_rasters[var] = get_climate_variable_raster(cmip6_scenario, year, var, None, None, untreated=True)
        climate_rasters[var] = climate_rasters[var].resample_to(fhs_pop_raster).clip(fhs_shapefile)

    admin_dfs = []
    for _, admin2_row in loc_mapping.iterrows():
        lsae_location_id = admin2_row.lsae_location_id
        lsae_shapefile = admin2_row.lsae_shape

        pop_raster = fhs_pop_raster.clip(lsae_shapefile).mask(lsae_shapefile)
        pop_array = pop_raster.set_no_data_value(np.nan).to_numpy()

        rasters = {'population': pop_array.flatten()}

        for var in climate_vars_to_match_bins:
            climate_raster = climate_rasters[var].clip(lsae_shapefile).mask(lsae_shapefile)
            climate_array = climate_raster.to_numpy()
            assert pop_array.shape == climate_array.shape
            binned_climate_array = np.digitize(climate_array, model.var_info[var]['bin_edges'], right=False) - 1
            rasters[var+'_bin_idx'] = binned_climate_array.flatten()

        for var in continuous_climate_vars:
            climate_raster = climate_rasters[var].clip(lsae_shapefile).mask(lsae_shapefile)
            climate_array = climate_raster.to_numpy()
            assert pop_array.shape == climate_array.shape
            if var in model.scaled_vars:
                varmin, varmax = model.scaling_bounds[var]
                climate_array = scale_like_input_data(climate_array, varmin, varmax)
            rasters[var] = climate_array.flatten()

        # Alternative approach is to group pixels to lsae
        # Keeping it as pixels
        pixels_df = pd.DataFrame(rasters)
        pixels_df = pixels_df.dropna(subset=['population'])  # TODO: Reconsider this, pretty sure it's right
        pixels_df['lsae_pop'] = np.nansum(pop_array)
        pixels_df['lsae_location_id'] = lsae_location_id
        pixels_df['worldpop_iso3'] = admin2_row.worldpop_iso3

        local_income_df = income_df.query('lsae_location_id == @lsae_location_id')
        for var in climate_vars_to_match_bins:
            pixels_df = pixels_df.merge(model.var_info[var]['bins'], left_on=var + '_bin_idx', right_index=True, how='inner')
        pixels_df = pixels_df.merge(local_income_df, on='lsae_location_id', how='left')

        # The lines from now on have coefficients and so are age_group and sex_id - specific
        # So model-specific. Parallelizing by them
        for m in models:
            model = m['model']
            px_for_model_df = pixels_df.copy()
            model.var_info['ihme_loc_id']['coefs'].rename(columns={'ihme_loc_id': 'worldpop_iso3'}, inplace=True)

            if model.has_grid:
                px_for_model_df = px_for_model_df.merge(model.grid_spec['grid_definition'], how='left')
                px_for_model_df = px_for_model_df.merge(model.var_info['grid_cell']['coefs'], how='left')

            for var in continuous_climate_vars:
                assert(len(model.var_info[var]['coefs'].columns) == 1)
                colname = model.var_info[var]['coefs'].columns[0]
                px_for_model_df[colname] = model.var_info[var]['coefs'][colname].item()

            px_for_model_df = px_for_model_df.merge(model.var_info['ihme_loc_id']['coefs'], how='left', on='worldpop_iso3')
            px_for_model_df.ihme_loc_id_coef = px_for_model_df.ihme_loc_id_coef.fillna(0)
        
            # build the logistic input one variable at a time
            px_for_model_df['logistic_input'] = model.var_info['intercept']['coef']
            px_for_model_df['logistic_input'] += px_for_model_df['ihme_loc_id_coef']
            for var in climate_vars_to_match_bins:
                if var in model.grid_spec['grid_order']: continue
                px_for_model_df['logistic_input'] += px_for_model_df[var+'_bin_coef']
            for var in continuous_climate_vars:
                px_for_model_df['logistic_input'] += px_for_model_df[var+'_coef']
            if model.has_grid:
                px_for_model_df['logistic_input'] += px_for_model_df['grid_cell_coef']
            px_for_model_df['prediction'] = expit(px_for_model_df['logistic_input'])
            px_for_model_df['age_group_id'] = m['age_group_id']
            px_for_model_df['sex_id'] = m['sex_id']
            admin_dfs.append(px_for_model_df)


    fhs_df = pd.concat(admin_dfs)
    #24s to 92s
    #2:27 to
    #24s to 92s

    fhs_df['population_at_income'] = fhs_df['population'] * fhs_df['proportion_at_income']
    fhs_df['population_proportion_at_income'] = fhs_df['population_at_income'] / fhs_df['lsae_pop'] / len(loc_mapping)
    fhs_df['affected_proportion'] = fhs_df['population_proportion_at_income'] * fhs_df['prediction']

    result_df = fhs_df.groupby(['fhs_location_id', 'age_group_id', 'sex_id']).agg(
        #susceptible_proportion = ('population_proportion_at_income', 'sum'),  #For testing, HAS to be == 1
        affected_proportion = ('affected_proportion', 'sum'))
    result_df['year_id'] = year
    result_df['scenario'] = cmip6_scenario
    result_df['measure'] = measure

    cm_data.save_results(
        result_df,
        model_id=model_id,
        measure=measure,
        scenario=cmip6_scenario,
        year=year,
    )


@click.command()
@clio.with_output_directory(DEFAULT_ROOT)
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
@clio.with_output_directory(DEFAULT_ROOT)
@clio.with_model_id()
@clio.with_measure(allow_all=True)
@clio.with_location_id(allow_all=True)
@clio.with_cmip6_scenario(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_queue()
def model_inference(
    output_dir: str,
    model_id: str,
    measure: list[str],
    location_id: list[str],
    cmip6_scenario: list[str],
    year: list[str],
    queue: str,
) -> None:
    """Run model inference."""
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    jobmon.run_parallel(
        runner="sttask",
        task_name="inference",
        node_args={
            "measure": measure,
            "location-id": location_id,
            "cmip6-scenario": cmip6_scenario,
            "year": year,
        },
        task_args={
            "output-dir": output_dir,
            "model-id": model_id,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "15Gb",
            "runtime": "60m",
            "project": "proj_rapidresponse",
            "constraints": "archive",

        },
        max_attempts=1,
    )
