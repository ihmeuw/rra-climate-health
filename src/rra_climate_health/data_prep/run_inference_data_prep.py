from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterra as rt
from rasterio.features import rasterize
from rra_tools import jobmon

from rra_climate_health import cli_options as clio
from rra_climate_health.data import DEFAULT_ROOT, ClimateMalnutritionData
from rra_climate_health.data_prep import upstream_paths


def run_ldi_prep_main(
    output_root: str | Path,
    year: int,
    version: str,
) -> None:
    """Run LDI data preparation."""
    # Measure doesn't matter for this task
    cm_data = ClimateMalnutritionData(Path(output_root) / "stunting")
    print("Loading admin2 shapes and raster template")
    admin2 = cm_data.load_lbd_admin2_shapes()
    raster_template = cm_data.load_raster_template()

    print("Loading LDI data")
    ldi = cm_data.load_ldi_distributions('admin2', version)
    year_col = "year_id" if "year_id" in ldi.columns else "year"
    nat_col = "national_ihme_loc_id" if "national_ihme_loc_id" in ldi.columns else "iso3"
    # Fill in missing values with national mean
    national_mean = ldi.groupby(
        ["scenario",year_col, nat_col, "population_percentile"]
    ).ldipc.transform("mean")
    null_mask = ldi.ldipc.isna()
    ldi.loc[null_mask, "ldipc"] = national_mean.loc[null_mask]
    # Convert to daily, and drop 0th percentile, which is just 0.
    ldi["ldi_pc_pd"] = ldi["ldipc"] / 365.25
    ldi = ldi[ldi.population_percentile > 0]

    print("Building shape map")
    ldi_locs = ldi["location_id"].unique().tolist()
    shape_map = (
        admin2.loc[admin2.loc_id.isin(ldi_locs), ["loc_id", "geometry"]]
        .rename(columns={"loc_id": "location_id"})
        .set_index("location_id")
        .geometry
    )

    print("Rasterizing LDI data")
    scenarios = ldi["scenario"].unique().tolist()
    percentiles = ldi["population_percentile"].unique().tolist()
    for scenario in scenarios:
        for percentile in percentiles:
            print(f"Rasterizing percentile: {percentile}")
            p_scenario_year_mask = (
                (ldi.population_percentile == percentile)
                & (ldi.scenario == scenario)
                & (ldi.year_id == year)
            )
            ldi_pc_pd = ldi.loc[p_scenario_year_mask].set_index("location_id").ldi_pc_pd
            ldi_pc_pd = ldi_pc_pd[~ldi_pc_pd.index.duplicated()]
            shapes = [
                (shape_map.loc[loc], ldi_pc_pd.loc[loc]) for loc in ldi_pc_pd.index
            ]
            ldi_arr = rasterize(
                shapes,
                out=np.zeros_like(raster_template),
                transform=raster_template.transform,
            )
            ldi_raster = rt.RasterArray(
                ldi_arr,
                transform=raster_template.transform,
                crs=raster_template.crs,
                no_data_value=np.nan,
            )
            cm_data.save_ldi_raster(ldi_raster, scenario, year, percentile, version)


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_year()
@clio.with_wealth_version()
def run_ldi_prep_task(output_root: str, year: str, wealth_version :str) -> None:
    """Run LDI data preparation."""
    run_ldi_prep_main(Path(output_root), int(year), wealth_version)


@click.command()  # type: ignore[arg-type]
@clio.with_output_root(DEFAULT_ROOT)
@clio.with_year(allow_all=True)
@clio.with_queue()
@clio.with_wealth_version()
def run_ldi_prep(output_root: str, year: list[str], queue: str, wealth_version: str) -> None:
    """Prep LDI rasters from admin2 data"""
    jobmon.run_parallel(
        runner="sttask",
        task_name="ldi_prep",
        node_args={"year": year},
        task_args={
            "output-root": output_root,
            "wealth-version": wealth_version,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "35Gb",
            "runtime": "4h",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=str(output_root),
    )
