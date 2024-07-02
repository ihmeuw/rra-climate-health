import datetime
from pathlib import Path
import typing
import itertools
import pickle

import pandas as pd
import rasterra as rt
import geopandas as gpd
import xarray as xr
from rra_tools.shell_tools import mkdir, touch

from spatial_temp_cgf.model_specification import ModelSpecification


if typing.TYPE_CHECKING:
    from pymer4 import Lmer

DEFAULT_ROOT = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition"


class ClimateMalnutritionData:

    def __init__(
        self,
        measure_root: str | Path,
    ) -> None:
        self._root = Path(measure_root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def training_data(self) -> Path:
        return self.root / "training_data"

    def new_training_version(self) -> str:
        run_directory = get_run_directory(self.training_data)
        mkdir(run_directory)
        return run_directory.name

    def save_training_data(self, data: pd.DataFrame, version: str) -> None:
        path = self.training_data / version
        data_path = path / 'data.parquet'
        touch(data_path)
        data.to_parquet(data_path)

    def load_training_data(self, version: str) -> pd.DataFrame:
        path = self.training_data / version / 'data.parquet'
        return pd.read_parquet(path)

    @property
    def models(self) -> Path:
        return self.root / "models"

    def new_model_version(self) -> str:
        run_directory = get_run_directory(self.models)
        mkdir(run_directory)
        return run_directory.name

    def save_model_specification(self, model_spec: ModelSpecification, version: str) -> None:
        model_root = self.models / version
        model_spec_path = model_root / f"specification.yaml"
        touch(model_spec_path)
        model_spec.to_yaml(model_spec_path)

    def load_model_specification(self, version: str) -> ModelSpecification:
        model_spec_path = self.models / version / "specification.yaml"
        return ModelSpecification.from_yaml(model_spec_path)

    def save_model(
        self,
        model: "Lmer",
        version: str,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> None:
        model_root = self.models / version
        mkdir(model_root, exist_ok=True)
        model_filepath = model_root / f'{age_group_id}_{sex_id}.pkl'
        touch(model_filepath, exist_ok=True)
        with model_filepath.open("wb") as f:
            pickle.dump(model, f)    

    def load_model_family(
        self,
        version: str,
    ) -> list[dict]:
        models = []
        model_id_dir = self.models / version
        for age_group_id, sex_id in itertools.product([4, 5], [1, 2]):
            model_filepath = (
                model_id_dir / f'{age_group_id}_{sex_id}.pkl'
            )
            with model_filepath.open("rb") as f:
                models.append({
                    'model': pickle.load(f),
                    'age_group_id': age_group_id,
                    'sex_id': sex_id
                })
        return models

    @property
    def results(self) -> Path:
        return self.root / "results"

    def raster_results_path(
        self,
        model_version: str,
        scenario: str,
        year: str | int,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> Path:
        return self.results / model_version / f"{year}_{scenario}_{age_group_id}_{sex_id}.tif"

    def save_raster_results(
        self,
        results: rt.RasterArray,
        model_version: str,
        scenario: str,
        year: str | int,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> None:
        path = self.raster_results_path(model_version, scenario, year, age_group_id, sex_id)
        mkdir(path.parent, parents=True, exist_ok=True)
        save_raster(results, path)

    def save_results_table(
        self,
        results: pd.DataFrame,
        model_version: str,
        scenario: str,
        year: str | int,
    ) -> None:
        path = self.results / model_version / f"{year}_{scenario}.parquet"
        touch(path, exist_ok=True)
        results.to_parquet(path)

    @property
    def shared_inputs(self) -> Path:
        return self.root.parent / 'input'

    def ldi_path(self, year: int | str, percentile: float | str) -> Path:
        return self.shared_inputs / "ldi" / f"{year}_{percentile}.tif"

    def load_ldi(self, year: int | str, percentile: float | str) -> rt.RasterArray:
        return rt.load_raster(self.ldi_path(year, percentile))

    def save_ldi_raster(
        self,
        ldi: rt.RasterArray,
        year: int | str,
        percentile: float | str,
    ) -> None:
        path = self.ldi_path(year, percentile)
        mkdir(path.parent, parents=True, exist_ok=True)
        save_raster(ldi, path)

    #########################
    # Upstream paths we own #
    #########################

    _POP_DATA_ROOT = Path('/mnt/team/rapidresponse/pub/population/data')
    _RAW_DATA_ROOT = _POP_DATA_ROOT / '01-raw-data'
    _PROCESSED_DATA_ROOT = _POP_DATA_ROOT / '02-processed-data'
    _CLIMATE_DATA_ROOT = Path("/mnt/share/erf/climate_downscale/results/annual")

    def load_lbd_admin2_shapes(self) -> gpd.GeoDataFrame:
        path = self._PROCESSED_DATA_ROOT / 'ihme' / 'lbd_admin2.parquet'
        return gpd.read_parquet(path)

    def load_raster_template(self) -> rt.RasterArray:
        path = self._RAW_DATA_ROOT / 'other-gridded-pop-projects' / 'global-human-settlement-layer' / '1km_template.tif'
        return rt.load_raster(path)

    def load_climate_raster(self, variable: str, scenario: str, year: int | str) -> xr.DataArray:
        scenario_folder = 'historical' if year < 2024 else scenario
        path = self._CLIMATE_DATA_ROOT / scenario_folder / variable / f"{year}.nc"
        return xr.open_dataset(path).sel(year=year)['value']


def get_run_directory(output_root: str | Path) -> Path:
    """Gets a path to a datetime directory for a new output.

    Parameters
    ----------
    output_root
        The root directory for all outputs.

    """
    output_root = Path(output_root).resolve()
    launch_time = datetime.datetime.now().strftime("%Y_%m_%d")
    today_runs = [
        int(run_dir.name.split(".")[1])
        for run_dir in output_root.iterdir()
        if run_dir.name.startswith(launch_time)
    ]
    run_version = max(today_runs) + 1 if today_runs else 1
    datetime_dir = output_root / f"{launch_time}.{run_version:0>2}"
    return datetime_dir


def save_raster(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    **kwargs,
) -> None:
    """Save a raster to a file with standard parameters."""
    save_params = {
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "ZSTD",
        "predictor": 2,  # horizontal differencing
        "num_threads": num_cores,
        "bigtiff": "yes",
        **kwargs,
    }
    touch(output_path, exist_ok=True)
    raster.to_file(output_path, **save_params)