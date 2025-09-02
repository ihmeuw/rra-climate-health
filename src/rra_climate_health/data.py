import datetime
import itertools
import pickle
import typing

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterra as rt
import xarray as xr
from rra_tools.shell_tools import mkdir, touch

from rra_climate_health.transforms import transform_column
from rra_climate_health.model_specification import ModelSpecification
from rra_climate_health.results_specification import (
    ResultsSpecification,
    ResultsVersionSpecification,
)

# if typing.TYPE_CHECKING:
#     from pymer4 import Lmer

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
        data_path = path / "data.parquet"
        touch(data_path)
        data.to_parquet(data_path)

    def load_training_data(self, version: str) -> pd.DataFrame:
        path = self.training_data / version / "data.parquet"
        return pd.read_parquet(path)
    
    def load_training_data_types(self, version: str):
        training_data = self.load_training_data(version)
        return training_data.dtypes.copy()

    def prepare_model_data(
        self,
        raw_model_data: pd.DataFrame,
        model_spec: ModelSpecification,
    ) -> tuple[pd.DataFrame, dict[str, typing.Any]]:
        # Do all required data transformations
        transformed_data = {}
        var_info = {}
        for var, transform_spec in model_spec.transform_map.items():
            transformed, transformer = transform_column(raw_model_data, var, transform_spec)
            transformed_data[var] = transformed
            var_info[var] = {
                "transformer": transformer,
                "transform_spec": transform_spec,
            }

        df = pd.DataFrame(transformed_data)

        if model_spec.grid_predictors:
            grid_spec: dict[str, typing.Any] = model_spec.grid_predictors.grid_spec
            grid_vars = grid_spec["grid_order"]
            df["grid_cell"] = df[grid_vars].astype(str).apply("_".join, axis=1)

            grid_spec["grid_definition_categorical"] = (
                df[grid_vars + ["grid_cell"]].drop_duplicates().sort_values(grid_vars)
            )
            grid_spec["grid_definition"] = grid_spec["grid_definition_categorical"].astype(
                str
            )
            var_info["grid_cell"] = grid_spec

        for random_effect in model_spec.random_effects:
            if random_effect not in df:
                df[random_effect] = raw_model_data[random_effect]

        df[model_spec.measure] = raw_model_data[model_spec.measure]

        return df, var_info

    @property
    def models(self) -> Path:
        return self.root / "models"

    def new_model_version(self) -> str:
        run_directory = get_run_directory(self.models)
        mkdir(run_directory)
        return run_directory.name

    def save_model_specification(
        self, model_spec: ModelSpecification, version: str
    ) -> None:
        model_root = self.models / version
        model_spec_path = model_root / "specification.yaml"
        touch(model_spec_path)
        model_spec.to_yaml(model_spec_path)

    def load_model_specification(self, version: str) -> ModelSpecification:
        model_spec_path = self.models / version / "specification.yaml"
        return ModelSpecification.from_yaml(model_spec_path)

    def save_results_specification(self, results_spec: ResultsSpecification) -> None:
        model_root = self.results / results_spec.version.results
        results_spec_path = model_root / "results_spec.yaml"
        touch(results_spec_path)
        results_spec.to_yaml(results_spec_path)

    def load_results_specification(self, version: str) -> ResultsSpecification:
        results_spec_path = self.results / version / "results_spec.yaml"
        return ResultsSpecification.from_yaml(results_spec_path)

    SUBMODEL_VARIABLE_SEPARATOR = "___"
    SUBMODEL_VALUE_SEPARATOR = "__"

    def save_model(
        self,
        model: "Lmer",
        version: str,
        submodel: list[tuple[str, str]] | None = None,
    ) -> None:
        model_root = self.models / version
        mkdir(model_root, exist_ok=True)
        if submodel:
            submodel_str = self.SUBMODEL_VARIABLE_SEPARATOR.join(
                [
                    f"{name}{self.SUBMODEL_VALUE_SEPARATOR}{value}"
                    for name, value in submodel
                ]
            )
            model_filename_base = f"{submodel_str}"
        else:
            model_filename_base = "base_model"
        model_filepath = model_root / (model_filename_base + ".pkl")
        touch(model_filepath, exist_ok=True)
        with model_filepath.open("wb") as f:
            pickle.dump(model, f)
        
        coefs_filepath = model_root / (model_filename_base + "_coefs.parquet")
        touch(coefs_filepath, exist_ok=True)
        model.coefs.to_parquet(coefs_filepath)

        random_effects_filepath = model_root / (model_filename_base + "_ranef.parquet")
        touch(random_effects_filepath, exist_ok=True)
        model.ranef.to_parquet(random_effects_filepath)

    
    def load_model_family(
        self,
        version: str,
    ) -> list[dict[str, typing.Any]]:
        models = []
        model_id_dir = self.models / version

        filepaths = model_id_dir.glob("*.pkl")
        for filepath in filepaths:
            model_dict = {}

            if filepath.stem == "base_model":
                # No submodels
                submodel_def = []
            else:
                submodel_def = [
                    tuple(var_str.split(self.SUBMODEL_VALUE_SEPARATOR))
                    for var_str in filepath.stem.split(self.SUBMODEL_VARIABLE_SEPARATOR)
                ]
            print(submodel_def)
            for var_name, var_value in submodel_def:
                model_dict[var_name] = var_value

            with filepath.open("rb") as f:
                model_dict["model"] = pickle.load(f)  # noqa: S301

            models.append(model_dict)

        return models

    def load_submodel(
        self, version: str, submodel: list[tuple[str, typing.Any]] | None = None
    ) -> "Lmer":
        model_id_dir = self.models / version
        possible_submodel_strs = ["base_model"]
        # We don't require submodel variables to be in a specific order, so we need to check all permutations
        if submodel:
            submodel = [(name, str(value)) for name, value in submodel]
            possible_submodel_strs += [
                self.SUBMODEL_VARIABLE_SEPARATOR.join(
                    [
                        f"{name}{self.SUBMODEL_VALUE_SEPARATOR}{value}"
                        for name, value in perm
                    ]
                )
                for perm in itertools.permutations(submodel)
            ]
        for submodel_str in possible_submodel_strs:
            model_filepath = model_id_dir / f"{submodel_str}.pkl"
            if model_filepath.exists():
                with model_filepath.open("rb") as f:
                    model = pickle.load(f)  # noqa: S301
                return model
        message = f"Model for subm  odel {submodel} not found."
        raise FileNotFoundError(message)

    def load_submodel_coefficients(
        self, version: str, submodel: list[tuple[str, typing.Any]] | None = None
    ) -> "Lmer":
        model_id_dir = self.models / version
        possible_submodel_strs = ["base_model"]
        # We don't require submodel variables to be in a specific order, so we need to check all permutations
        if submodel:
            submodel = [(name, str(value)) for name, value in submodel]
            possible_submodel_strs += [
                self.SUBMODEL_VARIABLE_SEPARATOR.join(
                    [
                        f"{name}{self.SUBMODEL_VALUE_SEPARATOR}{value}"
                        for name, value in perm
                    ]
                )
                for perm in itertools.permutations(submodel)
            ]
        for submodel_str in possible_submodel_strs:
            model_coefs_filepath = model_id_dir / f"{submodel_str}_coefs.parquet"
            model_ranef_filepath = model_id_dir / f"{submodel_str}_ranef.parquet"
            if model_coefs_filepath.exists() and model_ranef_filepath.exists():
                model_coefs = pd.read_parquet(model_coefs_filepath)['Estimate']
                model_ranef = pd.read_parquet(model_ranef_filepath)
                return model_coefs, model_ranef
        message = f"Model coefficients for sub model {submodel}  at {model_coefs_filepath} not found."
        raise FileNotFoundError(message)


    def load_model_variable_info(
        self, version: str) -> dict[str, typing.Any]:
        model_spec = self.load_model_specification(version)
        full_training_data = self.load_training_data(model_spec.version.training_data)
        full_training_data = full_training_data.reset_index(drop=True)
        full_training_data["intercept"] = 1.0
        raw_df = full_training_data.loc[:, model_spec.raw_variables]
        _, var_info = self.prepare_model_data(raw_df, model_spec)

        return var_info

    @property
    def results(self) -> Path:
        return self.root / "results"

    def new_results_version(self, 
        model_version: str,
        age_groups: list[int],
        sex_ids: list[int],
        years: list[int], 
        scenarios: list[str],
        draws: int) -> str:
        run_directory = get_run_directory(self.results)
        mkdir(run_directory)
        # create results specification file
        self.save_results_specification(
            ResultsSpecification(
                version=ResultsVersionSpecification(
                    model=model_version, results=run_directory.name
                ),
                draws=draws,
                age_groups=age_groups,
                scenarios=scenarios,
                years=years,
                sex_ids = sex_ids,
            )
        )
        return run_directory.name

    def raster_results_path(
        self,
        results_version: str,
        scenario: str,
        year: str | int,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> Path:
        return (
            self.results
            / results_version
            / f"{year}_{scenario}_{age_group_id}_{sex_id}.tif"
        )

    def save_raster_results(
        self,
        results: rt.RasterArray,
        results_version: str,
        scenario: str,
        year: str | int,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> None:
        path = self.raster_results_path(
            results_version, scenario, year, age_group_id, sex_id
        )
        mkdir(path.parent, parents=True, exist_ok=True)
        save_raster(results, path)

    def save_results_table(
        self,
        results: pd.DataFrame,
        model_version: str,
        scenario: str,
        year: str | int,
        age_group_id: str | int,
        sex_id: str | int,
        draw: int,
    ) -> None:
        path = self.results / model_version / f"{year}_{scenario}_{sex_id}_{age_group_id}_{draw}.parquet"
        touch(path, exist_ok=True)
        results.to_parquet(path)
    
    def load_results_table(
        self,
        model_version: str,
        scenarios: list[str],
        years: list[str | int],
        age_group_ids: list[str | int],
        sex_ids: list[str | int],
        draws: int,
    ) -> pd.DataFrame:
        dfs = []
        for scenario in scenarios:
            for year in years:
                for age_group_id in age_group_ids:
                    for sex_id in sex_ids:
                        for draw in range(0, draws):
                            path = self.results / model_version / \
                                f"{year}_{scenario}_{sex_id}_{age_group_id}_{draw}.parquet"
                            dfs.append(pd.read_parquet(path))
        return pd.concat(dfs)

    def save_forecast(
        self,
        forecast: pd.DataFrame,
        results_version: str,
    ) -> None:
        path = self.results / results_version / "forecast.parquet"
        touch(path, exist_ok=True)
        forecast.to_parquet(path)

    def load_forecast(
        self,
        results_version: str,
    ) -> pd.DataFrame:
        path = self.results / results_version / "forecast.parquet"
        return pd.read_parquet(path)

    @property
    def shared_inputs(self) -> Path:
        return self.root.parent / "input"

    def load_ldi_distributions(self, geospecificity: str, version: str) -> pd.DataFrame:
        if geospecificity != "national" and geospecificity != "admin2":
            error_message = f"geospecificity must be 'national' or 'admin2', not {geospecificity}"
            raise ValueError(error_message)

        path = self.shared_inputs / "ldi" / version / f"{geospecificity}_estimates.parquet"
        return pd.read_parquet(path)

    def ldi_raster_path(self, scenario: int | str, year: int | str, percentile: float | str, version: str) -> Path:
        return self.shared_inputs / "ldi_raster" / version / str(scenario) / f"{year}_{percentile}.tif"

    def load_ldi_raster(self, scenario: int | str, year: int | str, percentile: float | str, version: str) -> rt.RasterArray:
        # Temporary: we don't actually use the scenarios for income/consumption so just use reference/4.5
        return rt.load_raster(self.ldi_raster_path(0, year, percentile, version)).astype(np.float32)

    def save_ldi_raster(
        self,
        ldi: rt.RasterArray,
        scenario: int | str,
        year: int | str,
        percentile: float | str,
        version: str,
    ) -> None:
        path = self.ldi_raster_path(scenario, year, percentile, version)
        mkdir(path.parent, parents=True, exist_ok=True)
        save_raster(ldi, path)

    def rasterized_variable_path(self, variable: str, year: int | str) -> Path:
        return self.shared_inputs / variable / f"{year}.tif"

    def load_rasterized_variable(
        self, variable: str, year: int | str
    ) -> rt.RasterArray:
        return rt.load_raster(self.rasterized_variable_path(variable, year)).astype(np.float32)

    def save_rasterized_variable_raster(
        self,
        variable_name: str,
        variable_raster: rt.RasterArray,
        year: int | str,
    ) -> None:
        path = self.rasterized_variable_path(variable_name, year)
        mkdir(path.parent, parents=True, exist_ok=True)
        save_raster(variable_raster, path)
    
    def rasterized_intercept_path(self, model_version: str) -> Path:
        return self.models / model_version / "intercept" / "intercept.tif"

    def load_rasterized_intercept(
        self, model_version: str,
    ) -> rt.RasterArray:
        return rt.load_raster(self.rasterized_intercept_path(model_version)).astype(np.float32)

    def save_rasterized_intercept(
        self,
        model_version: str,
        variable_raster: rt.RasterArray,
        **kwargs: dict[str, typing.Any],
    ) -> None:
        path = self.rasterized_intercept_path(model_version)
        mkdir(path.parent, parents=True, exist_ok=True)
        save_raster(variable_raster, path, **kwargs)

    def load_elevation(self) -> rt.RasterArray:
        return rt.load_raster(self.shared_inputs / "GLOBE_DEM_MOSAIC_Y2016M02D09.TIF").set_no_data_value(-32768).astype(np.float32)

    #########################
    # Upstream paths we own #
    #########################

    _POP_DATA_ROOT = Path("/mnt/team/rapidresponse/pub/population/data")
    _RAW_DATA_ROOT = _POP_DATA_ROOT / "01-raw-data"
    _PROCESSED_DATA_ROOT = _POP_DATA_ROOT / "02-processed-data"
    _CLIMATE_DATA_ROOT = Path("/mnt/share/erf/climate_downscale/results/annual")

    def save_lbd_admin2_shapes(self, gdf: gpd.GeoDataFrame) -> None:
        path = self._PROCESSED_DATA_ROOT / "ihme" / "lbd_admin2.parquet"
        touch(path, exist_ok=True)
        gdf.to_parquet(path)

    def load_lbd_admin2_shapes(self) -> gpd.GeoDataFrame:
        path = self._PROCESSED_DATA_ROOT / "ihme" / "lbd_admin2.parquet"
        return gpd.read_parquet(path)

    def save_fhs_shapes(self, gdf: gpd.GeoDataFrame) -> None:
        path = self._PROCESSED_DATA_ROOT / "ihme" / "fhs_most_detailed.parquet"
        touch(path, exist_ok=True)
        gdf.to_parquet(path)

    def load_fhs_shapes(self, *, most_detailed_only: bool = True) -> gpd.GeoDataFrame:
        path = self._PROCESSED_DATA_ROOT / "ihme" / "fhs_most_detailed.parquet"
        fhs_shapes = gpd.read_parquet(path)

        hierarchy_path = self._PROCESSED_DATA_ROOT / "ihme" / "fhs_hierarchy.parquet"
        hierarchy = pd.read_parquet(hierarchy_path)
        most_detailed_locs = hierarchy.loc[
            hierarchy.most_detailed == 1, "location_id"
        ].tolist()

        fhs_shapes["most_detailed"] = 0
        fhs_shapes.loc[fhs_shapes.loc_id.isin(most_detailed_locs), "most_detailed"] = 1

        if most_detailed_only:
            return fhs_shapes[fhs_shapes.most_detailed == 1].reset_index()
        return fhs_shapes

    def load_fhs_hierarchy(self) -> pd.DataFrame:
        path = self._PROCESSED_DATA_ROOT / "ihme" / "fhs_hierarchy.parquet"
        hierarchy = pd.read_parquet(path)
        return hierarchy

    def load_raster_template(self) -> rt.RasterArray:
        path = (
            self._RAW_DATA_ROOT
            / "other-gridded-pop-projects"
            / "global-human-settlement-layer"
            / "1km_template.tif"
        )
        return rt.load_raster(path).astype(np.float32)
    
    def load_raster_template_spec(self) -> dict[str, typing.Any]:
        raster_template = self.load_raster_template()
        return {
            "shape": raster_template.shape,
            "transform": raster_template.transform,
            "crs": raster_template.crs,
        }

    def load_population_raster(self) -> rt.RasterArray:
        path = (
            self._RAW_DATA_ROOT
            / "other-gridded-pop-projects"
            / "global-human-settlement-layer"
            / "1km_population.tif"
        )
        return rt.load_raster(path).set_no_data_value(np.nan).astype(np.float32)

    def load_climate_raster(
        self, variable: str, scenario: str, year: int | str, draw: int,
    ) -> xr.DataArray:
        if scenario == "constant_climate":
            scenario = "ssp245"
        path = (
            self._CLIMATE_DATA_ROOT / scenario / variable / f"{draw:03}.nc"
        )
        return xr.open_dataset(path).sel(year=year)["value"]


def get_run_directory(output_root: str | Path) -> Path:
    """Gets a path to a datetime directory for a new output.

    Parameters
    ----------
    output_root
        The root directory for all outputs.

    """
    output_root = Path(output_root).resolve()
    launch_time = datetime.datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d")
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
    **kwargs: typing.Any,
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
