from pathlib import Path
import typing

import dill as pickle
import pandas as pd
from rra_tools.shell_tools import mkdir, touch


if typing.TYPE_CHECKING:
    from pymer4 import Lmer

DEFAULT_ROOT = "/mnt/team/rapidresponse/pub/population/modeling/climate_malnutrition"


class ClimateMalnutritionData:

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def models(self) -> Path:
        return self.root / "models"

    def save_model(
        self,
        model: "Lmer",
        model_id: str,
        measure: str,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> None:
        model_id_dir = self.models / model_id
        mkdir(model_id_dir, exist_ok=True)
        model_filepath = model_id_dir / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
        touch(model_filepath, exist_ok=True)
        with model_filepath.open("wb") as f:
            pickle.dump(model, f)

    def load_model(
        self,
        model_id: str,
        measure: str,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> "Lmer":
        model_id_dir = self.models / model_id
        model_filepath = model_id_dir / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
        with model_filepath.open("rb") as f:
            model = pickle.load(f)
        return model


    @property
    def results(self) -> Path:
        return self.root / "results"

    def save_results(
        self,
        results: pd.DataFrame,
        model_id: str,
        measure: str,
        scenario: str,
        year: str | int,
        age_group_id: str | int,
        sex_id: str | int,
    ) -> None:
        model_id_dir = self.results / model_id
        mkdir(model_id_dir, exist_ok=True)
        file_name = f"{measure}_{age_group_id}_{sex_id}_{scenario}_{year}.parquet"
        results_filepath = model_id_dir / file_name
        touch(results_filepath, exist_ok=True)
        results.to_parquet(results_filepath)

