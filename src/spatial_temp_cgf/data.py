from pathlib import Path
import typing
import itertools

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

    def load_model_family(
        self,
        model_id: str,
        measure: str,
    ) -> list[dict]:
        models = []
        model_id_dir = self.models / model_id
        for age_group_id, sex_id in itertools.product([4, 5], [1, 2]):
            model_filepath = (
                model_id_dir / f'model_{measure}_{age_group_id}_{sex_id}.pkl'
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

    def results_path(
        self,
        model_id: str,
        location_id: str | int,
        measure: str,
        scenario: str,
        year: str | int,
    ) -> Path:
        return self.results / model_id / f"{measure}_{location_id}_{scenario}_{year}.parquet"

    def save_results(
        self,
        results: pd.DataFrame,
        model_id: str,
        location_id: str | int,
        measure: str,
        scenario: str,
        year: str | int,
    ) -> None:
        path = self.results_path(model_id, location_id, measure, scenario, year)
        mkdir(path.parent, parents=True, exist_ok=True)
        touch(path, exist_ok=True)
        results.to_parquet(path)

