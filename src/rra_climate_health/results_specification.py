from pathlib import Path

import yaml
from pydantic import BaseModel

from rra_climate_health.model_specification import ModelSpecification


class ResultsVersionSpecification(BaseModel):
    model: str
    results: str


class ResultsSpecification(BaseModel):
    version: ResultsVersionSpecification

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ModelSpecification":
        with Path(yaml_path).open("r") as f:
            yaml_dict = yaml.safe_load(f)
        return ModelSpecification.parse_obj(yaml_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        with Path(yaml_path).open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)
