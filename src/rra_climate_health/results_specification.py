from pathlib import Path

import yaml
from pydantic import BaseModel


class ResultsVersionSpecification(BaseModel):
    model: str
    results: str
    draws: int = 1


class ResultsSpecification(BaseModel):
    version: ResultsVersionSpecification

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ResultsSpecification":
        with Path(yaml_path).open("r") as f:
            yaml_dict = yaml.safe_load(f)
        return ResultsSpecification.parse_obj(yaml_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        with Path(yaml_path).open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)
