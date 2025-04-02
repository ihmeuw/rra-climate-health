from pathlib import Path

import yaml
from pydantic import BaseModel


class ResultsVersionSpecification(BaseModel):
    model: str
    results: str

class ResultsSpecification(BaseModel):
    version: ResultsVersionSpecification
    draws: int = 1
    age_groups: list[int] = []
    sex_ids: list[int] = []
    scenarios: list[str] = []
    years: list[int] = []

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ResultsSpecification":
        with Path(yaml_path).open("r") as f:
            yaml_dict = yaml.safe_load(f)
        return ResultsSpecification.parse_obj(yaml_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        with Path(yaml_path).open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)
