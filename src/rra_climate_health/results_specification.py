from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class ResultsVersionSpecification(BaseModel):
    model: str
    results: str

class ResultsSpecification(BaseModel):
    version: ResultsVersionSpecification
    draws: int = 1
    age_groups: list[int | str] = []
    sex_ids: list[int] = []
    scenarios: list[str] = []
    years: list[int] = []

    @field_validator("age_groups", mode="before")
    @classmethod
    def _numeric_age_groups_are_ints(cls, age_groups: object) -> object:
        """Coerce numeric age group IDs to int, leaving named strata alone.

        The field has to stay a union because ``child_mortality`` is modeled on named
        survival intervals (``age_1_m`` ... ``age_60_m``).  But pydantic's smart union
        keeps ``"238"`` as a ``str``, and every consumer of this spec -- the forecast
        step's population merge, the residual step's GBD query -- matches against int64
        ``age_group_id``.  Normalizing here means new specs are written with bare ints
        and older specs holding quoted numbers heal themselves on read.
        """
        if not isinstance(age_groups, list):
            return age_groups
        return [
            int(a) if isinstance(a, str) and a.isdigit() else a for a in age_groups
        ]

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ResultsSpecification":
        with Path(yaml_path).open("r") as f:
            yaml_dict = yaml.safe_load(f)
        return ResultsSpecification.parse_obj(yaml_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        with Path(yaml_path).open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)
