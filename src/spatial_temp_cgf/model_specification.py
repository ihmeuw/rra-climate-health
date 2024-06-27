from pathlib import Path
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator
import yaml


class BinningCategory(StrEnum):
    HOUSEHOLD = 'household'
    LOCATION = 'location'
    COUNTRY = 'country'


class BinningStrategy(StrEnum):
    QUANTILES = 'quantiles'
    EQUAL = 'equal'
    READABLE_5 = 'readable_5'
    ZERO_ONE_MORE = '0_1_more'
    ZERO_MORE = '0_more'
    ZERO_MORE_READABLE = '0_more_readable'
    CUSTOM_DAYSOVER = 'custom_daysover'


class BinningSpecification(BaseModel):
    category: BinningCategory
    strategy: BinningStrategy
    nbins: int = Field(10, gt=0)


class OutcomeVariable(StrEnum):
    WASTING = 'wasting'
    STUNTING = 'stunting'
    UNDERWEIGHT = 'underweight'
    LOW_BMI = 'low_bmi'
    ANEMIA = 'anemia'


class PredictorSpecification(BaseModel):
    name: str
    binning: BinningSpecification | None = None
    random_effect: str | None = None


class GridSpecification(BaseModel):
    grid_predictor_x: PredictorSpecification
    grid_predictor_y: PredictorSpecification
    name: str = "grid_cell"
    random_effect: str | None = None

    @model_validator(mode="after")
    def check_no_predictor_random_effects(cls, v):
        if v.grid_predictor_x.random_effect is not None:
            msg = "Grid predictor x cannot have a random effect"
            raise ValueError(msg)
        if v.grid_predictor_y.random_effect is not None:
            msg = "Grid predictor y cannot have a random effect"
            raise ValueError(msg)
        return v


class HoldoutType(StrEnum):
    random = 'random'
    # To implement
    # temporal = 'temporal'
    # spatial = 'spatial'


class HoldoutSpecification(BaseModel):
    type: HoldoutType = HoldoutType.random
    proportion: float = 0.2
    seed: int = 42


class VersionSpecification(BaseModel):
    training_data: str
    model: str | None = None


class ModelSpecification(BaseModel):
    version: VersionSpecification
    measure: OutcomeVariable
    holdout: HoldoutSpecification = HoldoutSpecification()
    grid_predictors: GridSpecification | None = None
    other_predictors: list[PredictorSpecification] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'ModelSpecification':
        with Path(yaml_path).open('r') as f:
            yaml_dict = yaml.safe_load(f)
        return ModelSpecification.parse_obj(yaml_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        with Path(yaml_path).open('w') as f:
            yaml.dump(self.model_dump(mode='json'), f)

    @property
    def lmer_formula(self) -> str:
        formula = f"{self.response_measure.value} ~"

        predictors = [self.grid_predictors] if self.grid_predictors else []
        predictors += self.other_predictors
        for predictor in predictors:
            if predictor.random_effect:
                var_name = "1" if predictor.name == "intercept" else predictor.name
                formula += f" ({var_name} | {predictor.random_effect}) +"
            else:
                formula += f" {predictor.name} +"
        formula = formula.rstrip(" +")
        return formula


