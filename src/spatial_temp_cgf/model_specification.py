from pathlib import Path
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator
import yaml


class ScalingStrategy(StrEnum):
    IDENTITY = 'identity'
    MIN_MAX = 'min_max'


class ScalingSpecification(BaseModel):
    type: Literal['scaling']
    strategy: ScalingStrategy = ScalingStrategy.IDENTITY


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
    type: Literal['binning']
    strategy: BinningStrategy
    category: BinningCategory
    nbins: int = Field(10, gt=0)


class OutcomeVariable(StrEnum):
    WASTING = 'wasting'
    STUNTING = 'stunting'
    UNDERWEIGHT = 'underweight'
    LOW_BMI = 'low_bmi'
    ANEMIA = 'anemia'


class PredictorSpecification(BaseModel):
    name: str = "intercept"
    transform: BinningSpecification | ScalingSpecification = Field(
        ScalingSpecification(),
        discriminator='transformation_type',
    )
    random_effect: str = ""


class GridSpecification(BaseModel):
    x: PredictorSpecification
    y: PredictorSpecification
    name: str = "grid_cell"
    random_effect: str = ""

    @model_validator(mode="after")
    def check_no_predictor_random_effects(cls, v):
        if v.x.random_effect:
            msg = "Grid predictor x cannot have a random effect"
            raise ValueError(msg)
        if v.y.random_effect:
            msg = "Grid predictor y cannot have a random effect"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def check_transform_is_binning(cls, v):
        if v.x.transform.transformation_type != 'binning':
            msg = "Grid predictor x must be binned"
            raise ValueError(msg)
        if v.y.transform.transformation_type != 'binning':
            msg = "Grid predictor y must be binned"
            raise ValueError(msg)
        return v

    @property
    def raw_variables(self) -> list[str]:
        return [self.x.name, self.y.name]

    @property
    def transform_map(self) -> dict[str, BinningSpecification]:
        return {
            self.x.name: self.x.transform,
            self.y.name: self.y.transform,
        }

    @property
    def grid_spec(self) -> dict[str, list[str]]:
        return {'grid_order': self.raw_variables}


class HoldoutType(StrEnum):
    random = 'random'
    no_holdout = 'no_holdout'
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
    predictors: list[PredictorSpecification] = Field(default_factory=list)
    grid_predictors: GridSpecification | None = None

    @property
    def raw_variables(self) -> list[str]:
        variables = [self.measure]
        variables += [predictor.name for predictor in self.predictors]

        if self.grid_predictors:
            variables += self.grid_predictors.raw_variables
        return variables

    @property
    def random_effects(self) -> list[str]:
        random_effects = [predictor.random_effect for predictor in self.predictors]
        if self.grid_predictors:
            random_effects.append(self.grid_predictors.random_effect)
        return [re for re in random_effects if re]

    @property
    def transform_map(self) -> dict[str, BinningSpecification | ScalingSpecification]:
        transform_map = {
            predictor.name: predictor.transform for predictor in self.predictors
        }
        if self.grid_predictors:
            transform_map.update(self.grid_predictors.transform_map)
        return transform_map

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

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'ModelSpecification':
        with Path(yaml_path).open('r') as f:
            yaml_dict = yaml.safe_load(f)
        return ModelSpecification.parse_obj(yaml_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        with Path(yaml_path).open('w') as f:
            yaml.dump(self.model_dump(mode='json'), f)
