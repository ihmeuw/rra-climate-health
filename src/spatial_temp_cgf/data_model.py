from enum import StrEnum

from pydantic import BaseModel, Field


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
    binning_specification: BinningSpecification | None = None
    random_effect: str | None = None


class GridSpecification(BaseModel):
    grid_predictor_x: PredictorSpecification
    grid_predictor_y: PredictorSpecification
    random_effect: str | None = None


class HoldoutType(StrEnum):
    random = 'random'
    temporal = 'temporal'
    spatial = 'spatial'


class HoldoutSpecification(BaseModel):
    holdout_type: HoldoutType = HoldoutType.random
    holdout_fraction: float = 0.2


class ModelSpecification(BaseModel):
    model_id: str
    response_measure: OutcomeVariable
    holdout: HoldoutSpecification = HoldoutSpecification()
    grid_predictors: GridSpecification | None = None
    other_predictors: dict[str, PredictorSpecification] = Field(default_factory=dict)






