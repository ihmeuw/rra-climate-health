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


class ResponseMeasure(StrEnum):
    WASTING = 'wasting'
    STUNTING = 'stunting'
    UNDERWEIGHT = 'underweight'
    LOW_BMI = 'low_bmi'
    ANEMIA = 'anemia'


class PredictorSpecification(BaseModel):
    binning_specification: BinningSpecification | None = None


class GridSpecification(BaseModel):
    grid_predictor_x: PredictorSpecification
    grid_predictor_y: PredictorSpecification


class ModelSpecification(BaseModel):
    model_id: str
    response_measure: ResponseMeasure




