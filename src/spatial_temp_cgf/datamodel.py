from enum import StrEnum

from pydantic import BaseModel

class LabelStrategy(StrEnum):
    means = "means"


class BinStrategy(StrEnum):
    quantiles = "quantiles"
    equal = "equal"
    readable_5 = "readable_5"
    more_0_1 = "more_0_1"
    more_0 = "more_0"


class BinningSpecification(BaseModel):
    bin_category: str
    bin_strategy: str

