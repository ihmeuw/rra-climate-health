from enum import StrEnum

from pydantic import BaseModel


class BinCategory(StrEnum):
    HOUSEHOLD = 'household'
    LOCATION = 'location'
    COUNTRY = 'country'


class BinningSpecification(BaseModel):
    category: BinCategory
    bin_strategy: str

