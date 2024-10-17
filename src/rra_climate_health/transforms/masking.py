from typing import Any

import numpy.typing as npt
import pandas as pd

from rra_climate_health.model_specification import MaskingSpecification


class Masker:
    def __init__(self, spec: MaskingSpecification):
        self._threshold = spec.threshold

    def __call__(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return (data > self._threshold).astype(float)


def mask_column(
    df: pd.DataFrame,
    column: str,
    spec: MaskingSpecification,
) -> tuple["pd.Series[bool | int | float]", Masker]:
    masker = Masker(spec)
    masked = pd.Series(
        masker(df[spec.from_column].to_numpy().reshape(-1, 1)).flatten(),
        index=df.index,
        name=column,
    )
    return masked, masker
