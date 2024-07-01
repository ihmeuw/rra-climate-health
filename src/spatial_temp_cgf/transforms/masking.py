import numpy as np
import pandas as pd

from spatial_temp_cgf.model_specification import MaskingSpecification


class Masker:

    def __init__(self, spec: MaskingSpecification):
        self._threshold = spec.threshold

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return data > self._threshold


def mask_column(
    df: pd.DataFrame,
    column: str,
    spec: MaskingSpecification,
) -> tuple[pd.Series, Masker]:
    masker = Masker(spec)
    masked = pd.Series(
        masker(df[spec.from_column].values.reshape(-1, 1)).flatten(),
        index=df.index,
        name=column,
    )
    return masked, masker

