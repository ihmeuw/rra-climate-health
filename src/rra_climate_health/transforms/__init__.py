import pandas as pd

from rra_climate_health.model_specification import TransformSpecification
from rra_climate_health.transforms.binning import Binner, bin_column
from rra_climate_health.transforms.masking import Masker, mask_column
from rra_climate_health.transforms.scaling import Scaler, scale_column


def transform_column(
    df: pd.DataFrame,
    column: str,
    spec: TransformSpecification,
) -> tuple[pd.Series[int | float], Binner | Scaler | Masker]:
    if spec.type == "binning":
        return bin_column(df, column, spec)
    elif spec.type == "scaling":
        return scale_column(df, column, spec)
    elif spec.type == "masking":
        return mask_column(df, column, spec)
    elif spec.type == 'categorical':
        return df[column], None
    else:
        msg = f"Unknown transformation type {spec.type}"
        raise ValueError(msg)
