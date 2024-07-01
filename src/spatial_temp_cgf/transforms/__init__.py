import pandas as pd

from spatial_temp_cgf.model_specification import TransformSpecification

from spatial_temp_cgf.transforms.binning import bin_column, Binner
from spatial_temp_cgf.transforms.scaling import scale_column, Scaler
from spatial_temp_cgf.transforms.masking import mask_column, Masker


def transform_column(
    df: pd.DataFrame,
    column: str,
    spec: TransformSpecification,
) -> tuple[pd.Series, Binner | Scaler | Masker]:
    if spec.type == 'binning':
        return bin_column(df, column, spec)
    elif spec.type == 'scaling':
        return scale_column(df, column, spec)
    elif spec.type == 'masking':
        return mask_column(df, column, spec)
    else:
        msg = f"Unknown transformation type {spec.type}"
        raise ValueError(msg)
