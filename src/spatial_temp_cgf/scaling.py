import pandas as pd

from sklearn.preprocessing import MinMaxScaler, Binarizer

from spatial_temp_cgf.model_specification import (
    ScalingSpecification,
    ScalingStrategy,
)


def identity(data: pd.Series) -> pd.Series:
    return data


def min_max(data: pd.Series) -> pd.Series:
    return (data - data.min()) / (data.max() - data.min())


def scale_column(
    df: pd.DataFrame,
    column: str,
    spec: ScalingSpecification,
    info: dict | None = None,
) -> tuple[pd.Series, dict]:
    if spec.strategy == ScalingStrategy.IDENTITY:
        data = df[column].copy()
        info = {}
    elif spec.strategy == ScalingStrategy.MIN_MAX:
        scaler = MinMaxScaler()
        if info:
            scaler.set_params(**info)
        else:
            scaler.fit(df[[column]])
            info = scaler.get_params()
        data = pd.Series(scaler.transform(df[[column]])[:, 0], name=column)
    else:
        msg = f"Unknown scaling strategy {spec.strategy}"
        raise ValueError(msg)
    return data, info
