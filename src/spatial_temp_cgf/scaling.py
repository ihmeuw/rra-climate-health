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

def min_max_scale_to_range(to_scale, input_min, input_max):
    return (to_scale - input_min) / (input_max - input_min)


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
            data = min_max_scale_to_range(df[column], info['feature_range'][0], info['feature_range'][1])
            info = info
            #scaler.set_params(**info) #Not actually what we want, we want it to be able to go beyond the scale I think
        else:
            scaler.fit(df[[column]])
            info = {'feature_range' : (scaler.data_min_[0], scaler.data_max_[0])}
            data = pd.Series(scaler.transform(df[[column]])[:, 0], name=column)
    elif spec.strategy == ScalingStrategy.STANDARDIZE:
        if info:
            data = (df[column] - info['mean']) / info['std']
        else:
            info = {'mean': df[column].mean(), 'std': df[column].std()}
            data = (df[column] - df[column].mean()) / df[column].std()
    else:
        msg = f"Unknown scaling strategy {spec.strategy}"
        raise ValueError(msg)
    return data, info
