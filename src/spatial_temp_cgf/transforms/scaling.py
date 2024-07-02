import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)

from spatial_temp_cgf.model_specification import (
    ScalingSpecification,
    ScalingStrategy,
)


class IdentityScaler:

    def __init__(self):
        pass

    def fit(self, data: np.ndarray):
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data


class InnerNinetyFiveScaler:
    def __init__(self):
        self.two_point_five = None
        self.ninety_seven_point_five = None

    def fit(self, data: np.ndarray):
        self.two_point_five = np.percentile(data, 2.5)
        self.ninety_seven_point_five = np.percentile(data, 97.5)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.two_point_five) / (self.ninety_seven_point_five - self.two_point_five)


SCALING_STRATEGIES = {
    ScalingStrategy.IDENTITY: IdentityScaler,
    ScalingStrategy.MIN_MAX: MinMaxScaler,
    ScalingStrategy.STANDARDIZE: StandardScaler,
    ScalingStrategy.INNER_NINETY_FIVE: InnerNinetyFiveScaler,
}


class Scaler:

    def __init__(self, spec: ScalingSpecification):
        self._strategy = SCALING_STRATEGIES[spec.strategy]()
        self._fitted = False

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            self._strategy.fit(data)
            self._fitted = True
        # Scikit learn scalars can operate on multiple features at once.  We don't
        # want this.  We want the scalers to operate on 1d (df) and 2d (raster)
        # data for a single feature. Manipulate some internal state so we don't
        # raise errors here.
        self._strategy.n_features_in_ = data.shape[1]
        return self._strategy.transform(data)


def scale_column(
    df: pd.DataFrame,
    column: str,
    spec: ScalingSpecification,
) -> tuple[pd.Series, Scaler]:
    scaler = Scaler(spec)
    scaled = pd.Series(
        scaler(df[column].values.reshape(-1, 1)).flatten(),
        index=df.index,
        name=column,
    )
    return scaled, scaler
