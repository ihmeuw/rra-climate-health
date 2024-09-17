from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)

from rra_climate_health.model_specification import (
    ScalingSpecification,
    ScalingStrategy,
)


class IdentityScaler:
    def __init__(self) -> None:
        pass

    def fit(self, data: npt.NDArray[Any]) -> None:
        pass

    def transform(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return data


class InnerNinetyFiveScaler:
    def __init__(self) -> None:
        self.two_point_five = np.nan
        self.ninety_seven_point_five = np.nan

    def fit(self, data: npt.NDArray[Any]) -> None:
        self.two_point_five = np.percentile(data, 2.5)
        self.ninety_seven_point_five = np.percentile(data, 97.5)

    def transform(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return (data - self.two_point_five) / (
            self.ninety_seven_point_five - self.two_point_five
        )


SCALING_STRATEGIES = {
    ScalingStrategy.IDENTITY: IdentityScaler,
    ScalingStrategy.MIN_MAX: MinMaxScaler,
    ScalingStrategy.STANDARDIZE: StandardScaler,
    ScalingStrategy.INNER_NINETY_FIVE: InnerNinetyFiveScaler,
}


class Scaler:
    def __init__(self, spec: ScalingSpecification) -> None:
        self._strategy = SCALING_STRATEGIES[spec.strategy]()
        self._fitted = False

    def __call__(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if not self._fitted:
            self._strategy.fit(data)
            self._fitted = True
        # Scikit learn scalars can operate on multiple features at once.  We don't
        # want this.  We want the scalers to operate on 1d (df) and 2d (raster)
        # data for a single feature. Manipulate some internal state so we don't
        # raise errors here.
        self._strategy.n_features_in_ = data.shape[1]
        return self._strategy.transform(data)  # type: ignore[no-any-return]


def scale_column(
    df: pd.DataFrame,
    column: str,
    spec: ScalingSpecification,
) -> tuple[pd.Series[Any], Scaler]:
    scaler = Scaler(spec)
    scaled = pd.Series(
        scaler(df[column].to_numpy().reshape(-1, 1)).flatten(),
        index=df.index,
        name=column,
    )
    return scaled, scaler
