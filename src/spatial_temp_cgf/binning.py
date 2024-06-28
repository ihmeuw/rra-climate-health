import pandas as pd
import numpy as np

from spatial_temp_cgf.model_specification import (
    BinningSpecification,
    BinningCategory,
    BinningStrategy,
)


def get_qcut_bins(data: pd.Series, q: int, precise: bool) -> np.ndarray:
    qcut_kwargs = {
        "duplicates": "drop",
        "precision": 3 if precise else 1,
        "retbins": True,
    }
    _, bins = pd.qcut(data, q=q, **qcut_kwargs)
    return bins


def quantile_bins(data: pd.Series, nbins: int) -> np.ndarray:
    return get_qcut_bins(data, q=nbins, precise=nbins > 10)


def equal_bins(data: pd.Series, nbins: int) -> np.ndarray:
    return pd.cut(data, bins=nbins, include_lowest=True, retbins=True, right=False)[1]


def readable_5_bins(data: pd.Series, nbins: int) -> np.ndarray:
    return np.array([data.min(), 20] + list(range(22, 2+int(np.ceil(data.max())), 2)))


def zero_one_more_bins(data: pd.Series, nbins: int) -> np.ndarray:
    _ , otherbins = get_qcut_bins(data[data > 1], q=nbins-2, precise=nbins > 10)
    otherbins[-1] = otherbins[-1] + 1
    return np.array([0, 1] + list(otherbins))


def zero_more_bins(data: pd.Series, nbins: int) -> np.ndarray:
    _ , otherbins = get_qcut_bins(data[data > 1], q=nbins-1, precise=nbins > 10)
    otherbins[-1] = otherbins[-1] + 1
    return np.array([0,] + list(otherbins))


def zero_more_readable_bins(data: pd.Series, nbins: int) -> np.ndarray:
    bin_interval = 10
    max_bin = int((np.ceil(data.max()/bin_interval) + 1) * bin_interval)
    return np.array([0, 1] + list(range(bin_interval, max_bin, bin_interval)))


def custom_daysover_bins(data: pd.Series, nbins: int) -> np.ndarray:
    return np.array([0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 250])


BINNING_STRATEGIES = {
    BinningStrategy.QUANTILES: quantile_bins,
    BinningStrategy.EQUAL: equal_bins,
    BinningStrategy.READABLE_5: readable_5_bins,
    BinningStrategy.ZERO_ONE_MORE: zero_one_more_bins,
    BinningStrategy.ZERO_MORE: zero_more_bins,
    BinningStrategy.ZERO_MORE_READABLE: zero_more_readable_bins,
    BinningStrategy.CUSTOM_DAYSOVER: custom_daysover_bins,
}


def bin_column(
    df: pd.DataFrame,
    column: str,
    spec: BinningSpecification,
) -> tuple[pd.Series, dict]:
    # Why are we grouping anything here?
    grouped_df = (
        df.groupby(spec.groupby_columns + [column], as_index=False)
        .size()
        .drop(columns=['size'])
    )

    bins = BINNING_STRATEGIES[spec.strategy](grouped_df[column], spec.nbins)
    result_column = column + '_bin'
    grouped_df[result_column] = pd.cut(
        grouped_df[column],
        bins=bins,
        include_lowest=True,
        right=False,
    )
    # Hack to keep the shape right
    df = pd.merge(df, grouped_df, how='left')
    binned_column = df[result_column].rename(column)

    bins_categorical = pd.DataFrame({
        result_column: binned_column.unique().sort_values()
    })
    bin_info = {
        'bin_edges': bins,
        'bins_categorical': bins_categorical,
        'bins': bins_categorical.astype(str)
    }

    return binned_column, bin_info


