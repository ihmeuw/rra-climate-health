import pandas as pd
import numpy as np


def get_bins(
    data: pd.Series,
    nbins: int,
    bin_strategy: str = 'quantiles',
) -> list[float] | list[int]:
    qcut_kwargs = {
        "duplicates": "drop",
        "precision": 3 if nbins > 10 else 1,
        "retbins": True,
    }
    if bin_strategy == 'quantiles':
        _, bins = pd.qcut(data, q=nbins, **qcut_kwargs)
    elif bin_strategy == 'equal':
        _, bins = pd.cut(data, bins=nbins, include_lowest=True, retbins=True, right=False)
    elif bin_strategy == 'readable_5':
        bins = [data.min(), 20] + list(range(22, 2+int(np.ceil(data.max())), 2))
    elif bin_strategy == '0_1_more':
        _ , otherbins = pd.qcut(data[data > 1], q=nbins-2, **qcut_kwargs)
        otherbins[-1] = otherbins[-1]+1
        bins = [0, 1] + list(otherbins)
    elif bin_strategy == '0_more':
        _ , otherbins = pd.qcut(data[data > 1], q=nbins-2, **qcut_kwargs)
        otherbins[-1] = otherbins[-1]+1
        bins = [0,] + list(otherbins)
    elif bin_strategy == '0_more_readable':
        bin_interval = 10
        bins = [0, 1] + list(range(bin_interval, int((np.ceil((data.max()/bin_interval))+1)*bin_interval), bin_interval))
    elif bin_strategy == 'custom_daysover':
        bins = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 250]
    else:
        raise ValueError()

    return bins

def group_and_bin_column_definition(
    df: pd.DataFrame,
    bin_col: str,
    bin_category: str,
    nbins: int,
    bin_strategy: str = "quantiles",
) -> tuple[list[float] | list[int], pd.DataFrame]:
    group_cols = {
        "household": ['nid', 'hh_id', 'psu', 'year_start'],
        "location": ['lat', 'long'],
        "country": ['iso3'],
    }[bin_category]
    result_column = bin_col + '_bin'
    grouped_df = df.groupby(group_cols + [bin_col], as_index=False).size().drop(columns=['size'])
    bins = get_bins(grouped_df[bin_col], nbins, bin_strategy)
    grouped_df[result_column] = pd.cut(
        grouped_df[bin_col],
        bins=bins,
        include_lowest=True,
        right=False,
    )
    return bins, pd.merge(df, grouped_df, how='left')
