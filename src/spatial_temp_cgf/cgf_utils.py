import pandas as pd
import numpy as np


def bin_column(df, col, nbins, bin_strategy='quantiles', retbins = False):
    precision = 3 if nbins > 10 else 1
    drop_dupes = nbins > 10
    if bin_strategy == 'quantiles':
        binned, bins = pd.qcut(df[col], q=nbins, duplicates='drop', precision = precision, retbins=True)
    elif bin_strategy == 'equal':
        binned, bins = pd.cut(df[col], bins=nbins, include_lowest = True, retbins=True)
    elif bin_strategy == 'readable_5':
        bins = [df[col].min(), 20] + list(range(22, 2+int(np.ceil(df[col].max())), 2))#list(range(5*int(df[col].min() // 5), 5+5*int(df[col].max() // 5), 5))
        binned = pd.cut(df[col], bins = bins, include_lowest = True, right=False)
    elif bin_strategy == '0_1_more':
        _ , otherbins = pd.qcut(df.loc[df[col] > 1, col], q=nbins-2, retbins = True, duplicates = 'drop', precision = precision)
        otherbins[-1] = otherbins[-1]+1
        bins = [0,1] + list(otherbins)
        binned = pd.cut(df[col], bins=bins, include_lowest = True, right=False)
    elif bin_strategy == '0_more':
        _ , otherbins = pd.qcut(df.loc[df[col] >= 1, col], q=nbins-2, retbins = True, duplicates='drop', precision = precision)
        otherbins[-1] = otherbins[-1]+1
        bins = [0,] + list(otherbins)
        binned = pd.cut(df[col], bins=bins, include_lowest = True, right=False)
    elif bin_strategy == '0_more_readable':
        bin_interval = 10
        bins = [0,1] + list(range(bin_interval, int((np.ceil((df[col].max()/bin_interval))+1)*bin_interval), bin_interval))
        binned = pd.cut(df[col], bins = bins, include_lowest = True, right=False)
    elif bin_strategy == 'custom_daysover':
        bins = [0,1,5,10,20,30,40,50,60,70,80,90,100,110,120,250]
        binned = pd.cut(df[col], bins = bins, include_lowest = True, right=False)
    else:
        raise ValueError('bin_strategy must be either "quantiles" or "equal"')
    if retbins:
        return bins, binned
    return binned


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
    bins, binned_column = bin_column(grouped_df, bin_col, nbins, bin_strategy)
    grouped_df[result_column] = binned_column
    return bins, pd.merge(df, grouped_df, how='left')
