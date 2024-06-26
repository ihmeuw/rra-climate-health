import typing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd


printable_names = {
    'income_per_day_bin': 'Daily Income (2010 USD)',
    'temp_bin': 'Yearly Temperature (Mean)',
    'temp_bin_quants': 'Mean Yearly Temperature',
    'temp_avg_bin': 'Mean Temperature (5 year)',
    'precip_bin': 'Yearly Precipitation',
    'precip_avg_bin': 'Mean Yearly Precipitation (5 year avg)',
    'over30_bin': 'Days over 30° C',
    'over30_avg_bin': 'Days over 30° C (5 year avg)',
    'stunting': 'Stunting',
    'wasting': 'Wasting',
    'underweight': 'Underweight',
    'over30_avgperyear_bin': 'Lifetime Yearly Average Days Above 30°C',
    'over30_birth_bin': 'Days over 30 in year of birth',
    'temp_diff_birth_bin': 'temp diff birth',
    'bmi': 'BMI',
    'low_adult_bmi': 'Low adult BMI',
    'cgf_value': 'Stunting',
    'adjusted_pred': 'Controlling for Country Effects',
    'predict_nocountry': 'Prediction without country',
    'residual': 'residual',
    'grid_coef': 'dummy coefficient',
    'gdppc': 'GDP per capita',
    'ldi_pc_pd': 'Lag-distributed income per day',
    'ldi_pc_pd_bin': 'Lag-distributed income per day',
    'ldipc': 'Lag-distributed income per day',
}


def plot_heatmap(
    df: pd.DataFrame,
    temp_col: str,
    wealth_col: str = 'income_per_day_bin',
    country: str | None = None,
    year: int | None = None,
    margins: bool = False,
    filter: str | None = None,
    value_col: str = 'cgf_value',
    title_addin: str = '',
    file: str | None = None,
    pdf_handle: typing.Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    counts: bool = True,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    plot_df = df.copy()

    temp_value_col = temp_col.replace("_bin", "")
    temp_value_col = temp_value_col if temp_value_col in plot_df.columns else 'temp'
    wealth_value_col = wealth_col.replace("_bin", "")

    x_ticks = range(plot_df[temp_col].nunique() + 1)
    x_labs = plot_df.groupby(temp_col)[temp_value_col].min().apply(lambda x: f"{x:.1f}").values.tolist() + \
        [f"{df[temp_value_col].max():.1f}"]
    x_labs = [x for x in x_labs if x != 'nan']

    y_ticks = range(plot_df[wealth_col].nunique() + 1)
    y_labs = plot_df.groupby(wealth_col)[wealth_value_col].min().apply(lambda x: f"{x:.1f}").values.tolist() + \
        [f"{plot_df[wealth_value_col].max():.1f}"]
    y_labs = [x for x in y_labs if x != 'nan']

    if filter is not None:
        plot_df = plot_df.query(filter)
    if country:
        plot_df = plot_df[plot_df['iso3'] == country]
    if year:
        plot_df = plot_df[plot_df['year_start'] == year]

    #plot_df = plot_df.rename(columns = printable_names)

    #value_col = printable_names[value_col] if value_col in printable_names.keys() else value_col

    pivot_table_mean = plot_df.pivot_table(values=value_col, index=wealth_col,
        columns=temp_col, aggfunc='mean', dropna=False, margins=margins, observed=False)
    pivot_table_count = plot_df.pivot_table(values=value_col, index=wealth_col,
        columns=temp_col, aggfunc='count', dropna=False, margins=margins, observed=False)

    fig, ax = plt.subplots(figsize=figsize)

    if vmax:
        colorbin_interval = 0.025
        boundaries = np.arange(vmin, vmax + colorbin_interval, colorbin_interval)
        cmap = plt.get_cmap('RdYlBu_r', len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        sns.heatmap(pivot_table_mean, annot=True, fmt=".2f", annot_kws={"size": 14, "weight":'regular'}, #cmap='RdYlBu_r',
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, cbar_kws={"ticks": boundaries})
    else:
        sns.heatmap(pivot_table_mean, annot=True, fmt=".2f", cmap='RdYlBu_r')

    # Overlay the counts on the heatmap
    if counts:
        for i, row in enumerate(pivot_table_mean.values):
            for j, value in enumerate(row):
                plt.text(j+0.5, i+0.6, f'\n({pivot_table_count.values[i][j]})',
                        ha="center", va="center")

    label_naming = lambda x: printable_names[x] if x in printable_names.keys() else x

    ax.set_xlabel(label_naming(temp_col), fontsize=16)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labs, rotation=45)

    ax.set_ylabel(label_naming(wealth_col), fontsize=16)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labs)

    cbar = ax.collections[0].colorbar
    cbar.set_label(label_naming(value_col), size=16)
    ax.set_title(title_addin, fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=14)

    # plt.title((f'{printable_names[wealth_col]} x {printable_names[temp_col]} \n x {title_addin}{value_col}'
    # f'(Mean Proportion & Count) {country if country else ", All Locations"}'));
    if file:
        plt.savefig(file, dpi=300, bbox_inches="tight")
    if pdf_handle:
        pdf_handle.savefig(fig)
    else:
        plt.show()
    plt.close()
